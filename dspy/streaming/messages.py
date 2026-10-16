import asyncio
import concurrent.futures
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dspy.dsp.utils.settings import settings
from dspy.utils.callback import BaseCallback
from dspy.utils.lazy_import import require

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature

anyio = require("anyio")
logger = logging.getLogger(__name__)


@dataclass
class _StreamSignature:
    """The adapter's render signature, queued before that call's response chunks."""

    predict_id: int
    signature: type["Signature"]
    predict_call_id: str | None = None


@dataclass
class StreamResponse:
    """A field chunk, optionally associated with its Predict callback's call_id.

    ``predict_call_id`` distinguishes concurrent calls to the same predictor.
    It is None for responses constructed without a captured Predict call.
    """

    predict_name: str
    signature_field_name: str
    chunk: str
    is_last_chunk: bool
    predict_call_id: str | None = None


@dataclass
class StatusMessage:
    """Dataclass that wraps a status message for status streaming."""

    message: str


class _StreamDeliveryError(RuntimeError):
    """A synchronous owner-loop producer cannot wait for its own consumer."""


class _PredictStream:
    """Associate legacy chunks with the Predict whose context produced them."""

    def __init__(self, stream, predict_id, call_id):
        self.stream = stream
        self.predict_id = predict_id
        self.call_id = call_id

    def _tag(self, item):
        from dspy.clients.engines.streaming import EngineChunk

        cls = type(item)
        is_chunk = isinstance(item, (EngineChunk, _StreamSignature)) or (
            cls.__name__ == "ModelResponseStream" and cls.__module__.startswith("litellm")
        )
        if is_chunk and getattr(item, "predict_id", None) == self.predict_id:
            if getattr(item, "predict_call_id", None) is None:
                item.predict_call_id = self.call_id
        return item

    async def send(self, item):
        await self.stream.send(self._tag(item))

    def send_nowait(self, item):
        return self.stream.send_nowait(self._tag(item))

    def clone(self):
        return type(self)(self.stream.clone(), self.predict_id, self.call_id)

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def __enter__(self):
        self.stream.__enter__()
        return self

    def __exit__(self, *args):
        return self.stream.__exit__(*args)

    async def __aenter__(self):
        await self.stream.__aenter__()
        return self

    async def __aexit__(self, *args):
        return await self.stream.__aexit__(*args)


def sync_send_to_stream(stream, message):
    """Send message to stream in a sync context, regardless of event loop state."""

    async def _send():
        await stream.send(message)

    if settings.stream_portal is not None:
        # This portal belongs to this streamify invocation, not the worker or
        # a global loop. It also cancels blocked sends when the stream closes.
        if threading.get_ident() == settings.stream_owner_thread_id:
            try:
                return stream.send_nowait(message)
            except anyio.WouldBlock as exc:
                raise _StreamDeliveryError(
                    "A synchronous call on the streaming event loop filled its buffer. "
                    "Use the module/tool's acall() or run synchronous work in a worker thread."
                ) from exc
        return settings.stream_portal.call(_send)

    try:
        asyncio.get_running_loop()

        # If we're in an event loop, offload to a new thread with its own event loop
        def run_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(_send())
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_new_loop)
            return future.result()
    except RuntimeError:
        # Not in an event loop, safe to use a new event loop in this thread
        return anyio.from_thread.run(_send)


class StatusMessageProvider:
    """Provides customizable status message streaming for DSPy programs.

    This class serves as a base for creating custom status message providers. Users can subclass
    and override its methods to define specific status messages for different stages of program execution,
    each method must return a string.

    Examples:
    ```python
    class MyStatusMessageProvider(StatusMessageProvider):
        def lm_start_status_message(self, instance, inputs):
            return f"Calling LM with inputs {inputs}..."

        def module_end_status_message(self, outputs):
            return f"Module finished with output: {outputs}!"

    program = dspy.streamify(dspy.Predict("q->a"), status_message_provider=MyStatusMessageProvider())
    ```
    """

    def tool_start_status_message(self, instance: Any, inputs: dict[str, Any]):
        """Status message before a `dspy.Tool` is called."""
        return f"Calling tool {instance.name}..."

    def tool_end_status_message(self, outputs: Any):
        """Status message after a `dspy.Tool` is called."""
        return "Tool calling finished! Querying the LLM with tool calling results..."

    def module_start_status_message(self, instance: Any, inputs: dict[str, Any]):
        """Status message before a `dspy.Module` or `dspy.Predict` is called."""
        pass

    def module_end_status_message(self, outputs: Any):
        """Status message after a `dspy.Module` or `dspy.Predict` is called."""
        pass

    def lm_start_status_message(self, instance: Any, inputs: dict[str, Any]):
        """Status message before a `dspy.LM` is called."""
        pass

    def lm_end_status_message(self, outputs: Any):
        """Status message after a `dspy.LM` is called."""
        pass


class StatusStreamingCallback(BaseCallback):
    def __init__(self, status_message_provider: StatusMessageProvider | None = None):
        self.status_message_provider = status_message_provider or StatusMessageProvider()

    async def _acall(self, handler_name, **kwargs):
        """Internal async callback dispatch: await bounded sends on their owner loop.

        Public/user callbacks remain synchronous. Only inherited built-in
        handlers use this hook; overrides keep their existing dispatch.
        """
        stream = settings.send_stream
        if stream is None:
            return
        try:
            message = self._async_status_message(handler_name, **kwargs)
        except Exception as exc:
            # A user-supplied provider has the same warning policy as other
            # callbacks; it must not replace the program's original exception.
            logger.warning("Error when applying status message provider: %s", exc)
            return
        if message:
            # Delivery failures and cancellation are not provider failures.
            await stream.send(StatusMessage(message))

    def _async_status_message(self, handler_name, **kwargs):
        provider = self.status_message_provider
        if handler_name == "on_tool_start":
            if kwargs["instance"].name == "finish":
                return
            message = provider.tool_start_status_message(kwargs["instance"], kwargs["inputs"])
        elif handler_name == "on_tool_end":
            if kwargs["outputs"] == "Completed.":
                return
            message = provider.tool_end_status_message(kwargs["outputs"])
        elif handler_name in ("on_module_start", "on_lm_start"):
            method = getattr(provider, handler_name.removeprefix("on_") + "_status_message")
            message = method(kwargs["instance"], kwargs["inputs"])
        elif handler_name in ("on_module_end", "on_lm_end"):
            method = getattr(provider, handler_name.removeprefix("on_") + "_status_message")
            message = method(kwargs["outputs"])
        else:
            return
        return message

    def on_tool_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        stream = settings.send_stream
        if stream is None or instance.name == "finish":
            return

        status_message = self.status_message_provider.tool_start_status_message(instance, inputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))

    def on_tool_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: BaseException | None = None,
    ):
        stream = settings.send_stream
        if stream is None or outputs == "Completed.":
            return

        status_message = self.status_message_provider.tool_end_status_message(outputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))

    def on_lm_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        stream = settings.send_stream
        if stream is None:
            return

        status_message = self.status_message_provider.lm_start_status_message(instance, inputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))

    def on_lm_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: BaseException | None = None,
    ):
        stream = settings.send_stream
        if stream is None:
            return

        status_message = self.status_message_provider.lm_end_status_message(outputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))

    def on_module_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        stream = settings.send_stream
        if stream is None:
            return

        status_message = self.status_message_provider.module_start_status_message(instance, inputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))

    def on_module_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: BaseException | None = None,
    ):
        stream = settings.send_stream
        if stream is None:
            return

        status_message = self.status_message_provider.module_end_status_message(outputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))
