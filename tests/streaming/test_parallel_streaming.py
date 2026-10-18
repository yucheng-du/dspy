"""Exercise concurrent Predict calls through the public, offline engine interface."""

import asyncio
import threading
from collections import defaultdict
from typing import ClassVar

import anyio
import pytest

import dspy
from dspy import lm15
from dspy.streaming import StatusMessage, StatusMessageProvider, StreamListener, StreamResponse
from dspy.utils.callback import BaseCallback

HEADER = "[[ ## answer ## ]]\n\n"
END = "\n\n[[ ## completed ## ]]"
ANSWERS = {"A": "A-firstA-later", "B": "B-only"}


def question(request):
    content = request.messages[-1].text
    return content.split("[[ ## question ## ]]\n", 1)[1].split("\n", 1)[0].strip()


class InterleavedEngine:
    """A starts, B finishes, then A finishes; waits are bounded even on a broken baseline."""

    def __init__(self):
        self.a_started = threading.Event()
        self.b_finished = threading.Event()
        self.closed = set()

    def complete(self, request):
        text = HEADER + ANSWERS[question(request)] + END
        return lm15.Response(None, request.model, lm15.Message.assistant(text), "stop", lm15.Usage())

    def stream(self, request):
        name = question(request)
        try:
            if name == "B":
                assert self.a_started.wait(5), "A did not reach its first chunk"
            yield lm15.StreamStartEvent(model=request.model)
            yield lm15.StreamDeltaEvent(lm15.TextDelta(HEADER))
            yield lm15.StreamDeltaEvent(lm15.TextDelta("A-first" if name == "A" else "B-only"))
            if name == "A":
                self.a_started.set()
                assert self.b_finished.wait(5), "B did not finish"
                yield lm15.StreamDeltaEvent(lm15.TextDelta("A-later"))
            yield lm15.StreamDeltaEvent(lm15.TextDelta(END))
            yield lm15.StreamEndEvent(finish_reason="stop", usage=lm15.Usage())
        finally:
            self.closed.add(name)
            if name == "B":
                self.b_finished.set()


class AsyncInterleavedEngine:
    def __init__(self):
        self.a_started = asyncio.Event()
        self.b_finished = asyncio.Event()
        self.closed = set()

    async def complete(self, request):
        return InterleavedEngine().complete(request)

    async def stream(self, request):
        name = question(request)
        try:
            if name == "B":
                await asyncio.wait_for(self.a_started.wait(), 5)
            yield lm15.StreamStartEvent(model=request.model)
            yield lm15.StreamDeltaEvent(lm15.TextDelta(HEADER))
            yield lm15.StreamDeltaEvent(lm15.TextDelta("A-first" if name == "A" else "B-only"))
            if name == "A":
                self.a_started.set()
                await asyncio.wait_for(self.b_finished.wait(), 5)
                yield lm15.StreamDeltaEvent(lm15.TextDelta("A-later"))
            yield lm15.StreamDeltaEvent(lm15.TextDelta(END))
            yield lm15.StreamEndEvent(finish_reason="stop", usage=lm15.Usage())
        finally:
            self.closed.add(name)
            if name == "B":
                self.b_finished.set()


class CallTrace(BaseCallback):
    def __init__(self):
        self.predict_ids = {}
        self.lm_ids = set()

    def on_module_start(self, call_id, instance, inputs):
        if isinstance(instance, dspy.Predict):
            self.predict_ids[inputs["kwargs"]["question"]] = call_id

    def on_lm_start(self, call_id, instance, inputs):
        self.lm_ids.add(call_id)


class Statuses(StatusMessageProvider):
    def module_start_status_message(self, instance, inputs):
        if isinstance(instance, dspy.Predict):
            return "start " + inputs["kwargs"]["question"]

    def module_end_status_message(self, outputs):
        if isinstance(outputs, dspy.Prediction) and "answer" in outputs:
            return "end " + outputs.answer


class ParallelProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("question -> answer")

    def forward(self):
        pairs = [(self.predict, {"question": name}) for name in ANSWERS]
        results = dspy.Parallel(num_threads=2, disable_progress_bar=True, timeout=10)(pairs)
        return dspy.Prediction(results=results)

    async def aforward(self):
        results = await asyncio.gather(*(self.predict.acall(question=name) for name in ANSWERS))
        return dspy.Prediction(results=results)


@pytest.mark.anyio
@pytest.mark.parametrize("is_async", [False, True], ids=["parallel", "gather"])
async def test_same_predict_interleaved_calls(is_async, caplog):
    sync_engine, async_engine = InterleavedEngine(), AsyncInterleavedEngine()
    trace = CallTrace()
    lm = dspy.LM("custom/offline", engine=sync_engine, async_engine=async_engine, cache=False, num_retries=0)
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter(), callbacks=[trace]):
        program = dspy.streamify(
            ParallelProgram(),
            stream_listeners=[StreamListener("answer", allow_reuse=True)],
            status_message_provider=Statuses(),
            is_async_program=is_async,
        )
        with anyio.fail_after(15):
            values = [value async for value in program()]
    predictions = [value for value in values if isinstance(value, dspy.Prediction)]
    assert len(predictions) == 1
    assert all(result is not None for result in predictions[0].results)
    assert [result.answer for result in predictions[0].results] == list(ANSWERS.values())
    chunks = [value for value in values if isinstance(value, StreamResponse)]
    # Check existing content/end semantics before testing the new association field.
    assert sum(chunk.is_last_chunk for chunk in chunks) == 2
    assert sorted("".join(chunk.chunk for chunk in chunks)) == sorted("".join(ANSWERS.values()))
    by_call = defaultdict(list)
    for chunk in chunks:
        by_call[chunk.predict_call_id].append(chunk)
    assert set(by_call) == set(trace.predict_ids.values())
    assert len(by_call) == 2
    assert not (set(by_call) & trace.lm_ids)
    for name, call_id in trace.predict_ids.items():
        assert "".join(chunk.chunk for chunk in by_call[call_id]) == ANSWERS[name]
        assert sum(chunk.is_last_chunk for chunk in by_call[call_id]) == 1
    assert sorted(value.message for value in values if isinstance(value, StatusMessage)) == sorted(
        ["start A", "start B", "end A-firstA-later", "end B-only"]
    )
    assert "AnyIO worker thread" not in caplog.text
    assert (async_engine if is_async else sync_engine).closed == {"A", "B"}


class SerialEngine:
    def __init__(self):
        self.completed = []
        self.streamed = []

    def complete(self, request):
        self.completed.append(question(request))
        return InterleavedEngine().complete(request)

    def stream(self, request):
        name = question(request)
        self.streamed.append(name)
        yield lm15.StreamStartEvent(model=request.model)
        for text in (HEADER, ANSWERS[name], END):
            yield lm15.StreamDeltaEvent(lm15.TextDelta(text))
        yield lm15.StreamEndEvent(finish_reason="stop", usage=lm15.Usage())


class SerialProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("question -> answer")

    def forward(self):
        return dspy.Prediction(results=[self.predict(question=name) for name in ANSWERS])


@pytest.mark.anyio
@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("cached_first", [False, True])
async def test_serial_calls_and_cached_signature_do_not_claim_listener(reuse, cached_first):
    engine = SerialEngine()
    lm = dspy.LM("custom/offline", engine=engine, cache=cached_first, num_retries=0)
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        module = SerialProgram()
        if cached_first:
            assert module.predict(question="A").answer == ANSWERS["A"]
        stream = dspy.streamify(module, stream_listeners=[StreamListener("answer", allow_reuse=reuse)])
        with anyio.fail_after(10):
            values = [value async for value in stream()]
    chunks = [value for value in values if isinstance(value, StreamResponse)]
    expected = ANSWERS["B"] if cached_first else "".join(ANSWERS.values()) if reuse else ANSWERS["A"]
    assert "".join(chunk.chunk for chunk in chunks) == expected
    assert sum(chunk.is_last_chunk for chunk in chunks) == (2 if reuse and not cached_first else 1)
    assert [result.answer for result in values[-1].results] == list(ANSWERS.values())
    assert engine.completed == (["A"] if cached_first else [])
    assert engine.streamed == (["B"] if cached_first else ["A", "B"])


def test_stream_response_preserves_four_positional_arguments():
    response = StreamResponse("p", "answer", "hello", True)
    assert response.chunk == "hello"
    assert getattr(response, "predict_call_id", None) is None


class CustomListener(StreamListener):
    copies: ClassVar[list] = []

    def __init__(self, required_configuration, **kwargs):
        super().__init__("answer", **kwargs)
        self.configuration = required_configuration
        self.seen = []

    def __copy__(self):
        clone = type(self).__new__(type(self))
        clone.__dict__ = self.__dict__.copy()
        clone.seen = []
        self.copies.append(clone)
        return clone

    def receive(self, chunk):
        self.seen.append(chunk)
        result = super().receive(chunk)
        if result is not None:
            result.chunk = self.configuration + result.chunk
        return result


@pytest.mark.anyio
async def test_custom_listener_keeps_type_constructor_configuration_and_copy_protocol():
    import copy

    CustomListener.copies = []
    template = CustomListener("!", allow_reuse=True)
    identifiers = copy.deepcopy(template.adapter_identifiers)
    engine = SerialEngine()
    with dspy.context(lm=dspy.LM("custom/offline", engine=engine, cache=False), adapter=dspy.ChatAdapter()):
        module = SerialProgram()
        stream = dspy.streamify(module, stream_listeners=[template])
        with anyio.fail_after(10):
            values = [value async for value in stream()]
    assert len(CustomListener.copies) == 2
    assert all(type(item) is CustomListener and item.predict is module.predict for item in CustomListener.copies)
    assert all(item.seen for item in CustomListener.copies)
    assert CustomListener.copies[0].seen is not CustomListener.copies[1].seen
    assert not template.seen and not template.stream_start
    assert template.adapter_identifiers == identifiers
    assert all(value.chunk.startswith("!") for value in values if isinstance(value, StreamResponse))


@pytest.mark.anyio
async def test_overlapping_invocations_of_one_streamer_have_separate_listener_state():
    backend = AsyncInterleavedEngine()
    trace = CallTrace()
    lm = dspy.LM("custom/offline", engine=InterleavedEngine(), async_engine=backend, cache=False)
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter(), callbacks=[trace]):
        stream = dspy.streamify(
            dspy.Predict("question -> answer"), is_async_program=True, stream_listeners=[StreamListener("answer")]
        )

        async def collect(name):
            return [value async for value in stream(question=name)]

        with anyio.fail_after(10):
            results = await asyncio.gather(collect("A"), collect("B"))
    for name, values in zip(ANSWERS, results, strict=True):
        chunks = [value for value in values if isinstance(value, StreamResponse)]
        assert "".join(chunk.chunk for chunk in chunks) == ANSWERS[name]
        assert {chunk.predict_call_id for chunk in chunks} == {trace.predict_ids[name]}
        assert sum(chunk.is_last_chunk for chunk in chunks) == 1


class ToolProgram(dspy.Module):
    def __init__(self, count, *, inline=False, callbacks=None):
        super().__init__()
        self.count = count
        self.inline = inline
        self.cleaned = False
        self.tool = dspy.Tool(lambda value: value, name="echo")
        self.callbacks = callbacks or []

    async def aforward(self):
        try:
            for value in range(self.count):
                if self.inline:
                    self.tool(value=value)
                else:
                    await self.tool.acall(value=value)
            return dspy.Prediction(answer="done")
        finally:
            self.cleaned = True


class EndTrace(BaseCallback):
    def __init__(self):
        self.starts = []
        self.ends = []
        self.module_end_parents = []

    def on_module_end(self, call_id, outputs, exception):
        from dspy.utils.callback import ACTIVE_CALL_ID

        self.module_end_parents.append(ACTIVE_CALL_ID.get())

    def on_tool_start(self, call_id, instance, inputs):
        self.starts.append(call_id)

    def on_tool_end(self, call_id, outputs, exception):
        self.ends.append((call_id, exception))


@pytest.mark.anyio
@pytest.mark.parametrize("count", [2, 30])
async def test_inline_sync_tool_on_owner_loop_has_bounded_delivery_and_cleanup(count):
    trace = EndTrace()
    module = ToolProgram(count, inline=True)
    from dspy.streaming.messages import StatusStreamingCallback

    with dspy.context(callbacks=[StatusStreamingCallback(), trace]):
        stream = dspy.streamify(module, is_async_program=True)
    with anyio.fail_after(10):
        if count == 2:
            values = [value async for value in stream()]
            assert len([value for value in values if isinstance(value, StatusMessage)]) == 4
            assert values[-1].answer == "done"
        else:
            with pytest.raises(RuntimeError, match="streaming event loop filled its buffer"):
                _ = [value async for value in stream()]
    assert module.cleaned
    # If a start status cannot be sent the tool never starts; every entered
    # tool still receives its later user cleanup, even if its end status fails.
    assert set(trace.starts) == {call_id for call_id, _ in trace.ends}


@pytest.mark.anyio
async def test_inline_sync_tool_start_overflow_cleans_up_callbacks_that_started():
    from dspy.utils.callback import ACTIVE_CALL_ID

    trace = EndTrace()
    module = ToolProgram(30, inline=True)
    ended_parents, program_ids = [], []

    class ParentTrace(BaseCallback):
        def on_module_start(self, call_id, instance, inputs):
            if instance is module:
                program_ids.append(call_id)

        def on_tool_end(self, call_id, outputs, exception):
            if exception is not None:
                ended_parents.append(ACTIVE_CALL_ID.get())
                # Cleanup failure must not skip later cleanup or replace the delivery error.
                raise RuntimeError("cleanup failed after start delivery error")

    # streamify appends its status callback after user callbacks, so the user
    # starts run before the status start that overflows the buffer.
    with dspy.context(callbacks=[ParentTrace(), trace]):
        stream = dspy.streamify(module, is_async_program=True)
    with anyio.fail_after(10):
        with pytest.raises(RuntimeError, match="streaming event loop filled its buffer") as error:
            _ = [value async for value in stream()]
    assert module.cleaned
    # Eight tools fill the 16-message buffer; the ninth fails before its body runs.
    assert len(trace.starts) == len(trace.ends) == 9
    assert set(trace.starts) == {call_id for call_id, _ in trace.ends}
    assert trace.ends[-1][1] is error.value
    assert len(ended_parents) == 1 and ended_parents == program_ids


class CountingStatuses(StatusMessageProvider):
    def __init__(self):
        self.produced = 0
        self.blocked_end = asyncio.Event()
        self.stream = None

    def _next(self):
        self.produced += 1
        self.stream = dspy.settings.send_stream
        if self.produced == 18:
            self.blocked_end.set()
        return str(self.produced)

    def tool_start_status_message(self, instance, inputs):
        return self._next()

    def tool_end_status_message(self, outputs):
        return self._next()


@pytest.mark.anyio
@pytest.mark.parametrize("cancel", [False, True])
async def test_async_status_backpressure_and_cancelled_end_preserve_user_cleanup(cancel):
    from contextlib import aclosing

    from dspy.streaming.messages import StatusStreamingCallback
    from dspy.utils.callback import ACTIVE_CALL_ID

    class InheritedStatus(StatusStreamingCallback):
        pass

    provider = CountingStatuses()
    trace = EndTrace()
    module = ToolProgram(30)
    values = []
    consumer_ready = asyncio.Event()
    resume = asyncio.Event()

    async def consume(stream):
        token = ACTIVE_CALL_ID.set("outside")
        try:
            async with aclosing(stream()) as output:
                async for value in output:
                    values.append(value)
                    if len(values) == 1:
                        consumer_ready.set()
                        await resume.wait()
        finally:
            assert ACTIVE_CALL_ID.get() == "outside"
            ACTIVE_CALL_ID.reset(token)

    with dspy.context(callbacks=[InheritedStatus(provider), trace]):
        stream = dspy.streamify(module, is_async_program=True)
        with anyio.fail_after(10):
            task = asyncio.create_task(consume(stream))
            try:
                await consumer_ready.wait()
                await provider.blocked_end.wait()
                await anyio.sleep(0)
                assert provider.produced == 18
                assert provider.stream.statistics().current_buffer_used == 16
                assert provider.stream.statistics().tasks_waiting_send == 1
                if cancel:
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                else:
                    resume.set()
                    await task
            finally:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
    assert module.cleaned
    assert set(trace.starts) == {call_id for call_id, _ in trace.ends}
    assert trace.module_end_parents == ["outside"]
    if cancel:
        assert len(trace.starts) == len(trace.ends) == 9
    if not cancel:
        assert [item.message for item in values if isinstance(item, StatusMessage)] == [str(i) for i in range(1, 61)]
        assert values[-1].answer == "done"


@pytest.mark.anyio
async def test_user_callback_and_status_provider_errors_keep_warning_policy(caplog):
    class BrokenUser(BaseCallback):
        def on_tool_end(self, **kwargs):
            raise ValueError("user callback failed")

    class BrokenProvider(StatusMessageProvider):
        def tool_end_status_message(self, outputs):
            raise ValueError("status provider failed")

    module = ToolProgram(2)
    with dspy.context(callbacks=[BrokenUser()]):
        stream = dspy.streamify(module, is_async_program=True, status_message_provider=BrokenProvider())
    with anyio.fail_after(10):
        values = [value async for value in stream()]
    assert values[-1].answer == "done" and module.cleaned
    assert "user callback failed" in caplog.text and "status provider failed" in caplog.text


@pytest.mark.anyio
async def test_legacy_litellm_parallel_transport_uses_this_streams_owner_loop():
    import json
    from http.server import BaseHTTPRequestHandler, HTTPServer

    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            name = payload["messages"][-1]["content"].split("[[ ## question ## ]]\n", 1)[1].split("\n", 1)[0].strip()
            received.append(name)
            events = []
            for content in (HEADER, ANSWERS[name], END):
                events.append(
                    {
                        "id": "local-" + name,
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "local-test",
                        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                    }
                )
            events.append(
                {
                    "id": "local-" + name,
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "local-test",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
            body = ("".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    trace = CallTrace()
    try:
        lm = dspy.LM(
            "openai/local-test",
            engine="litellm",
            cache=False,
            num_retries=0,
            api_base=f"http://127.0.0.1:{server.server_port}/v1",
            api_key="local-only",
            timeout=5,
        )
        with dspy.context(lm=lm, adapter=dspy.ChatAdapter(), callbacks=[trace]):
            stream = dspy.streamify(ParallelProgram(), stream_listeners=[StreamListener("answer", allow_reuse=True)])
            with anyio.fail_after(15):
                values = [value async for value in stream()]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
    assert not thread.is_alive()
    assert sorted(received) == ["A", "B"]
    assert [result.answer for result in values[-1].results] == list(ANSWERS.values())
    for name, call_id in trace.predict_ids.items():
        chunks = [item for item in values if isinstance(item, StreamResponse) and item.predict_call_id == call_id]
        assert "".join(item.chunk for item in chunks) == ANSWERS[name]
        assert sum(item.is_last_chunk for item in chunks) == 1


@pytest.mark.anyio
async def test_legacy_custom_lm_with_only_predict_id_keeps_processed_cot_signature():
    from litellm import ModelResponse, ModelResponseStream
    from litellm.types.utils import Delta, StreamingChoices

    from dspy.streaming.messages import sync_send_to_stream

    text = "[[ ## plan ## ]]\nLocal reasoning.\n\n[[ ## answer ## ]]\n17" + END

    class LegacyLM(dspy.BaseLM):
        def forward(self, prompt=None, messages=None, **kwargs):
            for offset in range(0, len(text), 4):
                chunk = ModelResponseStream(choices=[StreamingChoices(delta=Delta(content=text[offset : offset + 4]))])
                chunk.predict_id = id(dspy.settings.caller_predict)
                # Existing plugins know only predict_id, not the new optional ID.
                sync_send_to_stream(dspy.settings.send_stream, chunk)
            return ModelResponse(
                model=self.model,
                choices=[{"message": {"role": "assistant", "content": text}, "finish_reason": "stop", "index": 0}],
            )

    lm = LegacyLM("openai/gpt-4o-mini", cache=False)
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        stream = dspy.streamify(
            dspy.ChainOfThought("question -> answer"), stream_listeners=[StreamListener("reasoning")]
        )
        with anyio.fail_after(10):
            values = [value async for value in stream(question="A")]
    chunks = [item for item in values if isinstance(item, StreamResponse)]
    assert "".join(item.chunk for item in chunks) == "Local reasoning."
    assert all(item.signature_field_name == "reasoning" for item in chunks)
    assert values[-1].reasoning == "Local reasoning." and values[-1].answer == "17"


@pytest.mark.anyio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_stream_failure_preserves_public_error_and_closes_provider(asynchronous):
    closed = []

    class Failing(SerialEngine):
        def stream(self, request):
            try:
                yield lm15.StreamStartEvent(model=request.model)
                yield lm15.StreamDeltaEvent(lm15.TextDelta(HEADER + "partial"))
                raise lm15.AuthError("local failure", provider="offline")
            finally:
                closed.append(True)

    class AsyncFailing:
        async def complete(self, request):
            return Failing().complete(request)

        async def stream(self, request):
            for event in Failing().stream(request):
                yield event

    class BrokenEnd(StatusMessageProvider):
        def module_end_status_message(self, outputs):
            raise ValueError("end provider must not replace the LM error")

    lm = dspy.LM("custom/offline", engine=Failing(), async_engine=AsyncFailing(), cache=False, num_retries=0)
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        stream = dspy.streamify(
            dspy.Predict("question -> answer"),
            is_async_program=asynchronous,
            status_message_provider=BrokenEnd(),
            stream_listeners=[StreamListener("answer")],
        )
        with anyio.fail_after(10), pytest.raises(dspy.LMAuthError, match="local failure"):
            _ = [value async for value in stream(question="A")]
    assert closed == [True]


@pytest.mark.anyio
async def test_cancelled_parallel_stream_unblocks_worker_sends_and_closes_engines():
    from contextlib import aclosing

    class FloodEngine(SerialEngine):
        def __init__(self):
            super().__init__()
            self.entered = threading.Barrier(2, timeout=5)
            self.closed = set()
            self.streams = []

        def stream(self, request):
            name = question(request)
            try:
                self.streams.append(dspy.settings.send_stream)
                self.entered.wait()
                yield lm15.StreamStartEvent(model=request.model)
                yield lm15.StreamDeltaEvent(lm15.TextDelta(HEADER))
                for _ in range(1000):
                    yield lm15.StreamDeltaEvent(lm15.TextDelta(name + "-content "))
                yield lm15.StreamDeltaEvent(lm15.TextDelta(END))
                yield lm15.StreamEndEvent(finish_reason="stop", usage=lm15.Usage())
            finally:
                self.closed.add(name)

    engine = FloodEngine()
    ready = asyncio.Event()
    hold = asyncio.Event()

    async def consume(stream):
        async with aclosing(stream()) as output:
            async for value in output:
                if isinstance(value, StreamResponse):
                    ready.set()
                    await hold.wait()

    lm = dspy.LM("custom/offline", engine=engine, cache=False, num_retries=0)
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        stream = dspy.streamify(ParallelProgram(), stream_listeners=[StreamListener("answer", allow_reuse=True)])
        with anyio.fail_after(10):
            task = asyncio.create_task(consume(stream))
            try:
                await ready.wait()
                while len(engine.streams) != 2 or engine.streams[0].statistics().tasks_waiting_send < 2:
                    await anyio.sleep(0.001)
                assert engine.streams[0].statistics().current_buffer_used == 16
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                while engine.closed != {"A", "B"}:
                    await anyio.sleep(0.001)
                assert engine.streams[0].statistics().tasks_waiting_send == 0
            finally:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass


@pytest.mark.anyio
@pytest.mark.parametrize("listen", [False, True])
async def test_distinct_predictors_and_raw_engine_chunks_keep_their_routes(listen):
    from dspy.clients.engines.streaming import EngineChunk

    class DistinctProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.first = dspy.Predict("question -> answer")
            self.second = dspy.Predict("question -> answer")

        def forward(self):
            results = dspy.Parallel(num_threads=2, disable_progress_bar=True, timeout=10)(
                [(self.first, {"question": "A"}), (self.second, {"question": "B"})]
            )
            return dspy.Prediction(results=results)

    engine = SerialEngine()
    trace = CallTrace()
    lm = dspy.LM("custom/offline", engine=engine, cache=False, num_retries=0)
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter(), callbacks=[trace]):
        module = DistinctProgram()
        listeners = [StreamListener("answer", predict=module.first), StreamListener("answer", predict=module.second)]
        stream = dspy.streamify(module, stream_listeners=listeners if listen else None)
        with anyio.fail_after(10):
            values = [value async for value in stream()]
    assert all(result is not None for result in values[-1].results)
    assert [result.answer for result in values[-1].results] == list(ANSWERS.values())
    for name, call_id in trace.predict_ids.items():
        chunks = [
            item
            for item in values
            if isinstance(item, StreamResponse if listen else EngineChunk) and item.predict_call_id == call_id
        ]
        if listen:
            assert "".join(item.chunk for item in chunks) == ANSWERS[name]
            assert sum(item.is_last_chunk for item in chunks) == 1
        else:
            assert "".join(item.choices[0].delta.content or "" for item in chunks) == HEADER + ANSWERS[name] + END
            assert sum(item.choices[0].finish_reason == "stop" for item in chunks) == 1


@pytest.mark.anyio
async def test_cancelled_start_status_cleans_up_only_callbacks_that_started():
    from contextlib import aclosing

    from dspy.streaming.messages import StatusStreamingCallback
    from dspy.utils.callback import ACTIVE_CALL_ID

    class StartOnly(CountingStatuses):
        def tool_end_status_message(self, outputs):
            return None

    provider = StartOnly()
    trace, later_trace = EndTrace(), EndTrace()
    module = ToolProgram(30)
    ready, hold = asyncio.Event(), asyncio.Event()
    ended_parents, program_ids = [], []

    class ParentTrace(BaseCallback):
        def on_module_start(self, call_id, instance, inputs):
            if instance is module:
                program_ids.append(call_id)

        def on_tool_end(self, call_id, outputs, exception):
            if exception is not None:
                ended_parents.append(ACTIVE_CALL_ID.get())
                # Cleanup failure must not replace the original cancellation.
                raise RuntimeError("cleanup failed after start cancellation")

    async def consume(stream):
        token = ACTIVE_CALL_ID.set("outside")
        try:
            async with aclosing(stream()) as output:
                async for _ in output:
                    ready.set()
                    await hold.wait()
        finally:
            assert ACTIVE_CALL_ID.get() == "outside"
            ACTIVE_CALL_ID.reset(token)

    # The trace precedes status, as in streamify's normal automatic callback
    # installation. A later trace also checks that an unstarted callback does
    # not receive a fabricated end notification for the cancelled call.
    with dspy.context(callbacks=[trace, ParentTrace(), StatusStreamingCallback(provider), later_trace]):
        stream = dspy.streamify(module, is_async_program=True)
        with anyio.fail_after(10):
            task = asyncio.create_task(consume(stream))
            try:
                await ready.wait()
                await provider.blocked_end.wait()
                await anyio.sleep(0)
                assert provider.produced == 18
                assert provider.stream.statistics().current_buffer_used == 16
                assert provider.stream.statistics().tasks_waiting_send == 1
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            finally:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
    assert module.cleaned
    assert len(trace.starts) == len(trace.ends) == 18
    assert set(trace.starts) == {call_id for call_id, _ in trace.ends}
    assert isinstance(trace.ends[-1][1], asyncio.CancelledError)
    assert len(later_trace.starts) == len(later_trace.ends) == 17
    assert set(later_trace.starts) == {call_id for call_id, _ in later_trace.ends}
    assert len(ended_parents) == 1 and ended_parents == program_ids
    assert trace.module_end_parents == ["outside"]


@pytest.mark.anyio
async def test_async_callback_inputs_preserve_a_user_parameter_named_instance():
    from dspy.utils.callback import with_callbacks

    captured = []

    class Trace(BaseCallback):
        def on_module_start(self, call_id, instance, inputs):
            captured.append(inputs)

    class Operation:
        @with_callbacks
        async def run(self, instance):
            return instance

    payload = object()
    with dspy.context(callbacks=[Trace()]):
        assert await Operation().run(payload) is payload
    assert captured == [{"instance": payload}]
