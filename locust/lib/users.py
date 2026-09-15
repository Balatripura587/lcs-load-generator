"""Locust user classes simulating LCS query and streaming endpoint traffic."""

import json
import logging
import random
import time

from locust import HttpUser, constant, events, task

from lib.config import LCS_MODEL, LCS_PROVIDER, LCS_TOKEN, REQUEST_TIMEOUT
from lib.metrics import record_bytes, record_stream_time, record_ttft
from lib.questions import QUESTIONS

logger = logging.getLogger("lcs.users")

# Responses API uses combined "provider/model" format
_RESPONSES_MODEL = f"{LCS_PROVIDER}/{LCS_MODEL}"


def _build_query_payload(conversation_id=None):
    """Build payload for /v1/query and /v1/streaming_query."""
    payload = {
        "query": random.choice(QUESTIONS),
        "provider": LCS_PROVIDER,
        "model": LCS_MODEL,
        "no_tools": True,
        "generate_topic_summary": False,
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id
    return payload


def _build_responses_payload(stream, previous_response_id=None):
    """Build payload for /v1/responses."""
    payload = {
        "input": random.choice(QUESTIONS),
        "model": _RESPONSES_MODEL,
        "stream": stream,
        "store": True,
    }
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id
    return payload


def _emit_sse_metrics(base_name, total_stream_ms, ttft, response_length=0):
    """Emit synthetic Locust events for full stream time and TTFT."""
    events.request.fire(
        request_type="SSE",
        name=f"{base_name} [full stream]",
        response_time=total_stream_ms,
        response_length=response_length,
        exception=None,
        context={"synthetic": True},
    )
    if ttft is not None:
        events.request.fire(
            request_type="SSE",
            name=f"{base_name} [TTFT]",
            response_time=ttft,
            response_length=0,
            exception=None,
            context={"synthetic": True},
        )


def _iter_sse_events(lines, on_line=None):
    """Yield (event type, parsed data) from an SSE response."""
    event_type = None
    data_lines = []

    def flush_event():
        if not data_lines:
            return None
        data = "\n".join(data_lines)
        if data == "[DONE]":
            return None
        try:
            return event_type or "message", json.loads(data)
        except json.JSONDecodeError:
            logger.debug("Ignoring invalid SSE data: %s", data)
            return None

    for line in lines:
        if on_line:
            on_line(line)
        text = line.decode() if isinstance(line, bytes) else line
        text = text.rstrip("\r")
        if not text:
            parsed = flush_event()
            if parsed is not None:
                yield parsed
            event_type = None
            data_lines = []
        elif text.startswith("event:"):
            event_type = text[6:].strip()
        elif text.startswith("data:"):
            data_lines.append(text[5:].lstrip())

    parsed = flush_event()
    if parsed is not None:
        yield parsed


class LCSBaseUser(HttpUser):
    """Base user with auth headers and zero wait time between requests."""

    abstract = True
    wait_time = constant(0)

    def on_start(self):
        self.conversation_id = None
        self.previous_response_id = None
        self.headers = {"Content-Type": "application/json"}
        if LCS_TOKEN:
            self.headers["Authorization"] = f"Bearer {LCS_TOKEN}"
        logger.debug("User started: provider=%s model=%s token=%s",
                      LCS_PROVIDER, LCS_MODEL, "set" if LCS_TOKEN else "none")


class LCSQueryClient(LCSBaseUser):
    """Simulated user sending POST /v1/query requests."""

    @task
    def query(self):
        payload = _build_query_payload(self.conversation_id)

        payload_bytes = len(json.dumps(payload).encode())

        with self.client.post(
            "/v1/query",
            json=payload,
            headers=self.headers,
            timeout=REQUEST_TIMEOUT,
            name="/v1/query",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    new_cid = data.get("conversation_id")
                    if new_cid and not self.conversation_id:
                        logger.debug("New conversation: %s", new_cid)
                    self.conversation_id = new_cid or self.conversation_id
                except Exception:
                    pass
                record_bytes(len(response.content or b""), payload_bytes)
                response.success()
            else:
                logger.debug("Query failed: status=%d", response.status_code)
                response.failure(f"Status {response.status_code}")


class LCSStreamingClient(LCSBaseUser):
    """Simulated user sending POST /v1/streaming_query SSE requests with TTFT tracking."""

    def on_start(self):
        super().on_start()
        self.headers["Accept"] = "text/event-stream"

    @task
    def streaming_query(self):
        payload = _build_query_payload(self.conversation_id)

        payload_bytes = len(json.dumps(payload).encode())
        start = time.perf_counter()
        ttft = None
        response_bytes = 0

        with self.client.post(
            "/v1/streaming_query",
            json=payload,
            headers=self.headers,
            stream=True,
            catch_response=True,
            timeout=REQUEST_TIMEOUT,
            name="/v1/streaming_query",
        ) as response:
            if response.status_code != 200:
                logger.debug("Streaming failed: status=%d", response.status_code)
                response.failure(f"Status {response.status_code}")
                return

            stream_failed = False

            def count_bytes(line):
                nonlocal response_bytes
                response_bytes += len(line) if isinstance(line, bytes) else len(line.encode())

            for event_type, event_data in _iter_sse_events(response.iter_lines(), count_bytes):
                if not isinstance(event_data, dict):
                    continue

                # /v1/streaming_query usually carries the event name in JSON body
                # as {"event": "...", "data": {...}} rather than SSE "event:" lines.
                payload_event = event_data.get("event") or event_type

                if payload_event == "token" and ttft is None:
                    ttft = (time.perf_counter() - start) * 1000
                    record_ttft(ttft)

                if payload_event == "error":
                    stream_failed = True

                if not self.conversation_id:
                    payload_data = event_data.get("data")
                    if isinstance(payload_data, dict):
                        cid = payload_data.get("conversation_id")
                        if cid:
                            self.conversation_id = cid

            total_stream_ms = (time.perf_counter() - start) * 1000
            logger.debug("Stream complete: ttft=%.1fms total=%.1fms bytes=%d",
                          ttft or 0, total_stream_ms, response_bytes)
            record_stream_time(total_stream_ms)
            record_bytes(response_bytes, payload_bytes)

            if stream_failed:
                response.failure("Streaming query reported a failure")
            else:
                response.success()

            _emit_sse_metrics(
                base_name="/v1/streaming_query",
                total_stream_ms=total_stream_ms,
                ttft=ttft,
                response_length=response_bytes,
            )


class LCSResponsesClient(LCSBaseUser):
    """Simulated user sending POST /v1/responses (non-streaming)."""

    @task
    def responses(self):
        payload = _build_responses_payload(False, self.previous_response_id)

        payload_bytes = len(json.dumps(payload).encode())

        with self.client.post(
            "/v1/responses",
            json=payload,
            headers=self.headers,
            timeout=REQUEST_TIMEOUT,
            name="/v1/responses",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    self.previous_response_id = data.get("id")
                except (TypeError, ValueError):
                    response.failure("Invalid JSON response")
                    return
                record_bytes(len(response.content or b""), payload_bytes)
                response.success()
            else:
                logger.debug("Responses failed: status=%d", response.status_code)
                response.failure(f"Status {response.status_code}")


class LCSStreamingResponsesClient(LCSBaseUser):
    """Simulated user sending POST /v1/responses with stream=True."""

    def on_start(self):
        super().on_start()
        self.headers["Accept"] = "text/event-stream"

    @task
    def streaming_responses(self):
        payload = _build_responses_payload(True, self.previous_response_id)

        payload_bytes = len(json.dumps(payload).encode())
        start = time.perf_counter()
        ttft = None
        response_bytes = 0
        stream_failed = False

        def count_bytes(line):
            nonlocal response_bytes
            response_bytes += len(line) if isinstance(line, bytes) else len(line.encode())

        with self.client.post(
            "/v1/responses",
            json=payload,
            headers=self.headers,
            stream=True,
            catch_response=True,
            timeout=REQUEST_TIMEOUT,
            name="/v1/responses [streaming]",
        ) as response:
            if response.status_code != 200:
                logger.debug("Streaming responses failed: status=%d", response.status_code)
                response.failure(f"Status {response.status_code}")
                return

            for event_type, event_data in _iter_sse_events(response.iter_lines(), count_bytes):
                if not isinstance(event_data, dict):
                    continue
                response_data = event_data.get("response", {})
                if event_type in {"response.created", "response.completed"}:
                    response_id = response_data.get("id")
                    if response_id:
                        self.previous_response_id = response_id

                # Primary TTFT trigger for responses streaming.
                if event_type == "response.output_text.delta" and ttft is None:
                    ttft = (time.perf_counter() - start) * 1000
                    record_ttft(ttft)

                # Fallback TTFT trigger for blocked/moderated responses that may
                # emit output items without output_text.delta events.
                if event_type == "response.output_item.added" and ttft is None:
                    item = event_data.get("item")
                    if isinstance(item, dict):
                        content = item.get("content")
                        if isinstance(content, list):
                            has_visible_text = any(
                                (
                                    isinstance(part, dict)
                                    and (
                                        (part.get("type") == "output_text" and bool(part.get("text")))
                                        or (part.get("type") == "refusal" and bool(part.get("refusal")))
                                    )
                                )
                                for part in content
                            )
                            if has_visible_text:
                                ttft = (time.perf_counter() - start) * 1000
                                record_ttft(ttft)

                if event_type in {
                    "error",
                    "response.error",
                    "response.failed",
                    "response.incomplete",
                    "response.cancelled",
                }:
                    stream_failed = True

            total_stream_ms = (time.perf_counter() - start) * 1000
            logger.debug("Responses stream complete: ttft=%.1fms total=%.1fms bytes=%d",
                         ttft or 0, total_stream_ms, response_bytes)
            record_stream_time(total_stream_ms)
            record_bytes(response_bytes, payload_bytes)

            if stream_failed:
                response.failure("Responses stream reported a failure")
            else:
                response.success()

            _emit_sse_metrics(
                base_name="/v1/responses",
                total_stream_ms=total_stream_ms,
                ttft=ttft,
                response_length=response_bytes,
            )