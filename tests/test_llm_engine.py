import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import tests.fakes  # noqa: F401 - puts the repo root on sys.path

import intelligence
from intelligence import LlamaDecisionEngine, extract_openai_text, strip_reasoning


class ParsingTest(unittest.TestCase):
    def test_chat_completion_message_content(self):
        data = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "BTC: 0.42\nETH: -0.10"}}]}
        self.assertEqual(extract_openai_text(data), "BTC: 0.42\nETH: -0.10")

    def test_reasoning_preamble_is_dropped(self):
        data = {"choices": [{"message": {"content": "<think>BTC looks like 0.9? no…</think>\n\nBTC: 0.30"}}]}
        self.assertEqual(extract_openai_text(data), "BTC: 0.30")

    def test_content_parts_form(self):
        data = {"choices": [{"message": {"content": [{"type": "text", "text": "SOL: "}, {"type": "text", "text": "0.5"}]}}]}
        self.assertEqual(extract_openai_text(data), "SOL: 0.5")

    def test_legacy_completions_text(self):
        self.assertEqual(extract_openai_text({"choices": [{"text": " ADA: 0.1 "}]}), "ADA: 0.1")

    def test_reasoning_only_answer_is_empty_not_the_reasoning(self):
        data = {"choices": [{"message": {"content": None, "reasoning_content": "BTC: 0.99 maybe"}}]}
        self.assertEqual(extract_openai_text(data), "")

    def test_malformed_responses_are_none(self):
        for data in (None, [], {}, {"choices": []}, {"choices": ["x"]}, {"error": {"message": "loading model"}},
                     {"choices": [{"finish_reason": "stop"}]}):
            with self.subTest(data=data):
                self.assertIsNone(extract_openai_text(data))

    def test_strip_reasoning_edge_cases(self):
        self.assertEqual(strip_reasoning(None), "")
        self.assertEqual(strip_reasoning("<think>still thinking about BTC: 0.8"), "")  # budget ran out mid-thought
        self.assertEqual(strip_reasoning("weighing it…</think>BTC: 0.2"), "BTC: 0.2")  # opening tag was in the prompt
        self.assertEqual(strip_reasoning("<THINK>a</THINK><think>b</think> ETH: 0.1"), "ETH: 0.1")
        self.assertEqual(strip_reasoning("BTC: 0.2"), "BTC: 0.2")

    def test_unknown_style_is_refused_with_a_sentence(self):
        with self.assertRaises(ValueError) as ctx:
            LlamaDecisionEngine(model_name="m", base_url="http://127.0.0.1:1", api_style="anthropic")
        self.assertIn("LLAMA_API_STYLE", str(ctx.exception))


class _Stub(BaseHTTPRequestHandler):
    """A model server: records requests; answers from server.responses (a list of (status, body))."""

    def log_message(self, *a):
        pass

    def _answer(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        self.server.requests.append({
            "method": self.command, "path": self.path, "headers": dict(self.headers),
            "body": json.loads(raw) if raw else None,
        })
        if self.path.endswith("/models") or self.path.endswith("/tags"):
            status, body = 200, {"data": [{"id": "local"}]}
        else:
            status, body = self.server.responses.pop(0) if self.server.responses else (500, {"error": "no answer queued"})
        payload = json.dumps(body).encode() if not isinstance(body, bytes) else body
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = _answer
    do_POST = _answer


class EngineWireTest(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
        self.server.requests = []
        self.server.responses = []
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]
        # No real sleeping between retries.
        patcher = mock.patch.object(intelligence.time, "sleep", lambda s: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def engine(self, style="openai", **kw):
        base = f"http://127.0.0.1:{self.port}/v1" if style == "openai" else f"http://127.0.0.1:{self.port}/api"
        kw.setdefault("max_retries", 3)
        return LlamaDecisionEngine(model_name="local", base_url=base, api_style=style,
                                   options={"num_predict": 32, "temperature": 0.7, "top_p": 0.9, "raw": True, "mirostat": 0},
                                   **kw)

    def both_transports(self, make):
        """Run a check with requests (when installed) and with the urllib fallback."""
        engines = [make()]
        fallback = make()
        fallback._session = None
        engines.append(fallback)
        return engines if engines[0]._session is not None else engines[1:]

    def test_openai_request_shape_and_answer(self):
        for eng in self.both_transports(lambda: self.engine(system_prompt="You rate coins.", api_key="sk-test-0123456789")):
            with self.subTest(transport="requests" if eng._session is not None else "urllib"):
                self.server.requests.clear()
                self.server.responses.append((200, {"choices": [{"message": {"content": "<think></think>BTC: 0.25"}}]}))
                self.assertEqual(eng.generate("NOW RATE THESE COINS:\nBTC"), "BTC: 0.25")
                req = self.server.requests[-1]
                self.assertEqual(req["path"], "/v1/chat/completions")
                body = req["body"]
                self.assertEqual(body["model"], "local")
                self.assertEqual(body["messages"], [
                    {"role": "system", "content": "You rate coins."},
                    {"role": "user", "content": "NOW RATE THESE COINS:\nBTC"},
                ])
                self.assertEqual(body["max_tokens"], 32)
                self.assertEqual(body["temperature"], 0.7)
                self.assertFalse(body["stream"])
                self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
                self.assertNotIn("raw", body)
                self.assertNotIn("mirostat", body)
                self.assertEqual(req["headers"].get("Authorization"), "Bearer sk-test-0123456789")

    def test_thinking_can_be_left_on(self):
        eng = self.engine(disable_thinking=False)
        self.server.responses.append((200, {"choices": [{"message": {"content": "ETH: 0.1"}}]}))
        eng.generate("x")
        self.assertNotIn("chat_template_kwargs", self.server.requests[-1]["body"])
        self.assertNotIn("Authorization", self.server.requests[-1]["headers"])

    def test_ollama_style_is_unchanged(self):
        eng = self.engine(style="ollama")
        self.server.responses.append((200, {"response": "ETH: -0.20", "done": True}))
        self.assertEqual(eng.generate("rate ETH"), "ETH: -0.20")
        req = self.server.requests[-1]
        self.assertEqual(req["path"], "/api/generate")
        self.assertEqual(req["body"]["prompt"], "rate ETH")
        self.assertEqual(req["body"]["options"]["num_predict"], 32)
        self.assertIn("keep_alive", req["body"])

    def test_non_200_answers_are_retried_and_then_give_up(self):
        # Regression: a non-200 used to loop forever without counting as an attempt.
        for eng in self.both_transports(lambda: self.engine(max_retries=2)):
            with self.subTest(transport="requests" if eng._session is not None else "urllib"):
                self.server.requests.clear()
                self.server.responses[:] = [(503, {"error": "Loading model"}), (503, {"error": "Loading model"})]
                self.assertIsNone(eng.generate("x"))
                self.assertEqual(len(self.server.requests), 2)
                self.assertIn("HTTP 503", eng.last_error)

    def test_a_retry_that_succeeds_clears_the_error(self):
        eng = self.engine(max_retries=3)
        self.server.responses[:] = [(500, {"error": "busy"}), (200, {"choices": [{"message": {"content": "BTC: 0.1"}}]})]
        self.assertEqual(eng.generate("x"), "BTC: 0.1")
        self.assertIsNone(eng.last_error)

    def test_200_without_an_answer_counts_as_a_failure(self):
        eng = self.engine(max_retries=1)
        self.server.responses.append((200, {"error": {"message": "context full"}}))
        self.assertIsNone(eng.generate("x"))

    def test_unreachable_model_returns_none_and_logs_once(self):
        self.server.shutdown()
        self.server.server_close()
        eng = self.engine(max_retries=1)
        with self.assertLogs("LlamaDecisionEngine", level="DEBUG") as logs:
            self.assertIsNone(eng.generate("x"))
            self.assertIsNone(eng.generate("x"))
        errors = [r for r in logs.records if r.levelname == "ERROR"]
        self.assertEqual(len(errors), 1)  # the second failure of the streak is DEBUG
        self.assertFalse(eng.ping(timeout=0.5))

    def test_ping(self):
        self.assertTrue(self.engine().ping())
        self.assertEqual(self.server.requests[-1]["path"], "/v1/models")
        self.assertTrue(self.engine(style="ollama").ping())
        self.assertEqual(self.server.requests[-1]["path"], "/api/tags")

    def test_a_proxy_in_the_environment_is_not_used_for_a_local_model(self):
        dead_proxy = "http://127.0.0.1:9"  # discard port: nothing listens
        with mock.patch.dict(os.environ, {"HTTP_PROXY": dead_proxy, "http_proxy": dead_proxy, "NO_PROXY": "", "no_proxy": ""}):
            for eng in self.both_transports(lambda: self.engine(max_retries=1)):
                with self.subTest(transport="requests" if eng._session is not None else "urllib"):
                    self.server.responses.append((200, {"choices": [{"message": {"content": "BTC: 0.3"}}]}))
                    self.assertEqual(eng.generate("x"), "BTC: 0.3")
                    self.assertTrue(eng.ping())


if __name__ == "__main__":
    unittest.main()
