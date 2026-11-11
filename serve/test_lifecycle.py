"""Model discovery, on-demand loading and safe unloading over the actual HTTP API."""
import json
import io
import os
import subprocess
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.test_parallel import FakeEngineCase
from serve.server import ByteTokenizer, EngineDied, EngineStuck, MockEngine, Service, SessionRefused, StrataEngine, serve


class ResidentEngine(StrataEngine):
    def __init__(self, tok):
        self.reply = MockEngine(tok, "Hello.", max_context=4096)
        self.max_context, self.info, self.last = 4096, {}, {}
        self.loaded, self.starts, self.closes = False, 0, 0
        self.unloaded = True
        self.progress = None

    def alive(self):
        return self.loaded

    def exit_code(self):
        return None

    def restart(self):
        self.loaded = True
        self.starts += 1
        self.unloaded = False

    def close(self):
        self.loaded = False
        self.closes += 1

    def unload(self):
        self.close()
        self.unloaded = True

    def generate(self, *args, **kwargs):
        yield from self.reply.generate(*args, **kwargs)


class Lifecycle(unittest.TestCase):
    def test_close_cleans_up_after_a_broken_stdin_pipe(self):
        for running in (False, True):
            with self.subTest(running=running):
                engine = StrataEngine("missing-executable", [], lazy=True)
                read_fd, write_fd = os.pipe()
                stdin = os.fdopen(write_fd, "w")
                self.addCleanup(stdin.close)
                os.close(read_fd)                       # the engine's end of the pipe is already gone
                stdin.write("pending")                  # buffered: flush() and close() will hit the broken pipe
                proc = mock.Mock(stdin=stdin, stdout=io.StringIO())
                self.addCleanup(proc.stdout.close)
                proc.poll.side_effect = [None if running else 0, 0]
                engine.proc, engine.pump, engine.log = proc, mock.Mock(), io.StringIO()
                self.addCleanup(engine.log.close)
                engine.ended, engine.progress, engine.last = False, (1, 2), {"generated": 1}

                engine.close()

                self.assertTrue(stdin.closed)
                self.assertTrue(proc.stdout.closed)
                self.assertTrue(engine.log.closed)
                engine.pump.join.assert_called_once_with(timeout=2)
                self.assertIsNone(engine.proc)
                self.assertTrue(engine.ended)
                self.assertIsNone(engine.progress)
                self.assertEqual(engine.last, {})
                if running:
                    proc.terminate.assert_called_once()
                    proc.wait.assert_called_once_with(timeout=5)
                else:
                    proc.terminate.assert_not_called()
                    proc.wait.assert_not_called()
                proc.kill.assert_not_called()
                engine.close()                          # repeated cleanup is harmless

    def test_close_sends_eof_before_waiting_for_windows_reader(self):
        engine = StrataEngine("missing-executable", [], lazy=True)
        proc = mock.Mock()
        proc.stdin, proc.stdout = io.StringIO(), io.StringIO()
        proc.poll.side_effect = [None, 0]
        proc.wait.side_effect = lambda **kwargs: self.assertTrue(proc.stdin.closed)
        engine.proc = proc
        engine.close()
        self.assertIsNone(engine.proc)
        proc.kill.assert_not_called()

    def test_close_does_not_forget_a_process_still_exiting(self):
        engine = StrataEngine("missing-executable", [], lazy=True)
        proc = mock.Mock()
        proc.stdin, proc.stdout = io.StringIO(), io.StringIO()
        proc.poll.return_value = None
        proc.wait.side_effect = subprocess.TimeoutExpired("engine", 2)
        engine.proc = proc
        with self.assertRaisesRegex(EngineStuck, "still releasing"):
            engine.close()
        self.assertIs(engine.proc, proc)
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()

    def test_close_terminates_before_it_kills(self):
        # an engine that does not end on QUIT is terminated first (as the unload always did); kill is the last resort
        engine = StrataEngine("missing-executable", [], lazy=True)
        proc = mock.Mock()
        proc.stdin, proc.stdout = io.StringIO(), io.StringIO()
        proc.poll.side_effect = [None, 0]
        proc.wait.side_effect = [subprocess.TimeoutExpired("engine", 20), 0]
        engine.proc = proc
        engine.close()
        proc.terminate.assert_called_once()
        proc.kill.assert_not_called()
        self.assertIsNone(engine.proc)

    def test_native_lazy_constructor_does_not_start_a_process(self):
        engine = StrataEngine("missing-executable", ["--max-context", "16384"], lazy=True)
        self.assertFalse(engine.alive())
        self.assertEqual(engine.max_context, 16384)
        self.assertIsNone(engine.exit_code())
        engine.close()

    def setUp(self):
        tok = ByteTokenizer()
        self.engine = ResidentEngine(tok)
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read().decode()
            return response.status, data if "event-stream" in response.headers.get("Content-Type", "") else json.loads(data)

    def test_discover_chat_unload_and_stream_reload(self):
        self.assertEqual(self.request("/api/health")[1]["service"], "strata")
        self.assertFalse(self.request("/v1/status")[1]["loaded"])
        self.assertEqual(self.request("/v1/models")[1]["data"][0]["status"]["value"], "unloaded")
        self.assertTrue(self.request("/props")[1]["models_autoload"])
        self.assertEqual(self.engine.starts, 0)
        chat = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32, "reasoning_effort": "none"}
        for _ in range(2):
            code, answer = self.request("/v1/chat/completions", chat)
            self.assertEqual(code, 200)
            self.assertEqual(answer["choices"][0]["message"]["content"], "Hello.")
        self.assertEqual(self.engine.starts, 1)
        self.assertFalse(self.request("/v1/unload", {})[1]["loaded"])
        code, stream = self.request("/v1/chat/completions", {**chat, "stream": True})
        self.assertEqual(code, 200)
        self.assertIn("data: [DONE]", stream)
        self.assertEqual(self.engine.starts, 2)

    def test_busy_auth_and_foreign_origin_cannot_unload(self):
        self.engine.loaded = True
        with self.svc.fifo:
            self.assertEqual(self.request("/v1/unload", {})[0], 409)
        self.svc.api_key = "local-secret"
        self.assertEqual(self.request("/v1/unload", {})[0], 401)
        self.assertEqual(self.request("/v1/unload", {}, {"Authorization": "Bearer local-secret",
                                                        "Origin": "https://other.example"})[0], 403)
        self.assertEqual(self.engine.closes, 0)
        self.assertTrue(self.engine.loaded)
        self.assertEqual(self.request("/v1/unload", {}, {"Authorization": "Bearer local-secret"})[0], 200)

    def test_an_engine_that_died_on_load_is_reported_as_such(self):
        # EngineDied is a RuntimeError: it keeps its own answer ("the next request restarts it")
        def died():
            raise EngineDied("the engine ended while it started")
        self.engine.restart = died
        code, body = self.request("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}]})
        self.assertEqual(code, 503)
        self.assertIn("the next request restarts it", body["error"]["message"])

    def test_explicit_load_and_model_matching(self):
        self.assertEqual(self.request("/v1/load", {"model": "other-model"})[0], 404)
        self.assertEqual(self.engine.starts, 0)
        before = time.time()
        code, status = self.request("/v1/load", {"model": self.svc.model})
        self.assertEqual(code, 200)
        self.assertTrue(status["loaded"])
        self.assertEqual(self.engine.starts, 1)
        self.assertGreaterEqual(self.svc.last_request_at, before)
        self.svc.status["queued"] = 1
        self.assertEqual(self.request("/v1/load", {})[0], 409)
        self.assertEqual(self.request("/v1/unload", {})[0], 409)
        self.assertEqual(self.engine.closes, 0)
        self.assertEqual(self.request("/v1/load", {}, {"Content-Type": "text/plain"})[0], 415)


class CacheShutdown(FakeEngineCase):
    def test_closed_input_still_terminates_the_cache_process_before_reporting_failure(self):
        script = ('import os, time\nos.close(0)\n'
                  'print("INFO engine=0.1.39 session_cache_async=1", flush=True)\n'
                  'print("READY 4096 stop", flush=True)\ntime.sleep(60)\n')
        with mock.patch("serve.test_parallel.FAKE_BATCH", script):
            self.start(0)
        process = self.engine.proc
        try:
            with self.assertRaises(EngineDied):
                self.engine.close()
            self.assertIsNotNone(process.poll())
            self.assertIsNone(self.engine.proc)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)

    def test_cache_flush_commits_before_quit_and_shutdown_gets_its_bounded_wait(self):
        self.start(2, more=["--cache-async"])
        process = self.engine.proc
        self.engine.silence_s = 0.05
        with mock.patch("serve.server.ENGINE_QUIT_WAIT_S", 0.05):
            self.engine.close()
        self.assertEqual(self.log.read_text().splitlines(), ["FLUSH", "FLUSH committed", "SHUTDOWN committed"])
        self.assertEqual(process.returncode, 0)
        self.assertTrue(process.stdin.closed)
        self.assertIsNone(self.engine.proc)

    def test_close_during_serial_generation_leaves_done_to_its_owner(self):
        self.start(0, more=["--cache-async", "--session-files"])
        output, failures = [], []
        started = threading.Event()
        def generate():
            try:
                for token in self.engine.generate(list(b"LONGREPLY"), 1000, {}, threading.Event()):
                    if token is not None:
                        output.append(token)
                        started.set()
            except Exception as e:
                failures.append(e)
        worker = threading.Thread(target=generate)
        worker.start()
        self.assertTrue(started.wait(5))
        process = self.engine.proc
        with mock.patch("serve.server.ENGINE_QUIT_WAIT_S", 0.05):
            self.engine.close()
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertTrue(output)
        commands = self.log.read_text().splitlines()
        self.assertFalse(any(line.startswith("FLUSH") for line in commands))
        self.assertEqual(commands[-1], "SHUTDOWN committed")
        self.assertEqual(process.returncode, 0)
        self.assertFalse(self.engine.ctl.locked())

    def test_close_during_batch_admission_leaves_done_and_badm_to_the_owner(self):
        self.start(2, more=["--cache-async", "--session-files"])
        cancel = [threading.Event(), threading.Event()]
        started, failures = threading.Event(), []
        def generate(i):
            try:
                ids = list(b"LONGREPLY") + ([1] * 8000 if i else [])
                for token in self.engine.generate(ids, 1000, {}, cancel[i]):
                    if token is not None:
                        started.set()
            except EngineDied:
                pass                                    # the sibling's batch stream legitimately ends with the process
            except Exception as e:
                failures.append(e)
        workers = [threading.Thread(target=generate, args=(i,)) for i in range(2)]
        workers[0].start()
        self.assertTrue(started.wait(5))
        workers[1].start()
        try:
            deadline = time.monotonic() + 5
            while not (self.engine.progress and any(
                    live and live["state"] == "reading" and live["prompt_tokens"] > 8000
                    for live in self.engine.slot_live)) and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertIsNotNone(self.engine.progress)
            process = self.engine.proc
            with mock.patch("serve.server.ENGINE_QUIT_WAIT_S", 0.05):
                self.engine.close()
            for worker in workers:
                worker.join(5)
                self.assertFalse(worker.is_alive())
            self.assertEqual(process.returncode, 0)
            self.assertEqual(failures, [])
            commands = self.log.read_text().splitlines()
            self.assertFalse(any(line.startswith("FLUSH") for line in commands))
            self.assertEqual(commands[-1], "SHUTDOWN committed")
        finally:
            for event in cancel:
                event.set()
            for worker in workers:
                worker.join(5)

    def test_close_while_current_thread_holds_control_does_not_self_deadlock(self):
        self.start(2, more=["--cache-async"])
        with self.engine.ctl, mock.patch("serve.server.ENGINE_QUIT_WAIT_S", 0.05):
            self.engine.close()
        self.assertEqual(self.log.read_text().splitlines(), ["SHUTDOWN committed"])
        self.assertIsNone(self.engine.proc)

    def test_flush_error_is_reported_after_the_process_is_closed(self):
        self.start(2, more=["--cache-async", "--flush-error"])
        with self.assertRaisesRegex(SessionRefused, "pending write failed"):
            self.engine.close()
        self.assertIsNone(self.engine.proc)
        self.assertEqual(self.log.read_text().splitlines(), ["FLUSH", "SHUTDOWN committed"])

    def test_shutdown_error_is_reported_without_consuming_control_replies(self):
        self.start(2, more=["--cache-async", "--shutdown-error"])
        with self.engine.ctl:
            with self.assertRaisesRegex(SessionRefused, "shutdown write failed"):
                self.engine.close()
        self.assertIsNone(self.engine.proc)

    def test_missing_shutdown_barrier_is_not_a_successful_drain(self):
        self.start(0, more=["--cache-async", "--shutdown-early"])
        with self.engine.ctl:
            with self.assertRaisesRegex(EngineDied, "before its disk cache shutdown barrier"):
                self.engine.close()
        self.assertIsNone(self.engine.proc)

    def test_malformed_shutdown_wait_is_not_treated_as_a_successful_drain(self):
        self.start(0, more=["--cache-async", "--shutdown-invalid"])
        process = self.engine.proc
        with self.engine.ctl:
            with self.assertRaisesRegex(EngineDied, "invalid session wait"):
                self.engine.close()
        self.assertIsNone(self.engine.proc)
        self.assertNotEqual(process.returncode, 0)


if __name__ == "__main__":
    unittest.main()
