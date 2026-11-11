"""serve/test_slots.py - llama-server compatible /slots/{id}?action=save|restore over native session-file commands
(mock engine: no GPU, no pack).

    python -m unittest serve.test_slots -v
"""
from __future__ import annotations

import json
import os
import queue
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (ByteTokenizer, Service, StrataEngine, serve, slot_filename_problem,  # noqa: E402
                          slot_save_dir, engine_args, EngineDied, SessionRefused)
from serve.test_parallel import FakeEngineCase  # noqa: E402
from serve.test_stream_ownership import wait_until  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class FakeProc:
    """Stands in for the engine process: answers SAVE / RESTORE lines like `strata --serve` does."""

    def __init__(self, engine):
        self.engine = engine
        self.stdin = self
        self.sent: list[str] = []

    def write(self, s):
        self.sent.append(s)
        cmd, _, path = s.strip().partition(" ")
        script = self.engine.script
        lines = self.engine.lines
        if script is not None:                      # a scripted answer: a list of lines, None = the process ends
            # played by its own thread, like the real engine's output: write() returns at once, so the server's
            # wait (and its deadline) is already running when a delayed line arrives
            def play():
                for line in script:
                    if self.killed:
                        return
                    if isinstance(line, float):
                        time.sleep(line)
                    else:
                        lines.put(line)
            self.player = threading.Thread(target=play, daemon=True)
            self.player.start()
            return
        if cmd == "SAVE":
            if self.engine.fail:
                self.engine.lines.put("SERR invalid 0 no complete session to save\n")
            else:
                Path(path).write_bytes(b"x" * 1234)
                self.engine.lines.put("SAVED 62993 1234 401.5\n")
        elif cmd == "RESTORE":
            if not Path(path).exists() or self.engine.fail:
                self.engine.lines.put("SERR invalid 0 session file: saved with another model (model fingerprint differs)\n")
            else:
                self.engine.lines.put("RESTORED 62993 1234 560.2\n")
        elif cmd == "DIE":
            self.engine.lines.put(None)

    def flush(self):
        pass

    def poll(self):
        return -9 if self.killed else None

    killed = False

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return -9


class FakeEngine(StrataEngine):
    def __init__(self):
        super().__init__("strata", ["--max-context", "65536"], lazy=True)
        self.max_context = 65536
        self.unloaded, self.ended = False, False
        self.lines = queue.Queue()
        self.fail = False
        self.script = None
        self.proc = FakeProc(self)
        self.wlock = threading.Lock()


class Slots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        tok = ByteTokenizer()
        cls.engine = FakeEngine()
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.svc.slot_save_path = cls.dir.name
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.dir.cleanup()

    def setUp(self):
        self.engine.fail = False
        self.engine.script = None
        self.engine.silence_s = 300.0
        self.fresh_engine()
        self.svc.slot_save_path = self.dir.name

    def fresh_engine(self):
        self.engine.ended = False
        old = getattr(self.engine.proc, "player", None)
        if old is not None:                          # the previous scripted engine has finished...
            self.engine.proc.killed = True
            old.join(timeout=5)
        while not self.engine.lines.empty():         # ...and left no line for the next exchange
            self.engine.lines.get_nowait()
        self.engine.proc = FakeProc(self.engine)

    def post(self, path, body, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers=headers or {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_a_foreign_page_or_a_form_never_reaches_the_engine(self):
        for action in ("save", "restore"):
            for headers, code in (({"Content-Type": "application/json", "Origin": "http://evil.example.com"}, 403),
                                  ({"Content-Type": "text/plain"}, 415),
                                  ({"Content-Type": "application/x-www-form-urlencoded"}, 415)):
                s, b = self.post(f"/slots/0?action={action}", {"filename": "x.bin"}, headers)
                self.assertEqual(s, code, (action, headers, b))
        self.assertEqual(self.engine.proc.sent, [])
        self.assertFalse(Path(self.dir.name, "x.bin").exists())

    def test_save_writes_into_the_slot_save_path(self):
        s, b = self.post("/slots/0?action=save", {"filename": "a.bin"})
        self.assertEqual(s, 200, b)
        self.assertEqual(self.engine.proc.sent, [f"SAVE {Path(self.dir.name, 'a.bin')}\n"])
        self.assertEqual((b["id_slot"], b["filename"], b["n_saved"], b["n_written"]), (0, "a.bin", 62993, 1234))
        self.assertAlmostEqual(b["timings"]["save_ms"], 401.5)

    def test_restore_answers_like_llama_server(self):
        self.post("/slots/0?action=save", {"filename": "b.bin"})
        s, b = self.post("/slots/0?action=restore", {"filename": "b.bin"})
        self.assertEqual(s, 200, b)
        self.assertEqual((b["id_slot"], b["filename"], b["n_restored"], b["n_read"]), (0, "b.bin", 62993, 1234))
        self.assertAlmostEqual(b["timings"]["restore_ms"], 560.2)

    def test_refused_file_is_an_error_not_a_success(self):
        self.post("/slots/0?action=save", {"filename": "c.bin"})
        self.engine.fail = True
        s, b = self.post("/slots/0?action=restore", {"filename": "c.bin"})
        self.assertEqual(s, 400, b)
        self.assertIn("fingerprint", b["error"]["message"])

    def test_missing_file_is_not_found(self):
        s, b = self.post("/slots/0?action=restore", {"filename": "never-saved.bin"})
        self.assertEqual(s, 404, b)
        self.assertEqual(self.engine.proc.sent, [])

    def test_filename_cannot_leave_the_directory(self):
        for bad in ("../x.bin", "/etc/passwd", "a/b.bin", "", ".", "..", "x\ny.bin",
                    # Windows: another drive, a stream, devices with and without an extension, names Windows trims
                    "D:escape.bin", "C:\\x.bin", "a.bin:stream", "NUL", "nul.txt", "CON.bin", "com1.x", "LPT9",
                    "a.bin.", "a.bin ", ".hidden", ".a.bin.0123456789abcdef.tmp", "x\x7fy", "a*b", "a?b", 'a"b',
                    "a<b", "a>b", "a|b", "x" * 201, None, 5, ["a"]):
            s, b = self.post("/slots/0?action=save", {"filename": bad})
            self.assertEqual(s, 400, (bad, b))
        self.assertEqual(self.engine.proc.sent, [])

    def test_ordinary_names_are_accepted(self):
        for good in ("a.bin", "chat-2026-10-03.bin", "a.bin.tmp", "consul.bin", "nullable", "caf\u00e9.bin",
                     "\u4f1a\u8bdd.bin", "x" * 200):
            self.assertIsNone(slot_filename_problem(good), good)

    def test_relative_save_dir_becomes_absolute(self):
        with tempfile.TemporaryDirectory() as base:
            got = slot_save_dir("sessions", base)
            self.assertEqual(got, os.path.join(base, "sessions"))
            self.assertTrue(os.path.isabs(got) and os.path.isdir(got))
            if os.name != "nt":
                self.assertEqual(os.stat(got).st_mode & 0o777, 0o700)
            self.assertEqual(slot_save_dir(got, "/elsewhere"), got)     # an absolute path stays as it is
            cwd = os.getcwd()
            try:
                os.chdir(base)
                self.assertEqual(os.path.realpath(slot_save_dir("rel")), os.path.realpath(os.path.join(base, "rel")))
            finally:
                os.chdir(cwd)
            for bad in ("", "  ", None, 3, "a\nb"):
                with self.assertRaises(ValueError):
                    slot_save_dir(bad, base)

    def test_engine_gets_an_absolute_path_on_one_line(self):
        self.post("/slots/0?action=save", {"filename": "abs.bin"})
        line = self.engine.proc.sent[0]
        self.assertTrue(line.endswith("\n") and line.count("\n") == 1)
        self.assertTrue(os.path.isabs(line[len("SAVE "):-1]))

    def test_restore_of_a_link_reaches_the_engine_which_decides(self):
        """The server only checks the name exists (lstat); the engine opens it without following a link."""
        if os.name == "nt":
            self.skipTest("symlinks need privileges on Windows")
        target = Path(self.dir.name, "real.bin")
        target.write_bytes(b"x")
        link = Path(self.dir.name, "link.bin")
        link.symlink_to(target)
        dangling = Path(self.dir.name, "dangling.bin")
        dangling.symlink_to(Path(self.dir.name, "nothing"))
        self.engine.script = ["SERR invalid 0 session file: x: a symbolic link, not a file\n"]
        for name in ("link.bin", "dangling.bin"):
            s, b = self.post("/slots/0?action=restore", {"filename": name})
            self.assertEqual(s, 400, b)
            self.assertIn("symbolic link", b["error"]["message"])

    def test_progress_lines_keep_a_long_save_alive(self):
        self.engine.silence_s = 0.3
        self.engine.script = [0.2, "SESSION 268435456 1198691396\n", 0.2, "SESSION 536870912 1198691396\n", 0.2,
                              "SAVED 63000 1198691396 2000.0\n"]
        t0 = time.monotonic()
        s, b = self.post("/slots/0?action=save", {"filename": "long.bin"})
        self.assertEqual(s, 200, b)
        self.assertGreater(time.monotonic() - t0, 0.5)       # the lines arrived during the wait, past its 0.3 s
        self.assertEqual(b["n_written"], 1198691396)
        self.assertFalse(self.engine.ended)

    def test_a_small_save_reports_its_only_partial_block(self):
        """B: a file below one 16 MiB block still reports (the start, then its last partial block)."""
        self.engine.silence_s = 0.3
        self.engine.script = ["SESSION 0 56834\n", 0.2, "SESSION 56834 56834\n", 0.2, "SAVED 120 56834 410.0\n"]
        s, b = self.post("/slots/0?action=save", {"filename": "small.bin"})
        self.assertEqual(s, 200, b)
        self.assertEqual(b["n_written"], 56834)
        self.assertFalse(self.engine.ended)

    def test_a_blocking_step_gets_its_announced_allowance(self):
        """B: SWAIT gives one non-streaming step (a flush, the device transfer) its own bounded wait."""
        self.engine.silence_s = 0.3
        self.engine.script = ["SESSION 1198691396 1198691396\n", "SWAIT flush 1\n", 0.7, "SWAIT publish 1\n", 0.7,
                              "SAVED 63000 1198691396 2000.0\n"]
        t0 = time.monotonic()
        s, b = self.post("/slots/0?action=save", {"filename": "flush.bin"})
        self.assertEqual(s, 200, b)
        self.assertGreater(time.monotonic() - t0, 1.3)       # longer than the silence limit, twice
        self.assertFalse(self.engine.ended)
        Path(self.dir.name, "flush.bin").write_bytes(b"x")    # the scripted engine wrote nothing; restore needs a file
        self.engine.script = ["SESSION 16777216 1198691396\n", 0.2, "SWAIT validate 1\n", 0.7, "SWAIT transfer 1\n", 0.7,
                              "RESTORED 63000 1198691396 900.0\n"]
        s, b = self.post("/slots/0?action=restore", {"filename": "flush.bin"})
        self.assertEqual(s, 200, b)

    def test_a_blocking_step_that_never_ends_is_ended(self):
        """B: the allowance is a bound, not a heartbeat: a step that outlives it ends the engine."""
        self.engine.silence_s = 0.3
        self.engine.script = ["SWAIT flush 1\n"]               # then nothing
        t0 = time.monotonic()
        s, b = self.post("/slots/0?action=save", {"filename": "stuck.bin"})
        took = time.monotonic() - t0
        self.assertEqual(s, 500, b)
        self.assertGreater(took, 0.9)
        self.assertLess(took, 5)
        self.assertIn("blocking step allowed 1 s", b["error"]["message"])
        self.assertTrue(self.engine.ended and self.engine.proc.killed)

    def test_progress_ends_the_allowance(self):
        """B: after a SESSION line the ordinary silence limit applies again (the allowance was for one step)."""
        self.engine.silence_s = 0.3
        self.engine.script = ["SWAIT transfer 2\n", "SESSION 1 2\n", 0.8, "SAVED 1 2 1.0\n"]
        s, b = self.post("/slots/0?action=save", {"filename": "reset.bin"})
        self.assertEqual(s, 500, b)
        self.assertIn("said nothing", b["error"]["message"])
        self.assertNotIn("blocking step", b["error"]["message"])
        self.assertTrue(self.engine.ended)

    def test_malformed_allowances_end_the_engine(self):
        for script in (["SWAIT flush 0\n"], ["SWAIT flush 3601\n"], ["SWAIT 5\n"], ["SWAIT flush x\n"],
                       ["SWAIT flush -1\n"], ["SWAIT fl:sh 5\n"], ["SWAIT flush 5 extra\n"]):
            with self.subTest(script=script):
                self.fresh_engine()
                self.engine.script = script + [0.1, "SAVED 1 2 1.0\n"]
                s, b = self.post("/slots/0?action=save", {"filename": "bad.bin"})
                self.assertEqual(s, 500, b)
                self.assertTrue(self.engine.ended and self.engine.proc.killed)

    def test_a_silent_engine_is_ended_not_waited_for(self):
        self.engine.silence_s = 0.3
        self.engine.script = []                      # the engine never answers
        t0 = time.monotonic()
        s, b = self.post("/slots/0?action=save", {"filename": "silent.bin"})
        self.assertEqual(s, 500, b)
        self.assertLess(time.monotonic() - t0, 5)
        self.assertIn("said nothing", b["error"]["message"])
        self.assertTrue(self.engine.ended and self.engine.proc.killed)
        self.assertTrue(self.svc.fifo.acquire(blocking=False))   # the FIFO was released
        self.svc.fifo.release()
        self.assertEqual(self.svc.status["queued"], 0)
        self.assertFalse(self.svc.status["busy"])

    def test_engine_death_is_a_server_error(self):
        self.engine.script = [None]
        s, b = self.post("/slots/0?action=save", {"filename": "dead.bin"})
        self.assertEqual(s, 500, b)

    def test_failed_restore_transfer_is_a_server_error(self):
        Path(self.dir.name, "t.bin").write_bytes(b"x")
        self.engine.script = ["SESSION 268435456 1198691396\n",
                              "FATAL restoring the session file failed after the device state was changed: copy\n"]
        s, b = self.post("/slots/0?action=restore", {"filename": "t.bin"})
        self.assertEqual(s, 500, b)
        self.assertIn("device state", b["error"]["message"])

    def test_malformed_or_mismatched_reply_is_a_server_error(self):
        for script in (["RESTORED 1 2 3\n"], ["SAVED x y z\n"], ["SAVED 1 2\n"], ["T 5\n"]):
            self.engine.script = script
            self.engine.ended = False
            self.engine.proc = FakeProc(self.engine)
            s, b = self.post("/slots/0?action=save", {"filename": "m.bin"})
            self.assertEqual(s, 500, (script, b))
            self.assertTrue(self.engine.ended, script)           # out of step: the engine is ended

    def test_out_of_protocol_lines_end_the_engine(self):
        """R4: a line this exchange does not allow is not a heartbeat, and an answer must hold valid numbers."""
        for script in (["SESSION nonsense\n"], ["SESSION 5\n"], ["SESSION -1 10\n"], ["SESSION 11 10\n"],
                       ["SESSION 8 10\n", "SESSION 4 10\n"],      # progress going back
                       ["SAVED -1 -2 nan\n"], ["SAVED 1 2 nan\n"], ["SAVED 1 2 inf\n"], ["SAVED 1 2 -5\n"],
                       ["SAVED +1 2 3\n"], ["ERR bare refusal\n"], ["SERR weird 0 x\n"], ["SERR io 2 x\n"],
                       ["SERR io\n"], ["garbage\n"]):
            self.engine.script = script
            self.engine.ended = False
            self.engine.proc = FakeProc(self.engine)
            s, b = self.post("/slots/0?action=save", {"filename": "p.bin"})
            self.assertEqual(s, 500, (script, b))
            self.assertTrue(self.engine.ended and self.engine.proc.killed, script)
            json.dumps(b, allow_nan=False)                        # standard JSON

    def test_log_lines_are_not_answers(self):
        self.engine.script = ["INFO something\n", "SAVED 1 2 3.0\n"]
        s, b = self.post("/slots/0?action=save", {"filename": "i.bin"})
        self.assertEqual(s, 200, b)
        self.assertFalse(self.engine.ended)

    def test_typed_refusals_have_their_own_codes(self):
        """R5: the engine's category decides the status, never the words of the message."""
        cases = [("storage", "No space left on device", 507), ("storage", "espace insuffisant", 507),
                 ("memory", "out of memory for the write buffer", 503), ("invalid", "not enough disk space", 400),
                 ("invalid", "saved with another model", 400), ("io", "Permission denied", 500),
                 ("io", "RAM", 500)]
        for kind, msg, code in cases:
            self.engine.script = [f"SERR {kind} 0 {msg}\n"]
            s, b = self.post("/slots/0?action=save", {"filename": "k.bin"})
            self.assertEqual(s, code, (kind, msg, b))
            self.assertEqual(b["error"]["kind"], kind)
            self.assertEqual(b["error"]["message"], msg)
            self.assertNotIn("published", b["error"])
            self.assertFalse(self.engine.ended)                   # a refusal keeps the engine

    def test_published_failure_is_reported(self):
        """R3: the folder flush failed after the rename - the new file replaced the old one; the answer says so."""
        self.engine.script = ["SERR io 1 flushing the directory: Input/output error (the new file has already "
                              "replaced the old one)\n"]
        s, b = self.post("/slots/0?action=save", {"filename": "pub.bin"})
        self.assertEqual(s, 500, b)
        self.assertTrue(b["error"]["published"])
        self.assertIn("replaced", b["error"]["message"])

    def test_disk_and_ram_refusals_have_their_own_codes(self):
        self.engine.script = ["SERR storage 0 session file: not enough disk space (1143 MiB plus a reserve of 4096 "
                              "MiB needed, 10 MiB free)\n"]
        self.assertEqual(self.post("/slots/0?action=save", {"filename": "d.bin"})[0], 507)
        Path(self.dir.name, "r.bin").write_bytes(b"x")
        self.engine.script = ["SERR memory 0 session file: not enough RAM to read it (1143 MiB plus a floor of 2560 "
                              "MiB needed, 900 MiB available)\n"]
        self.assertEqual(self.post("/slots/0?action=restore", {"filename": "r.bin"})[0], 503)

    def test_slot_work_counts_as_activity_and_shows_in_status(self):
        seen = {}

        def during():
            time.sleep(0.15)
            with self.svc.status_lock:
                seen.update(self.svc.status)
        self.engine.script = [0.4, "SAVED 1 2 3.0\n"]
        self.svc.last_request_at = 0
        t = threading.Thread(target=during)
        t.start()
        s, _ = self.post("/slots/0?action=save", {"filename": "act.bin"})
        t.join()
        self.assertEqual(s, 200)
        self.assertTrue(seen.get("busy"))
        self.assertEqual(seen.get("phase"), "saving the session")
        self.assertGreater(self.svc.last_request_at, time.time() - 5)
        self.assertFalse(self.svc.status["busy"])
        self.assertEqual(self.svc.status["queued"], 0)
        self.assertEqual(self.svc.unload(idle_for=60), "busy")        # just active: the idle unload waits

    def test_only_slot_zero_and_known_actions(self):
        self.assertEqual(self.post("/slots/1?action=save", {"filename": "d.bin"})[0], 400)
        self.assertEqual(self.post("/slots/0?action=erase_all", {"filename": "d.bin"})[0], 400)
        self.assertEqual(self.engine.proc.sent, [])

    def test_disabled_without_a_slot_save_path(self):
        saved = self.svc.slot_save_path
        self.svc.slot_save_path = None
        try:
            s, b = self.post("/slots/0?action=save", {"filename": "e.bin"})
        finally:
            self.svc.slot_save_path = saved
        self.assertEqual(s, 501, b)

    def test_refused_with_parallel_requests(self):
        """#465: batch slots do not hold the FIFO, so a session file is refused before anything reaches the engine."""
        self.engine.batch = 2
        try:
            s, b = self.post("/slots/0?action=save", {"filename": "p.bin"})
        finally:
            self.engine.batch = 0
        self.assertEqual(s, 501, b)
        self.assertIn("parallel", b["error"]["message"])
        self.assertEqual(self.engine.proc.sent, [])

    def test_waits_for_a_running_request(self):
        """A save never interleaves with a generation: it takes the same FIFO."""
        self.svc.fifo.acquire()
        out = {}
        t = threading.Thread(target=lambda: out.update(r=self.post("/slots/0?action=save", {"filename": "f.bin"})))
        t.start()
        t.join(0.5)
        self.assertTrue(t.is_alive())
        self.assertEqual(self.engine.proc.sent, [])
        self.svc.fifo.release()
        t.join(10)
        self.assertEqual(out["r"][0], 200)


    def test_flush_uses_session_protocol_and_exposes_writer_errors(self):
        self.engine.silence_s = 0.1
        self.engine.script = ["SWAIT flush 1\n", 0.3, "FLUSHED 0 0 300.0\n"]
        self.assertEqual(self.engine.session_file("flush", ""), {"tokens": 0, "bytes": 0, "ms": 300.0})
        self.assertEqual(self.engine.proc.sent, ["FLUSH\n"])
        for script, exception in ((["SERR io 0 writer failed\n"], SessionRefused), ([None], EngineDied),
                                  (["SAVED 0 0 0\n"], EngineDied), (["SWAIT flush 3601\n"], EngineDied)):
            self.fresh_engine()
            self.engine.script = script
            with self.subTest(script=script), self.assertRaises(exception):
                self.engine.session_file("flush", "")

    def test_flush_preserves_disabled_silence_limit_after_announced_wait(self):
        self.engine.silence_s = 0
        self.engine.script = ["SWAIT flush 1\n", 1.1, "FLUSHED 0 0 1100.0\n"]
        self.assertEqual(self.engine.session_file("flush", ""), {"tokens": 0, "bytes": 0, "ms": 1100.0})
        self.assertFalse(self.engine.proc.killed)

    def test_old_exchange_cannot_consume_or_kill_successor(self):
        for terminal in ("SAVED 1 2 3.0\n", "garbage\n"):
            with self.subTest(terminal=terminal):
                self.fresh_engine()
                old, lines = self.engine.proc, self.engine.lines
                self.engine.script = ["SWAIT transfer 1\n", 0.15, terminal]
                output = {}
                def exchange():
                    try:
                        output["result"] = self.engine.session_file("save", "old.bin")
                    except EngineDied as e:
                        output["error"] = e
                thread = threading.Thread(target=exchange)
                thread.start()
                self.assertTrue(wait_until(lambda: hasattr(old, "player")))
                successor = FakeProc(self.engine)
                self.engine.proc, self.engine.lines = successor, queue.Queue()
                self.engine.lines.put("SAVED 999 999 999.0\n")
                try:
                    thread.join(2)
                    self.assertFalse(thread.is_alive())
                    self.assertFalse(successor.killed or self.engine.ended)
                    if terminal == "garbage\n":
                        self.assertIsInstance(output.get("error"), EngineDied)
                        self.assertTrue(old.killed)
                    else:
                        self.assertEqual(output, {"result": {"tokens": 1, "bytes": 2, "ms": 3.0}})
                    self.assertEqual(self.engine.lines.get_nowait(), "SAVED 999 999 999.0\n")
                finally:
                    self.engine.proc, self.engine.lines = old, lines
                    old.player.join(2)

    def test_timeout_only_ends_captured_process(self):
        old, lines = self.engine.proc, self.engine.lines
        self.engine.silence_s = 0.02
        self.engine.script = []
        successor = FakeProc(self.engine)
        silent = self.engine._silent
        def restart_at_timeout(what, **kwargs):
            self.engine.proc, self.engine.lines = successor, queue.Queue()
            return silent(what, **kwargs)
        try:
            with mock.patch.object(self.engine, "_silent", side_effect=restart_at_timeout):
                with self.assertRaises(EngineDied):
                    self.engine.session_file("save", "old.bin")
            self.assertTrue(old.killed)
            self.assertFalse(successor.killed or self.engine.ended)
        finally:
            self.engine.proc, self.engine.lines = old, lines


class ConcurrentSlots(FakeEngineCase):
    def setUp(self):
        self.start(2, more=["--batch-session-files"])
        self.svc.slot_save_path = self.tmp.name

    def commands(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def post_slot(self, slot, action, filename):
        req = urllib.request.Request(self.base + f"/slots/{slot}?action={action}",
                                     data=json.dumps({"filename": filename}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_claims_before_control_without_fifo_and_restores_unknown_prefix(self):
        out = {}
        path = Path(self.tmp.name, "restore.bin")
        path.write_bytes(b"session")
        self.engine.slot_held[1] = [999, 998]
        threads = [threading.Thread(target=lambda: out.update(save=self.post_slot(0, "save", "a b.bin"))),
                   threading.Thread(target=lambda: out.update(restore=self.post_slot(1, "restore", "restore.bin")))]
        with self.svc.fifo:
            threads[0].start()
            self.assertTrue(wait_until(lambda: any(line.startswith("BSAVE 0 ") for line in self.commands())))
            with self.engine.slot_cv:
                self.assertEqual(self.engine.pick_slot([9]), 1)  # the file claim does not reserve its sibling
            threads[1].start()
            self.assertTrue(wait_until(lambda: self.engine.slot_claims == {0, 1}))
            with self.engine.slot_cv:
                self.assertIsNone(self.engine.pick_slot([9]))
            self.assertFalse(any(line.startswith("BRESTORE ") for line in self.commands()))
            self.assertEqual(self.post_slot(0, "save", "busy.bin")[0], 409)
            for thread in threads:
                thread.join(5)
                self.assertFalse(thread.is_alive())
        self.assertEqual(out["save"], (200, {"id_slot": 0, "filename": "a b.bin", "n_saved": 7, "n_written": 7,
                                           "timings": {"save_ms": 200.0}}))
        self.assertEqual(out["restore"][0], 200)
        self.assertEqual(out["restore"][1]["n_restored"], 7)
        self.assertEqual(self.commands(), [f"BSAVE 0 {Path(self.tmp.name, 'a b.bin')}", f"BRESTORE 1 {path}"])
        self.assertEqual(self.engine.slot_claims, set())
        self.assertEqual(self.engine.slot_busy, [False, False])
        self.assertEqual(self.engine.slot_held[1], [])

    def test_busy_target_refused_while_idle_file_keeps_sibling_running(self):
        cancel = [threading.Event(), threading.Event()]
        output, failures = [[], []], []
        def generate(i):
            try:
                for token in self.engine.generate(list(b"LONGREPLY"), 1000, {}, cancel[i]):
                    if token is not None:
                        output[i].append(token)
            except Exception as e:
                failures.append(e)
        threads = [threading.Thread(target=generate, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        try:
            self.assertTrue(wait_until(lambda: all(self.engine.slot_busy) and
                all(live and live["state"] == "decoding" for live in self.engine.slot_live)))
            self.assertEqual(self.post_slot(0, "save", "active.bin")[0], 409)
            self.assertFalse(any(line.startswith("BSAVE ") for line in self.commands()))
            cancel[1].set()
            threads[1].join(5)
            self.assertFalse(threads[1].is_alive())
            self.assertTrue(wait_until(lambda: sum(self.engine.slot_busy) == 1))
            active = self.engine.slot_busy.index(True)
            before = len(output[0])
            self.assertEqual(self.post_slot(1 - active, "save", "idle.bin")[0], 200)
            self.assertGreater(len(output[0]), before)
            self.assertTrue(threads[0].is_alive())
            with self.assertRaisesRegex(SessionRefused, "busy"):
                self.engine.session_file("save", str(Path(self.tmp.name, "busy.bin")), slot=active)
            self.assertTrue(self.engine.alive())
        finally:
            for event in cancel:
                event.set()
            for thread in threads:
                thread.join(5)
        self.assertEqual(failures, [])

    def test_restart_during_control_wait_cannot_send_to_or_release_successor(self):
        old_proc, old_lines = self.engine.proc, self.engine.lines
        old_busy, old_claims, old_held = self.engine.slot_busy, self.engine.slot_claims, self.engine.slot_held
        output = {}
        self.engine.ctl.acquire()
        worker = threading.Thread(target=lambda: output.update(result=self.svc.slot_action("0", "save", "old.bin")))
        worker.start()
        self.assertTrue(wait_until(lambda: old_claims == {0}))
        self.assertEqual(self.svc.unload(), "busy")
        successor = FakeProc(self.engine)
        self.engine.proc, self.engine.lines = successor, queue.Queue()
        self.engine.lines.put("SAVED 999 999 999.0\n")
        self.engine.slot_busy, self.engine.slot_claims, self.engine.slot_held = [True, False], {0}, [[999], []]
        self.engine.ctl.release()
        try:
            worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(output["result"][0], 503)
            self.assertEqual(successor.sent, [])
            self.assertEqual(self.engine.lines.get_nowait(), "SAVED 999 999 999.0\n")
            self.assertEqual(self.commands(), [])
            self.assertEqual(self.engine.slot_busy, [True, False])
            self.assertEqual(self.engine.slot_claims, {0})
            self.assertEqual(old_busy, [False, False])
            self.assertEqual(old_claims, set())
        finally:
            self.engine.proc, self.engine.lines = old_proc, old_lines
            self.engine.slot_busy, self.engine.slot_claims, self.engine.slot_held = old_busy, old_claims, old_held

    def test_new_process_cannot_take_slot_files_before_its_queues_and_tables_are_ready(self):
        old_proc, old_lines = self.engine.proc, self.engine.lines
        old_busy, old_claims = self.engine.slot_busy, self.engine.slot_claims
        paused, resume = threading.Event(), threading.Event()
        failures = []
        def restart():
            try:
                self.engine.restart(tries=1)
            except Exception as e:
                failures.append(e)
        worker = threading.Thread(target=restart)
        real_queue = queue.Queue
        made = 0
        def make_queue(*args, **kwargs):
            nonlocal made
            if threading.current_thread() is worker:
                made += 1
                if made == 2:                            # ready_q exists; self.lines and slot tables are still old
                    paused.set()
                    if not resume.wait(5):
                        raise RuntimeError("start publication barrier was not released")
            return real_queue(*args, **kwargs)
        from serve import server
        real_popen = server.subprocess.Popen
        script = Path(self.tmp.name, "fake_strata.py")
        with mock.patch("serve.server.queue.Queue", side_effect=make_queue), \
                mock.patch("serve.server.subprocess.Popen",
                           lambda cmd, **kw: real_popen([sys.executable, str(script), *cmd[1:]], **kw)):
            worker.start()
            try:
                self.assertTrue(paused.wait(5))
                successor = self.engine.proc
                self.assertIsNot(successor, old_proc)
                self.assertIs(self.engine.lines, old_lines)
                self.assertIs(self.engine.slot_busy, old_busy)
                with mock.patch.object(successor.stdin, "write", wraps=successor.stdin.write) as writes:
                    self.assertEqual(self.post_slot(0, "save", "not-ready.bin")[0], 503)
                    writes.assert_not_called()
                    with self.assertRaises(EngineDied):
                        self.engine.session_file("save", str(Path(self.tmp.name, "sdk-not-ready.bin")), slot=1)
                    writes.assert_not_called()
                self.assertEqual(old_busy, [False, False])
                self.assertEqual(old_claims, set())
            finally:
                resume.set()
                worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertTrue(self.engine.alive())
        self.assertIsNot(self.engine.lines, old_lines)
        self.assertIsNot(self.engine.slot_busy, old_busy)
        self.assertEqual(self.post_slot(0, "save", "ready.bin")[0], 200)
        self.assertEqual(self.post_slot(1, "save", "next.bin")[0], 200)
        self.assertEqual(self.commands(), [f"BSAVE 0 {Path(self.tmp.name, 'ready.bin')}",
                                           f"BSAVE 1 {Path(self.tmp.name, 'next.bin')}"])

    def test_unload_decision_rejects_new_claims_without_holding_slot_condition_during_close(self):
        closing, resume = threading.Event(), threading.Event()
        result = {}
        unload = self.engine.unload
        def paused_unload():
            closing.set()                               # Service finished checking slots, but close has not begun
            if not resume.wait(5):
                raise RuntimeError("unload barrier was not released")
            unload()
        def request_unload():
            req = urllib.request.Request(self.base + "/unload", data=b"{}",
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as response:
                result["response"] = response.status, json.load(response)
        worker = threading.Thread(target=request_unload)
        with mock.patch.object(self.engine, "unload", side_effect=paused_unload):
            worker.start()
            try:
                self.assertTrue(closing.wait(5))
                acquired = self.engine.slot_cv.acquire(blocking=False)
                self.assertTrue(acquired, "close must not hold slot_cv while waiting for I/O")
                if acquired:
                    self.engine.slot_cv.release()
                self.assertEqual(self.post_slot(0, "save", "too-late.bin")[0], 503)
                self.assertEqual(self.engine.slot_claims, set())
                self.assertEqual(self.engine.slot_busy, [False, False])
                self.assertEqual(self.commands(), [])
            finally:
                resume.set()
                worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result["response"], (200, {"status": "unloaded"}))
        self.assertIsNone(self.engine.proc)
        self.assertFalse(Path(self.tmp.name, "too-late.bin").exists())

    def test_grouped_slots_and_missing_capability_keep_existing_refusal(self):
        self.engine.slot_groups = 2
        self.assertEqual(self.post_slot(0, "save", "group.bin")[0], 501)
        self.engine.slot_groups = 1
        self.engine.info.pop("session_slots")
        self.assertEqual(self.post_slot(0, "save", "disabled.bin")[0], 501)
        self.assertEqual(self.commands(), [])


class DiskCacheArgs(unittest.TestCase):
    def test_native_settings_preserve_explicit_features_and_legacy_files(self):
        with tempfile.TemporaryDirectory() as base:
            directory = Path(base, "sessions")
            directory.mkdir()
            index = directory / "index.json"
            index.write_text("legacy index must not be opened or pruned")
            old = directory / "auto-legacy.bin"
            old.write_bytes(b"legacy")
            args = ["--unified-slots", "--prompt-cache", "6"]
            cfg = {"args": args, "cwd": base, "parallel": 2, "slot_save_path": "sessions",
                   "disk_cache": {"max_mib": 16384, "max_entries": 8, "idle_s": 0.5, "min_tokens": 4096}}
            self.assertEqual(engine_args(cfg), args + ["--batch", "2", "--session-cache-mib", "16384",
                             "--session-cache-slots", "8", "--session-cache-idle-seconds", "0.5",
                             "--session-cache-min-tokens", "4096", "--session-cache-dir", str(directory)])
            self.assertEqual(old.read_bytes(), b"legacy")
            self.assertEqual(index.read_text(), "legacy index must not be opened or pruned")
            cfg["user_args"] = ["--session-cache-mib", "32"]
            got = engine_args(cfg)
            self.assertEqual(got.count("--session-cache-mib"), 1)
            self.assertEqual(got[got.index("--session-cache-mib") + 1], "32")
            cfg.pop("user_args")
            cfg["disk_cache"] = False
            self.assertEqual(engine_args(cfg), args + ["--batch", "2"])
            self.assertEqual(cfg["args"], args)

    def test_cache_does_not_enable_fixed_slots_or_touch_storage_when_off(self):
        with tempfile.TemporaryDirectory() as base:
            cfg = {"args": [], "cwd": base, "parallel": 2, "slot_save_path": "sessions"}
            self.assertEqual(engine_args(cfg), ["--batch", "2"])
            self.assertFalse(Path(base, "sessions").exists())
            cfg["disk_cache"] = {"max_mib": 16, "max_entries": 8, "idle_s": 60, "min_tokens": 4096}
            self.assertNotIn("--unified-slots", engine_args(cfg))

    def test_invalid_native_settings_do_not_create_a_cache(self):
        good = {"max_mib": 16, "max_entries": 8, "idle_s": 60, "min_tokens": 4096}
        with tempfile.TemporaryDirectory() as base:
            for key, value in (("max_mib", True), ("max_entries", -1), ("min_tokens", "10"),
                               ("idle_s", float("inf")), ("idle_s", -0.1)):
                with self.subTest(key=key, value=value):
                    cfg = {"args": [], "cwd": base, "slot_save_path": "sessions",
                           "disk_cache": dict(good, **{key: value})}
                    with self.assertRaises(ValueError):
                        engine_args(cfg)
            self.assertFalse(Path(base, "sessions").exists())
            with self.assertRaisesRegex(ValueError, "slot_save_path"):
                engine_args({"args": [], "disk_cache": good})


if __name__ == "__main__":
    unittest.main()
