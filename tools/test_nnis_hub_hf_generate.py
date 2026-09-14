"""CPU-only process/transport regressions; fixtures do not execute a model."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nnis_hub_hf_generate as generation


@unittest.skipUnless(os.name == "posix", "bounded pipe supervision requires POSIX")
class GenerationTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "tokenizer.json").write_text("{}", encoding="utf-8")
        self.result = self.root / "result.json"
        self.pid_file = self.root / "pid"

    def fake(self, body: str) -> Path:
        executable = self.root / "native"
        executable.write_text(
            f"#!{sys.executable}\nimport os, sys, time\nfrom pathlib import Path\n"
            f"Path({str(self.pid_file)!r}).write_text(str(os.getpid()))\n{body}\n",
            encoding="utf-8",
        )
        executable.chmod(0o700)
        return executable

    def run_fake(self, body: str, *, timeout: float = 5.0) -> dict:
        return generation.execute(
            nnis_hf_bin=str(self.fake(body)), model_dir=self.model,
            tokenizer_file=None, prompt="prompt", device_ordinal=0,
            max_new_tokens=1, result_path=self.result, timeout_seconds=timeout,
        )

    def assert_reaped(self) -> None:
        pid = int(self.pid_file.read_text())
        with self.assertRaises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)

    def test_exact_limit_for_both_streams_preserves_bytes(self) -> None:
        with mock.patch.object(generation, "MAX_CAPTURE_BYTES", 131072):
            result = self.run_fake(
                "for _ in range(32):\n"
                "    sys.stdout.buffer.write(b'a' * 4096); sys.stdout.buffer.flush()\n"
                "    sys.stderr.buffer.write(b'b' * 4096); sys.stderr.buffer.flush()"
            )
        for name, byte in (("stdout", b"a"), ("stderr", b"b")):
            expected = byte * 131072
            self.assertEqual(result["output"][name + "_bytes"], len(expected))
            self.assertEqual(result["output"][name + "_sha256"], hashlib.sha256(expected).hexdigest())
            self.assertEqual(result["output"][name + "_utf8"].encode(), expected)
        self.assertEqual(json.loads(self.result.read_bytes()), result)
        self.assert_reaped()

    def test_overflow_stops_stdout_and_stderr_producers_before_exit(self) -> None:
        for stream in ("stdout", "stderr"):
            with self.subTest(stream=stream), mock.patch.object(generation, "MAX_CAPTURE_BYTES", 1024):
                with self.assertRaisesRegex(RuntimeError, stream + " exceeds.*capture budget"):
                    self.run_fake(
                        f"sys.{stream}.buffer.write(b'x' * 1025); sys.{stream}.buffer.flush()\n"
                        "time.sleep(60)", timeout=5.0,
                    )
                self.assertFalse(self.result.exists())
                self.assert_reaped()

    def test_timeout_stops_silent_child(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "timeout-seconds"):
            self.run_fake("time.sleep(60)", timeout=1.0)
        self.assertFalse(self.result.exists())
        self.assert_reaped()

    def test_timeout_also_covers_closed_pipes(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "timeout-seconds"):
            self.run_fake("os.close(1); os.close(2); time.sleep(60)", timeout=1.0)
        self.assertFalse(self.result.exists())
        self.assert_reaped()

    def test_invalid_utf8_never_publishes_result(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not valid UTF-8"):
            self.run_fake("sys.stdout.buffer.write(b'\\xff')")
        self.assertFalse(self.result.exists())
        self.assert_reaped()

    def test_native_failure_diagnostic_is_bounded(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exit 7") as failure:
            self.run_fake("sys.stderr.write('x' * 20000); sys.exit(7)")
        self.assertLess(len(str(failure.exception)), generation.MAX_ERROR_BYTES + 100)
        self.assertFalse(self.result.exists())
        self.assert_reaped()

    def test_result_not_visible_until_sync_and_link(self) -> None:
        real_sync = os.fsync
        def check_sync(fd: int) -> None:
            self.assertFalse(self.result.exists())
            real_sync(fd)
        with mock.patch.object(generation.os, "fsync", side_effect=check_sync):
            generation._write_new_json(self.result, {"test": "exact"})
        self.assertEqual(self.result.read_bytes(), b'{"test":"exact"}\n')
        self.assertEqual(list(self.root.glob(".nnis-result-*")), [])

    def test_sync_or_link_failure_leaves_no_partial_destination(self) -> None:
        for operation in ("fsync", "link"):
            with self.subTest(operation=operation), mock.patch.object(
                generation.os, operation, side_effect=OSError("injected publication failure")
            ):
                with self.assertRaisesRegex(RuntimeError, "cannot write generation result"):
                    generation._write_new_json(self.result, {"complete": True})
            self.assertFalse(self.result.exists())
            self.assertEqual(list(self.root.glob(".nnis-result-*")), [])

    def test_concurrent_publisher_is_neither_overwritten_nor_removed(self) -> None:
        real_link = os.link
        def competing_link(source: Path, destination: Path) -> None:
            destination.write_bytes(b"other publisher")
            real_link(source, destination)
        with mock.patch.object(generation.os, "link", side_effect=competing_link):
            with self.assertRaises(generation.ProcessContractError):
                generation._write_new_json(self.result, {"loser": True})
        self.assertEqual(self.result.read_bytes(), b"other publisher")
        self.assertEqual(list(self.root.glob(".nnis-result-*")), [])

    def test_dangling_symlink_is_not_overwritten(self) -> None:
        self.result.symlink_to(self.root / "missing")
        with self.assertRaises(generation.ProcessContractError):
            generation._write_new_json(self.result, {})
        self.assertTrue(self.result.is_symlink())
        self.assertFalse((self.root / "missing").exists())

    def test_invalid_timeouts_fail_before_native_start(self) -> None:
        for timeout in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(timeout=timeout), mock.patch.object(generation.subprocess, "Popen") as spawn:
                with self.assertRaises(generation.ProcessContractError):
                    self.run_fake("pass", timeout=timeout)
                spawn.assert_not_called()
        self.assertFalse(self.result.exists())

    def test_invalid_request_types_and_nul_fail_before_native_start(self) -> None:
        valid = dict(nnis_hf_bin="unused", model_dir=self.model, tokenizer_file=None,
                     prompt="x", device_ordinal=0, max_new_tokens=1, result_path=self.result)
        for change in ({"device_ordinal": True}, {"device_ordinal": 2**31},
                       {"max_new_tokens": True}, {"max_new_tokens": 1.0},
                       {"prompt": "a\0b"}, {"prompt": "\ud800"}):
            with self.subTest(change=change), mock.patch.object(generation.subprocess, "Popen") as spawn:
                with self.assertRaises(generation.ProcessContractError):
                    generation.execute(**(valid | change))
                spawn.assert_not_called()
        self.assertFalse(self.result.exists())


if __name__ == "__main__":
    unittest.main()
