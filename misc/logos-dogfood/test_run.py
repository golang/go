# pyright: strict
"""Checks that replay evidence cannot turn an incomplete analysis into success."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

try:
    import fcntl
except ImportError:
    fcntl = None

from run import (
    Assessment,
    Case,
    Run,
    Source,
    assess_build,
    assess_lint,
    canonical,
    compare_repeats,
    execute,
    file_hash,
    prepare_source,
    tree_hash,
    verify_policy,
    verify_source,
    within,
)


class EvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def run_record(self, code: int, stdout: str = "", stderr: str = "") -> Run:
        out = self.root / "stdout"
        err = self.root / "stderr"
        out.write_text(stdout)
        err.write_text(stderr)
        return Run(
            ("tool",),
            str(self.root),
            {},
            code,
            False,
            "",
            0.1,
            str(out),
            str(err),
            file_hash(out),
            file_hash(err),
        )

    @staticmethod
    def diagnostic(
        rule: str = "correctness::example", severity: str = "deny"
    ) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "rule_id": rule,
            "severity": severity,
            "message": "example finding",
            "file": "sample.go",
            "span": {"start_line": 1, "start_col": 1},
        }

    @staticmethod
    def finding(rule: str = "printf") -> dict[str, object]:
        return {
            "rule": rule,
            "path": "sample.go",
            "line": 1,
            "column": 1,
            "message": "example finding",
        }

    def test_nonzero_empty_lint_is_failure(self) -> None:
        run = self.run_record(1)
        self.assertEqual(assess_lint(run, "logos-lint").status, "invalid-report")

    def test_zero_empty_lint_is_clean(self) -> None:
        self.assertEqual(assess_lint(self.run_record(0), "logos-lint").status, "clean")

    def test_native_compiler_error_is_a_failed_load(self) -> None:
        error = (
            Path(__file__).parent / "testdata" / "compiler-error.jsonl"
        ).read_text()
        stderr = json.dumps(self.diagnostic(severity="warn")) + "\n" + error
        result = assess_lint(self.run_record(1, stderr=stderr), "logos-lint")
        self.assertEqual((result.status, result.diagnostics), ("load-failed", 2))
        self.assertEqual(
            assess_lint(self.run_record(0, stderr=stderr), "logos-lint").status,
            "invalid-report",
        )

    def test_malformed_untagged_diagnostic_is_not_a_compiler_error(self) -> None:
        error = (
            Path(__file__).parent / "testdata" / "compiler-error.jsonl"
        ).read_text()
        changes: tuple[tuple[str, object], ...] = (
            ("span", {}),
            ("provenance", {}),
            (
                "provenance",
                {"origin_expr": "", "origin_line": False, "inferred_type": ""},
            ),
            ("autofix_candidates", None),
            ("severity", "warn"),
            ("message", ""),
        )
        for field, value in changes:
            record: dict[str, object] = json.loads(error)
            record[field] = value
            with self.subTest(field=field):
                result = assess_lint(
                    self.run_record(1, stderr=json.dumps(record)), "logos-lint"
                )
                self.assertEqual(result.status, "invalid-report")
        empty_rule = json.dumps(self.diagnostic(rule=""))
        self.assertEqual(
            assess_lint(self.run_record(1, stderr=empty_rule), "logos-lint").status,
            "invalid-report",
        )

    def test_deny_diagnostic_explains_failure(self) -> None:
        run = self.run_record(1, stderr=json.dumps(self.diagnostic()))
        result = assess_lint(run, "logos-lint")
        self.assertEqual((result.status, result.diagnostics), ("findings", 1))

    def test_warning_does_not_explain_exit_one(self) -> None:
        warning = json.dumps(self.diagnostic(severity="warn"))
        self.assertEqual(
            assess_lint(self.run_record(0, stderr=warning), "logos-lint").status,
            "findings",
        )
        self.assertEqual(
            assess_lint(self.run_record(1, stderr=warning), "logos-lint").status,
            "invalid-report",
        )

    def test_plain_errors_are_not_hidden_by_one_valid_record(self) -> None:
        stderr = (
            json.dumps(self.diagnostic()) + "\n# broken\ninvalid C compiler input\n"
        )
        self.assertEqual(
            assess_lint(self.run_record(1, stderr=stderr), "logos-lint").status,
            "invalid-report",
        )

    def test_package_banners_require_following_diagnostics(self) -> None:
        diagnostic = json.dumps(self.diagnostic())
        valid = (
            "# example.com/module\n"
            + diagnostic
            + "\n# example.com/other [example.com/other.test]\n"
            + diagnostic
        )
        self.assertEqual(
            assess_lint(self.run_record(1, stderr=valid), "logos-lint").status,
            "findings",
        )
        trailing = diagnostic + "\n# example.com/missing\n"
        self.assertEqual(
            assess_lint(self.run_record(1, stderr=trailing), "logos-lint").status,
            "invalid-report",
        )
        consecutive = "# example.com/missing\n# example.com/other\n" + diagnostic
        self.assertEqual(
            assess_lint(self.run_record(1, stderr=consecutive), "logos-lint").status,
            "invalid-report",
        )

    def test_compiler_record_without_rule_is_not_a_lint_finding(self) -> None:
        run = self.run_record(1, stderr=json.dumps(self.diagnostic(rule="")))
        self.assertEqual(assess_lint(run, "logos-lint").status, "invalid-report")

    def test_invalid_utf8_is_an_invalid_report(self) -> None:
        run = self.run_record(0)
        Path(run.stderr).write_bytes(b"\xff")
        self.assertEqual(assess_lint(run, "logos-lint").status, "invalid-report")

    def test_gglint_counts_must_match_exit_status(self) -> None:
        empty = json.dumps({"mode": "standard", "findings": []})
        populated = json.dumps({"mode": "standard", "findings": [self.finding()]})
        self.assertEqual(
            assess_lint(self.run_record(1, empty), "gglint", "standard").status,
            "invalid-report",
        )
        self.assertEqual(
            assess_lint(self.run_record(0, populated), "gglint", "standard").status,
            "invalid-report",
        )
        self.assertEqual(
            assess_lint(self.run_record(1, populated), "gglint", "standard").status,
            "findings",
        )

    def test_gglint_typecheck_is_incomplete_analysis(self) -> None:
        document = json.dumps(
            {"mode": "strict", "findings": [self.finding("typecheck")]}
        )
        self.assertEqual(
            assess_lint(self.run_record(1, document), "gglint", "strict").status,
            "load-failed",
        )

    def test_gglint_mode_mismatch_is_not_success(self) -> None:
        document = json.dumps({"mode": "basic", "findings": []})
        self.assertEqual(
            assess_lint(self.run_record(0, document), "gglint", "strict").status,
            "invalid-report",
        )

    def test_gglint_duplicate_findings_violate_report_contract(self) -> None:
        document = json.dumps(
            {"mode": "basic", "findings": [self.finding(), self.finding()]}
        )
        self.assertEqual(
            assess_lint(self.run_record(1, document), "gglint", "basic").status,
            "invalid-report",
        )

    def test_repeat_ignores_order_but_preserves_duplicates(self) -> None:
        first = self.diagnostic("a")
        second = self.diagnostic("b")
        self.assertEqual(canonical([first, second], ""), canonical([second, first], ""))
        self.assertNotEqual(canonical([first], ""), canonical([first, first], ""))

    def test_repeat_change_fails(self) -> None:
        run = self.run_record(1)
        first = Assessment("findings", diagnostics=1, normalized_sha256="one")
        second = Assessment("findings", diagnostics=2, normalized_sha256="two")
        self.assertEqual(
            compare_repeats(first, second, run, run).status, "nondeterministic"
        )
        self.assertEqual(
            compare_repeats(first, Assessment("timeout"), run, run).status, "timeout"
        )

    def test_candidate_requires_successful_oracle(self) -> None:
        run = self.run_record(1)
        self.assertEqual(assess_build(run, "oracle", None).status, "upstream-failed")
        self.assertEqual(
            assess_build(run, "go", Assessment("pass")).status, "candidate-failed"
        )
        self.assertEqual(assess_build(run, "go", None).status, "uncompared")
        self.assertEqual(
            assess_build(replace(run, returncode=0), "go", None).status, "uncompared"
        )

    def test_oracle_panic_does_not_blame_candidate(self) -> None:
        run = self.run_record(1, stderr="panic: upstream test failed\n")
        self.assertEqual(assess_build(run, "oracle", None).status, "upstream-failed")

    def test_dependency_failure_is_blocked(self) -> None:
        run = self.run_record(1, stderr="module lookup disabled by GOPROXY=off\n")
        self.assertEqual(assess_build(run, "oracle", None).status, "blocked")

    def test_empty_package_pattern_is_not_a_compile_pass(self) -> None:
        for error in (
            'go: warning: "./..." matched no packages',
            "no Go files in /example",
            "no non-test Go files in /example",
        ):
            with self.subTest(error=error):
                run = self.run_record(0, stderr=error + "\n")
                self.assertEqual(
                    assess_build(run, "oracle", None).status, "load-failed"
                )

    def test_quoted_source_message_is_not_a_crash(self) -> None:
        record = self.diagnostic()
        record["message"] = (
            'error result of errors.New("internal compiler error") is dropped'
        )
        run = self.run_record(1, stderr=json.dumps(record))
        self.assertEqual(assess_lint(run, "logos-lint").status, "findings")

    def test_timeout_keeps_partial_output_and_is_not_success(self) -> None:
        run = execute(
            (
                sys.executable,
                "-c",
                "import time; print('started', flush=True); time.sleep(10)",
            ),
            self.root,
            {},
            self.root / "deadline",
            0.25,
        )
        self.assertTrue(run.timed_out)
        self.assertIn("started", Path(run.stdout).read_text())
        self.assertEqual(assess_lint(run, "logos-lint").status, "timeout")

    @unittest.skipUnless(fcntl is not None, "process-group lock probe requires POSIX")
    def test_interrupt_terminates_descendants_and_reaps_command(self) -> None:
        locks = tuple(self.root / f"process-{index}.lock" for index in range(3))
        child = ""
        for index in reversed(range(3)):
            spawn = (
                f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
                if child
                else "print('tree-ready', flush=True)\n"
            )
            child = (
                "import fcntl, subprocess, sys, time\n"
                f"lock = open({str(locks[index])!r}, 'w')\n"
                "fcntl.flock(lock, fcntl.LOCK_EX)\n"
                f"{spawn}time.sleep(30)\n"
            )
        processes: list[subprocess.Popen[bytes]] = []
        original_wait = subprocess.Popen[bytes].wait
        log = self.root / "interrupted"

        def cleanup() -> None:
            for process in processes:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                original_wait(process, timeout=5)

        self.addCleanup(cleanup)

        def interrupt_wait(
            process: subprocess.Popen[bytes], timeout: float | None = None
        ) -> int:
            if processes:
                return original_wait(process, timeout=timeout)
            processes.append(process)
            deadline = time.monotonic() + 5
            while "tree-ready" not in log.with_suffix(".stdout").read_text():
                if time.monotonic() >= deadline:
                    self.fail("grandchild did not start before interruption")
                time.sleep(0.01)
            raise KeyboardInterrupt

        with (
            patch.object(subprocess.Popen, "wait", interrupt_wait),
            self.assertRaises(KeyboardInterrupt),
        ):
            execute((sys.executable, "-c", child), self.root, {}, log, 10)
        self.assertIsNotNone(
            processes[0].returncode, "interrupted command was not reaped"
        )
        self.assertIn("tree-ready", log.with_suffix(".stdout").read_text())
        assert fcntl is not None
        for lock in locks:
            with lock.open("rb") as stream:
                deadline = time.monotonic() + 2
                while True:
                    try:
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            self.fail(f"descendant still holds {lock.name}")
                        time.sleep(0.01)

    def test_changed_source_changes_hash(self) -> None:
        source = self.root / "source.go"
        source.write_text("package source\n")
        initial = tree_hash(self.root, ("source.go",), "example.com/source@v1.0.0")
        source.write_text("package modified\n")
        self.assertNotEqual(
            initial, tree_hash(self.root, ("source.go",), "example.com/source@v1.0.0")
        )

    def test_source_path_cannot_escape(self) -> None:
        with self.assertRaises(ValueError):
            within(self.root, "../outside")

    def test_module_without_origin_requires_undeclared_revision_and_valid_sum(
        self,
    ) -> None:
        module = "example.com/source"
        version = "v1.0.0"
        source = self.root / f"{module}@{version}"
        source.mkdir(parents=True)
        (source / "go.mod").write_text(f"module {module}\n")
        digest = tree_hash(source, ("go.mod",), f"{module}@{version}")
        metadata = self.root / "cache" / "download" / module / "@v"
        metadata.mkdir(parents=True)
        (metadata / f"{version}.info").write_text(json.dumps({"Version": version}))
        case = Case(
            "source",
            "module",
            f"{module}@{version}",
            module,
            "https://example.com/source",
            "",
            version,
            digest,
            "reference",
            "go-only",
            0,
            (),
            ("./...",),
            "test",
        )
        self.assertEqual(
            verify_source(case, self.root, self.root, self.root).digest, digest
        )
        with self.assertRaises(TypeError):
            verify_source(
                replace(case, revision="a" * 40), self.root, self.root, self.root
            )
        (source / "go.mod").write_text("module example.com/changed\n")
        with self.assertRaises(ValueError):
            verify_source(case, self.root, self.root, self.root)

    def test_package_patterns_cannot_inject_flags(self) -> None:
        manifest: dict[str, object] = {
            "id": "source",
            "source_kind": "git",
            "source": "source",
            "module": "example.com/source",
            "upstream": "https://example.com/source",
            "revision": "a" * 40,
            "cohort": "reference",
            "backend": "go-only",
            "cgo": 0,
            "operation": "test",
            "patterns": ["-exec=unwanted"],
        }
        with self.assertRaises(ValueError):
            Case.parse(manifest)


class PolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.policy_file = self.root / "policies" / "source.toml"
        self.policy_file.parent.mkdir()
        self.policy_file.write_text(
            '[capabilities]\ntransitive_unsafe_allowlist = ["example.com/dependency"]\n'
        )
        self.manifest: dict[str, object] = {
            "id": "source",
            "source_kind": "git",
            "source": "source",
            "module": "example.com/source",
            "upstream": "https://example.com/source",
            "revision": "a" * 40,
            "cohort": "reference",
            "backend": "go-only",
            "cgo": 0,
            "operation": "test",
            "logos_policy": "policies/source.toml",
            "logos_policy_sha256": file_hash(self.policy_file),
        }
        self.case = Case.parse(self.manifest)
        policy = verify_policy(self.case, self.root)
        assert policy is not None
        self.policy = policy
        source = self.root / "source"
        source.mkdir()
        (source / "go.mod").write_text("module example.com/source\n")
        files = ("go.mod",)
        self.source = Source(source, files, tree_hash(source, files))
        self.work = self.root / "work"

    def test_policy_fields_require_a_valid_pair(self) -> None:
        changes: tuple[dict[str, object], ...] = (
            {"logos_policy": ""},
            {"logos_policy_sha256": ""},
            {"logos_policy_sha256": "z" * 64},
            {"logos_policy_sha256": "a" * 63},
            {"logos_policy": 1},
        )
        for change in changes:
            with (
                self.subTest(change=change),
                self.assertRaises((ValueError, TypeError)),
            ):
                Case.parse({**self.manifest, **change})

    def test_policy_paths_reject_traversal_and_symlinks(self) -> None:
        for name in ("../source.toml", str(self.policy_file)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                Case.parse({**self.manifest, "logos_policy": name})
        (self.root / "linked.toml").symlink_to(self.policy_file)
        (self.root / "linked").symlink_to(
            self.policy_file.parent, target_is_directory=True
        )
        for name in ("linked.toml", "linked/source.toml"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify_policy(replace(self.case, logos_policy=name), self.root)

    def test_policy_tampering_is_rejected(self) -> None:
        self.policy_file.write_text(self.policy_file.read_text() + "# changed\n")
        with self.assertRaisesRegex(ValueError, "pinned SHA-256"):
            verify_policy(self.case, self.root)

    def test_policy_cannot_weaken_defaults_or_grant_globs(self) -> None:
        for text in (
            '[capabilities]\ntransitive_unsafe = "allow"\n',
            '[capabilities]\ntransitive_exec = "allow"\n',
            '[capabilities]\ntransitive_unsafe_allowlist = ["example.com/**"]\n',
            '[capabilities]\ntransitive_exec_allowlist = ["example.com/**"]\n',
            '[capabilities]\ntransitive_unsafe_allowlist = ["example.com/../dependency"]\n',
            '[capabilities]\ntransitive_unsafe_allowlist = ["example.com/dep", "example.com/dep"]\n',
            '[capabilities]\nreflect_allowlist = ["example.com/dependency"]\n',
            '[lints]\nstyle = "allow"\n',
            "[capabilities]\ntransitive_unsafe_allowlist = []\n",
        ):
            self.policy_file.write_text(text)
            case = replace(self.case, logos_policy_sha256=file_hash(self.policy_file))
            with self.subTest(text=text), self.assertRaises(ValueError):
                verify_policy(case, self.root)

    def test_policy_accepts_reviewed_execution_grants(self) -> None:
        self.policy_file.write_text(
            '[capabilities]\ntransitive_exec_allowlist = ["example.com/driver"]\n'
        )
        case = replace(self.case, logos_policy_sha256=file_hash(self.policy_file))
        policy = verify_policy(case, self.root)
        assert policy is not None
        self.assertEqual(policy.content, self.policy_file.read_bytes())

    def test_policy_is_isolated_to_logos_stages(self) -> None:
        for stage in ("oracle", "go", "logos", "logos-lint", "gglint"):
            inputs = prepare_source(self.source, self.work, stage, self.policy)
            applied = stage in ("logos", "logos-lint")
            self.assertEqual((self.work / "logos.toml").exists(), applied)
            if applied:
                self.assertEqual(inputs["policy_sha256"], self.policy.sha256)
                self.assertNotEqual(
                    inputs["effective_source_digest"], self.source.digest
                )
            else:
                self.assertNotIn("policy_sha256", inputs)
                self.assertEqual(inputs["effective_source_digest"], self.source.digest)
            self.assertFalse((self.source.path / "logos.toml").exists())
            self.assertEqual(
                tree_hash(self.source.path, self.source.files), self.source.digest
            )

    def test_repeat_restores_pristine_source_and_exact_policy(self) -> None:
        first = prepare_source(self.source, self.work, "logos-lint", self.policy)
        (self.work / "go.mod").write_text("module modified\n")
        (self.work / "logos.toml").write_text("changed\n")
        repeated = prepare_source(self.source, self.work, "logos-lint", self.policy)
        self.assertEqual(first, repeated)
        self.assertEqual((self.work / "logos.toml").read_bytes(), self.policy.content)
        self.assertEqual(
            (self.work / "go.mod").read_bytes(),
            (self.source.path / "go.mod").read_bytes(),
        )

    def test_policy_never_overwrites_upstream_configuration(self) -> None:
        original = b"# upstream policy\n"
        (self.source.path / "logos.toml").write_bytes(original)
        files = (*self.source.files, "logos.toml")
        source = replace(
            self.source, files=files, digest=tree_hash(self.source.path, files)
        )
        with self.assertRaisesRegex(ValueError, "overwrite"):
            prepare_source(source, self.work, "logos", self.policy)
        self.assertEqual((self.source.path / "logos.toml").read_bytes(), original)
        self.assertEqual((self.work / "logos.toml").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
