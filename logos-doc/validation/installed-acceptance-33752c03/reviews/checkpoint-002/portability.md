# Execution-queue portability review

## Verdict

The frozen execution-queue test is not portable to ordinary Windows host
settings in its current form. Source proves two executable-boundary
failures: its helper files have no executable suffix, and its missing-file
assertion expects an error different from Windows lookup's result. No
Windows runtime was executed in this review. These findings do not change
the actual Linux RED, GREEN, mutation or restored-race observations.

The existing parity harness already has Windows host assumptions before
the new scenarios run. This review does not establish a newly introduced
Windows failure in an otherwise working whole-package gate, or a defect
in the channel limiter. Keep the frozen Linux proof unchanged and require
separate Windows evidence before claiming that host.

## Concrete executable trace

1. `TestRunBinaryExecutionQueue` supplies an explicit output path named
   `helper`. The non-directory `-o` branch in
   `src/cmd/go/internal/work/build.go` assigns `p.Target = cfg.BuildO`
   directly. Its automatic executable-suffix paths are different branches.
2. The existing `cloneBinary` in
   `src/cmd/internal/goparity/parity_test.go` hardlinks or copies bytes to
   the exact requested destination. It adds no suffix. The new destinations
   are `holder0`, `holder1`, `control`, `queued`, `probe` and `timeout0`
   through `timeout2`.
3. `runAll` and `runBinaryTimeout` pass those absolute paths to
   `exec.CommandContext`. On Windows, `src/os/exec/exec.go` resolves an
   absolute command through `lookExtensions` when constructing the command.
4. `src/os/exec/lp_windows.go` obtains a normal nonempty extension list from
   PATHEXT or its `.com`, `.exe`, `.bat`, `.cmd` defaults. For a path without
   an extension, `findExecutable` tries only the appended extensions. It
   does not accept the existing extensionless file. It returns
   `exec.ErrNotFound`, so the first holder cannot write a readiness marker.
5. The start-error scenario requests the likewise suffixless `missing0`
   through `missing2`. The same lookup returns `exec.ErrNotFound`, which is
   a separate `errors.New` sentinel. `exec.Error.Unwrap` exposes that
   sentinel, and the later `%w` wrapping preserves it. Therefore
   `errors.Is(err, fs.ErrNotExist)` is false. This assertion would fail even
   if earlier helper setup were made reachable on Windows.

## Existing host boundary

Before building the new helper, `forkTool` calls `os.Stat` on `bin/go`
without `.exe`. A normal Windows toolchain installs `go.exe`; no suffix
resolution is performed by `os.Stat`. The existing fake-toolchain tests
also write shell scripts with a `#!/bin/sh` header. Those are prior harness
limitations and are not repaired by an execution-admission change.

## Process-group fallback

The shared-queue and missing-executable scenarios do not require a process
group to establish their nominal assertions. Their helper program spawns
no children. The `!unix` implementation in `parity_kill_other.go` calls
`cmd.Process.Kill()` for the direct child. The timeout scenario checks group
disappearance only when `processGroupKillSupported()` is true. Its ordinary
timeout cancellation still kills the direct helper through the fallback.

The outer scenario watchdog can kill a process with helper descendants.
The test retains release-marker cleanup for all known helper names and a
30-second helper self-deadline. This source review found no additional
non-Unix deadlock that would justify a blanket process-group-support skip.

Plan 9's `lookExtensions` is a no-op, and its explicit-path lookup checks
the actual filename. Its syscall error mapping recognizes missing files
as `fs.ErrNotExist`. The Windows filename and error mismatch therefore
cannot be generalized to all non-Unix hosts. No Plan 9 acceptance result
is claimed.

## Preserved scope

No fixture, assertion, source file, timeout, host guard or manifest was
modified. The adjacent review receipt records the exact source hashes used
for this trace and the separate phase-two Linux gate results.
