# Native Windows parity requirement review

No explicit requirement to execute `cmd/internal/goparity` natively on
Windows appears in the reviewed SPEC or IMPLEMENTATION text.

M22-A-S3 defines the fast, release and oracle-off parity gates without
naming a host operating system. M22-A-S5 defines the complete stamped
release invocation and its assertions, again without a Windows host leg.
The release gate requires both driver optimization coverage, the plain-Go
regression, a second reproducibility host and actual arm64 execution. It
does not name native Windows parity execution. SPEC 17.3 names the
reproducibility targets linux/amd64, linux/arm64 and js/wasm. Compilation
for a target and execution on a native host are distinct evidence.

This finding does not narrow the toolchain's compatibility contract.
SPEC 0.1 keeps Go language semantics without a host-specific exception.
M0-S1 explicitly requires a Windows `.exe` suffix when installing the
logos driver. The other Windows references in these two documents concern
capability sink discovery for Windows APIs. They establish installation
and target-analysis obligations, rather than an additional native Windows
parity run in the current release matrix.

The source-derived Windows helper-naming and missing-error limitations
remain recorded in checkpoint-002/portability.md, alongside the existing
forkTool and fake-toolchain host assumptions. No guard, test assertion,
source file or acceptance exception was changed. Current actual proof is
native Linux. No Windows execution result is claimed.

This checkpoint also reviews the separate native void-hook mutation.
Only the call to `check.logosRecordVoidPanic(call)` is removed in an
external compiler overlay. The resulting compiler is separate from the
installed compiler. The same frozen assertion binary reports exactly the
two builtin void-function failures, while the other seven cells pass.
Restoring selection to the installed compiler passes all nine cells.
Historical layout metadata retains its preparation-time pending flag;
the later compile and execution receipts establish the completed outcome.
