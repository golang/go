# Provenance and fixture-scope corrections

The original evidence comes from failing head
`9909feb26fe92b124f17c11fdf0022be7a643188`. The archived old partition helper and
its test file are byte-identical to that head. The complete archived workflow
comes from implementation base `ed3b743476c4665e89f18f6c2f3d090bbf3610b8`; its
inventory, complement, and cmd/go execution block is byte-identical to the
failing workflow, while other steps have changed. The archived full ed3
workflow is not presented as the complete workflow used in the failed run.

An independent inert-file reproduction exposed a missing identity boundary:
the first receipt helper checked only `src`, so it accepted a tracked fixture
change under top-level `test/` with unchanged HEAD and tools. That helper and
the original inert-workspace test file are preserved byte-for-byte in
`identity-src-only/`, with the observed failing boundary in `red.json`.

The original synthetic workspace committed only `src/input.txt`. To represent
the real Go repository before requiring `HEAD:test`, its setup now also commits
one inert file under `test/`. Every existing test body and assertion remains
unchanged. Separate new controls require tracked and untracked fixture changes
to fail and require exact HEAD, src-tree, and test-tree receipts. These controls
are frozen before correcting the script; no missing-test-tree fallback is added.

The correction records both native trees and checks clean root-relative status
for `src test`. It changes no Go source or fixture assertion. This closes the
specific compiler-fixture provenance gap without claiming that the old hosted
run had installed-tool or fixture-tree receipts.
