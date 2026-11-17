# Correct two ambiguous proving-path phrases

Hosted compiler run 43939662751, job 133553484639, failed its final
`TestRedteamLedgerRows` gate at published head
`1a4c7635f38dbf303833ff8e718f95541558f8be`. All preceding configured compiler,
command, cgo/ejection, testdir and vet steps passed. The failed gate reported
two nonexistent proving paths: `test/root` in row #336 and `test/tool` in
row #441.

The unchanged validator recognizes path-shaped substrings in a proof cell.
The prose `shortest/root` therefore contains `test/root`; `source/test/tool`
contains `test/tool`. This follow-up spells those phrases out. It changes no
test path, test assertion, attack status, compiler source, specification rule,
or release requirement. The 346 attack rows and 302 task records retain their
statuses and identities.

## Direct validator replay

The test binary was compiled from the clean installed-acceptance source
`e96a570da271c43a80e23bb1a188aac054acff25`, using its freshly installed Go
driver and compiler. Both validator source files are byte-identical in the
published candidate and the build source; `frozen-tests.sha256` records them.
Compilation exited zero with an empty diagnostic log. The binary SHA-256 is
`59c88e7bd1cef0e36a5a1e559f12fead791a427afd1c75b704e4a43183d1f61d`.

For these file-only tests, `GOROOT` selects the candidate metadata tree through
the existing `forkGoroot` function. The selected tests read ledger rows and
check proving-file existence; they execute no compiler or parity program.
No binary from the candidate's historical `bin` link is invoked.

The exact uncached selection was:

```sh
GOROOT=/workspace/scratch/9949907ca71e/logos-acceptance-fixes \
  /workspace/scratch/9949907ca71e/recovery-evidence/ledger-prose-correction/goparity.test \
  -test.run='^TestRedteamLedgerRows$' -test.count=1 -test.v -test.timeout=5m
```

- Baseline RED reproduces exactly the two hosted diagnostics.
- The two prose replacements make the unchanged validator pass.
- Restoring only the first ambiguous phrase reproduces only row #336's error.
- Restoring only the second reproduces only row #441's error.
- Exact candidate restoration passes both `TestRedteamLedgerRows` and
  `TestRedteamLedgerRelease`, with `-goparity.release` selecting the latter.

The final filtered invocation is a ledger validation only. It is not the full
parity suite and creates no release stamp. `probes.json` records the actual
commands, exits, expected diagnostics, binary identity and final ledger hash.
The raw logs and exit records are preserved beside it. The compiled test binary
and full pre-correction ledger snapshot remain local intermediates; the exact
original ledger is already available at the published commit above.

## Hosted boundary

The corrected metadata requires a new PR tip and fresh hosted prerequisite and
stamp results. Prior outcomes remain attached to their actual commits. Source
tree `e237c399aae8f8065b724e8a0e27585686c7da18` is unchanged, so previously
completed installed behavioral evidence retains its scope. M6-S3b's blocked
rows and patch remain unchanged, and this documentation repair does not reopen
that implementation.
