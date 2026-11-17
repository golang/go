# CTFE map-field store acceptance

The frozen positive fixture passed the scoped installed-compiler checks on
2026-11-16 after the executor recovered. See
[the execution record](EXECUTED-2026-11-16.md) and `summary.json` for the
exact source, commands, raw-log identities, and observed tests.

This is a test-oracle correction for an already supported fresh local map
store. The original `storeField` declaration is preserved byte for byte.
The positive fixture requires the compile-time value 4 through a type-level
array assignment and requires output `4` followed by LF. No CTFE
implementation or normative specification changes.

The earlier unexecuted preparation is retained unchanged under
`preparation/`; its remote reconstruction predates these later execution
results. `const_func_unsupported_store.original.lgo` preserves the obsolete
negative input, and [the source review](source-review.md) explains its
conflict with the existing assignment contract. The initial misselected
testdir invocation and its generated-GOROOT failure remain in `testdir/`;
they are excluded from acceptance.

The tested compiler belongs to clean public `45c29ff2`. The temporary
candidate GOROOT view supplied the added testdir fixture while using that
same installed compiler. This does not replace a fresh build or complete
acceptance of subsequent cmd/go changes or upstream integrations.

`evidence.sha256` identifies every copied record. Test executables, caches,
and live temporary source trees are excluded.
