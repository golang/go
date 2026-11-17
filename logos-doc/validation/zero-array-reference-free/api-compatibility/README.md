# Frozen compiler-fixture API failure

Native CI run `43967559240`, unit job `133650880035`, tested
`f4242fed9cf0707a4103a0adbdb82e866bcabd69` with source tree
`ee5ae95824022485b74d4e3186a1540ef7124f76`.

Both classification phases failed to build because the fixture called
`NewInterfaceType(...).Complete()` at lines 19 and 99. Compiler-internal
`types2.Interface` has no exported `Complete` method. Its constructor returns
the already complete empty-interface singleton or sets `complete = true`;
type-set computation is lazy. Neither classification phase ran a named test.
The gate correctly rejected these build failures.

The separate old-mutation overlay ran its exact two named failures and emitted
the required `ParamMutated="1"` diagnostic. The candidate mutation phase ran
all 15 named tests successfully. Before/after source and tool identities match.
The production zero-length-array predicate and mutation expectations are not
implicated by this fixture API error.

This directory freezes the original type fixture, original checker, original
manifest, every zero-array phase receipt and raw stream, complete unit log,
complete original unit artifact ZIP, and GitHub/source provenance. All inputs
are listed in `FROZEN_FAILURES.sha256`. Their bytes remain unchanged after the
compatibility correction. The original `../FROZEN.sha256` also remains intact.

The authorized correction removes only the two known `.Complete()` call
suffixes from the active type fixture. A separate replacement manifest must
bind the 20 unchanged original inputs, archived original fixture, corrected
active fixture, original manifest, and retained failure inputs. The hosted
gate must additionally enforce that exact byte transformation. Every test name,
assertion, expected classification, mutation mask, and negative-control outcome
remains unchanged.

No local Go command was executed. The corrected classification phases require
a fresh hosted run; this failed run remains preserved as failed.
