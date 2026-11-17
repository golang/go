# Transitive SIMD compiler imports

The exact native head `9909feb26fe92b124f17c11fdf0022be7a643188` fails
`TestCompileSIMDTaggedMethodFacts` in both driver profiles. The producer is
the unchanged real `src/simd/testdata/iface/iface.go`; the ordinary consumer
imports only that producer and invokes its exported method. The installed
build reports a missing `simd/internal/bridge` archive and an undefined
`simd` qualifier. The full failed method event stream and before/after
source and executable identities are frozen here from run 43951103189,
unit job 133593503956, artifact 14272030204. Mutation-mask failures in that
same event stream belong to the separately recorded zero-array repair.

## Preconditions and intended invariants

The input has already passed source type checking. SIMD rewriting on
amd64, arm64, and wasm can introduce uses of SIMD support packages even
when the source imports only a SIMD-dependent producer. The build action
must expose the already-built transitive simd and bridge archives to that
compiler and hash their contents. This must not authorize source imports
of internal packages, add unrelated standard-library archives, use stale
or absent archives, or change dry-run and experiment-off behavior.

Every generated qualifier must have a binding in its own file. Compiler
aliases must survive source aliases, dot/blank imports, and user variables
named simd or bridge. Type-only rewrites require the same imports as
function rewrites. Keeping an import used must introduce no new runtime
call. The simd package's special internal rewriting keeps its established
bridge binding.

## Frozen controls

The new tests precede the production change. There are 37 named tests:
11 work/action-identity names, seven rewriter names, and 19 installed
compiler names. These include 32 leaf cases. All real-source consumers
compile only; no generated SIMD instructions execute. The original tagged
method test and actual producer remain byte-identical.

The work tests cover supported and unsupported architectures, absent or
disabled experiments, direct imports, dry runs, unbuilt archive markers,
and build-cache content identity. The rewriter tests type-check generated
ASTs under six namespace arrangements and reject added init-time calls.
Installed cases use eight consumer forms under both fork Go and LoGOs,
including a type-only wrapper and an explicit internal-import denial.

`frozen.json` binds the old source, tests, actual producer, and original
hosted failure. The archived importcfg implementation must fail the exact
amd64 archive-set case. The archived rewriter must fail the exact missing
binding case. Both failures must be semantic, with exact named events and
no build failure, skip, crash, timeout, or unrelated diagnostic. Candidate
acceptance must execute the complete frozen inventory and retain the
original real-source tagged-method gate.

## Status

The original hosted failure is observed. New local Go execution is
prohibited for this task; only source inspection and formatting have run.
The frozen negative controls, candidate inventory, fresh installed build,
and full upstream gate remain pending. This checkpoint claims no new Go
pass.
