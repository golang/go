# GoX implementation plan — three milestones

Status: experimental implementation delivered across all three milestones.
Checked tasks have implementation and local evidence; unchecked tasks remain
release gates or precision/coverage follow-ups. Contract: [SPECIFICATION.md](SPECIFICATION.md).

Base inspected: `64f1787400cb9507e355a13bf5f25c7158dabe4d` on
`feature/shared-toolchain`. Develop in an isolated branch/worktree; do not switch,
reset or modify another session's checkout. Reconcile fresh upstream changes
before implementing the integration milestone, particularly launcher changes.

## Milestone 1 — Complete explicit syntax, lowering and component runtime

Deliver all discussed feature families together: nested SEs, capturing named
inner functions, typed block expressions, structured X markup, and component
declarations with compiler-injected instance/hook context. Deliver the minimal
headless component runtime in this milestone. Bare JSX spelling follows in M2.
This is deliberately not an SE-only milestone. Completion means compilable,
semantically tested Go output, not just token recognition.

### M1.1 Establish the reusable module and upstream baseline

- [x] Create root `gox/` module, separate from src/go.mod and src/cmd/go.mod.
- [x] Copy scanner/parser implementation and relevant tests from the pinned
  repository baseline; retain BSD notices and LICENSE, record import adaptations.
- [x] Inventory dependencies on APIs added since the intended minimum host Go
  version. Set and verify the actual minimum; baseline source currently uses
  Go 1.28 module directives and scanner.End.
- [ ] Separate scanner, parser/extension records, lowering, source maps and API
  packages; keep CLI, filesystem orchestration and subprocesses outside the core.
- [x] Add ordinary-Go corpus differential harness and deterministic fixtures.

### M1.2 Adapt scanner for every syntax family

- [ ] Introduce extension token wrapper and mode stack without changing global
  standard token definitions.
- [x] Implement uppercase F quoted/raw SE prefixes, text escapes, doubled braces
  and nested interpolation code; retain source spans and original spelling.
- [x] Preserve ordinary Go tokenization for strings, runes, comments, numbers,
  Unicode identifiers and composite literals inside interpolations.
- [x] Recognize contextual do/component forms without reserving ordinary identifiers;
  add structured X wrapper/tag/text modes, quoted attributes and code holes.
- [x] Treat quotes inside attributes and interpolations as nested syntax, not outer
  terminators. Close the X wrapper only after a complete root element. Test
  multiline X, nested F/X and missing inner/outer delimiters.
- [x] Preserve Go semicolon insertion in code; specify end-expression tokens and
  the optional terminal interpolation semicolon interaction.
- [x] Bound nesting, recover from EOF and malformed input, guarantee progress.

### M1.3 Adapt parser for every syntax family

- [x] Parse SE pieces into records holding recursively parsed Go expressions.
- [x] Parse local named declarations as statement constructs, with receivers,
  local generics and missing bodies rejected explicitly.
- [x] Parse do Type Block, including brace-bearing and instantiated type syntax.
- [x] Parse X elements/components, ordered attrs/children, expression holes and
  matching tags; reject duplicate attrs and unsupported shorthand.
- [x] Parse component declarations, typed parameters and optional children input;
  represent component definitions separately from ordinary function declarations.
- [ ] Represent extensions via node-identity records and placeholders, retaining
  standard Go AST nodes for unaffected syntax; avoid impossible external ast.Expr
  implementations and magic source identifier conventions.
- [ ] Preserve comments/directives and original positions through recursive parsing.

### M1.4 Lower and validate semantics

- [x] Implement deterministic identifier allocation and import handling across
  all original identifiers; add helper strategy for shadowed imports.
- [x] Lower SEs with sequential expression evaluation and immediate per-value
  formatting. Cover literal percent signs, typed nil, untyped nil, multi-result
  rejection and no-interpolation constants.
- [x] Lower named functions in place to declaration plus assignment, allowing
  self-recursion without adding forward visibility or automatic hoisting.
- [x] Lower typed blocks to IIFEs with ordinary Go control-flow boundaries.
- [x] Lower X into parameterized template functions with sequential single-value
  slots; preserve typed property checks, nil handling and callbacks without
  stringifying values into source. Component nodes describe deferred invocations.
- [x] Generate component render functions with an explicit instance parameter;
  keep local declarations/captures in place. Generate stable definition/slot IDs.
- [x] Resolve hook forms lexically, inject instance/slot arguments and diagnose
  prohibited hook positions rather than rewriting ordinary user functions.
- [x] Check named component props, required/unknown attrs, children contract and
  reserved key metadata; generate typed render adapters before erasing inputs.
- [ ] Assert no extension placeholders survive, emit go/format output and map
  diagnostics to original spans. Compile emitted source with the target compiler.

### M1.5 Persistent component runtime

- [x] Implement root-owned registry and persistent instances separate from VNodes;
  store current inputs/callbacks and refresh captures on parent updates.
- [x] Reconcile by parent + definition + key/position, reject duplicate keys, and
  retain keyed instances across reorder. Do not use source sites or closure
  addresses as occurrence identities.
- [x] Implement typed state/ref slots, replacement and functional state updates,
  mount-only storage initialization and stale-setter lifetime protection.
- [ ] Implement typed contexts, defaults, nested providers, consumer subscriptions
  and provider-driven invalidation.
- [x] Serialize scheduling/render/commit per root; enqueue updates, expose Flush,
  reject render-phase setters, and make the cross-goroutine enqueue path safe.
- [x] Implement unmount cleanup, panic bookkeeping recovery and headless host
  tree/event adapter; keep host mutations outside render.
- [x] Document v1 hook placement restrictions, snapshot callbacks, ref threading,
  no live-edit state migration and deferred effects/production renderers.

### M1.6 Required semantic and error matrix

| Area | Required examples/assertions |
| --- | --- |
| Ordinary Go | Token/AST compatibility on upstream corpus; comments and semicolons |
| SE text | Quoted/raw, braces, percent signs, escapes, UTF-8, CRLF, empty text |
| Recursive syntax | SE inside SE, block inside SE, X inside block/SE, SE in attrs |
| Delimiters | Composite literals, funcs, comments, nested quotes, unterminated modes |
| Ordering | Calls plus variable reads, mutating Stringer/Formatter, exactly-once use |
| Type errors | Multi-value operands, missing return, invalid result type, const contexts |
| Capture | Outer mutation before/after calls, shadowing, escape after outer return |
| Functions | Self-recursion, duplicate name, before-declaration use, explicit mutual recursion |
| Loops | := per-iteration capture, = shared variable, pointer/slice aliasing |
| Blocks | Local loops, defer, panic/recover, illegal outer return/break/goto assumptions |
| Markup | X nested quotes, native tags, component props, nil/text/Node/slice children, attrs order |
| Instances | Two occurrences of one definition isolate state; local captures refresh |
| Identity | Keyed reorder, unkeyed position, duplicate keys, type/key reset, remount |
| Hooks | State/ref persistence, context defaults/shadowing/subscriptions, invalid placements |
| Scheduling | State rerenders, refs do not, ordered updates, latest-state updater, queued goroutines |
| Lifetime | Unmount cleanup, stale setter ignored, render panic recovery, event callback refresh |
| Hygiene | User names resembling generated names, shadowed fmt/strings, import aliases |
| Recovery | Multiple diagnostics with correct positions; no hangs or bogus successful output |

Exit gate: every syntax family has valid and invalid fixtures, mixed recursive
cases execute correctly, copied Go tests pass under the supported host toolchain,
and generated outputs compile without a modified compiler. A headless counter
rerenders after an event, keyed siblings preserve isolated state, refs persist
without scheduling, and context updates reach the correct consumers. Core API
is usable from tests; user-facing build orchestration is milestone 2. Record unsupported
cases as explicit diagnostics rather than silently dropping syntax.

## Milestone 2 — Bare JSX and standalone developer tooling

Depends on M1. Deliver an independently usable transformer and documented local
module/workspace workflow before touching the Go command's package loader.

### M2.1 Bare JSX as an alternative entry syntax

- [x] Let the parser enter markup at expression-start < without consuming it as
  ordinary Go lookahead; preserve comparisons, <= and channel receive <-.
- [x] Reuse X element records, hole handling, component lowering and runtime.
  Do not create a second JSX implementation or change ordinary .go parsing.
- [ ] Test paired X/bare fixtures for equivalent normalized lowering, side-effect
  ordering, diagnostics, hook identity within each build and runtime output.
- [ ] Cover return/assignment/call/composite contexts, generic syntax, comments,
  newlines/semicolons and malformed tags. Report markup errors without fallback.
- [x] Document uppercase F, structured X and bare JSX together; old s/view syntax
  is not accepted as an alias.

### M2.2 API and commands

- [ ] Stabilize TransformFile/TransformPackage inputs, dependency metadata,
  diagnostics and source maps; add cancellation at package orchestration level.
- [x] Implement emit/check/build/run/test, --help, output options and argument
  forwarding after --. Distinguish syntax checking from full type checking.
- [x] Select the Go command explicitly; support executable and this repository's
  Python-launcher distributions through an argv-based launcher abstraction.
- [x] Preserve stdout/stderr/stdin and process exit status without shell quoting.
- [x] Make output writes atomic and explicit, never overwrite source input.

### M2.3 Package transformation and dependencies

- [x] Select files using target suffixes, build tags, test mode and language
  version before transforming; include mixed .go/.gox and .gox-only packages.
- [x] Use a temporary filesystem overlay over the existing module/workspace
  graph, preserving relative replacements, embeds/assets and test package separation.
- [x] Resolve module/workspace/replacement roots through Go module metadata;
  document eager standalone preparation and the runtime dependency requirement.
- [ ] Aggregate sibling declarations for name hygiene and import validation.
- [x] Pin the goxview runtime dependency; provide reproducible local examples
  without requiring automatic module edits/downloads.
- [x] Cache transformation by source bytes, configuration and transformer identity;
  use atomic writes and concurrent-process tests.

### M2.4 Diagnostics, robustness and documentation

- [ ] Map compiler/type-check errors to GoX locations with source snippets; test
  errors in nested expressions and generated helpers separately.
- [x] Add fuzz targets for scanner/parser/lowering progress and nesting limits.
- [ ] Test output determinism across directory locations and cache hits.
- [x] Document syntax, evaluation/capture rules, component ABI, installation,
  first project, generated output inspection, known editor limitations and runtime.
- [ ] Include interactive headless examples for keyed lists, state/ref/context and
  current props/captures; explain instance ownership and explicit host adapters.
- [x] Add examples combining all features and an ordinary Go consumer of emitted
  source; explain that check/build don't execute compile-time user macros.
- [ ] Add standalone CI scoped to gox paths; measure transform latency/output size
  and cold/warm builds without inventing performance targets before a baseline.

Exit gate: a clean environment can install the tool, emit/build/run/test the
documented examples with stock Go, receive source-mapped errors, and repeat
builds safely. Both X and bare JSX examples must exercise the same runtime and
produce equivalent behavior. Validate Linux and Windows paths/launching; test
additional OSes
only when supported runners exist, and label untested targets accurately.

## Milestone 3 — Integrate into the fork's Go CLI and distributions

Depends on M2. Reuse the exact tested engine in-process. This is package loader
integration as well as a build step; a compile-only interception is insufficient.

### M3.1 Dependency and discovery design

- [x] Refresh the base branch and review contemporaneous changes from the other
  work session. Keep shared-toolchain behavior and launchers intact.
- [ ] Pin/vendor the engine into the cmd module through the normal dependency
  workflow. Verify bootstrap Go requirements; avoid hand-maintained copied engines.
- [x] Implement selected .gox and _test.gox discovery in cached and uncached
  loading, including GoX-only packages and packages from dependencies.
- [x] Audit go/build, modindex, load and module import scanning paths identified
  in the spec. Resolve the cmd/internal visibility boundary before modifying
  common loaders; do not create a dependency from std into cmd.
- [x] Transform before final import graph construction; include generated runtime
  imports, import cycles, workspaces, vendor mode and local replacements.
- [x] Preserve excluded-file behavior and avoid parser errors from inactive files.

### M3.2 Build orchestration and observability

- [x] Pass generated sources through cgo, coverage, compile and vet in the correct
  order; preserve build/embed/compiler directives and relative resource lookup.
- [x] Include source selection and transformer identity in build/cache actions;
  verify invalidation for source, options, version, target and runtime changes.
- [x] Add explicit-file command handling, module package patterns and test modes.
- [x] Expose original/generated inputs in documented list metadata; handle editor
  overlays deliberately, including packages with no on-disk .go files.
- [x] Add `go gox emit` for inspection using the same engine as build/run/test.
- [ ] Preserve reproducibility and trimpath behavior; test concurrent builds and
  machine-independent generated names/maps. Don't leak temporary absolute paths.

### M3.3 Regression and release gates

- [ ] Add Go command script tests for build/run/test/list/vet, test coverage,
  build constraints, cross-compilation, embeds, modules/workspaces, vendor mode,
  indexed/uncached loading, import cycles and cache invalidation.
- [x] Compile and run mixed Go/GoX integration fixtures with the native toolchain,
  including F, X, bare JSX and headless state/ref/context component scenarios.
- [ ] Repeat relevant smoke tests with shared-library distributions: Windows
  Python/batch invocation and Linux shared/native launcher paths.
- [ ] Confirm GoX requires no new .exe launcher on Windows and does not change
  existing argv0, GOROOT detection, tool discovery or exit propagation behavior.
- [x] Verify ordinary Go package behavior and bootstrap/build scripts remain
  functional with no GoX sources present.
- [ ] Extend existing release packaging only after standalone and integration
  gates pass; publish supported-target and editor compatibility documentation.

Exit gate: this fork's ordinary `go build`, `go run`, `go test`, `go list` and
`go vet` process supported GoX packages transparently; standard Go regressions
and supported shared-distribution smoke tests pass. No merge or release is
implied by completion of this planning branch.

## Execution and review policy

Keep commits reviewable by task group; use the milestone exit gates as the
definition of done, not token recognition or documentation alone. Update this
checklist with evidence and concrete test commands during implementation.
If a proposed syntax changes, update the specification, fixtures and diagnostics
together. Maintain the complete M1 feature scope; simplify semantics explicitly
instead of postponing a feature family without a scope decision.

Major risks to resolve early: upstream API version drift, semicolon behavior in
embedded code, scope-safe generated imports, formatting side effects, component
identity and hook lifetime, parser-directed bare JSX entry, package discovery occurring before compile, and module-index/cache divergence. None
requires changing Go closure implementation or the compiler backend.


## Implementation evidence (2026-10-11)

The implementation is in `gox/`; `src/cmd/go/internal/goxcmd` calls it in-process.
The ordinary compiler backend and shared launcher sources are unchanged. The
base branch was fetched again and still points at the inspected baseline.

Locally verified on Linux amd64:

- The full standalone suite under stock Go 1.26 and this fork's Go 1.28 build,
  including the copied scanner/parser corpus, ordinary-Go differential tests,
  recursive syntax, closure/loop capture, formatting order, static property
  failures, nil values, hook scope/placement and generated-name/import hygiene.
- `go test -race ./...`; additional runtime tests for workers queuing during
  rendering, root keys, nil contexts, updater/render panics and queue recovery.
- `go test -run=^$ -fuzz=FuzzTransform -fuzztime=5s -parallel=2` completed
  109,692 executions with no failure. This is a bounded smoke run, not exhaustive
  parser verification.
- `go test cmd/go/internal/fsys cmd/go/internal/modindex` and targeted Go command
  tests for GoX build/run/test/list/vet, coverage, selected/inactive files,
  source-edit cache invalidation, imported components, embeds, trimpath, cgo,
  workspace/local replacements, vendor mode, Windows cross-compilation and
  ordinary overlay behavior. Command help and `-C` registration regressions pass.
- A native bootstrap, `src/make-shared.sh`, the Linux shared-toolchain smoke suite
  (including C-ABI entry points and Windows cross-compilation), and the GoX counter
  running through the Linux shared Go command.
- `.github/workflows/gox.yml` adds Linux/Windows standalone checks and native Linux
  integration/shared-launcher checks. Windows execution awaits CI; cross-building
  a Windows executable is not a substitute for Windows launcher testing.

### Implementation choices and outstanding release gates

- The parser uses its original standard AST plus structured markup records and
  lowers extensions directly into ordinary nodes. No sentinel identifiers or
  external implementations of `ast.Expr` are used. A separate general-purpose IR
  was unnecessary for these forms and has not been added.
- Templates have an outer ordered evaluator and an inner explicitly parameterized
  constructor. The component ABI uses typed per-property adapters plus a structural
  property-set check, including for imported definitions.
- Generated identifiers use a logical-file digest and check the source for
  collisions. Package hook bindings are collected before rewriting; complete
  aggregation of every user identifier for adversarial cross-file name collisions
  is still a hardening task.
- Source diagnostics use Go `//line` directives and API line mappings. Precise
  byte-range sidecars, source snippets, and polished generated-helper diagnostics
  remain open. Error locations for generated code are approximate.
- The headless runtime rerenders the whole root after a queued update. Context
  values and identity are correct across those renders; selective consumer
  subscription/invalidation remains an optimization task. Render and host tree
  access have a single-owner contract; workers use the explicit enqueue methods.
- Standalone commands prepare modules eagerly, so syntax errors in unused selected
  packages can fail a command. Native discovery transforms encountered packages.
  Neither path edits user module manifests or runs user code at compile time.
- The cmd vendor copy is generated by `gox/tools/sync_vendor.py`, checked in CI.
  It is not a second hand-maintained engine. Integrating that step into upstream's
  broader dependency-update workflow remains a maintenance task.
- Full editor integration, exhaustive semantic-matrix coverage, long fuzz runs,
  cold/warm build benchmarks, Windows shared/Python/batch execution and release packaging
  remain unchecked. This branch is suitable for feature review and experimentation,
  not a claim that every release gate above has passed.

### Initial transform baseline

On the local AMD EPYC 9V74 Linux amd64 environment with stock Go 1.26, a
128-byte component/template fixture measured approximately 738 µs per transform,
113 KB allocated and 2,015 allocations (`BenchmarkTransform`, 200 ms sample).
This is a development baseline, not a performance target or a general build-time
claim. Source mapping/preamble handling and later changes can affect the result.
