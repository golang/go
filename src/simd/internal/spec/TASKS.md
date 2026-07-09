This spec is quite incomplete! The plan is to migrate everything from simdgen's
categories.yaml files into the spec and replace categories.yaml. As we're doing
so, we should think carefully about:

1. The doc comment for each operation. Let's start moving toward a more formal
   style for mathematically specified operations.

2. How every operation generalizes across types, sizes, and especially to
   scalable vectors. The spec is meant to be *maximalist*, circumscribing what
   every architecture can do.

3. In what cases operations are architecture-dependent.

Tasks
- [ ] Hook specgen into simdgen, probably unioning the categories.yaml input and
  the specgen input while we transition
- [ ] Migrate categories.yaml to spec
- [ ] Figure out how to represent compiler-only (non-API) simdgen operations
- [ ] Delete categories.yaml
- [ ] Incorporate specgen into other generators, so the spec becomes the sole
  source of truth for all exported APIs.
  - [ ] tmplgen
  - [ ] wasmgen
- [ ] Build a tool to ensure hand-written APIs (emulations, etc) match the spec.
  Maybe the hand-written code still lives directly in archsimd, but as
  unexported function, and the generator writes the trivial exported API glue
  for these.
- [ ] Generate a full reference implementation for testing that provides the
  SIMD API but just wraps the spec package.
- [ ] Generate conformance tests of the archsimd API against the spec testing
  layer.
