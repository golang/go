# Working in the SIMD packages

What these packages are, how the generators fit together, how to run them is in
[archsimd/_gen/README.md](archsimd/_gen/README.md). This file is only about not
burning your context window.

## Most of this tree, by volume, is generated — and some of it is enormous

Generated files here are machine-uniform: hundreds to thousands of
near-identical stanzas, each a doc comment plus one declaration.

```go
// Add adds corresponding elements of two vectors.
//
// Asm: VPADDD, CPU Feature: AVX
func (x Int32x4) Add(y Int32x4) Int32x4
```

**Two or three stanzas tell you what the whole file would.** Reading one end to
end can consume a large fraction of your context and teach you nothing the
sample didn't. This stays true as the API grows: these files get longer, not
more varied.

A file is generated if its first line says so:

```sh
head -1 <file> | grep 'Code generated'
```

Don't rely on a remembered list of which files are big — measure:

```sh
find . -name '*.go' -size +50k -not -path '*/_gen/*' | xargs ls -lS
```

`archsimd/ops_*.go`, `internal/simdref/simdref.go`, `internal/bridge/decls_*.go`
and the `simd_test` helpers are usually on that list. Treat it as a starting
point, not an inventory.

## Sample; don't read

Anchor the pattern, ask for little context:

```sh
# the shape of one declaration, with its full doc comment
GOEXPERIMENT=simd go doc simd/archsimd.Int32x4.Add

# which receivers carry an op — count first, list if you need to
grep -c '^func ([^)]*) Add(' archsimd/ops_amd64.go
```

An unanchored `grep Add` also matches `AddSub`, `MulAdd`, `PairwiseAdd` and
every `// Asm:` line mentioning one. On a file of uniform stanzas, `-A5`/`-B5`
returns most of the file.

## Size tool output before you read it

`specls` prints the expanded spec API. Its output is as large as the files it
describes, so use `-f` (a regexp) to narrow it, and anchor the pattern:

```sh
go run ./cmd/specls -f '^Add$'   # one spec function
go run ./cmd/specls -f Add       # every name containing "Add" — much larger
go run ./cmd/specls              # the entire expanded API — very large
```

Same caution for diffs. The README says to use `-diff` constantly, and that's
right — but regenerating a family can produce thousands of lines. Get the shape
first, then read what matters:

```sh
git diff --stat
go run . -tools <tool> -diff | wc -l
```

## Don't edit generated files

Edit the generator input and regenerate. The first line of each generated file
names the inputs it was built from; use those to find what to edit, but
regenerate with `go generate` (see the README), not by re-running the command
in that line. Commit regenerated output in the same commit as the change that
caused it.

## If you were handed one spec transition task

[archsimd/_gen/SPEC-TRANSITION.md](archsimd/_gen/SPEC-TRANSITION.md) is the
active plan. Read Part 2 (the rationale; tasks cite it by section number) and
your own task from Part 4. **Don't read Part 4 end to end** — it is every task
in the project, and you were given one.
