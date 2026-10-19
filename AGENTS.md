# loGos

loGos is a fork of Go 1.27.2 that adds the loGo/s language (`.lgo` files) and the `logos` command. Plain `go` keeps upstream behavior.

## Commands

Run from `src/` with the fork first on PATH: `export PATH="$(cd .. && pwd)/bin:$PATH"`.

- Build the toolchain: `./make.bash`. Rerun it after every compiler, runtime, or cmd/go change, because tests use the installed binaries.
- Test one package: `go test -count=1 -short <pkg>`. Cached results are not evidence.
- Add `-race` for concurrency, runtime, and cmd/go changes.
- Script tests: `go test -count=1 cmd/go -run 'Script/^logos_<name>$'`.
- Compiler run and errorcheck tests: `go test -count=1 cmd/internal/testdir -run 'Test/logos/'`.
- Changed files: `gofmt -l <files>` prints nothing, and `go vet <pkgs>` passes.
- Milestone gate: `./run.bash --no-rebuild`.

## Gotchas

- Upstream files take hook lines only: one call into a `*_logos.go` file or a new package, guarded by a logos condition. Put the logic in new files, because every upstream line costs rebase work.
- Leave `src/vendor` and `src/cmd/vendor` unchanged; they are vendored copies.
- A module holds only `.go` packages or only `.lgo` packages, and `.lgo` code imports only the standard library and loGo/s packages.

## Workflow

- Work slice by slice from `logos-docs/IMPLEMENTATION.md`: write the slice's attack tests, see them fail, build, then pass the slice gate (rules G1 and G2).
- A changed language rule updates its Table T row in IMPLEMENTATION.md and the matching SPEC.md sentence in the same commit.
- A slice that stays red after three attempts is recorded as blocked (rule G1, item 7); tests are never weakened to reach green.

## References

- Language spec: `logos-docs/SPEC.md`
- Build plan, gates, and lexicon: `logos-docs/IMPLEMENTATION.md`
- Red-team test ledger: `logos-docs/REDTEAM.md`
- Original spec text (historical, not normative): `logos-docs/archive/`
