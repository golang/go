# loGos

loGos is a fork of Go 1.27.4 that adds the loGo/s language (`.lgo` files) and the `logos` command. It is one toolchain for Go and loGo/s. Backend, runtime, cgo, and tooling improvements apply to Go packages too.

## Commands

Run from `src/` with the fork first on PATH: `export PATH="$(cd .. && pwd)/bin:$PATH"`.

- Build the toolchain: `./make.bash` installs `bin/go` and `bin/logos`. Rerun it after every compiler, runtime, or cmd/go change, because tests use the installed binaries.
- Smoke: `logos version` prints `logos version go1.27.4`, and `logos help` prints the `logos <command> [arguments]` usage.
- Test one package: `go test -count=1 -short <pkg>`. Cached results are not evidence.
- Add `-race` for concurrency, runtime, and cmd/go changes.
- Script tests: `go test -count=1 cmd/go -run 'Script/^logos_<name>$'`.
- Compiler run and errorcheck tests: `go test -count=1 cmd/internal/testdir -run 'Test/logos/'`.
- Changed files: `gofmt -l <files>` prints nothing, and `go vet <pkgs>` passes.
- Lint new packages only (golangci-lint 2.15.0 at `/home/linuxbrew/.linuxbrew/bin/golangci-lint`): from `src/cmd` with the fork's `bin` first on PATH, `golangci-lint run --config ../../logos-doc/golangci.yml ./go/internal/logos/...`.
- Milestone gate: `./run.bash --no-rebuild`.

## Gotchas

- Plain `go` keeps Go language semantics and its upstream command text (`go version`, `go help`), and `run.bash` is its regression floor. It may differ from upstream in performance, code generation, runtime, and tooling.
- Fork logic SHOULD stand in a `*_logos.go` file or a new package, and an upstream file SHOULD take only the hook line that calls it, because every upstream line costs rebase work. Fork code MAY stand in an upstream file when a move gives no clear gain: a field of an upstream struct, a changed signature, a generated file, or a declaration or helper of about 20 lines or fewer that its upstream caller uses directly. Guard loGo/s language behavior with a logos condition. Backend, runtime, and cgo improvements take no logos guard. A changed default for plain `go` lands only after `run.bash` passes with it.
- Leave `src/vendor` and `src/cmd/vendor` unchanged; they are vendored copies.
- A module holds only `.go` packages or only `.lgo` packages. Go and loGo/s packages can import each other. Only `logos` builds `.lgo` files.
- The fork's own libraries under `src/logos/` MUST answer a question about a type at compile time (comptime, generated code, or a compiler check) whenever the lowering knows the type, and MUST NOT call `reflect` for it. `reflect` MAY stay only where the type is known only at run time (an interface value or an unmodeled field) or where a Go API takes a `reflect.Type`.

## Workflow

- Work slice by slice from `logos-doc/IMPLEMENTATION.md`: write the slice's attack tests, see them fail, build, then pass the slice gate (rules G1 and G2).
- A changed language rule updates its Table T row in IMPLEMENTATION.md and the matching SPEC.md sentence in the same commit.
- A slice that stays red after three attempts is recorded as blocked (rule G1, item 7); tests are never weakened to reach green.

## References

- Language spec: `logos-doc/SPEC.md`
- Build plan, gates, and lexicon: `logos-doc/IMPLEMENTATION.md`
- Red-team test ledger: `logos-doc/REDTEAM.md`
