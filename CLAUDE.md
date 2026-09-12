# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the Go programming language source repository (`golang/go`). The current working branch is `fix-x509-algorithm-error-message` off `master`, tracking Go 1.27 development. The bootstrap toolchain required is Go ≥ 1.24.6.

## Build

All commands run from `src/`:

```bash
cd src

# Build the toolchain (produces bin/go and bin/gofmt)
./make.bash

# Build + run all tests
./all.bash
```

The built `go` binary lives at `bin/go` (repo root), not on `$PATH`. Always use `../bin/go` or `../../bin/go` when running from subdirectories.

## Running Tests

```bash
# Run tests for a single standard library package
../bin/go test strings
../bin/go test -v -run TestFoo strings

# Run tests for a cmd package
../bin/go test cmd/go
../bin/go test -v -run TestName cmd/compile/...

# Run cmd/go script tests (*.txt files in testdata/script/)
../bin/go test cmd/go -run=Script/^foo$

# Run the compiler/runtime test suite (src/test/ directory)
../bin/go test cmd/internal/testdir
../bin/go test cmd/internal/testdir -run='Test/(file1.go|file2.go)'

# Run all stdlib tests via dist (what all.bash does after building)
../bin/go tool dist test
../bin/go tool dist test -list          # list all test shards
../bin/go tool dist test strings        # run a specific shard by name
```

## Making Changes

### Standard library changes

For bug fixes with no new API: edit the package source and its `_test.go` files. No extra steps needed.

### New exported API

When adding a new exported symbol, three files must be updated together:

1. **`api/next/<issue>.txt`** — one line per new symbol, e.g.:
   ```
   pkg net/url, method (*URL) Clone() *URL #73450
   ```

2. **`doc/next/6-stdlib/99-minor/<pkg>/<issue>.md`** — release note prose, e.g.:
   ```
   The new [URL.Clone] method creates a deep copy of a URL.
   ```

3. The implementation itself.

The `go tool api` checker enforces that `api/next/` entries match the actual exported API. A CI failure mentioning "go tool api" means one of these is out of sync.

## Repository Structure

```
src/               Standard library + toolchain source
  cmd/
    go/            The go command (build, test, mod, work, …)
    compile/       Go compiler (frontend: noder/typecheck/ir → SSA → arch backends)
    link/          Linker
    asm/           Assembler
    dist/          Bootstrap tool (builds the toolchain itself)
  runtime/         Go runtime (scheduler, GC, channels, maps, …)
  internal/        Internal packages shared across stdlib
api/               API compatibility surface files
  go1.N.txt        Frozen API for released versions
  next/            Proposed API for the upcoming release (mutable)
doc/next/          Release notes for the upcoming release (mutable)
test/              Compiler/runtime black-box tests (run via cmd/internal/testdir)
misc/              Miscellaneous files (cgo, wasm, android, …)
```

## Key Architecture Notes

### Compiler pipeline (`cmd/compile`)
Source → `noder` (syntax+types2) → `ir` (AST) → `typecheck` → `escape` → `inline` → `ssa` (SSA IR) → arch-specific backends (amd64, arm64, …) → object files.

### Runtime (`src/runtime`)
Implements goroutine scheduling (`proc.go`), GC (`mgc.go`, `mheap.go`), channels (`chan.go`), maps (`map.go` + `internal/runtime/maps/`), and low-level OS interfaces. Uses `//go:nosplit`, `//go:noescape`, and `//go:linkname` annotations for unsafe operations.

### `cmd/go`
Organized as subcommands under `internal/`. Module resolution is in `internal/modload/`, proxy/fetching in `internal/modfetch/`, build actions in `internal/work/`. Script-based integration tests live in `testdata/script/*.txt`.

### Platform-specific files
Files named `foo_GOOS.go`, `foo_GOARCH.go`, or `foo_GOOS_GOARCH.go` are automatically selected by the build system. Stub implementations for unsupported platforms follow the same pattern with `_stub` or platform names.

## Contribution Conventions

- Commit messages use the format: `package/path: short description` (e.g., `crypto/x509: improve error message when signature algorithm is unsupported`).
- This repo uses Gerrit for code review (not GitHub PRs). The upstream remote is `https://go.googlesource.com/go`.
- The `codereview.cfg` sets `branch: master` — all work targets master regardless of local branch names.
- `//go:linkname` usage outside the standard library is restricted; changes touching it require care.
- GODEBUG settings for behavioral changes must be documented and wired through `internal/godebugs`.
