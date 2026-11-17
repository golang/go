---
name: building-logos-first-from-go
description: Build the LoGOs toolchain from source with an upstream Go as the bootstrap, the first build that produces bin/logos. Use when the user says build LoGOs or logos from source, run make.bash, set GOROOT_BOOTSTRAP, or has no logos binary yet. Not for rebuilding logos with logos or for writing .lgo code.
license: BSD-3-Clause
compatibility: Needs Bash, git, and an upstream Go that can compile the fork (see step 1), run inside a LoGOs checkout.
metadata:
  version: "0.22.0"
---

# Build LoGOs first from Go

An upstream Go compiles the LoGOs sources once. `make.bash` installs `bin/go` and `bin/logos`; they are the same binary and pick their mode from the name they run under. That result is stage 1. Every later step starts from it.

This skill writes only inside the checkout (`bin/`, `pkg/`, build caches). It never edits `src/vendor` or `src/cmd/vendor`, never installs outside the checkout, and never rewrites git history.

## Procedure

1. Pick the bootstrap Go. Take the latest stable upstream Go from go.dev/dl or the system package manager, and read the Install sentence in `README.md` for the release it names as known to work. Go 1.26.8 fails to compile the fork's compiler sources. Done when `go version` prints a release at least as new as the README names and `go env GOROOT` prints a directory outside the checkout.
2. Settle the source. Run `git fetch origin`, then `git rev-list --left-right --count HEAD...origin/<branch>`; pull only when the left count is zero and the right count is above zero. Leave unrelated modified or deleted files alone. Done when HEAD is known and `sed -n 1p VERSION` prints the baseline, for example `go1.27.4`.
3. Clean the environment. Run `unset GOROOT GOTOOLCHAIN`, then `export GOROOT_BOOTSTRAP="$(go env GOROOT)"` with the upstream Go first on `PATH`. The checkout's own `bin/go` is never a bootstrap. Done when `echo "$GOROOT_BOOTSTRAP"` prints the upstream root from step 1.
4. Build from `src/`: `cd src && ./make.bash`. The script exists only in `src/`, so a run from the checkout root cannot start. Run it as a background job with no deadline; it took about 200 seconds at load average 14. Done when the output ends with `Installed commands in <checkout>/bin`.
5. Verify from the checkout root. All three must hold:
   - `bin/logos version` prints `logos version go<VERSION> <os>/<arch>` with the line 1 value of `VERSION`.
   - `bin/logos help` prints `logos <command> [arguments]`.
   - `bin/go version` prints `go version go<VERSION> <os>/<arch>`; plain `go` keeps the upstream text.
   Done when each output matches.
6. Use it. For the current shell run `export PATH="$PWD/bin:$PATH"` from the checkout root. Do not copy `bin/logos` to another directory: it is built trimmed and finds its GOROOT from the real path of the executable, so a copy fails with `go: cannot find GOROOT directory: 'go' binary is trimmed and GOROOT is not set`. To install it elsewhere, link it (a symlink resolves to the real path).

## Rebuild rule

Rerun step 4 after every change to the compiler, the runtime, or `cmd/go`, because tests run the installed binaries and cached results are not evidence. Never run `make.bash` while a test or build job still uses `bin/` or `pkg/tool/`; it rewrites both and breaks the running job.

## When it goes wrong

| Symptom | Action |
|---|---|
| `make.bash: No such file or directory` | You are in the checkout root. Change to `src/`. |
| Compile errors in compiler or runtime sources during toolchain1 | The bootstrap Go is too old. Install the release the README names and repeat from step 1. |
