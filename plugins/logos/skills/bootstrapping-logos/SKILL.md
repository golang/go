---
name: bootstrapping-logos
description: Rebuild the logos command with the logos toolchain itself at -O=3, the stage 2 binary at bin/bootstrap/logos. Use when the user says bootstrap logos with logos, self-host, build logos with the built logos, or build logos at -O3. Needs a built stage 1 (bin/logos). Not for the first build from Go.
license: BSD-3-Clause
compatibility: Needs a current stage 1 build (bin/logos from make.bash) inside a LoGOs checkout.
metadata:
  version: "0.22.0"
---

# Bootstrap logos with logos

Stage 1 is the toolchain `make.bash` built from an upstream Go (`bin/logos`). Stage 2 is the `logos` command, the `cmd/go` package, compiled by stage 1 with the fork's own optimizer at `-O=3`. This skill writes only `bin/bootstrap/` (git ignores `/bin/`). It never touches `bin/logos`, `bin/go`, or `pkg/tool/`.

Stage 2 replaces only the go command. The compiler and linker in `pkg/tool/` still come from stage 1, so a changed compiler, runtime, or `cmd/go` needs a fresh stage 1 first.

## Procedure

1. Check stage 1. From the checkout root run `bin/logos version` and `sed -n 1p VERSION`. Done when the version output contains the `VERSION` value. When `bin/logos` is missing or older than the sources, stop and tell the user to build stage 1 with `cd src && ./make.bash` using an upstream Go as `GOROOT_BOOTSTRAP`.
2. Build stage 2 from the checkout root:
   `mkdir -p bin/bootstrap && ./bin/logos build -O=3 -o ./bin/bootstrap/logos cmd/go`
   The flag form is `-O=3`; `-O3` is rejected. The levels are `0`, `1`, `2`, `3`, and `s`. Do not pass `-trimpath`: an untrimmed binary bakes in its GOROOT and runs from any directory. It took about 40 seconds at load average 14. Done when the command exits 0 and `bin/bootstrap/logos` exists.
3. Prove `-O=3` reached the compiler. The build info does not record `-O`, so run
   `./bin/logos build -O=3 -a -n -o /tmp/x cmd/go/internal/base 2>&1 | grep -o 'inlbudget=[0-9]*' | sort -u`
   Done when it prints `inlbudget=400`. An invalid level, for example `-O=9`, must fail with `invalid -O level`.
4. Smoke the result from a directory outside the checkout:
   - `bin/bootstrap/logos version` prints `logos version go<VERSION>-X:logosarena <os>/<arch>`. The suffix is expected: the `logos` command applies `GOEXPERIMENT=logosarena` by default, and stage 1 was built without it.
   - `bin/bootstrap/logos help` prints `logos <command> [arguments]`.
   - `bin/bootstrap/logos env GOROOT` prints the checkout root.
   - `bin/bootstrap/logos run test/logos/run/option_index_elements.lgo`, run from the checkout, prints `ok ...` lines and exits 0.
   Done when all four hold.
5. Report the stage 2 file size and the build time. The file is about twice the size of `bin/logos`; the likely cause is that `make.bash` strips debug data (`-ldflags=-w`) and this build does not [INFERENCE].

## Rules

- Never overwrite `bin/logos` with stage 2. Stage 1 stays as the reference for comparing behavior.
- Rebuild stage 2 after every stage 1 rebuild; keep the same command.
- A failure at step 2 is a compiler or build defect to report with the exact error, never a reason to change `-O` silently.
