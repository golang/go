---
name: dogfooding-logos
description: Install the bootstrapped logos (bin/bootstrap/logos) as the everyday logos command, test it with the repository's logos suites, and compare it with stage 1. Use when the user says dogfood logos, install the bootstrapped logos, update the installed logos command, or test the bootstrapped binary. Needs a built stage 2.
license: BSD-3-Clause
compatibility: Needs bin/bootstrap/logos from a bootstrap build and bin/logos from make.bash, inside a LoGOs checkout, plus an upstream Go on PATH.
metadata:
  version: "0.22.0"
---

# Dogfood logos

Dogfooding means the logos you use every day was built by logos. Stage 1 is `bin/logos` (built from an upstream Go). Stage 2 is `bin/bootstrap/logos` (built by stage 1 at `-O=3`). This skill installs stage 2 on the user's `PATH` and checks that it behaves like stage 1.

It installs one symlink (step 2) and writes temporary directories and test logs. In the install directory it never copies a binary and never links a `go` command. It never changes `bin/logos`. The stage 2 root of step 3 is a temporary directory and holds copies of stage 2.

## Procedure

1. Check stage 2. From the checkout root run `bin/bootstrap/logos version`. Done when it prints `logos version go<VERSION>-X:logosarena <os>/<arch>`. When the file is missing, stop and tell the user to build it from the checkout root with `./bin/logos build -O=3 -o ./bin/bootstrap/logos cmd/go`.
2. Install by symlink. Set `bindir="${GOBIN:-$(go env GOPATH)/bin}"` and run `ls -l "$bindir/logos"`, keeping the output. When the target is a regular file and not a symlink, stop and ask first, because replacing it cannot be undone. Otherwise run `mkdir -p "$bindir" && ln -sfn "$PWD/bin/bootstrap/logos" "$bindir/logos"`. Link rather than copy: a copy keeps the old build after the next rebuild, and a copy of the trimmed stage 1 binary fails with `go: cannot find GOROOT directory`, because that binary finds its GOROOT from the real path of the executable. Done when, from a directory outside the checkout, `"$bindir/logos" version` and `"$bindir/logos" env GOROOT` print the stage 2 version and the checkout root, and `command -v logos` prints `$bindir/logos`. When another `logos` comes first on `PATH`, report its path.
3. Run the suites with stage 2 as the subject. `go test` puts `$GOROOT/bin` first on the `PATH` of every test process, so a `PATH` override alone leaves stage 1 (`bin/go`, `bin/logos`) as the tools the `testdir` and script groups run. Build a stage 2 root instead, from the checkout root:
   `R=$PWD; T=$(mktemp -d); mkdir "$T/bin"; for e in $(ls "$R"); do [ "$e" = bin ] || ln -s "$R/$e" "$T/$e"; done; cp "$R/bin/bootstrap/logos" "$T/bin/go"; ln "$T/bin/go" "$T/bin/logos"; cp "$R/bin/gofmt" "$T/bin/gofmt"; unset GOROOT GOTOOLCHAIN; export PATH="$T/bin:$PATH"`
   The root holds symlinks to the checkout's files and copies of stage 2 as `bin/go` and `bin/logos`. Copy, do not link: a symlink resolves back into the checkout and returns stage 1's `bin` as `$GOROOT/bin`. Done when `go env GOROOT` prints `$T`, and `go version` and `logos version` both print the stage 2 version.
   On Linux, prove the subject on a small run before the long one. From `$T/src` run `strace -f -qq -e trace=execve -o "$T/exec.log" go test -count=1 cmd/internal/testdir -run 'Test/logos/run/option_index_elements'`, then `grep -oE 'execve\("[^"]*/bin/(go|logos)"' "$T/exec.log" | sort | uniq -c`. Done when every path starts with `$T/bin/` and none starts with `$R/bin/`.
   The script tests compare ejected output against upstream toolchains. Set `UPROOT1272` and `UPROOT1273` to upstream Go 1.27.2 and 1.27.3 roots and `UPROOT1274` to the upstream baseline root. An upstream Go prints a root with `GOTOOLCHAIN=go1.27.2 go env GOROOT`, which downloads that release on first use. Without these variables over 200 `logos_` script tests fail on `go version go1.27.2` assertions or on a system `go` that is too old. Done when `$UPROOT1272/bin/go version`, `$UPROOT1273/bin/go version`, and `$UPROOT1274/bin/go version` print their own releases.
   Then run the three groups from `$T/src`, each in its own subshell so they run in parallel, and `wait`:
   - `( go test -count=1 -short cmd/compile/internal/logos/... cmd/go/internal/logos/... cmd/internal/logos/... cmd/internal/logosowned internal/logosfile logos/... cmd/cgo/internal/testlogos > "$T/pkgs.log" 2>&1; echo "exit=$?" >> "$T/pkgs.log" ) &`
   - `( go test -count=1 cmd/go -run 'Script/^logos_' > "$T/script.log" 2>&1; echo "exit=$?" >> "$T/script.log" ) &`
   - `( go test -count=1 cmd/internal/testdir -run 'Test/logos/' > "$T/testdir.log" 2>&1; echo "exit=$?" >> "$T/testdir.log" ) &`
   The package group runs under stage 2 as the driver, the `testdir` group runs `$T/bin/logos`, and the script group builds its own `go` from the sources with stage 2 as the builder, so it checks the sources more than the stage 2 binary. The form `cmd > log; echo ... &` backgrounds only the `echo` and runs the groups one after another. Run the job in the background with no deadline; the script group took about eight minutes at a high load average. Done when each log ends with `exit=0`, or each failure is named with its test and reason.
4. Attribute every failure. Rerun the failing group with stage 1 first on `PATH` (the checkout's `bin` directory) and the same variables. Done when each failure is classed as environment (a missing `UPROOT` variable or toolchain), stage 2 only, or both stages. Call a failure pre-existing only with that baseline run.
5. Use it. Do day-to-day work with the installed `logos`. When a result looks wrong, rerun the same command with `bin/logos` and compare stdout, stderr, and exit code. Done when each difference is reported with the exact command, both outputs, and the stage that produced each.
6. Refresh. After any stage 1 rebuild or baseline change, rebuild stage 2 from the checkout root and rerun step 3. The symlink needs no change.

## Rules

- A stale copy, a symlink target outside the checkout, or a linked `go` is a failed install.
- Never rebuild stage 1 while a test job runs: `make.bash` rewrites `bin/` and `pkg/tool/`.
- Stage 2 is a different binary from stage 1 (`-O=3`, the `logosarena` experiment, larger file). A version suffix or a size difference is expected and is not a defect.
