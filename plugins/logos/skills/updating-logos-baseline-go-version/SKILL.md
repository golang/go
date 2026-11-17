---
name: updating-logos-baseline-go-version
description: Update the LoGOs baseline Go version to the latest stable upstream Go. Grounds the release, syncs every baseline mention, rebuilds stage 1 and stage 2, and reinstalls logos. Use when a new Go patch release ships, VERSION changed, or the user says update or bump the baseline Go version. Not for the go 1.27 language line or the ejection floor.
license: BSD-3-Clause
compatibility: Needs a LoGOs checkout, network access to go.dev and endoflife.date, and an upstream Go for the rebuild.
metadata:
  version: "0.22.0"
---

# Update the LoGOs baseline Go version

The baseline is the upstream Go release the fork is based on. Line 1 of `VERSION` (`go1.N.P`) is the single source of truth, and `logos version` prints it. Every other baseline mention follows `VERSION`. Merging an upstream release tag changes `VERSION`; this skill covers what must follow that merge, plus the rebuild.

This skill edits tracked text files and rebuilds `bin/`. It never edits `src/vendor`, `src/cmd/vendor`, or upstream `doc/`, never commits, and never pushes.

## Procedure

1. Ground the release. Record the URL and date for each source, because recall is not a source:
   - Release channel: `https://go.dev/doc/devel/release` (the `go1.N.P` entry and its date) and `https://go.dev/dl/?mode=json`.
   - Support window: `https://endoflife.date/api/v1/products/go/` (the cycle's `isMaintained` and `latest`).
   - Advisories for the exact release: `govulncheck -mode=binary <root>/bin/go` on the new release prints `No vulnerabilities found.` plus the database date, and the `[security]` announcement on `golang-announce` names the fixes.
   Done when the newest stable release, its date, its support state, and its advisory result are written down. When a source is unreachable, mark that item unconfirmed and continue. When an applicable unresolved critical advisory affects the release, report the blocked release and the advisory identifier and stop.
2. Find the stale mentions. With `n` set to `VERSION` line 1 without `go`, run from the checkout root:
   - `rg -n 'forkVersionPin|upstreamVersionPin' src/cmd/internal/goparity/parity_test.go`
   - `rg -n 'is known to work' README.md`
   - `rg -n 'Base: Go|fork of Go|current compiler base|Go baseline: the tree' AGENTS.md logos-doc/SPEC.md logos-doc/IMPLEMENTATION.md`
   - `rg -n 'go1\.[0-9]+\.[0-9]+|"1\.[0-9]+\.[0-9]+"' .github`
   Done when every hit is classed as a baseline site (step 3), a floor or oracle (step 4), or evidence (step 4).
3. Set the baseline sites to `n`:
   - `src/cmd/internal/goparity/parity_test.go`: `forkVersionPin` and `upstreamVersionPin`. Before you raise `upstreamVersionPin`, run the audit its failure message names (`git diff --stat <old tag> <new oracle tag> -- src test`) and read the result.
   - `README.md`: the bootstrap sentence `Go N is known to work` (true only after a `make.bash` run with that Go) and the sample outputs of `logos version` and `go version`.
   - `AGENTS.md`: the first sentence `fork of Go N`.
   - `logos-doc/SPEC.md`: the header `Base: Go N` and sections 0.1 and 18.1.
   - `logos-doc/IMPLEMENTATION.md`: the G2 baseline line, `Base: Go N` and `fork of Go N` in slice texts, and the compiler-base sentences. State version assertions as the `$GOROOT/VERSION` text, because `src/cmd/go/testdata/script/logos_command.txt` compares against `$GOROOT/VERSION` and not a literal.
   - `.github/scripts/install-ci-tool.sh` (`release=go<n>`, the `go version go<n>` check, and its checksum), the `go<n>-ci` stock root in `.github/scripts/feature-gate.sh` and `release-acceptance.sh`, and `go-version: "<n>"` in `.github/workflows/*.yml`. Take checksums from `https://go.dev/dl/?mode=json`.
   Done when the drift check of step 2 shows no baseline site naming another release.
4. Leave these alone:
   - `go 1.27` in `src/go.mod` and `src/cmd/go.mod`, and the default `"1.27"` in `goVersionOf` (`src/cmd/go/internal/logos/eject/eject.go`): the language line, not a patch pin.
   - The ejected-code floor (SPEC 0.3), `UPROOT1272` and `UPROOT1273`, and their checksum rows in `.github/scripts/install-upstream-oracles.sh` and `.github/workflows/cgo.yml`: ejection oracles that stay old on purpose.
   - Dated tables, `logos-doc/REDTEAM.md`, `logos-doc/validation/**`, `logos-doc/measurements/**`, and tool facts such as the Go version a linter was built with.
   Done when no edit touches these.
5. Rebuild and reinstall. Run only when no test or build job uses `bin/`, because `make.bash` rewrites `bin/` and `pkg/tool/`.
   - Stage 1: `unset GOROOT GOTOOLCHAIN; cd src && GOROOT_BOOTSTRAP="$(go env GOROOT)" ./make.bash` with the new upstream Go first on `PATH`, as a background job with no deadline. Done when the output ends with `Installed commands in <checkout>/bin`.
   - Stage 2: `mkdir -p bin/bootstrap && ./bin/logos build -O=3 -o ./bin/bootstrap/logos cmd/go` from the checkout root. Done when the command exits 0.
   - Install: set `bindir="${GOBIN:-$(go env GOPATH)/bin}"` and run `ls -l "$bindir/logos"`. When the target is a regular file and not a symlink, stop and ask first, because `ln -sfn` deletes it and nothing can undo that. Otherwise run `ln -sfn "$PWD/bin/bootstrap/logos" "$bindir/logos"`. Never copy the binary.
   Done when, from a directory outside the checkout, `logos version` prints `logos version go<n>-X:logosarena <os>/<arch>`, `logos help` prints `logos <command> [arguments]`, and `logos env GOROOT` prints the checkout root.
6. Run the narrow tests that cover the version, from `src/` with `bin` first on `PATH`: `go test -count=1 cmd/go -run 'Script/^logos_command$'`, and `UPROOT1274=<upstream root of release n> go test -count=1 -short cmd/internal/goparity`. Done when the script test exits 0 and the goparity output has no `version skew not audited` message. Name every other goparity failure with its test and reason, and call one pre-existing only with a run on an unchanged checkout.
7. Hand back a dated table, one row per release checked: version, release date, support state, release-channel link, advisory result with its source, and the decision (kept or raised). Done when every row cites a source fetched in this run.
