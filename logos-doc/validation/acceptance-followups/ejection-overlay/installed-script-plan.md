# Installed overlay-ejection script plan

Prepared 2026-11-16. Not executed. The final make is active; installed new-source script GREEN remains pending.

## Binding and prerequisites

- Final source: e96a570da271c43a80e23bb1a188aac054acff25; full tree 0841c3d2a92b7d6091a5e248b82bd820326947a8; src tree e237c399aae8f8065b724e8a0e27585686c7da18. All 18 ejection package files and both frozen inputs match the completed component evidence.
- Wait for successful make and root’s explicit Go-lane release. Record the final installed compiler/tool identities.
- Reuse the final cmd/go race test binary if root or language builds it. Verify its build source is the final commit and its compiler belongs to that successful make. The earlier 45c29ff2 binary is baseline evidence only.
- Set LOGOS_FINAL_CMDGO_RACE to that verified binary and LOGOS_FINAL_GOCACHE to root’s selected disk cache. If the binary does not yet exist, root should schedule one shared cmd/go race-binary build, not a second build solely for this script.
- The frozen script SHA-256 must remain 2a445178e1ca9c86fcd59718ab809a2c07a7b2e751193e29fb113ba1f56bb173. Keep all fixture flags and assertions unchanged.

## One serial execution

Run from the final checkout so the normal harness reads the frozen script. The test binary serves as the go/logos command and uses the final installed compiler. Compilation/cache remain on disk; source temporaries use a unique real home directory.

```bash
set -u
: "${LOGOS_FINAL_CMDGO_RACE:?Set the verified final cmd/go race binary}"
: "${LOGOS_FINAL_GOCACHE:?Set the root-selected disk cache}"
LOGOS_EJECT_GATE_EVIDENCE=/workspace/scratch/9949907ca71e/hosted-evidence/runtime/ejection-overlay-installed-e96a570d
mkdir -p "$LOGOS_EJECT_GATE_EVIDENCE" || exit $?
LOGOS_EJECT_SOURCE_TMP=$(mktemp -d /home/agent/logos-eject-final-9949907ca71e-XXXXXX) || exit $?
cd /workspace/scratch/9949907ca71e/logos-final-acceptance/src/cmd/go || exit $?
set +e
env -u GOTOOLDIR -u CMDGO_TEST_RUN_MAIN -u GO_GCFLAGS -u BOOT_GO_GCFLAGS -u GODEBUG \
  GOENV=off GOTOOLCHAIN=local GOWORK=off GOPROXY=off GOSUMDB=off \
  GOROOT=/workspace/scratch/9949907ca71e/logos-final-acceptance \
  PATH=/workspace/scratch/9949907ca71e/logos-final-acceptance/bin:/usr/local/bin:/usr/bin:/bin \
  GOCACHE="$LOGOS_FINAL_GOCACHE" TMPDIR="$LOGOS_EJECT_SOURCE_TMP" GOTMPDIR="$LOGOS_EJECT_SOURCE_TMP" \
  GOMAXPROCS=1 GOMEMLIMIT=512MiB GOGC=50 GOFLAGS=-p=1 GOEXPERIMENT= \
  GOOS=linux GOARCH=amd64 GOAMD64=v1 CGO_ENABLED=1 CC=/usr/bin/gcc-14 \
  GOPATH=/workspace/scratch/9949907ca71e/acceptance-gopath \
  timeout --kill-after=10s 660s "$LOGOS_FINAL_CMDGO_RACE" \
  -test.v -test.count=1 -test.short -test.parallel=1 \
  '-test.run=^TestScript$/^logos_eject_overlay_sources$' -test.timeout=10m \
  > "$LOGOS_EJECT_GATE_EVIDENCE/script.log" 2>&1
LOGOS_EJECT_STATUS=$?
printf "%s\n" "$LOGOS_EJECT_STATUS" > "$LOGOS_EJECT_GATE_EVIDENCE/script.exit"
test "$LOGOS_EJECT_STATUS" -eq 0
```

Accept only exit 0 plus the named script PASS, with no skip, no-tests warning, race report, or timeout. Preserve the complete log and exact binary/source identities. The existing script asserts emitted/deleted/ignored files, hidden backing-file exclusion, and execution output. Stop and report a failure before changing any source, fixture, or budget.

No baseline rerun, component-suite expansion, or additional mutation is needed for this single remaining gate. No command above ran during preparation.
