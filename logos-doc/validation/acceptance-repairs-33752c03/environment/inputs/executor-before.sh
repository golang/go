#!/usr/bin/env bash
set -euo pipefail

# Each invocation owns one hosted job. Tests, their exclusions, and the source
# tree are inputs, never edited here. This script is not a release waiver.
mode=${1:?expected matrix, parity, repro, compare, runtimes, closure, or retain}
shift
: "${RUNNER_TEMP:?}"
: "${GITHUB_WORKSPACE:?}"
: "${GITHUB_JOB:?}"
repo=$(realpath "$GITHUB_WORKSPACE")
job_root="$RUNNER_TEMP/logos-release-$GITHUB_JOB"
evidence="$job_root/evidence"
mkdir -p "$evidence"
cd "$repo"

run() {
  local label=$1 started ended
  local -a statuses
  shift
  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf '%q ' "$@" > "$evidence/$label.command"
  printf '\n' >> "$evidence/$label.command"
  set +e
  "$@" 2>&1 | tee "$evidence/$label.log"
  statuses=("${PIPESTATUS[@]}")
  set -e
  ended=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  jq -n --arg start "$started" --arg end "$ended" \
    --argjson exit "${statuses[0]}" --argjson log_exit "${statuses[1]}" \
    '{start_utc:$start,end_utc:$end,exit_code:$exit,log_exit_code:$log_exit}' \
    > "$evidence/$label.result.json"
  test "${statuses[1]}" -eq 0 || return "${statuses[1]}"
  return "${statuses[0]}"
}

clean_tip() {
  test "$(git rev-parse HEAD)" = "$TESTED_REVISION"
  test -z "$(git status --porcelain)"
}

identity() {
  : "${TESTED_REVISION:?}"
  [[ $TESTED_REVISION =~ ^[0-9a-f]{40}$ ]]
  clean_tip
  local head tree src_tree requested fingerprint
  head=$(git rev-parse HEAD)
  tree=$(git rev-parse 'HEAD^{tree}')
  src_tree=$(git rev-parse HEAD:src)
  requested=$(jq -er '.src_tree' logos-doc/validation/release-acceptance-request.json)
  test "$src_tree" = "$requested"
  jq -er '.frozen_inputs | to_entries[] | "\(.value)  \(.key)"' \
    logos-doc/validation/release-acceptance-request.json > "$evidence/frozen-inputs.sha256"
  sha256sum --check --strict "$evidence/frozen-inputs.sha256"
  fingerprint=$({ hostname; cat /proc/sys/kernel/random/boot_id; } | sha256sum | cut -d ' ' -f1)
  jq -n --arg head "$head" --arg tree "$tree" --arg src_tree "$src_tree" \
    --arg event_sha "${GITHUB_SHA:-}" --arg run "${GITHUB_RUN_ID:-}" \
    --arg attempt "${GITHUB_RUN_ATTEMPT:-}" --arg job "$GITHUB_JOB" \
    --arg runner "${RUNNER_NAME:-}" --arg image "${ImageOS:-}" \
    --arg host_fingerprint "$fingerprint" --arg retention "${GITHUB_RETENTION_DAYS:-unreported}" \
    '{head:$head,tree:$tree,src_tree:$src_tree,event_sha:$event_sha,run_id:$run,
      run_attempt:$attempt,job:$job,runner:$runner,image:$image,
      host_fingerprint:$host_fingerprint,log_retention_days:$retention}' \
    > "$evidence/identity.json"
  cat "$evidence/identity.json"
}

preflight() {
  local mem disk min_mem min_disk fs directory
  mem=$(awk '/^MemAvailable:/ {print $2 * 1024}' /proc/meminfo)
  disk=$(df -B1 --output=avail "$RUNNER_TEMP" | tail -n 1 | tr -d ' ')
  min_mem=$((6 * 1024 * 1024 * 1024))
  min_disk=$((8 * 1024 * 1024 * 1024))
  if [[ $mode == parity ]]; then
    min_mem=$((10 * 1024 * 1024 * 1024))
    min_disk=$((10 * 1024 * 1024 * 1024))
  fi
  printf '%s mode=%s mem_available=%s disk_available=%s required_mem=%s required_disk=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$mode" "$mem" "$disk" "$min_mem" "$min_disk" \
    | tee -a "$evidence/resources.log"
  awk -v have="$mem" -v need="$min_mem" 'BEGIN {exit !(have >= need)}'
  test "$disk" -ge "$min_disk"
  fs=$(stat -f -c %T "$RUNNER_TEMP")
  test "$fs" != tmpfs
  directory=$(realpath "$RUNNER_TEMP")
  while :; do
    test ! -e "$directory/.git"
    test "$directory" != / || break
    directory=$(dirname "$directory")
  done
}

bootstrap() {
  preflight
  mkdir "$job_root/tmp" "$job_root/go-tmp" "$job_root/build-cache" "$job_root/postbuild-cache"
  export TMPDIR="$job_root/tmp" GOTMPDIR="$job_root/go-tmp"
  export GOCACHE="$job_root/build-cache" GOENV=off GOTOOLCHAIN=local GOWORK=off
  export GOMAXPROCS=2 GOGC=50 GOMEMLIMIT=2GiB GOFLAGS=-p=2
  unset GOOS GOARCH GOEXPERIMENT GO_GCFLAGS BOOT_GO_GCFLAGS CGO_ENABLED
  run bootstrap-go bash .github/scripts/install-ci-tool.sh go
  run bootstrap-node bash .github/scripts/install-ci-tool.sh node
  stock="$RUNNER_TEMP/go1.27.4-ci"
  export PATH="$stock/bin:$RUNNER_TEMP/node-v26.13.1-ci/bin:$PATH"
  run upstream-oracles bash .github/scripts/install-upstream-oracles.sh
  export UPROOT1272="$RUNNER_TEMP/go1.27.2" UPROOT1273="$RUNNER_TEMP/go1.27.3" UPROOT1274="$stock"
  export GOROOT_BOOTSTRAP="$stock"
  export BOOT_GO_GCFLAGS=-c=1 GO_GCFLAGS=-c=1
  cd "$repo/src"
  run make ./make.bash
  unset GO_GCFLAGS BOOT_GO_GCFLAGS
  export GOROOT="$repo" PATH="$repo/bin:$repo/lib/wasm:$PATH"
  export GOCACHE="$job_root/postbuild-cache"
  test -z "$(ls -A "$GOCACHE")"
  run installed-go go version
  run installed-logos logos version
  go env GOAMD64 GOARM64 GOEXPERIMENT CGO_ENABLED > "$evidence/installed-settings.txt"
  sha256sum "$repo/bin/go" "$repo/bin/logos" \
    "$repo/pkg/tool/linux_amd64/"{compile,link,asm,cgo} > "$evidence/installed-tools.sha256"
  local compiler
  compiler=$(command -v gcc-14)
  mkdir "$job_root/c23"
  ln -s "$compiler" "$job_root/c23/cc"
  printf 'int main(void) { return 0; }\n' > "$job_root/c23/check.c"
  run c23-compile "$job_root/c23/cc" -std=c23 "$job_root/c23/check.c" -o "$job_root/c23/check"
  run c23-run "$job_root/c23/check"
  export PATH="$job_root/c23:$PATH" CC="$compiler"
  clean_tip
  preflight
}

verify_script() {
  local name=$1 log=$2
  grep -Eq "^[[:space:]]*--- PASS: TestScript/$name \\(" "$log" || return 1
  if grep -Eq "^[[:space:]]*--- SKIP: TestScript/$name( |/)" "$log"; then
    return 1
  fi
}

matrix_gate() {
  local driver=${1:?} level=${2:?} skip
  local -a packages
  [[ $driver == go || $driver == logos ]]
  [[ $level =~ ^[123]$ ]]
  bootstrap
  skip=$(sed -n "s/^env REDTEAM80_SKIP='\(.*\)'$/\1/p" cmd/go/testdata/script/logos_redteam_80.txt)
  test -n "$skip"
  test "$(tr '|' '\n' <<< "$skip" | wc -l)" -eq 31
  printf '%s\n' "$skip" > "$evidence/explicit-skip.txt"
  run skip-contract go test -count=1 -v -timeout=20m cmd/go -run '^TestScript$/^logos_redteam_80$'
  verify_script logos_redteam_80 "$evidence/skip-contract.log"
  go list std cmd > "$evidence/packages-all.txt"
  grep -Fx os "$evidence/packages-all.txt"
  mapfile -t packages < <(awk '$0 != "os"' "$evidence/packages-all.txt")
  test "${#packages[@]}" -gt 0
  run matrix-bulk "$driver" test "-O=$level" -count=1 -p=3 -timeout=60m -skip "$skip" "${packages[@]}"
  run matrix-os "$driver" test "-O=$level" -count=1 -parallel=1 -timeout=60m -skip "$skip" os
  clean_tip
}

ledger_no_planned() {
  awk -F ' \\| ' '/^\| #/ && $5 == "planned |" {print; bad=1} END {exit bad}' \
    "$repo/logos-doc/REDTEAM.md" > "$evidence/planned-rows.txt"
}

parity_gate() {
  ledger_no_planned
  bootstrap
  clean_tip
  printf '%s\n' "$TESTED_REVISION" > "$evidence/expected-tip.txt"
  run parity go test -count=1 -timeout=3h -v cmd/internal/goparity \
    -goparity.release "-goparity.stamp=$evidence/goparity.stamp"
  cmp "$evidence/expected-tip.txt" "$evidence/goparity.stamp"
  clean_tip
}

txtar_file() {
  local archive=$1 name=$2 output=$3
  awk -v name="$name" '
    /^-- .* --$/ {if (active) exit; active=($0 == "-- " name " --"); next}
    active {print}
  ' "$archive" > "$output"
  test -s "$output"
}

repro_gate() {
  bootstrap
  export CGO_ENABLED=0 GOAMD64=v3 GOARM64=v8.2 GOEXPERIMENT=
  run repro-script go test -count=1 -parallel=1 -v -timeout=60m cmd/go -run '^TestScript$/^logos_repro$'
  verify_script logos_repro "$evidence/repro-script.log"
  run example-contract go test -count=1 -parallel=1 -v -timeout=30m cmd/go -run '^TestScript$/^logos_eject_example$'
  verify_script logos_eject_example "$evidence/example-contract.log"
  local -a recorded
  mapfile -t recorded < <(awk '$1 == "sha256:" && length($2) == 64 {print $2}' "$evidence/repro-script.log")
  test "${#recorded[@]}" -eq 8
  test "${recorded[0]}" = "${recorded[1]}"
  test "${recorded[2]}" = "${recorded[3]}"
  test "${recorded[4]}" = "${recorded[5]}"
  test "${recorded[6]}" = "${recorded[0]}"
  test "${recorded[7]}" = "${recorded[0]}"
  printf 'repro/linux/amd64\t%s\nrepro/linux/arm64\t%s\nrepro/js/wasm\t%s\n' \
    "${recorded[0]}" "${recorded[2]}" "${recorded[4]}" > "$evidence/digests.tsv"
  local module="$job_root/order" target copy cache output first second label
  mkdir "$module"
  cp "$repo/test/logos/example/order_service.lgo" "$module/order_service.lgo"
  txtar_file "$repo/src/cmd/go/testdata/script/logos_eject_example.txt" m/go.mod "$module/go.mod"
  sha256sum "$module/order_service.lgo" "$module/go.mod" > "$evidence/order-inputs.sha256"
  cd "$module"
  for target in linux/amd64 linux/arm64 js/wasm; do
    label=${target//\//-}
    for copy in a b; do
      cache=$(mktemp -d "$job_root/order-cache-$label-$copy-XXXXXX")
      output="$job_root/order-$label-$copy"
      run "order-$label-$copy" env GOOS="${target%/*}" GOARCH="${target#*/}" "GOCACHE=$cache" \
        logos build -trimpath -buildvcs=false -o "$output" .
      sha256sum "$output" > "$evidence/order-$label-$copy.sha256"
      # This exact cache was created above, is task-owned, and has no readers.
      rm -rf -- "$cache"
    done
    first=$(cut -d ' ' -f1 "$evidence/order-$label-a.sha256")
    second=$(cut -d ' ' -f1 "$evidence/order-$label-b.sha256")
    test "$first" = "$second"
    printf 'order/%s\t%s\n' "$target" "$first" >> "$evidence/digests.tsv"
  done
  jq -Rn '[inputs | split("\t") | {key:.[0],value:.[1]}] | from_entries' \
    < "$evidence/digests.tsv" > "$evidence/digests.json"
  jq --slurpfile digests "$evidence/digests.json" \
    '. + {digests:$digests[0],recipe:{goamd64:"v3",goarm64:"v8.2",cgo:"0",experiment:"",flags:["-trimpath","-buildvcs=false"]}}' \
    "$evidence/identity.json" > "$evidence/repro-result.json"
  printf 'result=%s\n' "$(jq -c . "$evidence/repro-result.json")" >> "$GITHUB_OUTPUT"
  cd "$repo"
  clean_tip
}

compare_gate() {
  : "${REPRO_A:?}" "${REPRO_B:?}"
  printf '%s\n' "$REPRO_A" > "$evidence/host-a.json"
  printf '%s\n' "$REPRO_B" > "$evidence/host-b.json"
  jq -e -s '
    .[0].host_fingerprint != .[1].host_fingerprint and
    .[0].head == .[1].head and .[0].tree == .[1].tree and
    .[0].src_tree == .[1].src_tree and .[0].recipe == .[1].recipe and
    (.[0].digests | length) == 6 and .[0].digests == .[1].digests
  ' "$evidence/host-a.json" "$evidence/host-b.json" | tee "$evidence/compare.log"
}

install_runtimes() {
  local sum='h1:nPO/LffYa4sP62Ueuema5PPnZPGaZjGl+vZGvbTek4g='
  local archive="$job_root/qemu.deb"
  mkdir "$job_root/stock-cache" "$job_root/gopls-bin" "$job_root/gopls-modules" "$job_root/gopls-work" "$job_root/qemu"
  local -a stock_env=(env "GOROOT=$stock" "GOCACHE=$job_root/stock-cache" \
    "GOMODCACHE=$job_root/gopls-modules" "GOPATH=$job_root/gopls-work" \
    "GOBIN=$job_root/gopls-bin" GOPROXY=https://proxy.golang.org GOSUMDB=sum.golang.org GOEXPERIMENT= CGO_ENABLED=0)
  cd "$job_root"
  run gopls-download "${stock_env[@]}" "$stock/bin/go" mod download -json golang.org/x/tools/gopls@v0.24.0
  jq -e --arg sum "$sum" '.Path == "golang.org/x/tools/gopls" and .Version == "v0.24.0" and .Sum == $sum' "$evidence/gopls-download.log"
  run gopls-install "${stock_env[@]}" "$stock/bin/go" install golang.org/x/tools/gopls@v0.24.0
  run gopls-version "$job_root/gopls-bin/gopls" version
  run gopls-build-info "$stock/bin/go" version -m "$job_root/gopls-bin/gopls"
  grep -F "golang.org/x/tools/gopls v0.24.0" "$evidence/gopls-version.log"
  grep -F $'mod\tgolang.org/x/tools/gopls\tv0.24.0\t'"$sum" "$evidence/gopls-build-info.log"
  run qemu-download curl --fail --silent --show-error --location --output "$archive" \
    'https://ftp.debian.org/debian/pool/main/q/qemu/qemu-user_10.0.13+ds-0+deb13u1_amd64.deb'
  printf '%s  %s\n' ca6ede739327a20ae5a3498230c8a3a9a072c586e131bc6bcc1201eb1725e336 "$archive" | sha256sum --check --strict -
  dpkg-deb --extract "$archive" "$job_root/qemu"
  printf '%s  %s\n' 73cc2584f119f9e74c85c0a8549dbb6fd104ae3897a06a7500797f9e5b4d0943 \
    "$job_root/qemu/usr/bin/qemu-aarch64" | sha256sum --check --strict -
  run qemu-version "$job_root/qemu/usr/bin/qemu-aarch64" --version
  export PATH="$job_root/gopls-bin:$job_root/qemu/usr/bin:$PATH"
  sha256sum "$job_root/gopls-bin/gopls" "$job_root/qemu/usr/bin/qemu-aarch64" > "$evidence/runtime-tools.sha256"
  cd "$repo/src"
}

runtime_gate() {
  bootstrap
  install_runtimes
  run node-version node --version
  grep -Fx v26.13.1 "$evidence/node-version.log"
  command -v gopls qemu-aarch64 node > "$evidence/runtime-paths.txt"
  export LOGOS_REQUIRE_WASMFX=1
  local -a scripts=(logos_lsp logos_sum_crossarch logos_go_parity_wasm)
  local file name pattern normalized="$evidence/runtime-normalized.log"
  for file in cmd/go/testdata/script/logos_wasm_*.txt; do
    name=${file##*/}
    scripts+=("${name%.txt}")
  done
  printf '%s\n' "${scripts[@]}" > "$evidence/required-scripts.txt"
  pattern=$(IFS='|'; printf '%s' "${scripts[*]}")
  test "$(go env CGO_ENABLED)" = 1
  run runtimes go test -race -v -count=1 -parallel=1 -timeout=90m cmd/go \
    -run "^TestScript$/^($pattern)$"
  for name in "${scripts[@]}"; do
    verify_script "$name" "$evidence/runtimes.log"
  done
  sed 's/^[[:space:]]*//' "$evidence/runtimes.log" > "$normalized"
  sed -n "/^-- /q; s/^stdout '\(.*\)'$/\1/p" cmd/go/testdata/script/logos_lsp.txt > "$evidence/lsp-markers.txt"
  test "$(wc -l < "$evidence/lsp-markers.txt")" -eq 11
  while IFS= read -r pattern; do
    grep -Eq "$pattern" "$normalized"
  done < "$evidence/lsp-markers.txt"
  # The script transcript preserves its variable names literally.
  # shellcheck disable=SC2016
  grep -Fx '> exec qemu-aarch64 $WORK/sum-arm64$GOEXE' "$normalized"
  # shellcheck disable=SC2016
  grep -Fx '> cmp $WORK/native.txt $WORK/arm64.txt' "$normalized"
  # shellcheck disable=SC2016
  grep -F '> exec node --experimental-wasm-wasmfx run.mjs $WORK/exe.wasm' "$normalized"
  sha256sum --check --strict "$evidence/runtime-tools.sha256"
  clean_tip
}

closure_gate() {
  local status id bad=0 rows=0
  while IFS=$'\t' read -r id status; do
    rows=$((rows + 1))
    case "$status" in
      tested|'satisfied upstream') ;;
      blocked)
        case "$id" in
          216|217|218|226) ;;
          *) printf 'M22 closure remains blocked by #%s; retain its mechanism and frozen attacks.\n' "$id"; bad=1 ;;
        esac
        ;;
      *) printf 'M22 closure rejects #%s status %s.\n' "$id" "$status"; bad=1 ;;
    esac
  done < <(awk -F ' \\| ' '/^\| #/ {sub(/^\| #/, "", $1); sub(/ \|$/, "", $5); print $1 "\t" $5}' "$repo/logos-doc/REDTEAM.md")
  test "$rows" -gt 0
  return "$bad"
}

retain() {
  local archive="$job_root/evidence.tar.gz"
  cd "$evidence"
  find . -type f ! -name files.sha256 -print0 | sort -z | xargs -0 -r sha256sum > "$job_root/files.sha256"
  mv "$job_root/files.sha256" files.sha256
  tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -cf - . | gzip -n -9 > "$archive"
  printf 'LOGOS_RELEASE_EVIDENCE_BEGIN %s\n' "$GITHUB_JOB"
  sha256sum "$archive"
  base64 -w 76 "$archive"
  printf 'LOGOS_RELEASE_EVIDENCE_END %s\n' "$GITHUB_JOB"
}

if [[ $mode == retain ]]; then
  retain
  exit
fi
identity
trap 'status=$?; printf "%s\n" "$status" > "$evidence/executor-exit.txt"' EXIT
case "$mode" in
  matrix) matrix_gate "$@" ;;
  parity) parity_gate ;;
  repro) repro_gate ;;
  compare) compare_gate ;;
  runtimes) runtime_gate ;;
  closure) closure_gate ;;
  *) printf 'unknown release gate: %s\n' "$mode" >&2; exit 2 ;;
esac
