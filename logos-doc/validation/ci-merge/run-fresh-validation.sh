#!/usr/bin/env bash
set -euo pipefail
task_root=/workspace/scratch/9949907ca71e
source_root=$task_root/logos-json-integration
evidence_root=$task_root/language-evidence/ci-merge
fresh_root=$evidence_root/fresh
tool_root=$task_root/language-ci-tools/bin
runtime_root=/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin
export PATH="$tool_root:$PATH" GIT_OPTIONAL_LOCKS=0 PYTHONDONTWRITEBYTECODE=1 UV_PYTHON_DOWNLOADS=never
mkdir -p "$fresh_root/mutation"
cd "$source_root"
source_head=$(git rev-parse HEAD)
merge_head=$(git rev-parse MERGE_HEAD)
cp "$evidence_root/recovered-inputs.sha256" "$fresh_root/validated-inputs.sha256"
run_record() {
  local name=$1 status=0 started ended stdout_hash stderr_hash
  shift
  started=$(date +%s%3N)
  "$@" >"$fresh_root/$name.stdout" 2>"$fresh_root/$name.stderr" || status=$?
  ended=$(date +%s%3N)
  stdout_hash=$(sha256sum "$fresh_root/$name.stdout")
  stderr_hash=$(sha256sum "$fresh_root/$name.stderr")
  jq -n --arg cwd "$source_root" --arg sourceHead "$source_head" --arg mergeHead "$merge_head" \
    --argjson status "$status" --argjson started "$started" --argjson ended "$ended" \
    --arg stdoutSha256 "${stdout_hash%% *}" --arg stderrSha256 "${stderr_hash%% *}" --args \
    '{command:$ARGS.positional,cwd:$cwd,sourceHead:$sourceHead,mergeHead:$mergeHead,status:$status,signal:null,error:"",seconds:(($ended-$started)/1000),stdoutSha256:$stdoutSha256,stderrSha256:$stderrSha256,provenance:"fresh validation after exact input reconstruction"}' \
    -- "$@" >"$fresh_root/$name.json"
  cat "$fresh_root/$name.json"
}
require_status() {
  [[ $(jq -r .status "$fresh_root/$1.json") == "$2" ]]
}
run_record inputs-before sha256sum --check "$fresh_root/validated-inputs.sha256"
require_status inputs-before 0
run_record final-actionlint "$tool_root/actionlint" -color
require_status final-actionlint 0
run_record final-shellcheck "$tool_root/shellcheck" .github/scripts/install-ci-tool.sh .github/scripts/install-upstream-oracles.sh
require_status final-shellcheck 0
run_record final-yaml-bash "$runtime_root/uv" run --offline --no-cache --no-project --python "$runtime_root/python3" python -B "$evidence_root/check-yaml-shell.py" "$source_root"
require_status final-yaml-bash 0
"$runtime_root/uv" run --offline --no-cache --no-project --python "$runtime_root/python3" python -B - "$source_root/.github/workflows/compiler-stability.yml" "$fresh_root/mutation/compiler-stability.yml" <<'PY'
from pathlib import Path
import sys
original: str = Path(sys.argv[1]).read_text()
before: str = "          GOROOT_BOOTSTRAP=$(go env GOROOT)\n          export GOROOT_BOOTSTRAP\n"
after: str = '          export GOROOT_BOOTSTRAP="$(go env GOROOT)"\n'
assert original.count(before) == 1
Path(sys.argv[2]).write_text(original.replace(before, after, 1))
PY
run_record mutation-actionlint "$tool_root/actionlint" -color -config-file "$source_root/.github/actionlint.yaml" "$fresh_root/mutation/compiler-stability.yml"
require_status mutation-actionlint 1
sed -E 's/\x1b\[[0-9;]*m//g' "$fresh_root/mutation-actionlint.stdout" >"$fresh_root/mutation-normalized.txt"
rg -q 'SC2155' "$fresh_root/mutation-normalized.txt"
[[ $(rg -c -F '[shellcheck]' "$fresh_root/mutation-normalized.txt") == 1 ]]
run_record inputs-after sha256sum --check "$fresh_root/validated-inputs.sha256"
require_status inputs-after 0
run_record unchanged-actionlint "$tool_root/actionlint" -color
require_status unchanged-actionlint 0
cat "$fresh_root/final-yaml-bash.stdout" "$fresh_root/mutation-normalized.txt"
