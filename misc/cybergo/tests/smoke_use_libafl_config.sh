#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT_DIR}/misc/cybergo/tests/smoke_use_libafl_common.sh"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
export GOCACHE="${tmp_dir}/gocache"

cpu_count="$(
  getconf _NPROCESSORS_ONLN 2>/dev/null || true
)"
if [[ -z "${cpu_count}" ]]; then
  cpu_count="1"
fi

cores="0"
if [[ "${cpu_count}" -ge 2 ]]; then
  cores="0,1"
fi

cfg_path="${tmp_dir}/libafl-config.jsonc"
cat >"${cfg_path}" <<EOF
{
  // cores: CPU cores to bind LibAFL clients to
  "cores": "${cores}"
}
EOF

cd "${ROOT_DIR}/test/cybergo/examples/reverse"
in_dir="$(libafl_input_dir)"
mkdir -p "${in_dir}"
printf 'FUZZING!' > "${in_dir}/seed-crash"

output_file="${tmp_dir}/output.txt"
run_expect_crash reverse FuzzReverse 2m "${output_file}" --libafl-config="${cfg_path}"

if ! grep -Fq "GOLIBAFL_CONFIG_APPLIED cores_ids=${cores}" "${output_file}"; then
  echo "expected output to confirm applied LibAFL config cores_ids=${cores}"
  exit 1
fi
