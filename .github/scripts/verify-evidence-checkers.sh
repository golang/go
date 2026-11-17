#!/usr/bin/env bash
# Runs the Python-based regression-evidence checker suites' own tests, types,
# and lint under one uv invocation prefix. Requires `uv` on PATH.
set -euo pipefail
cd "${GITHUB_WORKSPACE:?}"

helper=logos-doc/validation/tagged-package-facts
method=logos-doc/validation/tagged-method-facts
capability=logos-doc/validation/capability-inline-context
host=logos-doc/validation/upstream-host
offset=logos-doc/validation/field-offset-lint
simd_imports=logos-doc/validation/simd-transitive-imports
zero=logos-doc/validation/zero-array-reference-free

uvx() {
  uv run --no-project --no-config --python 3.12 --no-python-downloads \
    --with-requirements "$helper/test-requirements.txt" "$@"
}

for suite in "$helper" "$method"; do
  uvx python -m pytest -q "$suite/test_check_ci.py"
  uvx pyright "$suite/check_ci.py" "$suite/test_check_ci.py"
done
uvx python -m pytest -q "$capability/test_check_ci.py"
uvx pyright "$capability/test_check_ci.py"
uvx python -m pytest -q "$host"
uvx pyright "$host"/*.py
uvx python -m pytest -q "$offset/test_field_offset_ci.py" logos-doc/validation/test_native_frozen_gate.py
uvx pyright --project "$offset/pyrightconfig.json"
uvx python -m pytest -q "$simd_imports/test_simd_imports_ci.py"
uvx pyright --project "$simd_imports/pyrightconfig.json"
uvx python -m pytest -q "$zero/test_zero_array_ci.py"
uvx pyright --project "$zero/pyrightconfig.json"
uvx ruff check "$simd_imports"/*.py
uvx ruff format --check "$simd_imports"/*.py
uvx ruff check "$helper/check_ci.py" "$helper/test_check_ci.py" "$method/check_ci.py" "$method/test_check_ci.py" "$capability/test_check_ci.py" "$host"/*.py "$zero"/*.py
uvx ruff format --check "$helper/check_ci.py" "$helper/test_check_ci.py" "$method/check_ci.py" "$method/test_check_ci.py" "$capability/test_check_ci.py" "$host"/*.py "$zero"/*.py
uvx ruff check "$offset"/*.py logos-doc/validation/native_frozen_gate.py logos-doc/validation/test_native_frozen_gate.py
uvx ruff format --check "$offset"/*.py logos-doc/validation/native_frozen_gate.py logos-doc/validation/test_native_frozen_gate.py
shellcheck "$capability/check_ci.sh" .github/scripts/logos-test-env.sh .github/scripts/logos-upstream-identity.sh
