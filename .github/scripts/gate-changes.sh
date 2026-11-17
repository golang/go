#!/usr/bin/env bash
# Classify the pull-request diff into gate names. Prints one gate per line on
# stdout. On push or workflow_dispatch all gates are emitted; every gate body
# still verifies its own contract, so an over-wide trigger only costs time.
set -euo pipefail

base=${1:-}
if [[ -z $base ]]; then
  # workflow_dispatch and push: run every gate.
  printf '%s\n' deferred-callees generic-type-arguments simd-host json-profiles
  exit 0
fi
head=${GITHUB_SHA:?}
git -c protocol.version=2 fetch --no-tags --depth=1 origin "$base"
changed=$(git diff --name-only "$base" "$head")

any() { grep -Eq "$1" <<< "$changed"; }

# Shared infrastructure that every gate depends on.
if any '^\.github/scripts/(install-ci-tool|install-upstream-oracles|gate-changes|feature-gate|gate-setup|gates/)\.?' \
   || any '^\.github/workflows/feature-gates\.yml$'; then
  shared=1
else
  shared=0
fi

if [[ $shared == 1 ]] \
  || any '^src/cmd/compile/internal/logos/' \
  || any '^src/cmd/go/testdata/script/logos_eject_defer_(indexed_callee|generic_receiver)\.txt$' \
  || any '^test/logos/run/defer_(indexed_callee|generic_receiver)\.'; then
  echo deferred-callees
fi
if [[ $shared == 1 ]] \
  || any '^src/cmd/compile/internal/(syntax|types2|logos)/' \
  || any '^src/cmd/go/testdata/script/logos_eject_generic_type_arguments\.txt$' \
  || any '^test/logos/run/generic_type_arg'; then
  echo generic-type-arguments
fi
if [[ $shared == 1 ]] \
  || any '^src/cmd/go/scriptconds(_simd_logos)?_test\.go$' \
  || any '^src/cmd/go/internal/logos/config/targetcpu' \
  || any '^src/cmd/compile/internal/ssa/' \
  || any '^src/cmd/go/testdata/script/logos_(simd_|eject_simd_slice)'; then
  echo simd-host
fi
if [[ $shared == 1 ]] \
  || any '^src/cmd/compile/internal/logos/' \
  || any '^src/logos/rt/json/' \
  || any '^src/cmd/go/testdata/script/logos_.*json.*\.txt$' \
  || any '^test/logos/run/(json|comptime_json).*\.lgo$'; then
  echo json-profiles
fi
