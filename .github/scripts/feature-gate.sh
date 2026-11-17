#!/usr/bin/env bash
# Shared body of the feature-gate CI jobs: pinned checkout, toolchain build,
# then the gate-specific checks. Runs inside the repo root; the script cds
# into src/ for build and test steps.
set -euo pipefail
gate=${1:?expected deferred-callees, generic-type-arguments, simd-host, or json-profiles}
repo=$PWD

setup() {
  cd "$repo/src"
  bash ../.github/scripts/install-ci-tool.sh go
  # GITHUB_PATH/GITHUB_ENV only take effect in later steps; this step must
  # export the pinned roots itself.
  stock="$RUNNER_TEMP/go1.27.4-ci"
  PATH="$stock/bin:$PATH"
  export PATH
  bash ../.github/scripts/install-upstream-oracles.sh
  export UPROOT1272="$RUNNER_TEMP/go1.27.2" UPROOT1273="$RUNNER_TEMP/go1.27.3" UPROOT1274="$stock"
  GOROOT_BOOTSTRAP=$stock
  export GOROOT_BOOTSTRAP
  GOEXPERIMENT="${GOEXPERIMENT-}" ./make.bash
  PATH="$repo/bin:$PATH"
  export PATH
}

case "$gate" in
  deferred-callees)
    setup
    test -z "$(gofmt -l cmd/compile/internal/logos/defer.go cmd/compile/internal/logos/scopedefer.go)"
    go vet cmd/compile/internal/logos
    go test -count=1 -short -timeout 20m cmd/compile/internal/logos
    go test -count=1 -timeout 20m cmd/internal/testdir -run 'Test/logos/run/(defer_indexed_callee|defer_generic_receiver|errdefer_recv|defer_try_semantics|scope_defer)'
    go test -count=1 -race -timeout 20m cmd/go -run 'Script/^logos_(eject_defer_indexed_callee|eject_defer_generic_receiver|defer_try_eject_parity|eject_defer_go_option)$'
    target=cmd/compile/internal/logos/defer.go
    saved="$RUNNER_TEMP/defer.go.saved"
    cp "$target" "$saved"
    trap 'cp "$saved" "$target"' EXIT
    sha256sum ../test/logos/run/defer_indexed_callee.lgo ../test/logos/run/defer_indexed_callee.out cmd/go/testdata/script/logos_eject_defer_indexed_callee.txt > "$RUNNER_TEMP/deferred-fixtures.sha256"
    perl -0pi -e 's/\Qfun.start = fl.exprStart(call.Fun)\E/fun.start = fl.off(leftmostOf(call.Fun))/ == 1 or die "expected one callee span\n";' "$target"
    if go test -count=1 -short cmd/compile/internal/logos -run '^TestNoLogosNodesSurvive$/^defer_indexed_callee.lgo$' > "$RUNNER_TEMP/deferred-probe.log" 2>&1; then
      cat "$RUNNER_TEMP/deferred-probe.log"
      exit 1
    fi
    cat "$RUNNER_TEMP/deferred-probe.log"
    grep -F 'TestNoLogosNodesSurvive/defer_indexed_callee.lgo' "$RUNNER_TEMP/deferred-probe.log"
    grep -F 'syntax error' "$RUNNER_TEMP/deferred-probe.log"
    cp "$saved" "$target"
    cmp "$saved" "$target"
    sha256sum --check --strict "$RUNNER_TEMP/deferred-fixtures.sha256"
    go test -count=1 -short cmd/compile/internal/logos -run '^TestNoLogosNodesSurvive$/^defer_indexed_callee.lgo$'
    ;;
  generic-type-arguments)
    setup
    test -z "$(gofmt -l cmd/compile/internal/syntax/nodes.go cmd/compile/internal/syntax/parser.go cmd/compile/internal/syntax/parser_tuple_logos.go cmd/compile/internal/syntax/parser_typeargs_logos.go cmd/compile/internal/syntax/typeargs_logos.go cmd/compile/internal/types2/expr.go cmd/compile/internal/types2/index.go cmd/compile/internal/types2/typexpr.go cmd/compile/internal/types2/typeargs_logos_test.go cmd/compile/internal/types2/typeargs_nested_logos_test.go cmd/compile/internal/logos/defer.go cmd/compile/internal/logos/tuple.go)"
    go vet cmd/compile/internal/syntax cmd/compile/internal/types2 cmd/compile/internal/logos
    sha256sum cmd/compile/internal/types2/typeargs_logos_test.go cmd/compile/internal/types2/typeargs_nested_logos_test.go ../test/logos/run/generic_type_arguments.lgo ../test/logos/run/generic_type_arguments.out ../test/logos/run/generic_type_argument_index_controls.lgo ../test/logos/run/generic_type_argument_index_controls.out cmd/go/testdata/script/logos_eject_generic_type_arguments.txt > "$RUNNER_TEMP/typeargs-fixtures.sha256"
    go test -count=1 -timeout 20m cmd/internal/testdir -run 'Test/logos/run/(generic_type_arguments|generic_type_argument_index_controls|defer_generic_receiver|defer_indexed_callee)'
    go test -count=1 -race -timeout 20m cmd/go -run 'Script/^logos_(eject_generic_type_arguments|eject_defer_generic_receiver|eject_defer_indexed_callee)$'
    target=cmd/compile/internal/types2/index.go
    saved="$RUNNER_TEMP/typeargs-index.go.saved"
    cp "$target" "$saved"
    trap 'cp "$saved" "$target"' EXIT
    perl -0pi -e 's/\Qsyntax.LogosTypeArguments(e)\E/\/\/ mutation: skip confirmed function type arguments/ == 1 or die "expected one generic hook\n";' "$target"
    if go test -count=1 -short cmd/compile/internal/types2 -run '^TestLogos(GenericTypeArgumentPositions|NestedTypeArgumentPositions|TypeArgumentIndexControls)$' > "$RUNNER_TEMP/typeargs-generic-probe.log" 2>&1; then
      cat "$RUNNER_TEMP/typeargs-generic-probe.log"
      exit 1
    fi
    cat "$RUNNER_TEMP/typeargs-generic-probe.log"
    grep -F 'int? is not a type' "$RUNNER_TEMP/typeargs-generic-probe.log"
    cp "$saved" "$target"
    cmp "$saved" "$target"
    trap - EXIT
    target=cmd/compile/internal/logos/defer.go
    saved="$RUNNER_TEMP/typeargs-defer.go.saved"
    cp "$target" "$saved"
    trap 'cp "$saved" "$target"' EXIT
    perl -0pi -e 's/\Qhead := drainOwnedAt(fl.ownerOfInsertion(part.start), part.start, 0)\E/head := otext{}/ == 1 or die "expected one type prefix capture\n";' "$target"
    if go test -count=1 -short cmd/compile/internal/logos -run '^TestNoLogosNodesSurvive$/^generic_type_arguments.lgo$' > "$RUNNER_TEMP/typeargs-prefix-probe.log" 2>&1; then
      cat "$RUNNER_TEMP/typeargs-prefix-probe.log"
      exit 1
    fi
    cat "$RUNNER_TEMP/typeargs-prefix-probe.log"
    grep -F 'syntax error: unexpected ]' "$RUNNER_TEMP/typeargs-prefix-probe.log"
    cp "$saved" "$target"
    cmp "$saved" "$target"
    sha256sum --check --strict "$RUNNER_TEMP/typeargs-fixtures.sha256"
    go test -count=1 -short -timeout 20m cmd/compile/internal/syntax cmd/compile/internal/types2 cmd/compile/internal/logos
    ;;
  simd-host)
    GOEXPERIMENT=
    GOAMD64=v1
    export GOAMD64
    setup
    files=(
      cmd/go/scriptconds_test.go
      cmd/go/scriptconds_simd_logos_test.go
      cmd/compile/internal/ssa/zcse.go
      cmd/compile/internal/ssa/branchelim.go
      cmd/compile/internal/ssa/*logos*.go
    )
    gofmt -d "${files[@]}"
    test -z "$(gofmt -l "${files[@]}")"
    go vet cmd/go cmd/compile/internal/ssa
    go test -count=1 -short cmd/compile/internal/ssa
    GOAMD64=v1 GOEXPERIMENT= go test -count=1 -race -timeout=40m -v cmd/go -run '^(TestLogosHostAMD64Condition|TestScript)$/(README|logos_simd_(feature_gate|gate_avx512|gate_branch|nested_guard|zero_dominance)|logos_eject_simd_slice)$'
    ;;
  json-profiles)
    GOEXPERIMENT=
    setup
    test -z "$(gofmt -l logos/rt/json cmd/compile/internal/logos)"
    for variant in jsonv2 nojsonv2; do
      GOEXPERIMENT=$variant go vet logos/rt/json cmd/compile/internal/logos
      GOEXPERIMENT=$variant go test -count=1 -race -timeout=20m logos/rt/json cmd/compile/internal/logos
      GOEXPERIMENT=$variant go test -count=1 -timeout=30m cmd/internal/testdir -run 'Test/logos/run/(json|comptime_json)'
      GOEXPERIMENT=$variant go test -count=1 -race -timeout=20m cmd/go -run 'Script/^logos_(eject_json_(embedded|cycle)|json_fallback_string)$'
    done
    ;;
  *) printf 'unknown feature gate: %s\n' "$gate" >&2; exit 2 ;;
esac
