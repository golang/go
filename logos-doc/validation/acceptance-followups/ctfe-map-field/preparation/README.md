# Frozen positive CTFE map-field fixture

Status: prepared and frozen for later execution. No Go process, commit,
implementation edit, normative-spec edit, or ledger edit was performed by this
lane. Root accepted this separate test-oracle correction after the read-only
review in `../const-func-store-review.md`.

## Change and proof obligation

The obsolete errorcheck input is preserved byte-for-byte as
`const_func_unsupported_store.original.lgo`, SHA-256
`0997677cba3f15d6be766c190860622bbba3104a1482d4f4293a63640c0bf02a`.

The acceptance worktree replaces it with
`test/logos/run/const_func_map_field_store.lgo` and its `.out` file. The
`storeField` function body is byte-identical; its SHA-256 is
`5c37c42ae72003063a84baa513f2daf13a7fbf4288bce6de2a23d7ca6b0faae1`.
`frozen-positive.sha256` records both new input hashes.

The assignment `var _ [4]struct{} = [stored]struct{}{}` requires `stored` to be
a compile-time constant equal to 4. The program's required output is `4` and
LF. Its additional call to `storeField` must agree, but that call may itself
be folded. This fixture therefore does not claim to force an independent
unfolded runtime execution of the function body.

SPEC lines 448-451 require Go assignment identity and map-reference sharing;
line 462 requires rejection only for forms that remain unsupported. The
review identifies c3733e8a as the change that evaluates and captures arbitrary
map-valued destination bases, replacing the old bare-name restriction. This
is an oracle correction for already supported fresh local state, with no new
language rule or artificial production rejection.

## Preservation checks

`preparation.json` records exact hashes and a byte comparison of the function
body. All ten CTFE entries in the historical compiler-stability freeze manifest
still match. The genuine map-key rejection fixture and the frozen nil-map and
assignment-failure controls match `rejection-controls.sha256`.

`frozen-before.log` and `frozen-after.log` are byte-identical. Both show the
same two pre-existing non-CTFE mismatches, for the variant-host-tags and
variant-tags tests. The tooling lane owns those authorized mechanical adapter
changes and their separate exact-transform proof. This lane did not edit those
files or rewrite the historical manifest, and does not claim that the entire
old manifest passed in that shared worktree.

## Planned checks, not executed

After root releases the local process lane, use the freshly built frozen fork
and a parent-approved task cache. Set `UPROOT1274` to the pinned pristine Go
1.27.4 tree. Keep the temporary module outside any workspace Git ancestry.

```bash
set -euo pipefail
fork_root=/workspace/scratch/9949907ca71e/logos-json-integration
accept_root=/workspace/scratch/9949907ca71e/logos-acceptance-fixes
case_root=$(mktemp -d /var/tmp/logos-map-field.XXXXXX)
: "${UPROOT1274:?set the pinned pristine Go 1.27.4 tree}"
: "${GOCACHE:?set the parent-approved task cache}"
export GOROOT="$fork_root" GOPROXY=off GOTOOLCHAIN=local GOEXPERIMENT=""
export PATH="$fork_root/bin:$PATH"
mkdir "$case_root/m"
cp "$accept_root/test/logos/run/const_func_map_field_store.lgo" "$case_root/m/main.lgo"
cp "$accept_root/test/logos/run/const_func_map_field_store.out" "$case_root/expected.txt"
printf 'module example.com/constmapfield\n\ngo 1.27\n' > "$case_root/m/go.mod"
printf '[build]\narena = false\ntarget_cpu = "baseline"\n' > "$case_root/m/logos.toml"
cd "$case_root/m"
"$fork_root/bin/logos" run . > "$case_root/native.stdout" 2> "$case_root/native.stderr"
cmp "$case_root/expected.txt" "$case_root/native.stdout"
"$fork_root/bin/logos" eject -o out > "$case_root/eject.stdout" 2> "$case_root/eject.stderr"
cd out
"$fork_root/bin/go" run . > "$case_root/ejected-fork.stdout" 2> "$case_root/ejected-fork.stderr"
cmp "$case_root/expected.txt" "$case_root/ejected-fork.stdout"
env -u GOROOT "$UPROOT1274/bin/go" run . > "$case_root/ejected-upstream.stdout" 2> "$case_root/ejected-upstream.stderr"
cmp "$case_root/expected.txt" "$case_root/ejected-upstream.stdout"
cd "$fork_root/src"
"$fork_root/bin/go" test -count=1 -v cmd/internal/testdir -run '^Test/logos/compile/const_func_map_store_key\.lgo$'
"$fork_root/bin/go" test -count=1 -short cmd/compile/internal/logos/ctfe -run '^TestAssignment(EvaluationOrder|FailureOrder|NilMapReportsError|Storage|StorageReturnedValues)$'
cd "$accept_root"
sha256sum --check /workspace/scratch/9949907ca71e/hosted-evidence/language/const-map-field-oracle/frozen-positive.sha256
sha256sum --check /workspace/scratch/9949907ca71e/hosted-evidence/language/const-map-field-oracle/rejection-controls.sha256
```

Record the fresh compiler/source provenance, commands, exit statuses, exact
outputs, and unchanged hashes when these checks actually run. Do not describe
this plan as completed validation.
