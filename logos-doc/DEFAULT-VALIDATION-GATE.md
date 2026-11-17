# Native struct-default validation gate

This gate adds installed-compiler evidence to the retained M6-S3a
declaration validator. It adds no language rule and no binding
implementation. The reviewed candidate is PR63 at
`9a656c234a06169bee5a880a4fb116cfec38c98b`; M6-S3b remains blocked.

## Why the native gate is required

Native noder creates `Info{StoreTypesInSyntax: true}` without a `Types`
map. The retained declaration tests use `Info{}` or `ejectCheck`, which
allocates a `Types` map. Their printed native-tree runs compile generated
Go source after direct lowering. Those tests establish useful checker and
lowering behavior, but do not exercise the installed LoGOs checker's
storage configuration.

The retained validator temporarily allocates a map while checking each
default. `recordTypeAndValue` writes fresh results to both requested
stores, and the validator restores the prior nil map. A non-nil map does
hide older syntax-only facts from `Info.TypeOf`, so callers must preserve
the distinction. Source review did not establish a reachable defect in
this retained path: package constants are checked before delayed field
defaults, and the validator checks the default's fresh expressions before
reading its temporary map. This observation does not substitute for the
native gate.

## Frozen script

`src/cmd/go/testdata/script/logos_struct_default_validation.txt` builds
eleven packages in one LoGOs module through the installed `logos`
command. Each tested type is declared in an uncalled function. No value
of a default-bearing type is constructed, and no serializer is invoked.
The build still checks each declaration, including its default.

Function-local types also keep this gate independent of generated JSON
and binary methods. `LogosStaticStruct` and `LogosBinaryStruct` require a
package-scope type. An initial package-level version of the accepted
blank-field controls instead reached the known binding defect:
generated JSON code emitted `_logos_v._` for a blank field, despite no
source constructor call. That probe was preserved separately. Before
freezing the declaration-validation RED, the declarations were moved into
uncalled functions with all acceptance, rejection and diagnostic
expectations retained. No production code was changed for that correction.

Frozen script SHA-256:

```text
4cd169dc2ab9b2fd0e5c751b75bdd4f68ad55ba72a4df63a446d214372b3aab0
```

## Expectations and observed baseline

The source baseline is the completed direct-C component at
`584c7ece750d3e9508bb21c5a95d751dbf18cce2`, based on `b0bbbc05`, before
the retained PR63 validator. No compiler rebuild or mutation was made for
this gate. Its installed binary hashes are:

| Binary | SHA-256 |
|---|---|
| `bin/go` and `bin/logos` | `3d9780d7f455d7b75920b5ce61f8cd34ed6a4f1c3bc539c394ee2b91d3ee2baf` |
| `pkg/tool/linux_amd64/compile` | `85d01f687bcd896258299e4e0a74b21d2e817f1ab1393b20109bf96ba26954ee` |

Every row below was run independently using the exact file contents
extracted from the frozen script. The script itself also ran uncached: it
passed `control`, then failed at `imported` with
`main.lgo:5:60: undefined: time`.

| Package | Unchanged expectation | Observed S3 baseline |
|---|---|---|
| `control` | Accept `uint8` values 0, 1, 254 and 255; constant slice, struct and map values; valid repeated blank fields and groups beginning or ending with `_` | Accepted |
| `imported` | Accept `time.Second` in its declaration's file scope | Rejected: `main.lgo:5:60: undefined: time` |
| `local` | Accept the enclosing local constant | Rejected: `main.lgo:5:35: undefined: local` |
| `overflow_low` | Reject `uint8 = -1` with `overflows` at `bad.lgo:3` | Incorrectly accepted |
| `overflow_high` | Reject `uint8 = 256` with `overflows` at `bad.lgo:3` | Incorrectly accepted |
| `scalar` | Reject a string default for an `int` with `cannot use` at `bad.lgo:3` | Incorrectly accepted |
| `composite` | Reject a `[]string` default for `[]int` with `cannot use` at `bad.lgo:3` | Incorrectly accepted |
| `map_key` | Reject an effectful map-key call with `correctness::default_not_constant` at `bad.lgo:5` | Incorrectly accepted |
| `blank` | Check the second `_` against its own `int` type | Incorrectly accepted a string default |
| `blank_defaults` | Check repeated default-bearing `_` fields independently | Incorrectly accepted a string default for the second `int` field |
| `blank_group` | Check a group beginning with `_` against that group's type | Incorrectly accepted a string default for an `int` group |

The negative blank-field cases require `cannot use` at `bad.lgo:3`, with a
source column. The map-key function changes a package variable if called;
the declarations and the function itself are never executed. The test
requires compile-time rejection of that expression.

The per-case result set has one passing accepted-control package and ten
semantic RED results. These are missing declaration-validation behavior,
not VCS, resource, package-purity or compilation-environment failures.
The recorded local evidence is `default-validation-script-red.log`, the
eleven `default-validation-*-baseline.log` files, and
`default-validation-baseline.json`. The separate initial binding probe is
`default-validation-binding-probe.log`.

## Integration command and result boundary

After merging the retained validator, rebuild the installed toolchain
and run the unchanged script from `src` with the fork first on `PATH`:

```sh
./make.bash
export PATH="$(cd .. && pwd)/bin:$PATH"
go test -count=1 cmd/go -run 'Script/^logos_struct_default_validation$'
```

For the recorded baseline, a freshly built core command-test driver ran
the same selector against the S3 installed toolchain. The command used
`-test.count=1 -test.parallel=1 -test.timeout=10m -test.v` with
`GOROOT` pointing to that component. Both `TMPDIR` and `GOTMPDIR` pointed
to a genuine directory outside workspace VCS ancestry. Native per-case
builds used the ordinary VCS behavior and the same checked-in source.

This test-only commit claims the baseline RED and accepted controls. The
retained validator's native GREEN, the combined `make.bash`, and the full
upstream regression remain integration gates. Direct checker results or
a test binary rebuilt with a different compiler do not replace the
installed `.lgo` path in this script.

## Remaining binding contracts

M6-S3b's declaration-origin capture, private producer defaults, generic
origins and blank-field materialization remain blocked. Package-level
blank defaults can reach that binding path through synthesized JSON
methods before any source constructor call. Nested default composition
and exported anonymous default-bearing types inside containers also
remain outside the retained validator. No fourth binding repair is
attempted or implied by this gate.
