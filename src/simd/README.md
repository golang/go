# The Go SIMD packages

This directory holds Go's SIMD support: the portable `simd` package, the
platform-specific `simd/archsimd` package, and the internal packages and
generators behind them.

- [archsimd/_gen/README.md](archsimd/_gen/README.md) describes what each
  package is, the rules the API follows, and the generators that produce most
  of the code.
- [AGENTS.md](AGENTS.md) is advice for coding agents on working with the large
  generated files.

The rest of this file is about testing.

## Testing every configuration

The `simd` package picks an implementation when the program starts, based on
the CPU: emulation if the CPU has no SIMD that the package can use, and
otherwise the largest vector size the CPU supports. By default, `go test simd`
only tests the one it picks on your machine, but this behavior can be
overridden with `GODEBUG` settings. To test every configuration your machine
can run, use:

```sh
# From $GOROOT/src. The package has to come before the flag.
GOEXPERIMENT=simd go test simd -run-godebugs
```

`-run-godebugs` runs the tests once in each configuration: first under the
`GODEBUG` setting it was started with, and then under each setting that selects
another implementation that this CPU can run, such as `GODEBUG=simd=128` for a
smaller vector size or `GODEBUG=simd=0` for emulation. It prints
`# GODEBUG=...` before each run, and it passes the other test flags, like
`-run` and `-v`, on to each run. To list those other settings without running
anything:

```sh
GOEXPERIMENT=simd go test -v simd -list-godebugs
```

To reproduce a failure under one of them, set it yourself, as in
`GOEXPERIMENT=simd GODEBUG=simd=128 go test simd`.

`go tool dist test`, and so `all.bash` and the builders, also runs the tests
under each of these settings. Each is a dist test of its own, named like
`simd:GODEBUG=simd=128`, so `go tool dist test -run '^simd:GODEBUG='` runs just
these.

These settings can only select what the CPU at hand can run. A machine without
AVX-512 can't test 512-bit vectors this way, for example. For the CPUs and
platforms you don't have, see the next section.

## Testing on other platforms

Every target can be tested from a linux/amd64 workstation except AVX512, which
needs real hardware. The commands below run from `$GOROOT/src` and test the
packages that exercise SIMD operations.

`IN_QEMU=1` tells the tests that they are running under QEMU (some Go builders
set it, too), so they skip the checks that QEMU is known to fail; see the known
limitations below.

`-run-godebugs`, and the tests that run the test binary again themselves, start
it directly rather than with the `-exec` program. An amd64 test binary on an
amd64 host then runs on the host's CPU, not the emulated one. To test a
`GODEBUG` setting under an emulator, set it on the command line instead.

```sh
# amd64 AVX2 without AVX512 (the same features as the gotip-linux-amd64 builder)
IN_QEMU=1 GOEXPERIMENT=simd go test -exec 'qemu-x86_64 -cpu Haswell' simd simd/archsimd/internal/simd_test

# amd64 AVX without AVX2, the minimum for hardware SIMD in the simd package.
# Only simd: many archsimd tests of 256-bit types don't check for AVX2
# (go.dev/issue/82893). +fma because simd's MulAdd currently uses FMA, which
# Sandy Bridge lacks (go.dev/issue/82892).
IN_QEMU=1 GOEXPERIMENT=simd go test -exec 'qemu-x86_64 -cpu SandyBridge,+fma' simd

# amd64 without carryless multiply. Haswell above has it only for 128-bit
# vectors, so 256-bit CarrylessMul uses two 128-bit instructions. Removing
# PCLMULQDQ leaves none, so CarrylessMul falls back to software.
IN_QEMU=1 GOEXPERIMENT=simd go test -exec 'qemu-x86_64 -cpu Haswell,-pclmulqdq' simd simd/archsimd/internal/simd_test

# arm64 NEON only (the same features as the gotip-linux-arm64 builder)
IN_QEMU=1 GOEXPERIMENT=simd GOARCH=arm64 go test -exec 'qemu-aarch64 -cpu neoverse-n1' simd simd/archsimd/internal/simd_test

# arm64 SVE with 128-bit vectors
IN_QEMU=1 GOEXPERIMENT=simd GOARCH=arm64 go test -exec 'qemu-aarch64 -cpu neoverse-n2' simd simd/archsimd/internal/simd_test

# arm64 SVE with 256-bit vectors
IN_QEMU=1 GOEXPERIMENT=simd GOARCH=arm64 go test -exec 'qemu-aarch64 -cpu neoverse-v1' simd simd/archsimd/internal/simd_test

# wasm under node. go test finds go_js_wasm_exec on $PATH.
PATH=$(go env GOROOT)/lib/wasm:$PATH GOEXPERIMENT=simd GOOS=js GOARCH=wasm go test simd simd/archsimd/internal/simd_test

# wasm under wazero (the default WASI runtime is wasmtime)
PATH=$(go env GOROOT)/lib/wasm:$PATH GOWASIRUNTIME=wazero GOEXPERIMENT=simd GOOS=wasip1 GOARCH=wasm go test simd simd/archsimd/internal/simd_test
```

To run one build under several CPU models, build the test binary once with
`go test -c` and run it with, for example, `qemu-aarch64 -cpu neoverse-v1
./simd_test.test`.

Known limitations:

- **AVX512 cannot be emulated with QEMU.** QEMU's TCG emulator does not
  implement AVX512 (as of QEMU 11.1, `-cpu SapphireRapids` warns "TCG doesn't
  support requested feature: avx512f" and the AVX512 tests skip). Use the
  gotip-linux-amd64 gomote, which has AVX512 (as of October 2026, an AMD EPYC
  9B45 with every AVX512 extension the API uses), with `gomote put` and `gomote
  run` on cross-compiled test binaries. Intel SDE (`sde64 -spr -- ./x.test`)
  should also work, but is untested.

- **`qemu-x86_64` does not suppress faults in masked-off lanes of masked
  loads** (VMASKMOVPS/PD, which back the `Load*Part` functions for 32- and
  64-bit elements), so their page-boundary tests fault on the harness's guard
  page. With `IN_QEMU=1`, those tests skip when they fault.

- **`qemu-aarch64` 11.1 miscomputes SVE SQSUB on 32- and 64-bit elements**, so
  `Int32s.SubSaturated` and `Int64s.SubSaturated` fail under `neoverse-n2` and
  `neoverse-v1`. They pass on real SVE hardware (an Axion gomote). With
  `IN_QEMU=1`, `TestSubSaturatedSVE` skips them.

- A few quirks are not represented in our testing set. Rosetta on Apple Silicon
  supports 128-bit CLMUL but not 256. Some Raspberry Pis have NEON, but not
  PMULL (thus no 64-bit CLMUL instruction). `CarrylessMul` falls back to
  narrower instructions or software on such CPUs. The tests cover the fallbacks
  on any CPU by disabling the CPU features, either directly or, under
  `-run-godebugs`, with `GODEBUG=cpu.pclmulqdq=off`, `cpu.vpclmulqdq=off`, and
  `cpu.pmull=off`, but that doesn't prove the fallbacks avoid the missing
  instructions. The QEMU CPU models above do that for amd64; we have no arm64
  CPU model without PMULL.

### Builder and emulator capabilities

Generated by `./cmd/hostinfo` in `archsimd/_gen`:

| amd64           | gotip-linux-amd64 | gotip-linux-amd64_avx512 |
| --------------- | ----------------- | ------------------------ |
| AVX             | ✔                 | ✔                        |
| AVX2            | ✔                 | ✔                        |
| AVX2AES         | ✘                 | ✔                        |
| AVX2CLMUL       | ✘                 | ✔                        |
| AVX512          | ✘                 | ✔                        |
| AVX512AES       | ✘                 | ✔                        |
| AVX512BITALG    | ✘                 | ✔                        |
| AVX512CLMUL     | ✘                 | ✔                        |
| AVX512GFNI      | ✘                 | ✔                        |
| AVX512VBMI      | ✘                 | ✔                        |
| AVX512VBMI2     | ✘                 | ✔                        |
| AVX512VNNI      | ✘                 | ✔                        |
| AVX512VPOPCNTDQ | ✘                 | ✔                        |
| AVXAES          | ✔                 | ✔                        |
| AVXCLMUL        | ✔                 | ✔                        |
| AVXVNNI         | ✘                 | ✔                        |
| FMA             | ✔                 | ✔                        |
| SHA             | ✘                 | ✔                        |

| arm64           | gotip-linux-arm64 | qemu-aarch64 -cpu neoverse-n2 | qemu-aarch64 -cpu neoverse-v1 |
| --------------- | ----------------- | ----------------------------- | ----------------------------- |
| PMULL           | ✔                 | ✔                             | ✔                             |
| SVE             | ✘                 | ✔                             | ✔                             |
| SVE2            | ✘                 | ✔                             | ✔                             |
| SVE vector size | n/a               | 128                           | 256                           |

Note: `qemu-aarch64 -cpu neoverse-n1` matches the capabilities of gotip-linux-arm64.

The gotip-linux-arm64_c4as16-perf_vs_release gomote is a Google Axion
(Neoverse V2) machine with SVE and SVE2 at a 128-bit vector size. It is not a
general-purpose builder, but it is useful for checking SVE results on real
hardware: `gomote put` cross-compiled test binaries (`go test -c`) and `gomote
run` them.

| wasm    | js-node | wasip1-wazero |
| ------- | ------- | ------------- |
| SIMD128 | ✔       | ✔             |
