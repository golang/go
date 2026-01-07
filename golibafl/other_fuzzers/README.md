# Comparative Fuzzers

## Introduction
This subfolder contains several Go fuzzers that we used to benchmark and evaluate our tool's performance and capabilities.
The fuzzers are:
- [go_fuzz](./go_fuzz) which is based on the [go-fuzz](https://github.com/dvyukov/go-fuzz) tool.
- [go_fuzz_build](./go_fuzz_build/) which is based on the [Go-118-fuzz-build](https://github.com/AdamKorcz/go-118-fuzz-build) tool
- [native_fuzzing](./native_fuzzing/) which is based on the [native](https://go.dev/doc/security/fuzz/) fuzzing infrastructure of Go.

### Targets
The targets for all fuzzers are the same as described [here](../harnesses/README.md).