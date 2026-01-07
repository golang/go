# Go-118-fuzz-build based fuzzer

## Prerequisites
To build and use this fuzzer, install the `Go-118-fuzz-build` tooling:
```bash
git clone https://github.com/AdamKorcz/go-118-fuzz-build
cd go-118-fuzz-build
go build
```
This generates a `go-118-fuzz-build` binary. Add it to your `PATH` for the following steps to work correctly.

## Fuzzing
To fuzz a target, navigate to its directory (e.g., `cd caddy`) and run the following commands:

**Build the fuzzer**

```bash
./build.sh
```

**Start Fuzzing**

```bash
./fuzz.sh
```
By default, `fuzz.sh` applies the same `libFuzzer` settings as `oss-fuzz`. To customize them, modify the script.

**Generate Coverage Report**

```bash
./get_cov.sh
```
The script produces a `cover.html` file in the target directory.

