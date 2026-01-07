## Go-fuzzbased fuzzer

## Prerequisites
To build and use this fuzzer, install the `Go-fuzz` tooling:
```bash
go install github.com/dvyukov/go-fuzz/go-fuzz@latest github.com/dvyukov/go-fuzz/go-fuzz-build@latest
```

## Fuzzing
To fuzz a target, navigate to its directory (e.g., `cd path_constraint`) and run the following commands:

**Build the fuzzer**

```bash
./build.sh
```

**Start Fuzzing**

```bash
./fuzz.sh
```
By default, `fuzz.sh` fuzzes with a single worker. To increase this, modify the `-procs` parameter in the script.

**Generate Coverage Report**

```bash
./get_cov.sh
```
The script produces a `cover.html` file in the target directory.


## Note
Since `go-fuzz` is deprecated and lacks support for Go 1.18+ features, only the path_constraint target is included in this directory. All other targets that we used for testing didn't compile.
