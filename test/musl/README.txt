Go non-glibc Compatibility Test Suite

This directory contains tests for verifying Go's compatibility with
non-glibc Unix systems, particularly for c-shared and c-archive build
modes.

Background

Go makes assumptions about the C runtime that are specific to glibc but
not guaranteed by standards:
1. DT_INIT_ARRAY functions receive (argc, argv, envp) - only glibc does
   this
2. Initial-exec TLS model for shared libraries - incompatible with
   dlopen on non-glibc systems

Test Structure

Unit Tests
- src/runtime/cgo_musl_test.go - Go test that builds and loads a shared
  library
- src/runtime/testdata/testprogcgo/musl_sharedlib.go - Test shared
  library
- src/runtime/testdata/testprogcgo/musl_loader.c - C program that uses
  dlopen

Docker Tests
- Dockerfile.alpine-amd64 - Alpine Linux (musl) on x86-64
- Dockerfile.alpine-arm64 - Alpine Linux (musl) on ARM64
- Dockerfile.ubuntu-amd64 - Ubuntu (glibc) regression test
- test-cshared.sh - Shell script that runs inside containers
- run-docker-tests.sh - Orchestrates all Docker tests

Running Tests

Quick Test (if on Linux)
cd src
go test -run TestMuslSharedLibrary runtime

Comprehensive Docker Tests
cd test/musl
./run-docker-tests.sh

Individual Docker Test
docker build --platform linux/amd64 -f Dockerfile.alpine-amd64 -t test ../..
docker run --rm test

Expected Results

On Unpatched Go

Alpine/musl:
- dlopen fails with SIGSEGV in runtime initialization
- Error occurs in runtime.sysargs() dereferencing null argv

Ubuntu/glibc:
- Works correctly (glibc passes argc/argv to DT_INIT_ARRAY)

With Our Patches

Alpine/musl:
- dlopen succeeds
- Shared library initializes without SIGSEGV
- argc/argv may be 0/null but doesn't crash

Ubuntu/glibc:
- Continues to work correctly
- No regressions

Test Coverage

The tests verify:
1. Initialization - No SIGSEGV when argv is null
2. dlopen Success - Library can be loaded dynamically  
3. Function Calls - Exported functions can be called
4. argc/argv Access - Graceful handling when not available

Adding New Tests

To test on additional platforms:
1. Create Dockerfile.platform-arch
2. Add to run-docker-tests.sh
3. Use same test-cshared.sh script

Debugging Failures

If tests fail:
1. Check /tmp/docker-build-*.log for build errors
2. Run container interactively: docker run -it --rm image /bin/sh
3. Check for core dumps indicating SIGSEGV
4. Use strace to see where the crash occurs

Standards References

- ELF specification: DT_INIT_ARRAY behavior
- ELF TLS specification: Dynamic loading requirements