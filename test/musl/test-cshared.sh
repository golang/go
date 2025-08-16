#!/bin/bash
# Test script for Go c-shared libraries on musl and glibc

set -e

echo "=== System Information ==="
uname -a
echo

# Detect libc type
if ldd --version 2>&1 | grep -q musl; then
    echo "Detected musl libc"
    LIBC_TYPE="musl"
elif [ -f /lib/ld-musl-*.so.1 ]; then
    echo "Detected musl libc (via ld-musl)"
    LIBC_TYPE="musl"
elif [ -f /etc/alpine-release ]; then
    echo "Detected Alpine Linux (musl)"
    LIBC_TYPE="musl"
else
    echo "Detected glibc"
    LIBC_TYPE="glibc"
fi
echo

# Create test directory
TEST_DIR=$(mktemp -d)
cd "$TEST_DIR"

echo "=== Creating test shared library ==="
cat > testlib.go << 'EOF'
package main

import "C"
import (
    "fmt"
    "os"
)

//export Init
func Init() {
    fmt.Println("INIT_SUCCESS: Go shared library initialized")
}

//export GetEnv
func GetEnv(key *C.char) *C.char {
    k := C.GoString(key)
    v := os.Getenv(k)
    return C.CString(v)
}

//export GetArgCount
func GetArgCount() C.int {
    return C.int(len(os.Args))
}

func main() {}
EOF

# Build shared library
echo "Building shared library..."
go build -buildmode=c-shared -o testlib.so testlib.go

echo
echo "=== Creating test loader ==="
cat > loader.c << 'EOF'
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    printf("=== Loading Go shared library ===\n");
    
    void *handle = dlopen("./testlib.so", RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "ERROR: dlopen failed: %s\n", dlerror());
        return 1;
    }
    printf("SUCCESS: dlopen succeeded\n");
    
    // Test initialization
    void (*init)(void) = dlsym(handle, "Init");
    if (init) {
        init();
    } else {
        printf("WARNING: Init function not found\n");
    }
    
    // Test argc access
    int (*get_argc)(void) = dlsym(handle, "GetArgCount");
    if (get_argc) {
        int go_argc = get_argc();
        printf("ARGC: C=%d, Go=%d\n", argc, go_argc);
    }
    
    dlclose(handle);
    printf("\n=== All tests completed ===\n");
    return 0;
}
EOF

# Build loader
echo "Building loader..."
gcc -o loader loader.c -ldl

echo
echo "=== Running tests ==="
./loader

# Check for core dumps (indicates SIGSEGV)
if [ -f core ]; then
    echo
    echo "ERROR: Core dump detected - likely SIGSEGV"
    exit 1
fi

echo
echo "=== Test Summary ==="
echo "Platform: $(uname -m)"
echo "Libc: $LIBC_TYPE"
echo "Status: SUCCESS"
