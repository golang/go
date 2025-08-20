#!/bin/bash
# Test TLS General Dynamic model for c-shared on musl

set -e

echo "=== Testing TLS GD model for c-shared ==="

# Create temp directory
TESTDIR=$(mktemp -d)
cd "$TESTDIR"
echo "Working in: $TESTDIR"

# Create test shared library with TLS
cat > testlib.go << 'EOF'
package main

import "C"
import (
	"fmt"
	"runtime"
)

//export TestTLS
func TestTLS() {
	// This accesses TLS via runtime.g
	fmt.Printf("TLS_SUCCESS: goroutine %d\n", runtime.NumGoroutine())
}

func main() {}
EOF

# Build shared library
echo "Building shared library with TLS..."
go build -buildmode=c-shared -o testlib.so testlib.go

# Create loader that uses dlopen
cat > loader.c << 'EOF'
#include <dlfcn.h>
#include <stdio.h>

int main() {
    void *h = dlopen("./testlib.so", RTLD_NOW);
    if (!h) {
        printf("dlopen failed: %s\n", dlerror());
        return 1;
    }
    
    void (*test)(void) = dlsym(h, "TestTLS");
    if (test) {
        test();
    } else {
        printf("TestTLS not found: %s\n", dlerror());
        return 1;
    }
    
    dlclose(h);
    return 0;
}
EOF

# Build loader
echo "Building loader..."
cc -o loader loader.c -ldl

# Run test
echo "Running dlopen test..."
./loader

echo "Test completed successfully!"

# Cleanup
cd /
rm -rf "$TESTDIR"
