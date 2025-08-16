#!/bin/bash
# Quick local test for the argc/argv fix

set -e

echo "=== Testing Go c-shared argc/argv fix locally ==="

# Create temp directory
TESTDIR=$(mktemp -d)
cd "$TESTDIR"
echo "Working in: $TESTDIR"

# Create simple test library
cat > testlib.go << 'EOF'
package main

import "C"
import "fmt"

//export TestInit
func TestInit() {
	fmt.Println("SUCCESS: Library initialized without SIGSEGV")
}

func main() {}
EOF

# Build it
echo "Building shared library..."
go build -buildmode=c-shared -o testlib.so testlib.go

# Create loader
cat > loader.c << 'EOF'
#include <dlfcn.h>
#include <stdio.h>

int main() {
    void *h = dlopen("./testlib.so", RTLD_NOW);
    if (!h) {
        printf("dlopen failed: %s\n", dlerror());
        return 1;
    }
    
    void (*init)(void) = dlsym(h, "TestInit");
    if (init) init();
    
    dlclose(h);
    return 0;
}
EOF

# Build loader
echo "Building loader..."
cc -o loader loader.c -ldl

# Run test
echo "Running test..."
./loader

echo "Test completed successfully!"

# Cleanup
cd /
rm -rf "$TESTDIR"
