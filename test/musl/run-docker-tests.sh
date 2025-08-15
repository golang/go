#!/bin/bash
# Run comprehensive Docker tests for musl compatibility

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Go musl compatibility test suite ==="
echo "Testing from: $GO_ROOT"
echo

# Function to run a single Docker test
run_docker_test() {
    local dockerfile="$1"
    local platform="$2"
    local name="$3"
    
    echo "=== Testing $name ==="
    echo "Platform: $platform"
    echo "Building Docker image..."
    
    # Build the image
    if docker build \
        --platform="$platform" \
        -f "$SCRIPT_DIR/$dockerfile" \
        -t "go-musl-test:$name" \
        "$GO_ROOT" > /tmp/docker-build-$name.log 2>&1; then
        echo "Build successful"
    else
        echo "Build FAILED. See /tmp/docker-build-$name.log"
        return 1
    fi
    
    # Run the test
    echo "Running tests..."
    if docker run --rm --platform="$platform" "go-musl-test:$name"; then
        echo "✓ $name: PASSED"
    else
        echo "✗ $name: FAILED"
        return 1
    fi
    echo
}

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Please install Docker Desktop."
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "ERROR: Docker daemon not running. Please start Docker Desktop."
    exit 1
fi

# Run tests
FAILED=0

# Alpine Linux (musl) tests
if ! run_docker_test "Dockerfile.alpine-amd64" "linux/amd64" "alpine-amd64"; then
    FAILED=$((FAILED + 1))
fi

if ! run_docker_test "Dockerfile.alpine-arm64" "linux/arm64" "alpine-arm64"; then
    FAILED=$((FAILED + 1))
fi

# Ubuntu (glibc) regression tests
if ! run_docker_test "Dockerfile.ubuntu-amd64" "linux/amd64" "ubuntu-amd64"; then
    FAILED=$((FAILED + 1))
fi

# Summary
echo "=== Test Summary ==="
if [ $FAILED -eq 0 ]; then
    echo "All tests PASSED!"
else
    echo "$FAILED test(s) FAILED"
    exit 1
fi
