#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$GO_ROOT"

run_test() {
    local platform="$1"
    local dockerfile="$2"
    local tag="go-cshared-${platform}-test"

    echo "=== Building Go and testing c-shared dlopen on ${platform} ==="
    echo "    (this builds Go from source — it will take a few minutes)"

    if ! docker build -f "$dockerfile" -t "$tag" . ; then
        echo "FAIL: Docker build failed for ${platform} (library build or compile failed)"
        return 1
    fi

    echo "--- Running dlopen test on ${platform} ---"
    if docker run --rm "$tag"; then
        echo "=== PASS: ${platform} ==="
        return 0
    else
        echo "=== FAIL: ${platform} ==="
        return 1
    fi
}

failures=0

if ! run_test "alpine-musl" "$SCRIPT_DIR/Dockerfile.alpine"; then
    failures=$((failures + 1))
fi

echo ""

if ! run_test "ubuntu-glibc" "$SCRIPT_DIR/Dockerfile.ubuntu"; then
    failures=$((failures + 1))
fi

echo ""
if [ "$failures" -eq 0 ]; then
    echo "=== ALL PLATFORMS PASSED ==="
else
    echo "=== ${failures} PLATFORM(S) FAILED ==="
    exit 1
fi
