#!/bin/bash

# This is an end-to-end test of Go SIMD. It updates all generated
# files in this repo and then runs several tests.

XEDDATA="${XEDDATA:-xeddata}"
if [[ ! -d "$XEDDATA" ]]; then
    echo >&2 "Must either set \$XEDDATA or symlink xeddata/ to the XED obj/dgen directory."
    exit 1
fi

# Ensure that goroot is the appropriate ancestor of this directory
which go >/dev/null || exit 1
goroot="$(go env GOROOT)"
ancestor="../../../../.."
if [[ ! $ancestor -ef "$goroot" ]]; then
    # We might be able to make this work but it's SO CONFUSING.
    echo >&2 "go command in path has GOROOT $goroot instead of" `(cd $ancestor; pwd)`
    exit 1
fi

set -ex

# Regenerate SIMD files
go run . -o godefs -goroot "$goroot" -xedPath "$XEDDATA" go.yaml types.yaml categories.yaml
# Regenerate SSA files from SIMD rules
go run -C "$goroot"/src/cmd/compile/internal/ssa/_gen .

# Rebuild compiler
cd "$goroot"/src
go install cmd/compile

# Tests
# Set the GOEXPERIMENT explicitly.
GOEXPERIMENT=simd GOARCH=amd64 go run -C simd/archsimd/testdata .
GOEXPERIMENT=simd GOARCH=amd64 go test -v simd/archsimd
GOEXPERIMENT=simd GOARCH=amd64 go test go/doc go/build
GOEXPERIMENT=simd GOARCH=amd64 go test cmd/api -v -check -run ^TestCheck$
GOEXPERIMENT=simd GOARCH=amd64 go test cmd/compile/internal/ssagen -simd=0

# Check tests without the GOEXPERIMENT
GOEXPERIMENT= go test go/doc go/build
GOEXPERIMENT= go test cmd/api -v -check -run ^TestCheck$
GOEXPERIMENT= go test cmd/compile/internal/ssagen -simd=0

# TODO: Add some tests of SIMD itself
