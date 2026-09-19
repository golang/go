#!/bin/bash

# This is an end-to-end test of Go SIMD. It updates all generated
# files in this repo and then runs several tests.

which go >/dev/null || exit 1

set -ex

# Regenerate SIMD files
go run . -w -o godefs -arch amd64 go_amd64.yaml types.yaml categories.yaml
# Regenerate SSA files from SIMD rules
go run -C ../../../../cmd/compile/internal/ssa/_gen .

# Rebuild compiler
go install cmd/compile

# Tests
# Set the GOEXPERIMENT explicitly.
GOEXPERIMENT=simd GOARCH=amd64 go run -C ../../../../simd/archsimd/testdata .
GOEXPERIMENT=simd GOARCH=amd64 go test -v ../../../../simd/archsimd
GOEXPERIMENT=simd GOARCH=amd64 go test go/doc go/build
GOEXPERIMENT=simd GOARCH=amd64 go test cmd/api -v -check -run ^TestCheck$
GOEXPERIMENT=simd GOARCH=amd64 go test cmd/compile/internal/ssagen -simd=0

# Check tests without the GOEXPERIMENT
GOEXPERIMENT= go test go/doc go/build
GOEXPERIMENT= go test cmd/api -v -check -run ^TestCheck$
GOEXPERIMENT= go test cmd/compile/internal/ssagen -simd=0

# TODO: Add some tests of SIMD itself
