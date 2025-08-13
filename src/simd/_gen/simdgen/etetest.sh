#!/bin/bash -x

cat <<\\EOF

This is an end-to-end test of Go SIMD. It checks out a fresh Go
repository from the go.simd branch, then generates the SIMD input
files and runs simdgen writing into the fresh repository.

After that it generates the modified ssa pattern matching files, then
builds the compiler.

\EOF

rm -rf go-test
git clone https://go.googlesource.com/go -b dev.simd go-test
go run . -xedPath xeddata  -o godefs -goroot ./go-test  go.yaml types.yaml categories.yaml
(cd go-test/src/cmd/compile/internal/ssa/_gen ; go run *.go )
(cd go-test/src ; GOEXPERIMENT=simd  ./make.bash )
(cd go-test/bin; b=`pwd` ; cd ../src/simd/testdata; GOARCH=amd64 $b/go run .)
(cd go-test/bin; b=`pwd` ; cd ../src ;
GOEXPERIMENT=simd GOARCH=amd64 $b/go test -v simd
GOEXPERIMENT=simd $b/go test go/doc
GOEXPERIMENT=simd $b/go test go/build
GOEXPERIMENT=simd $b/go test cmd/api -v -check
$b/go test go/doc
$b/go test go/build
$b/go test cmd/api -v -check

$b/go test cmd/compile/internal/ssagen -simd=0
GOEXPERIMENT=simd $b/go test cmd/compile/internal/ssagen -simd=0
)

# next, add some tests of SIMD itself
