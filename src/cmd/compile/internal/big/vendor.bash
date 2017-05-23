#!/usr/bin/env bash

# Copyright 2015 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Run this script to obtain an up-to-date vendored version of math/big.

BIGDIR=../../../../math/big

# Start from scratch.
rm *.go

# We don't want any assembly files.
cp $BIGDIR/*.go .

# Use pure Go arith ops w/o build tag.
sed 's|^// \+build math_big_pure_go$||' arith_decl_pure.go > arith_decl.go
rm arith_decl_pure.go

# Import vendored math/big in external tests (e.g., floatexample_test.go).
for f in *_test.go; do
	sed 's|"math/big"|"cmd/compile/internal/big"|' $f > foo.go
	mv foo.go $f
done

# gofmt to clean up after sed
gofmt -w .

# Test that it works
go test -short
