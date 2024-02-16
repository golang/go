// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file and importx_test.go make it possible to write tests in the runtime
// package, which is generally more convenient for testing runtime internals.
// For tests that mostly touch public APIs, it's generally easier to write them
// in the runtime_test package and export any runtime internals via
// export_test.go.
//
// There are a few limitations on runtime package tests that this bridges:
//
// 1. Tests use the signature "XTest<name>(t TestingT)". Since runtime can't import
// testing, test functions can't use testing.T, so instead we have the T
// interface, which *testing.T satisfies. And we start names with "XTest"
// because otherwise go test will complain about Test functions with the wrong
// signature. To actually expose these as test functions, this file contains
// trivial wrappers.
//
// 2. Runtime package tests can't directly import other std packages, so we
// inject any necessary functions from std.

// TODO: Generate this

package runtime_test

import (
	"fmt"
	"internal/testenv"
	"runtime"
	"testing"
)

func init() {
	runtime.FmtSprintf = fmt.Sprintf
	runtime.TestenvOptimizationOff = testenv.OptimizationOff
}

func TestInlineUnwinder(t *testing.T) {
	runtime.XTestInlineUnwinder(t)
}

func TestSPWrite(t *testing.T) {
	runtime.XTestSPWrite(t)
}
