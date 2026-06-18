// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package testdata_test

// For testing purposes, this should NOT compile, because
// it uses the unsafe.Sizeof of a "simd" type in a constant
// context (as an array size).

import (
	"simd"
	"testing"
	"unsafe"
)

var v_from_simd [1]simd.Int8s

func TestItDoesNotCompile(t *testing.T) {
	var u [unsafe.Sizeof(v_from_simd)]byte
	if len(u) != 16 {
		println("FAIL")
	}
}
