// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package main

// For testing purposes, this should NOT compile, because
// it uses the unsafe.Sizeof of a "simd" type in a constant
// context (as an array size).

import (
	"simd"
	"unsafe"
)

var v [1]simd.Int8s
var u [unsafe.Sizeof(v)]byte

func main() {
	if len(u) != 16 {
		println("FAIL")
	}
}
