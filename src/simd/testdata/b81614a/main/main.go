// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"b81614/lib"
)

var sink any

func main() {
	// lib.F is a function returning simd type, w/o importing "simd"
	_ = lib.F
	var v = lib.F()
	sink = v.Len() // v has a simd type, and simd methods.
}
