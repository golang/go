// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"b81614/lib"
	_ "simd"
)

func main() {
	// like b81614a, lib.F is a function returning simd type, but in this case "simd" is also underscore-imported.
	_ = lib.F
}
