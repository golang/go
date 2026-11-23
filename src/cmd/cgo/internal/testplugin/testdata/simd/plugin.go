// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "simd"

// Lanes uses package simd, but has no SIMD types in its signature.
func Lanes() int {
	return simd.Int32s{}.Len()
}

// F and V have SIMD types, so the compiler only generates versions of them
// that are specialized to each vector size. They must not keep the plugin
// from building or loading.
func F(x simd.Int32s) int {
	return x.Len()
}

var V simd.Int32s

func main() {}
