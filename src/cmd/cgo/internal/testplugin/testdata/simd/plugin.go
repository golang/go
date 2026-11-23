// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "simd"

// Lanes uses package simd, but has no SIMD types in its signature.
func Lanes() int {
	return simd.Int32s{}.Len()
}

func main() {}
