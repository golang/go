// asmcheck

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import "math/bits"

func f(n int) int {
	r := 0
	// arm64:-"MOVD R[0-9]+, R[0-9]+"
	// amd64:-"LEAQ" "INCQ"
	for i := range n {
		r += i
	}
	return r
}

// addVW has loop body with both a flag-producing op and an induction-variable increment.
// It ensures that the induction increment doesn't get pointlessly spilled.
func addVWLike(z, x []uint, y uint) uint {
	c := y
	// arm64:-"MOVD R[0-9]+, R[0-9]+"
	// amd64:-"LEAQ" "INCQ"
	for i := range z {
		zi, cc := bits.Add(x[i], c, 0)
		z[i] = zi
		c = cc
	}
	return c
}
