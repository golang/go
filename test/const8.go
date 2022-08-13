// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that identifiers in implicit (omitted) RHS
// expressions of constant declarations are resolved
// in the correct context; see issues #49157, #53585.

package main

const X = 2

func main() {
	const (
		A    = iota // 0
		iota = iota // 1
		B           // 1 (iota is declared locally on prev. line)
		C           // 1
	)
	if A != 0 || B != 1 || C != 1 {
		println("got", A, B, C, "want 0 1 1")
		panic("FAILED")
	}

	const (
		X = X + X
		Y
		Z = iota
	)
	if X != 4 || Y != 8 || Z != 1 {
		println("got", X, Y, Z, "want 4 8 1")
		panic("FAILED")
	}
}
