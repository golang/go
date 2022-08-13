// asmcheck

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

//go:registerparams
func f1(a, b int) {
	// amd64:"MOVQ\tBX, CX", "MOVQ\tAX, BX", "MOVL\t\\$1, AX", -"MOVQ\t.*DX"
	g(1, a, b)
}

//go:registerparams
func f2(a, b int) {
	// amd64:"MOVQ\tBX, AX", "MOVQ\t[AB]X, CX", -"MOVQ\t.*, BX"
	g(b, b, b)
}

//go:noinline
//go:registerparams
func g(int, int, int) {}
