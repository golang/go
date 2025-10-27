// asmcheck

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

//go:registerparams
func f1(a, b int) {
	// amd64:"MOVQ BX, CX", "MOVQ AX, BX", "MOVL [$]1, AX", -"MOVQ .*DX"
	g(1, a, b)
}

//go:registerparams
func f2(a, b int) {
	// amd64:"MOVQ BX, AX", "MOVQ [AB]X, CX", -"MOVQ .*, BX"
	g(b, b, b)
}

//go:noinline
//go:registerparams
func g(int, int, int) {}
