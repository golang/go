// errorcheck -0 -d=ssa/decompose_addr/debug=1

//go:build !386 && !arm && !mips && !mipsle

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// decomposeAddr cases whose word leaves are 8 bytes wide: on 32-bit
// platforms such a leaf does not fit a register and the variable is
// left alone, so these expectations only hold on 64-bit platforms.

package main

func bareComplexField(p *complex128) int {
	// A bare multi-word value (no enclosing struct) also decomposes.
	x := *p // ERROR "promoted x"
	q := &x
	return int(real(*q)) + int(imag(*q))
}

func main() {}
