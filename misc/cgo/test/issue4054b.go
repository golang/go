// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
typedef enum {
	A = 0,
	B,
	C,
	D,
	E,
	F,
	G,
	H,
	I,
	J,
} issue4054b;
*/
import "C"

var issue4054b = []int{C.A, C.B, C.C, C.D, C.E, C.F, C.G, C.H, C.I, C.J}
