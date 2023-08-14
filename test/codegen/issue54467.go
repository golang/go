// asmcheck

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func f1(x *[4]int, y *[4]int) {
	// amd64:".*memmove"
	*x = *y
}
func f2(x *[4]int, y [4]int) {
	// amd64:-".*memmove"
	*x = y
}
func f3(x *[4]int, y *[4]int) {
	// amd64:-".*memmove"
	t := *y
	// amd64:-".*memmove"
	*x = t
}
func f4(x *[4]int, y [4]int) {
	// amd64:-".*memmove"
	t := y
	// amd64:-".*memmove"
	*x = t
}

type T struct {
	a [4]int
}

func f5(x, y *T) {
	// amd64:-".*memmove"
	x.a = y.a
}
func f6(x *T, y T) {
	// amd64:-".*memmove"
	x.a = y.a
}
func f7(x *T, y *[4]int) {
	// amd64:-".*memmove"
	x.a = *y
}
func f8(x *[4]int, y *T) {
	// amd64:-".*memmove"
	*x = y.a
}

func f9(x [][4]int, y [][4]int, i, j int) {
	// amd64:-".*memmove"
	x[i] = y[j]
}

func f10() []byte {
	// amd64:-".*memmove"
	return []byte("aReasonablyBigTestString")
}
