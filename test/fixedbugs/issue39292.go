// errorcheck -0 -m -l

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type t [20000]*int

func (t) f() {
}

func x() {
	x := t{}.f // ERROR "t{}.f escapes to heap"
	x()
}

func y() {
	var i int       // ERROR "moved to heap: i"
	y := (&t{&i}).f // ERROR "\(&t{...}\).f escapes to heap" "&t{...} escapes to heap"
	y()
}

func z() {
	var i int    // ERROR "moved to heap: i"
	z := t{&i}.f // ERROR "t{...}.f escapes to heap"
	z()
}

// Should match cmd/compile/internal/ir/cfg.go:MaxStackVarSize.
const maxStack = 128 * 1024

func w(i int) byte {
	var x [maxStack]byte
	var y [maxStack + 1]byte // ERROR "moved to heap: y"
	return x[i] + y[i]
}
