// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A generated method with a return value large enough to be
// initialized by duffzero is not a leaf method, which violated
// assumptions made by cmd/internal/obj/ppc64.

package main

const N = 9 // values > 8 cause (Super).Method to use duffzero

type Base struct {
}

func (b *Base) Method() (x [N]uintptr) {
	return
}

type Super struct {
	Base
}

type T interface {
	Method() [N]uintptr
}

func f(q T) {
	q.Method()
}

func main() {
	var s Super
	f(&s)
}
