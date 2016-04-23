// run

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2582
package main

type T struct{}

//go:noinline
func (T) cplx() complex128 {
	return complex(1, 0)
}

func (T) cplx2() complex128 {
	return complex(0, 1)
}

type I interface {
	cplx() complex128
}

func main() {

	var t T

	if v := real(t.cplx()); v != 1 {
		panic("not-inlined complex call failed")
	}
	_ = imag(t.cplx())

	_ = real(t.cplx2())
	if v := imag(t.cplx2()); v != 1 {
		panic("potentially inlined complex call failed")
	}

	var i I
	i = t
	if v := real(i.cplx()); v != 1 {
		panic("potentially inlined complex call failed")
	}
	_ = imag(i.cplx())
}
