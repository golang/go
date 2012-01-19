// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"
import "reflect"

const (
	R = 5
	I = 6i

	C1 = R + I // ADD(5,6)
)

func main() {
	c0 := C1
	c0 = (c0 + c0 + c0) / (c0 + c0 + 3i)
	r, i := real(c0), imag(c0)
	d := r - 1.292308
	if d < 0 {
		d = - d
	}
	if d > 1e-6 {
		println(r, "!= 1.292308")
		panic(0)
	}
	d = i + 0.1384615
	if d < 0 {
		d = - d
	}
	if d > 1e-6 {
		println(i, "!= -0.1384615")
		panic(0)
	}

	c := *(*complex128)(unsafe.Pointer(&c0))
	if c != c0 {
		println(c, "!=", c)
		panic(0)
	}

	var a interface{}
	switch c := reflect.ValueOf(a); c.Kind() {
	case reflect.Complex64, reflect.Complex128:
		v := c.Complex()
		_, _ = complex128(v), true
	}
}
