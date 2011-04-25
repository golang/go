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
	println(c0)

	c := *(*complex128)(unsafe.Pointer(&c0))
	println(c)

	var a interface{}
	switch c := reflect.ValueOf(a); c.Kind() {
	case reflect.Complex64, reflect.Complex128:
		v := c.Complex()
		_, _ = complex128(v), true
	}
}
