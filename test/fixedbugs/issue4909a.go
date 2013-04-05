// errorcheck

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4909: compiler incorrectly accepts unsafe.Offsetof(t.x)
// where x is a field of an embedded pointer field.

package p

import (
	"unsafe"
)

type T struct {
	A int
	*B
}

func (t T) Method() {}

type B struct {
	X, Y int
}

var t T
var p *T

const N1 = unsafe.Offsetof(t.X)      // ERROR "indirection"
const N2 = unsafe.Offsetof(p.X)      // ERROR "indirection"
const N3 = unsafe.Offsetof(t.B.X)    // valid
const N4 = unsafe.Offsetof(p.B.X)    // valid
const N5 = unsafe.Offsetof(t.Method) // ERROR "method value"
const N6 = unsafe.Offsetof(p.Method) // ERROR "method value"
