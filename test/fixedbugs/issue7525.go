// errorcheck

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 7525: self-referential array types.

package main

import "unsafe"

var x struct {
	a [unsafe.Sizeof(x.a)]int   // ERROR "array bound|typechecking loop|invalid expression"
	b [unsafe.Offsetof(x.b)]int // ERROR "array bound"
	c [unsafe.Alignof(x.c)]int  // ERROR "array bound|invalid expression"
}
