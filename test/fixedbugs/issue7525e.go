// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 7525: self-referential array types.

package main

import "unsafe"

var x struct { // GC_ERROR "initialization cycle: x refers to itself"
	c [unsafe.Alignof(x.c)]int // GCCGO_ERROR "array bound|typechecking loop|invalid array"
}
