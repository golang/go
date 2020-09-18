// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type I1 interface { I2 }	// ERROR "interface"
type I2 int

type I3 interface { int }	// ERROR "interface"

type S struct {
	x interface{ S }	// ERROR "interface"
}
type I4 interface { // GC_ERROR "invalid recursive type"
	I4	// GCCGO_ERROR "interface"
}

type I5 interface { // GC_ERROR "invalid recursive type"
	I6	// GCCGO_ERROR "interface"
}

type I6 interface {
	I5	// GCCGO_ERROR "interface"
}
