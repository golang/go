// errchk $G $D/$F.go

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct {
	// accepted by both compilers, legal according to spec
	x int
	y (int)
	int
	*float
	// not accepted by both compilers, not legal according to spec
	(complex)  // ERROR "non-declaration|expected"
	(*string)  // ERROR "non-declaration|expected"
	*(bool)    // ERROR "non-declaration|expected"
}

// accepted by both compilers, legal according to spec
func (p T) m() {}

// accepted by 6g, not accepted by gccgo, not legal according to spec
func (p (T)) f() {}   // ERROR "expected"
func (p *(T)) g() {}  // ERROR "expected"
func (p (*T)) h() {}  // ERROR "expected"
