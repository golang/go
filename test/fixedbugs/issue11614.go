// errorcheck

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that incorrect expressions involving wrong anonymous interface
// do not generate panics in Type Stringer.
// Does not compile.

package main

type I interface {
	int // ERROR "interface contains embedded non-interface"
}

func n() {
	(I)
}

func m() {
	(interface{int}) // ERROR "interface contains embedded non-interface" "type interface { int } is not an expression"
}

func main() {
}
