// errchk $G -e $D/$F.go

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type I interface {
	M()
}

func bad() {
	var i I
	var s string

	switch i {
	case s: // ERROR "mismatched types string and I|incompatible types"
	}

	switch s {
	case i: // ERROR "mismatched types I and string|incompatible types"
	}

	var m, m1 map[int]int
	switch m {
	case nil:
	case m1: // ERROR "can only compare map m to nil|map can only be compared to nil"
	default:
	}

	var a, a1 []int
	switch a {
	case nil:
	case a1: // ERROR "can only compare slice a to nil|slice can only be compared to nil"
	default:
	}

	var f, f1 func()
	switch f {
	case nil:
	case f1: // ERROR "can only compare func f to nil|func can only be compared to nil"
	default:
	}
}

func good() {
	var i interface{}
	var s string

	switch i {
	case s:
	}

	switch s {
	case i:
	}
}
