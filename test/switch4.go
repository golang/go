// errorcheck

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that erroneous switch statements are detected by the compiler.
// Does not compile.

package main

type I interface {
	M()
}

func bad() {

	i5 := 5
	switch i5 {
	case 5:
		fallthrough // ERROR "cannot fallthrough final case in switch"
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
