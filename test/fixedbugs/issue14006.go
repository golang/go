// errorcheck

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Literals that happen to resolve to named constants
// may be used as label names (see issue 13684). Make
// sure that other literals don't crash the compiler.

package main

const labelname = 1

func main() {
	goto labelname
labelname:
}

func f() {
	var x int
	switch x {
	case 1:
		2:	// ERROR "unexpected :"
	case 2:
	}

	switch x {
	case 1:
		2: ;	// ERROR "unexpected :"
	case 2:
	}

	var y string
	switch y {
	case "foo":
		"bar":	// ERROR "unexpected :"
	case "bar":
	}

	switch y {
	case "foo":
		"bar": ;	// ERROR "unexpected :"
	case "bar":
	}

	var z bool
	switch {
	case z:
		labelname:	// ERROR "missing statement after label"
	case false:
	}

	switch {
	case z:
		labelname:
	}

	switch {
	case z:
		labelname: ;
	case false:
	}
}