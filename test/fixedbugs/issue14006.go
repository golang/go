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
		2:	// ERROR "unexpected :|expected .*;.* or .*}.* or newline|value computed is not used"
	case 2:
	}

	switch x {
	case 1:
		2: ;	// ERROR "unexpected :|expected .*;.* or .*}.* or newline|value computed is not used"
	case 2:
	}

	var y string
	switch y {
	case "foo":
		"bar":	// ERROR "unexpected :|expected .*;.* or .*}.* or newline|value computed is not used"
	case "bar":
	}

	switch y {
	case "foo":
		"bar": ;	// ERROR "unexpected :|expected .*;.* or .*}.* or newline|value computed is not used"
	case "bar":
	}

	var z bool
	switch {
	case z:
		labelname:	// ERROR "missing statement after label"
	case false:
	}
}

func g() {
	var z bool
	switch {
	case z:
		labelname:	// ERROR "label labelname defined and not used|previous definition|defined and not used"
	}

	switch {
	case z:
		labelname: ;	// ERROR "label labelname already defined at LINE-5|label .*labelname.* already defined"
	case false:
	}
}
