// errorcheck -0 -m

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 23521: improve early DCE for if without explicit else.

package p

//go:noinline
func nonleaf() {}

const truth = true

func f() int { // ERROR "can inline f"
	if truth {
		return 0
	}
	// If everything below is removed, as it should,
	// function f should be inlineable.
	nonleaf()
	for {
		panic("")
	}
}

func g() int { // ERROR "can inline g"
	return f() // ERROR "inlining call to f"
}

func f2() int { // ERROR "can inline f2"
	if !truth {
		nonleaf()
	} else {
		return 0
	}
	panic("")
}

func g2() int { // ERROR "can inline g2"
	return f2() // ERROR "inlining call to f2"
}
