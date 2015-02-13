// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func fail() // unimplemented, to test dead code elimination

// Test dead code elimination in if statements
func init() {
	if false {
		fail()
	}
	if 0 == 1 {
		fail()
	}
}

// Test dead code elimination in ordinary switch statements
func init() {
	const x = 0
	switch x {
	case 1:
		fail()
	}

	switch 1 {
	case x:
		fail()
	}

	switch {
	case false:
		fail()
	}

	const a = "a"
	switch a {
	case "b":
		fail()
	}

	const snowman = '☃'
	switch snowman {
	case '☀':
		fail()
	}

	const zero = float64(0.0)
	const one = float64(1.0)
	switch one {
	case -1.0:
		fail()
	case zero:
		fail()
	}

	switch 1.0i {
	case 1:
		fail()
	case -1i:
		fail()
	}

	const no = false
	switch no {
	case true:
		fail()
	}
}

func main() {
}
