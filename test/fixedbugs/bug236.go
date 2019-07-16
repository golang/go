// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var gen = 'a'

func f(n int) string {
	s := string(gen) + string(n+'A'-1)
	gen++
	return s
}

func g(x, y string) string { return x + y }

var v1 = f(1) + f(2)
var v2 = g(f(3), f(4))
var v3 = f(5) + f(6) + f(7) + f(8) + f(9)

func main() {
	gen = 'a'

	if v1 != "aAbB" {
		panic("BUG: bug236a")
	}
	if v2 != "cCdD" {
		panic("BUG: bug236b")
	}
	if v3 != "eEfFgGhHiI" {
		panic("BUG: bug236c")
	}

	switch "aAbB" {
	case f(1) + f(2):
	default:
		panic("BUG: bug236d")
	}

	switch "cCdD" {
	case g(f(3), f(4)):
	default:
		panic("BUG: bug236e")
	}

	switch "eEfFgGhHiI" {
	case f(5) + f(6) + f(7) + f(8) + f(9):
	default:
		panic("BUG: bug236f")
	}
}
