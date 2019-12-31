// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the stringintconv checker.

package a

type A string

type B = string

type C int

type D = uintptr

func StringTest() {
	var (
		i int
		j rune
		k byte
		l C
		m D
		n = []int{0, 1, 2}
		o struct{ x int }
	)
	const p = 0
	_ = string(i) // want `^conversion from int to string yields a string of one rune$`
	_ = string(j)
	_ = string(k)
	_ = string(p)    // want `^conversion from untyped int to string yields a string of one rune$`
	_ = A(l)         // want `^conversion from C \(int\) to A \(string\) yields a string of one rune$`
	_ = B(m)         // want `^conversion from uintptr to B \(string\) yields a string of one rune$`
	_ = string(n[1]) // want `^conversion from int to string yields a string of one rune$`
	_ = string(o.x)  // want `^conversion from int to string yields a string of one rune$`
}
