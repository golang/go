// errorcheck -0 -m -l

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for escaping variable-sized allocations.
// In particular, we need to make sure things assigned into
// variable-sized allocations escape even when the variable-sized
// allocations themselves don't escape.

package foo

type T string

func f1(n int, v T) { // ERROR "leaking param: v"
	s := make([]T, n) // ERROR "make\(\[\]T, n\) does not escape"
	s[0] = v
	g(s)
}

func f2(n int, v T) { // ERROR "leaking param: v"
	s := make([]T, n) // ERROR "make\(\[\]T, n\) does not escape"
	p := &s[0]
	*p = v
	g(s)
}

func f3(n int, v T) { // ERROR "leaking param: v"
	s := make([]T, n) // ERROR "make\(\[\]T, n\) does not escape"
	t := (*[4]T)(s)
	t[0] = v
	g(s)
}

// TODO: imprecise: this does not need to leak v.
func f4(v T) { // ERROR "leaking param: v"
	s := make([]T, 4) // ERROR "make\(\[\]T, 4\) does not escape"
	s[0] = v
	g(s)
}

// TODO: imprecise: this does not need to leak v.
func f5(v T) { // ERROR "leaking param: v"
	var b [4]T
	s := b[:]
	s[0] = v
	g(s)
}

func f6(v T) { // ERROR "v does not escape"
	var b [4]T
	s := b[:]
	b[0] = v
	g(s)
}

func g(s []T) { // ERROR "s does not escape"
}
