// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import "testing"

var (
	n = [16]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	m = [16]int{2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}
)

func TestEqual(t *testing.T) {
	if r := move2(n, m, 0); r != n {
		t.Fatalf("%v != %v", r, n)
	}
	if r := move2(n, m, 1); r != m {
		t.Fatalf("%v != %v", r, m)
	}
	if r := move2p(n, m, 0); r != n {
		t.Fatalf("%v != %v", r, n)
	}
	if r := move2p(n, m, 1); r != m {
		t.Fatalf("%v != %v", r, m)
	}
}

//go:noinline
func move2(a, b [16]int, c int) [16]int {
	e := a
	f := b
	var d [16]int
	if c%2 == 0 {
		d = e
	} else {
		d = f
	}
	r := d
	return r
}

//go:noinline
func move2p(a, b [16]int, c int) [16]int {
	e := a
	f := b
	var p *[16]int
	if c%2 == 0 {
		p = &e
	} else {
		p = &f
	}
	r := *p
	return r
}
