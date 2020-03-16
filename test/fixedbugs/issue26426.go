//run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:noinline
func f(p *int, v int, q1, q2 *int, r *bool) {
	x := *r
	if x {
		*q1 = 1
	}
	*p = *p + v // This must clobber flags. Otherwise we keep x in a flags register.
	if x {
		*q2 = 1
	}
}

func main() {
	var p int
	var q1, q2 int
	var b bool
	f(&p, 1, &q1, &q2, &b)
	if q1 != 0 || q2 != 0 {
		panic("bad")
	}
}
