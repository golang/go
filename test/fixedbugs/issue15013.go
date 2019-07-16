// compile

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// CL 21202 introduced a compiler crash in the handling of a varargs
// function in the same recursive group as a function that calls it.
// Nothing in the standard library caught the problem, so adding a test.

package p

func F1(p *int, a ...*int) (int, *int) {
	if p == nil {
		return F2(), a[0]
	}
	return 0, a[0]
}

func F2() int {
	var i0, i1 int
	a, _ := F1(&i0, &i1)
	return a
}
