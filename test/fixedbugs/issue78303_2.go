// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var n = 1

//go:noinline
func bug(a func(int)) {
	m := int64(len("x"[:n]))

	for i, j := int64(1<<63-1), 0; i > m; i, j = i-(1<<62), j+1 {
		if j == 2 {
			a(3)
			return
		}
	}
	a(2)
}

func main() {
	var r int
	var set bool
	bug(func(x int) {
		if set {
			panic("called twice")
		}
		set = true
		r = x
	})
	if !set {
		panic("not called")
	}
	if r != 2 {
		panic("got wrong result")
	}
}
