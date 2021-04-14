// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 19710: mishandled defer delete(...)

package main

func main() {
	if n := len(f()); n != 0 {
		println("got", n, "want 0")
		panic("bad defer delete")
	}
}

func f() map[int]bool {
	m := map[int]bool{}
	for i := 0; i < 3; i++ {
		m[i] = true
		defer delete(m, i)
	}
	return m
}
