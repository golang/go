// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type S struct {
	a []int
}

var s = &S{make([]int, 10)}

func main() {
	s.a[f()] = 1 // 6g used to call f twice here
}

var n int

func f() int {
	if n++; n > 1 {
		println("f twice")
		panic("fail")
	}
	return 0
}
