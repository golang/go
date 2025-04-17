// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Exchanging two struct fields was compiled incorrectly.

package main

type S struct {
	i int
}

func F(c bool, s1, s2 S) (int, int) {
	if c {
		s1.i, s2.i = s2.i, s1.i
	}
	return s1.i, s2.i
}

func main() {
	i, j := F(true, S{1}, S{20})
	if i != 20 || j != 1 {
		panic(i+j)
	}
}
