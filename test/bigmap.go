// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func seq(x, y int) [1000]byte {
	var r [1000]byte
	for i := 0; i < len(r); i++ {
		r[i] = byte(x + i*y)
	}
	return r
}

func cmp(x, y [1000]byte) {
	for i := 0; i < len(x); i++ {
		if x[i] != y[i] {
			panic("BUG mismatch")
		}
	}
}

func main() {
	m := make(map[int][1000]byte)
	m[1] = seq(11, 13)
	m[2] = seq(2, 9)
	m[3] = seq(3, 17)

	cmp(m[1], seq(11, 13))
	cmp(m[2], seq(2, 9))
	cmp(m[3], seq(3, 17))
}
