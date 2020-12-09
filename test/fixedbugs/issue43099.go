// compile

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check to make sure we don't try to constant fold a divide by zero.
// This is a tricky test, as we need a value that's not recognized as 0
// until lowering (otherwise it gets handled in a different path).

package p

func f() {
	var i int
	var s string
	for i > 0 {
		_ = s[0]
		i++
	}

	var c chan int
	c <- 1 % i
}

func f32() uint32 {
	s := "\x00\x00\x00\x00"
	c := uint32(s[0]) | uint32(s[1])<<8 | uint32(s[2])<<16 | uint32(s[3])<<24
	return 1 / c
}
func f64() uint64 {
	s := "\x00\x00\x00\x00\x00\x00\x00\x00"
	c := uint64(s[0]) | uint64(s[1])<<8 | uint64(s[2])<<16 | uint64(s[3])<<24 | uint64(s[4])<<32 | uint64(s[5])<<40 | uint64(s[6])<<48 | uint64(s[7])<<56
	return 1 / c
}
