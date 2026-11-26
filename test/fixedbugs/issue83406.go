// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:noinline
func sarb() int8 {
	s := "\xfe"
	return int8(s[0]) >> 7
}

//go:noinline
func sarw() int16 {
	s := "\xfe\xff"
	return int16(uint16(s[0])|uint16(s[1])<<8) >> 15
}

func main() {
	if got := sarb(); got != -1 {
		panic(got)
	}
	if got := sarw(); got != -1 {
		panic(got)
	}
}
