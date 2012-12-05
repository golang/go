// run

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4396. Arrays of bytes are not required to be
// word aligned. 5g should use MOVB to load the address
// of s.g[0] for its nil check.
//
// This test _may_ fail on arm, but requires the host to 
// trap unaligned loads. This is generally done with
//
// echo "4" > /proc/cpu/alignment

package main

var s = struct {
	// based on lzw.decoder
	a, b, c, d, e uint16
	f             [4096]uint8
	g             [4096]uint8
}{}

func main() {
	s.g[0] = 1
}
