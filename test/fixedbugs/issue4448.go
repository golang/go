// run

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4448: 64-bit indices that are statically known
// to be bounded make 5g and 8g generate a dangling branch.

package main

const b26 uint64 = 0x022fdd63cc95386d

var bitPos [64]int

func init() {
	for p := uint(0); p < 64; p++ {
		bitPos[b26<<p>>58] = int(p)
	}
}

func MinPos(w uint64) int {
	if w == 0 {
		panic("bit: MinPos(0) undefined")
	}
	return bitPos[((w&-w)*b26)>>58]
}

func main() {
	const one = uint64(1)
	for i := 0; i < 64; i++ {
		if MinPos(1<<uint(i)) != i {
			println("i =", i)
			panic("MinPos(1<<uint(i)) != i")
		}
	}
}
