// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// WARNING: Please avoid updating this file. If this file needs to be updated,
// then a new inline_hot.pprof file should be generated:
//
//  $ cd $GOROOT/src/cmd/compile/internal/test/testdata/pgo/inline/
//  $ go test -bench=. -cpuprofile ./inline_hot.pprof
package main

import (
	"time"
)

type BS struct {
	length uint
	s      []uint64
}

const wSize = uint(64)
const lWSize = uint(6)

func D(i uint) int {
	return int((i + (wSize - 1)) >> lWSize)
}

func N(length uint) (bs *BS) {
	bs = &BS{
		length,
		make([]uint64, D(length)),
	}

	return bs
}

func (b *BS) S(i uint) *BS {
	b.s[i>>lWSize] |= 1 << (i & (wSize - 1))
	return b
}

var jn = [...]byte{
	0, 1, 56, 2, 57, 49, 28, 3, 61, 58, 42, 50, 38, 29, 17, 4,
	62, 47, 59, 36, 45, 43, 51, 22, 53, 39, 33, 30, 24, 18, 12, 5,
	63, 55, 48, 27, 60, 41, 37, 16, 46, 35, 44, 21, 52, 32, 23, 11,
	54, 26, 40, 15, 34, 20, 31, 10, 25, 14, 19, 9, 13, 8, 7, 6,
}

func T(v uint64) uint {
	return uint(jn[((v&-v)*0x03f79d71b4ca8b09)>>58])
}

func (b *BS) NS(i uint) (uint, bool) {
	x := int(i >> lWSize)
	if x >= len(b.s) {
		return 0, false
	}
	w := b.s[x]
	w = w >> (i & (wSize - 1))
	if w != 0 {
		return i + T(w), true
	}
	x = x + 1
	for x < len(b.s) {
		if b.s[x] != 0 {
			return uint(x)*wSize + T(b.s[x]), true
		}
		x = x + 1

	}
	return 0, false
}

func A() {
	s := N(100000)
	for i := 0; i < 1000; i += 30 {
		s.S(uint(i))
	}
	for j := 0; j < 1000; j++ {
		c := uint(0)
		for i, e := s.NS(0); e; i, e = s.NS(i + 1) {
			c++
		}
	}
}

func main() {
	time.Sleep(time.Second)
	A()
}
