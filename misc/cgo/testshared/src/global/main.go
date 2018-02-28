// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"globallib"
)

//go:noinline
func testLoop() {
	for i, s := range globallib.Data {
		if s != int64(i) {
			panic("testLoop: mismatch")
		}
	}
}

//go:noinline
func ptrData() *[1<<20 + 10]int64 {
	return &globallib.Data
}

//go:noinline
func testMediumOffset() {
	for i, s := range globallib.Data[1<<16-2:] {
		if s != int64(i)+1<<16-2 {
			panic("testMediumOffset: index mismatch")
		}
	}

	x := globallib.Data[1<<16-1]
	if x != 1<<16-1 {
		panic("testMediumOffset: direct mismatch")
	}

	y := &globallib.Data[1<<16-3]
	if y != &ptrData()[1<<16-3] {
		panic("testMediumOffset: address mismatch")
	}
}

//go:noinline
func testLargeOffset() {
	for i, s := range globallib.Data[1<<20:] {
		if s != int64(i)+1<<20 {
			panic("testLargeOffset: index mismatch")
		}
	}

	x := globallib.Data[1<<20+1]
	if x != 1<<20+1 {
		panic("testLargeOffset: direct mismatch")
	}

	y := &globallib.Data[1<<20+2]
	if y != &ptrData()[1<<20+2] {
		panic("testLargeOffset: address mismatch")
	}
}

func main() {
	testLoop()

	// SSA rules commonly merge offsets into addresses. These
	// tests access global data in different ways to try
	// and exercise different SSA rules.
	testMediumOffset()
	testLargeOffset()
}
