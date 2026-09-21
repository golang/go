// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"hash/maphash"
)

func main() {
	testEmpty()
	testNonEmpty()
}

func testEmpty() {
	s := maphash.MakeSeed()
	hashes := map[uint64]struct{}{}
	addHash := func(x any) {
		var h maphash.Hash
		h.SetSeed(s)
		var c maphash.ComparableHasher[any]
		c.Hash(&h, x)
		hashes[h.Sum64()] = struct{}{}
	}
	addHash(int64(0))
	addHash(uint64(0))
	addHash(int(0))
	addHash(uint(0))
	if len(hashes) < 3 {
		panic(fmt.Sprintf("too few hashes %v\n", hashes))
	}
}

type I interface {
	foo()
}

type I1 int64
type I2 uint64
type I3 int
type I4 uint

func (i I1) foo() {}
func (i I2) foo() {}
func (i I3) foo() {}
func (i I4) foo() {}

func testNonEmpty() {
	s := maphash.MakeSeed()
	hashes := map[uint64]struct{}{}
	addHash := func(x I) {
		var h maphash.Hash
		h.SetSeed(s)
		var c maphash.ComparableHasher[I]
		c.Hash(&h, x)
		hashes[h.Sum64()] = struct{}{}
	}
	addHash(I1(0))
	addHash(I2(0))
	addHash(I3(0))
	addHash(I4(0))
	if len(hashes) < 3 {
		panic(fmt.Sprintf("too few hashes %v\n", hashes))
	}
}
