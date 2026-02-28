// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Code was miscompiled on ppc64le due to incorrect zero-extension
// that was CSE'd.

package main

//go:noinline
func g(i uint64) uint64 {
	return uint64(uint32(i))
}

var sink uint64

func main() {
	for i := uint64(0); i < 1; i++ {
		i32 := int32(i - 1)
		sink = uint64((uint32(i32) << 1) ^ uint32((i32 >> 31)))
		x := g(uint64(i32))
		if x != uint64(uint32(i32)) {
			panic(x)
		}
	}
}
