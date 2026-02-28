// run

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var zero int

var sink any

func main() {
	var objs [][]*byte
	for i := 10; i < 200; i++ {
		// The objects we're allocating here are pointer-ful. Some will
		// max out their size class, which are the ones we want.
		// We also allocate from small to large, so that the object which
		// maxes out its size class is the last one allocated in that class.
		// This allocation pattern leaves the next object in the class
		// unallocated, which we need to reproduce the bug.
		objs = append(objs, make([]*byte, i))
	}
	sink = objs // force heap allocation

	// Bug will happen as soon as the write barrier turns on.
	for range 10000 {
		sink = make([]*byte, 1024)
		for _, s := range objs {
			s = append(s, make([]*byte, zero)...)
		}
	}
}
