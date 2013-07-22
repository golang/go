// run

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 5820: register clobber when clearfat and 64 bit arithmetic is interleaved.

package main

func main() {
	array := make([][]int, 2)
	index := uint64(1)
	array[index] = nil
	if array[1] != nil {
		panic("array[1] != nil")
	}
}
