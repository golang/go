// run

// Copyright 2019 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that a binary with a large data section can load. This failed on wasm.

package main

var test = [100 * 1024 * 1024]byte{42}

func main() {
	if test[0] != 42 {
		panic("bad")
	}
}
