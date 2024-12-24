// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"io"
)

// Expect 8 MB of memory usage for a small wasm program.
// This reflects the current allocator. We test an exact
// value here, but if the allocator changes, we can update
// or relax this.
const want = 8 << 20

var w = io.Discard

func main() {
	fmt.Fprintln(w, "hello world")

	const pageSize = 64 * 1024
	sz := uintptr(currentMemory()) * pageSize
	if sz != want {
		fmt.Printf("FAIL: unexpected memory size %d, want %d\n", sz, want)
	}
}

func currentMemory() int32 // implemented in assembly
