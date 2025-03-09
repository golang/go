// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"io"
)

// Wasm page size.
const pageSize = 64 * 1024

// Expect less than 3 MB + 1 page of memory usage for a small wasm
// program. This reflects the current allocator. If the allocator
// changes, update this value.
const want = 3<<20 + pageSize

var w = io.Discard

func main() {
	fmt.Fprintln(w, "hello world")

	sz := uintptr(currentMemory()) * pageSize
	if sz > want {
		fmt.Printf("FAIL: unexpected memory size %d, want <= %d\n", sz, want)
	}
}

func currentMemory() int32 // implemented in assembly
