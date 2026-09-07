// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func currentMemory() int32
func growMemory(pages int32) int32

func main() {
	// -T sets the first logical PC to 4 MiB. Use a small limit so
	// the test can grow memory to the boundary.
	const limit = 0x400000 / (64 << 10)
	initial := currentMemory()
	if initial == 0 || initial >= limit {
		panic("unexpected initial memory size")
	}
	if growMemory(limit-initial+1) != -1 || currentMemory() != initial {
		panic("memory grew past the logical PC range")
	}
	if growMemory(limit-initial) != initial || currentMemory() != limit {
		panic("memory did not grow to the logical PC range")
	}
	if growMemory(1) != -1 || currentMemory() != limit {
		panic("memory grew into the logical PC range")
	}
}
