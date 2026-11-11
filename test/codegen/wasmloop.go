// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// Loops without calls are WebAssembly loops.
func wasmLoop(s []int) int {
	t := 0
	// wasm:`Loop`
	for _, x := range s {
		t += x
	}
	return t
}

func wasmLoopNested(s []int) int {
	t := 0
	// wasm:`Loop`
	for _, x := range s {
		if x < 0 {
			// wasm:`Loop`
			for x < 0 {
				x += 2
			}
			t++
		}
		t += x
	}
	return t
}
