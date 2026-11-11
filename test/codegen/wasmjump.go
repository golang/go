// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// Branch directly to basic blocks located after the jump.
func wasmJumpForward(x int) int {
	// wasm:`Br\s` -`Set\sPC_B`
	if x > 0 {
		return x + 1
	}
	return x * 3
}
