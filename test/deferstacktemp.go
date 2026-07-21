// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This tests an SSA-able stack temporary whose address must refer to its stack
// slot. Without the addrtaken mark, (*state).addr returns &runtime.zerobase and
// runtime.deferprocStack follows a corrupted _defer record.
//
// On 64-bit systems, runtime._defer is 48 bytes with 7 fields and fits the
// current SSA limits. Nine defers force the non-open-coded deferprocStack path.

package main

var n int

func sink() { n++ }

//go:noinline
func nineDefers() {
	defer sink()
	defer sink()
	defer sink()
	defer sink()
	defer sink()
	defer sink()
	defer sink()
	defer sink()
	defer sink()
}

func main() {
	nineDefers()
	if n != 9 {
		println("n =", n, "want 9")
		panic("defer chain corrupted")
	}
}
