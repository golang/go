// compile

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 80097: ICE "invalid heap allocated var without Heapaddr"
// when a heap-escaping variable is declared in unreachable code.
// Escape analysis marks the variable as heap-allocated, but because
// the declaration is dead, SSA generation never assigns it a heap
// address. DWARF generation must tolerate this state.

package p

var foo = func() int {
label:
	goto label
	x := [1024 * 64]*[2]*int{}
	if x != x {
		_ = x
	}
	return 1
}()
