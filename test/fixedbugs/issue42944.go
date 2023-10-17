// errorcheck -0 -live

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 42944: address of callee args area should only be short-lived
// and never across a call.

package p

type T [10]int // trigger DUFFCOPY when passing by value, so it uses the address

func F() {
	var x T
	var i int
	for {
		x = G(i) // no autotmp live at this and next calls
		H(i, x)
	}
}

func G(int) T
func H(int, T)
