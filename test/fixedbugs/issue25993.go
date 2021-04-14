// compile -d=ssa/check/on

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 25993: SSA check fails on ARM.

package p

func f() {
	var x int
	var B0 bool
	B0 = !B0 || B0
	if B0 && B0 {
		x = -1
	}
	var AI []int
	var AB []bool
	_ = AI[x] > 0 && AB[x]
}
