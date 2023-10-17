// compile -d=ssa/check/seed

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This code caused an internal consistency error due to a bad shortcircuit optimization.

package p

func f() {
	var b bool
	if b {
		b = true
	}
l:
	for !b {
		b = true
		goto l
	}
}
