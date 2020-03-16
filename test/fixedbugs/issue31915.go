// compile -d=ssa/check/on

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f()

func g() {
	var a []int
	var b bool
	for {
		b = (b && b) != (b && b)
		for b && b == b || true {
			f()
			_ = a[0]
		}
		_ = &b
		a = []int{}
	}
}
