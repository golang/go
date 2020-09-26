//errorcheck -0 -m -m

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f() { // ERROR ""
	b1 := make([]byte, 1<<17)      // ERROR "too large for stack" ""
	b2 := make([]byte, 100, 1<<17) // ERROR "too large for stack" ""

	n, m := 100, 200
	b1 = make([]byte, n)      // ERROR "non-constant size" ""
	b2 = make([]byte, 100, m) // ERROR "non-constant size" ""

	_, _ = b1, b2
}
