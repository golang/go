//errorcheck -0 -m -m

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f() { // ERROR ""
	n, m := 100, 200
	_ = make([]byte, 1<<17)      // ERROR "too large for stack" ""
	_ = make([]byte, 100, 1<<17) // ERROR "too large for stack" ""
	_ = make([]byte, n, 1<<17)   // ERROR "too large for stack" ""

	_ = make([]byte, n)      // ERROR "non-constant size" ""
	_ = make([]byte, 100, m) // ERROR "non-constant size" ""
}
