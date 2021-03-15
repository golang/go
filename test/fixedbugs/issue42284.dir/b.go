// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import "./a"

func g() {
	h := a.E() // ERROR "inlining call to a.E" "a.I\(a.T\(0\)\) does not escape"
	h.M()      // ERROR "devirtualizing h.M to a.T"

	// BAD: T(0) could be stack allocated.
	i := a.F(a.T(0)) // ERROR "inlining call to a.F" "a.T\(0\) escapes to heap"

	// Testing that we do NOT devirtualize here:
	i.M()
}
