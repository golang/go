// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import "./a"

func g() {
	h := a.E() // ERROR "inlining call to a.E" "T\(0\) does not escape"
	h.M()      // ERROR "devirtualizing h.M to a.T" "inlining call to a.T.M"

	i := a.F(a.T(0)) // ERROR "inlining call to a.F" "a.T\(0\) does not escape"

	// It is fine that we devirtualize here, as we add an additional nilcheck.
	i.M() // ERROR "devirtualizing i.M to a.T" "inlining call to a.T.M"
}
