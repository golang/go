// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type I interface{ M() }
type T int

func (T) M() {} // ERROR "can inline T.M"

func E() I { // ERROR "can inline E"
	return T(0) // ERROR "T\(0\) escapes to heap"
}

func F(i I) I { // ERROR "can inline F" "leaking param: i to result ~r0 level=0"
	i = nil
	return i
}

func g() {
	h := E() // ERROR "inlining call to E" "T\(0\) does not escape"
	h.M()    // ERROR "devirtualizing h.M to T" "inlining call to T.M"

	// BAD: T(0) could be stack allocated.
	i := F(T(0)) // ERROR "inlining call to F" "T\(0\) escapes to heap"

	// Testing that we do NOT devirtualize here:
	i.M()
}
