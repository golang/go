// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type I interface{ M() }
type T int

func (T) M() {} // ERROR "can inline T.M"

func F(i I) I { // ERROR "can inline F" "leaking param: i to result ~r1 level=0"
	i = nil
	return i
}

func g() { // ERROR "can inline g"
	// BAD: T(0) could be stack allocated.
	i := F(T(0)) // ERROR "inlining call to F" "T\(0\) escapes to heap"

	// Testing that we do NOT devirtualize here:
	i.M()
}
