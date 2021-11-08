// errorcheck

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that comparisons of slice/map/func values against converted nil
// values are properly rejected.

package p

func bug() {
	type S []byte
	type M map[int]int
	type F func()

	var s S
	var m M
	var f F

	_ = s == S(nil) // ERROR "compare.*to nil|operator \=\= not defined for .|cannot compare"
	_ = S(nil) == s // ERROR "compare.*to nil|operator \=\= not defined for .|cannot compare"
	switch s {
	case S(nil): // ERROR "compare.*to nil|operator \=\= not defined for .|cannot compare"
	}

	_ = m == M(nil) // ERROR "compare.*to nil|operator \=\= not defined for .|cannot compare"
	_ = M(nil) == m // ERROR "compare.*to nil|operator \=\= not defined for .|cannot compare"
	switch m {
	case M(nil): // ERROR "compare.*to nil|operator \=\= not defined for .|cannot compare"
	}

	_ = f == F(nil) // ERROR "compare.*to nil|operator \=\= not defined for .|cannot compare"
	_ = F(nil) == f // ERROR "compare.*to nil|operator \=\= not defined for .|cannot compare"
	switch f {
	case F(nil): // ERROR "compare.*to nil|operator \=\= not defined for .|cannot compare"
	}
}
