// errorcheck

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Ideal vs non-ideal bool. See issue 3915, 3923.

package p

type mybool bool
type mybool1 bool

var (
	x, y int = 1, 2
	c1 bool = x < y
	c2 mybool = x < y
	c3 mybool = c2 == (x < y)
	c4 mybool = c2 == (1 < 2)
	c5 mybool = 1 < 2
	c6 mybool1 = x < y
	c7 = c1 == c2 // ERROR "mismatched types"
	c8 = c2 == c6 // ERROR "mismatched types"
	c9 = c1 == c6 // ERROR "mismatched types"
	_ = c2 && (x < y)
	_ = c2 && (1 < 2)
	_ = c1 && c2 // ERROR "mismatched types"
	_ = c2 && c6 // ERROR "mismatched types"
	_ = c1 && c6 // ERROR "mismatched types"
)
