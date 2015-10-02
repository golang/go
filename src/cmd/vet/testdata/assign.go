// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the useless-assignment checker.

package testdata

type ST struct {
	x int
}

func (s *ST) SetX(x int) {
	// Accidental self-assignment; it should be "s.x = x"
	x = x // ERROR "self-assignment of x to x"
	// Another mistake
	s.x = s.x // ERROR "self-assignment of s.x to s.x"
}
