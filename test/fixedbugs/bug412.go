// errorcheck

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type t struct {
	x int  // ERROR "duplicate field x|duplicate field name .x."
	x int
}

func f(t *t) int {
	return t.x  // GC_ERROR "ambiguous selector t.x"
}
