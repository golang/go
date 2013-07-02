// errorcheck

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 5698: can define a key type with slices.

package main

type Key struct {
	a int16 // the compiler was confused by the padding.
	b []int
}

type Val struct{}

type Map map[Key]Val // ERROR "invalid map key type"
