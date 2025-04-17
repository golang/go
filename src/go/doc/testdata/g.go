// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The package g is a go/doc test for mixed exported/unexported values.
package g

const (
	A, b = iota, iota
	c, D
	E, f
	G, H
)

var (
	c1, C2, c3 = 1, 2, 3
	C4, c5, C6 = 4, 5, 6
	c7, C8, c9 = 7, 8, 9
	xx, yy, zz = 0, 0, 0 // all unexported and hidden
)

var (
	x, X = f()
	y, z = f()
)
