// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spec

// map1 returns the vector z[i] = f(x[i]).
//
// If z has more lanes than x, the extra lanes of z are 0. If z has fewer lanes
// than x, the extra lanes of x are ignored.
func map1[From any, FromW Width, To any, ToW Width](x Vec[From, FromW], f func(x From) To) (z Vec[To, ToW]) {
	z = makeVec[To, ToW]()
	for i := range min(len(x), len(z)) {
		z[i] = f(x[i])
	}
	return z
}

// map2 returns the vector z[i] = f(x[i], y[i]).
//
// If z has more lanes than x, the extra lanes of z are 0. If z has fewer lanes
// than x, the extra lanes of x and y are ignored.
func map2[From any, FromW Width, To any, ToW Width](x, y Vec[From, FromW], f func(x, y From) To) (z Vec[To, ToW]) {
	z = makeVec[To, ToW]()
	for i := range min(len(x), len(z)) {
		z[i] = f(x[i], y[i])
	}
	return z
}

// grouped128 divides x, y, and z into groups of 128 bits and returns the vector
// whose n'th group is f applied to the n'th groups of x and y.
func grouped128[E1, E2, E3 any, W Width](f func(x Vec[E1, Width128], y Vec[E2, Width128]) Vec[E3, Width128], x Vec[E1, W], y Vec[E2, W]) (z Vec[E3, W]) {
	z = makeVec[E3, W]()
	xLanes, yLanes, zLanes := lanes[E1, Width128](), lanes[E2, Width128](), lanes[E3, Width128]()
	for g := range len(z) / zLanes {
		xg := Vec[E1, Width128](x[g*xLanes:][:xLanes])
		yg := Vec[E2, Width128](y[g*yLanes:][:yLanes])
		copy(z[g*zLanes:], f(xg, yg))
	}
	return z
}
