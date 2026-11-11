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

func grouped128[E1, E2 any, W Width](f func(x Vec[E1, Width128], y Vec[E2, Width128]) Vec[E1, Width128], x Vec[E1, W], y Vec[E2, W]) (z Vec[E1, W]) {
	z = makeVec[E1, W]()
	groupLanes := lanes[E1, Width128]()
	for base := 0; base < len(z); base += groupLanes {
		copy(z[base:], f(Vec[E1, Width128](x[base:][:groupLanes]), Vec[E2, Width128](y[base:][:groupLanes])))
	}
	return z
}
