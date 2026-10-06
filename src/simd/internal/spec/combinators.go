// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spec

// map1 returns the vector z[i] = f(x[i]).
//
// z may have more lanes than x, in which case they will be 0.
func map1[From EltOrMask, FromW Width, To EltOrMask, ToW Width](x Vec[From, FromW], f func(x From) To) (z Vec[To, ToW]) {
	z = makeVec[To, ToW]()
	// Loop over x, not z, because width rounding may have expanded z.
	for i, xi := range x {
		z[i] = f(xi)
	}
	return z
}

// map2 returns the vector z[i] = f(x[i], y[i]).
//
// z may have more lanes than x, in which case they will be 0.
func map2[From EltOrMask, FromW Width, To EltOrMask, ToW Width](x, y Vec[From, FromW], f func(x, y From) To) (z Vec[To, ToW]) {
	z = makeVec[To, ToW]()
	for i := range x {
		z[i] = f(x[i], y[i])
	}
	return z
}

func grouped128[E Elt, W Width](f func(x, y Vec[E, Width128]) Vec[E, Width128], x, y Vec[E, W]) (z Vec[E, W]) {
	z = makeVec[E, W]()
	groupLanes := lanes[E, Width128]()
	for base := 0; base < len(z); base += groupLanes {
		copy(z[base:], f(Vec[E, Width128](x[base:][:groupLanes]), Vec[E, Width128](y[base:][:groupLanes])))
	}
	return z
}
