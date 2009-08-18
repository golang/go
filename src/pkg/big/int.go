// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements signed multi-precision integers.

package big

// An Int represents a signed multi-precision integer.
// The zero value for an Int represents the value 0.
type Int struct {
	neg bool;  // sign
	abs []Word;  // absolute value of the integer
}


// New allocates and returns a new Int set to x.
func (z *Int) New(x int64) *Int {
	z.neg = false;
	if x < 0 {
		z.neg = true;
		x = -x;
	}
	z.abs = newN(z.abs, uint64(x));
	return z;
}


// Set sets z to x.
func (z *Int) Set(x *Int) *Int {
	z.neg = x.neg;
	z.abs = setN(z.abs, x.abs);
	return z;
}


// Add computes z = x+y.
func (z *Int) Add(x, y *Int) *Int {
	if x.neg == y.neg {
		// x + y == x + y
		// (-x) + (-y) == -(x + y)
		z.neg = x.neg;
		z.abs = addNN(z.abs, x.abs, y.abs);
	} else {
		// x + (-y) == x - y == -(y - x)
		// (-x) + y == y - x == -(x - y)
		if cmpNN(x.abs, y.abs) >= 0 {
			z.neg = x.neg;
			z.abs = subNN(z.abs, x.abs, y.abs);
		} else {
			z.neg = !x.neg;
			z.abs = subNN(z.abs, y.abs, x.abs);
		}
	}
	if len(z.abs) == 0 {
		z.neg = false;  // 0 has no sign
	}
	return z
}


// Sub computes z = x-y.
func (z *Int) Sub(x, y *Int) *Int {
	if x.neg != y.neg {
		// x - (-y) == x + y
		// (-x) - y == -(x + y)
		z.neg = x.neg;
		z.abs = addNN(z.abs, x.abs, y.abs);
	} else {
		// x - y == x - y == -(y - x)
		// (-x) - (-y) == y - x == -(x - y)
		if cmpNN(x.abs, y.abs) >= 0 {
			z.neg = x.neg;
			z.abs = subNN(z.abs, x.abs, y.abs);
		} else {
			z.neg = !x.neg;
			z.abs = subNN(z.abs, y.abs, x.abs);
		}
	}
	if len(z.abs) == 0 {
		z.neg = false;  // 0 has no sign
	}
	return z
}


// Mul computes z = x*y.
func (z *Int) Mul(x, y *Int) *Int {
	// x * y == x * y
	// x * (-y) == -(x * y)
	// (-x) * y == -(x * y)
	// (-x) * (-y) == x * y
	z.abs = mulNN(z.abs, x.abs, y.abs);
	z.neg = len(z.abs) > 0 && x.neg != y.neg;  // 0 has no sign
	return z
}


// Neg computes z = -x.
func (z *Int) Neg(x *Int) *Int {
	z.abs = setN(z.abs, x.abs);
	z.neg = len(z.abs) > 0 && !x.neg;  // 0 has no sign
	return z;
}


// TODO(gri) Should this be x.Cmp(y) instead?

// CmpInt compares x and y. The result is
//
//   -1 if x <  y
//    0 if x == y
//   +1 if x >  y
//
func CmpInt(x, y *Int) (r int) {
	// x cmp y == x cmp y
	// x cmp (-y) == x
	// (-x) cmp y == y
	// (-x) cmp (-y) == -(x cmp y)
	switch {
	case x.neg == y.neg:
		r = cmpNN(x.abs, y.abs);
		if x.neg {
			r = -r;
		}
	case x.neg:
		r = -1;
	default:
		r = 1;
	}
	return;
}


func (x *Int) String() string {
	s := "";
	if x.neg {
		s = "-";
	}
	return s + stringN(x.abs, 10);
}
