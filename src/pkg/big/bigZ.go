// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements signed multi-precision integers.

package big

// A Z represents a signed multi-precision integer.
// The zero value for a Z represents the value 0.
type Z struct {
	neg bool;  // sign
	m []Word;  // mantissa
}


// NewZ sets z to x.
func NewZ(z Z, x int64) Z {
	z.neg = false;
	if x < 0 {
		z.neg = true;
		x = -x;
	}
	z.m = newN(z.m, uint64(x));
	return z;
}


// SetZ sets z to x.
func SetZ(z, x Z) Z {
	z.neg = x.neg;
	z.m = setN(z.m, x.m);
	return z;
}


// AddZZ computes z = x+y.
func AddZZ(z, x, y Z) Z {
	if x.neg == y.neg {
		// x + y == x + y
		// (-x) + (-y) == -(x + y)
		z.neg = x.neg;
		z.m = addNN(z.m, x.m, y.m);
	} else {
		// x + (-y) == x - y == -(y - x)
		// (-x) + y == y - x == -(x - y)
		if cmpNN(x.m, y.m) >= 0 {
			z.neg = x.neg;
			z.m = subNN(z.m, x.m, y.m);
		} else {
			z.neg = !x.neg;
			z.m = subNN(z.m, y.m, x.m);
		}
	}
	if len(z.m) == 0 {
		z.neg = false;  // 0 has no sign
	}
	return z
}


// AddZZ computes z = x-y.
func SubZZ(z, x, y Z) Z {
	if x.neg != y.neg {
		// x - (-y) == x + y
		// (-x) - y == -(x + y)
		z.neg = x.neg;
		z.m = addNN(z.m, x.m, y.m);
	} else {
		// x - y == x - y == -(y - x)
		// (-x) - (-y) == y - x == -(x - y)
		if cmpNN(x.m, y.m) >= 0 {
			z.neg = x.neg;
			z.m = subNN(z.m, x.m, y.m);
		} else {
			z.neg = !x.neg;
			z.m = subNN(z.m, y.m, x.m);
		}
	}
	if len(z.m) == 0 {
		z.neg = false;  // 0 has no sign
	}
	return z
}


// MulZZ computes z = x*y.
func MulZZ(z, x, y Z) Z {
	// x * y == x * y
	// x * (-y) == -(x * y)
	// (-x) * y == -(x * y)
	// (-x) * (-y) == x * y
	z.neg = x.neg != y.neg;
	z.m = mulNN(z.m, x.m, y.m);
	return z
}


// NegZ computes z = -x.
func NegZ(z, x Z) Z {
	z.neg = len(x.m) > 0 && !x.neg;  // 0 has no sign
	z.m = setN(z.m, x.m);
	return z;
}


// Cmp compares x and y. The result is an int value that is
//
//   <  0 if x <  y
//   == 0 if x == y
//   >  0 if x >  y
//
func CmpZZ(x, y Z) (r int) {
	// x cmp y == x cmp y
	// x cmp (-y) == x
	// (-x) cmp y == y
	// (-x) cmp (-y) == -(x cmp y)
	switch {
	case x.neg == y.neg:
		r = cmpNN(x.m, y.m);
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


func (x Z) String() string {
	s := "";
	if x.neg {
		s = "-";
	}
	return s + stringN(x.m, 10);
}
