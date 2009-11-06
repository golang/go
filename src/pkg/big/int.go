// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements signed multi-precision integers.

package big

// An Int represents a signed multi-precision integer.
// The zero value for an Int represents the value 0.
type Int struct {
	neg	bool;	// sign
	abs	[]Word;	// absolute value of the integer
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


// NewInt allocates and returns a new Int set to x.
func NewInt(x int64) *Int	{ return new(Int).New(x) }


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
		z.neg = false;	// 0 has no sign
	}
	return z;
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
		z.neg = false;	// 0 has no sign
	}
	return z;
}


// Mul computes z = x*y.
func (z *Int) Mul(x, y *Int) *Int {
	// x * y == x * y
	// x * (-y) == -(x * y)
	// (-x) * y == -(x * y)
	// (-x) * (-y) == x * y
	z.abs = mulNN(z.abs, x.abs, y.abs);
	z.neg = len(z.abs) > 0 && x.neg != y.neg;	// 0 has no sign
	return z;
}


// Div calculates q = (x-r)/y where 0 <= r < y. The receiver is set to q.
func (z *Int) Div(x, y *Int) (q, r *Int) {
	q = z;
	r = new(Int);
	div(q, r, x, y);
	return;
}


// Mod calculates q = (x-r)/y and returns r.
func (z *Int) Mod(x, y *Int) (r *Int) {
	q := new(Int);
	r = z;
	div(q, r, x, y);
	return;
}


func div(q, r, x, y *Int) {
	if len(y.abs) == 0 {
		panic("Divide by zero undefined");
	}

	if cmpNN(x.abs, y.abs) < 0 {
		q.neg = false;
		q.abs = nil;
		r.neg = y.neg;

		src := x.abs;
		dst := x.abs;
		if r == x {
			dst = nil;
		}

		r.abs = makeN(dst, len(src), false);
		for i, v := range src {
			r.abs[i] = v;
		}
		return;
	}

	if len(y.abs) == 1 {
		var rprime Word;
		q.abs, rprime = divNW(q.abs, x.abs, y.abs[0]);
		if rprime > 0 {
			r.abs = makeN(r.abs, 1, false);
			r.abs[0] = rprime;
			r.neg = x.neg;
		}
		q.neg = len(q.abs) > 0 && x.neg != y.neg;
		return;
	}

	q.neg = x.neg != y.neg;
	r.neg = x.neg;
	q.abs, r.abs = divNN(q.abs, r.abs, x.abs, y.abs);
	return;
}


// Neg computes z = -x.
func (z *Int) Neg(x *Int) *Int {
	z.abs = setN(z.abs, x.abs);
	z.neg = len(z.abs) > 0 && !x.neg;	// 0 has no sign
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


func (z *Int) String() string {
	s := "";
	if z.neg {
		s = "-";
	}
	return s + stringN(z.abs, 10);
}


// SetString sets z to the value of s, interpreted in the given base.
// If base is 0 then SetString attempts to detect the base by at the prefix of
// s. '0x' implies base 16, '0' implies base 8. Otherwise base 10 is assumed.
func (z *Int) SetString(s string, base int) (*Int, bool) {
	var scanned int;

	if base == 1 || base > 16 {
		goto Error;
	}

	if len(s) == 0 {
		goto Error;
	}

	if s[0] == '-' {
		z.neg = true;
		s = s[1:len(s)];
	} else {
		z.neg = false;
	}

	z.abs, _, scanned = scanN(z.abs, s, base);
	if scanned != len(s) {
		goto Error;
	}

	return z, true;

Error:
	z.neg = false;
	z.abs = nil;
	return nil, false;
}


// SetBytes interprets b as the bytes of a big-endian, unsigned integer and
// sets x to that value.
func (z *Int) SetBytes(b []byte) *Int {
	s := int(_S);
	z.abs = makeN(z.abs, (len(b)+s-1)/s, false);
	z.neg = false;

	j := 0;
	for len(b) >= s {
		var w Word;

		for i := s; i > 0; i-- {
			w <<= 8;
			w |= Word(b[len(b)-i]);
		}

		z.abs[j] = w;
		j++;
		b = b[0 : len(b)-s];
	}

	if len(b) > 0 {
		var w Word;

		for i := len(b); i > 0; i-- {
			w <<= 8;
			w |= Word(b[len(b)-i]);
		}

		z.abs[j] = w;
	}

	z.abs = normN(z.abs);

	return z;
}


// Bytes returns the absolute value of x as a big-endian byte array.
func (z *Int) Bytes() []byte {
	s := int(_S);
	b := make([]byte, len(z.abs)*s);

	for i, w := range z.abs {
		wordBytes := b[(len(z.abs)-i-1)*s : (len(z.abs)-i)*s];
		for j := s-1; j >= 0; j-- {
			wordBytes[j] = byte(w);
			w >>= 8;
		}
	}

	i := 0;
	for i < len(b) && b[i] == 0 {
		i++;
	}

	return b[i:len(b)];
}


// Len returns the length of the absolute value of x in bits. Zero is
// considered to have a length of one.
func (z *Int) Len() int {
	if len(z.abs) == 0 {
		return 0;
	}

	return len(z.abs)*int(_W) - int(leadingZeros(z.abs[len(z.abs)-1]));
}


// Exp sets z = x**y mod m. If m is nil, z = x**y.
// See Knuth, volume 2, section 4.6.3.
func (z *Int) Exp(x, y, m *Int) *Int {
	if y.neg || len(y.abs) == 0 {
		z.New(1);
		z.neg = x.neg;
		return z;
	}

	z.Set(x);
	v := y.abs[len(y.abs)-1];
	// It's invalid for the most significant word to be zero, therefore we
	// will find a one bit.
	shift := leadingZeros(v) + 1;
	v <<= shift;

	const mask = 1<<(_W-1);

	// We walk through the bits of the exponent one by one. Each time we see
	// a bit, we square, thus doubling the power. If the bit is a one, we
	// also multiply by x, thus adding one to the power.

	w := int(_W)-int(shift);
	for j := 0; j < w; j++ {
		z.Mul(z, z);

		if v&mask != 0 {
			z.Mul(z, x);
		}

		if m != nil {
			z.Mod(z, m);
		}

		v <<= 1;
	}

	for i := len(y.abs)-2; i >= 0; i-- {
		v = y.abs[i];

		for j := 0; j < int(_W); j++ {
			z.Mul(z, z);

			if v&mask != 0 {
				z.Mul(z, x);
			}

			if m != nil {
				z.Mod(z, m);
			}

			v <<= 1;
		}
	}

	z.neg = x.neg && y.abs[0] & 1 == 1;
	return z;
}


// GcdInt sets d to the greatest common divisor of a and b, which must be
// positive numbers.
// If x and y are not nil, GcdInt sets x and y such that d = a*x + b*y.
// If either a or b is not positive, GcdInt sets d = x = y = 0.
func GcdInt(d, x, y, a, b *Int) {
	if a.neg || b.neg {
		d.New(0);
		if x != nil {
			x.New(0);
		}
		if y != nil {
			y.New(0);
		}
		return;
	}

	A := new(Int).Set(a);
	B := new(Int).Set(b);

	X := new(Int);
	Y := new(Int).New(1);

	lastX := new(Int).New(1);
	lastY := new(Int);

	q := new(Int);
	temp := new(Int);

	for len(B.abs) > 0 {
		q, r := q.Div(A, B);

		A, B = B, r;

		temp.Set(X);
		X.Mul(X, q);
		X.neg = !X.neg;
		X.Add(X, lastX);
		lastX.Set(temp);

		temp.Set(Y);
		Y.Mul(Y, q);
		Y.neg = !Y.neg;
		Y.Add(Y, lastY);
		lastY.Set(temp);
	}

	if x != nil {
		*x = *lastX;
	}

	if y != nil {
		*y = *lastY;
	}

	*d = *A;
}
