// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements signed multi-precision integers.

package big

// An Int represents a signed multi-precision integer.
// The zero value for an Int represents the value 0.
type Int struct {
	neg bool // sign
	abs nat  // absolute value of the integer
}


// New allocates and returns a new Int set to x.
func (z *Int) New(x int64) *Int {
	z.neg = false
	if x < 0 {
		z.neg = true
		x = -x
	}
	z.abs = z.abs.new(uint64(x))
	return z
}


// NewInt allocates and returns a new Int set to x.
func NewInt(x int64) *Int { return new(Int).New(x) }


// Set sets z to x.
func (z *Int) Set(x *Int) *Int {
	z.neg = x.neg
	z.abs = z.abs.set(x.abs)
	return z
}


// Add computes z = x+y.
func (z *Int) Add(x, y *Int) *Int {
	if x.neg == y.neg {
		// x + y == x + y
		// (-x) + (-y) == -(x + y)
		z.neg = x.neg
		z.abs = z.abs.add(x.abs, y.abs)
	} else {
		// x + (-y) == x - y == -(y - x)
		// (-x) + y == y - x == -(x - y)
		if x.abs.cmp(y.abs) >= 0 {
			z.neg = x.neg
			z.abs = z.abs.sub(x.abs, y.abs)
		} else {
			z.neg = !x.neg
			z.abs = z.abs.sub(y.abs, x.abs)
		}
	}
	if len(z.abs) == 0 {
		z.neg = false // 0 has no sign
	}
	return z
}


// Sub computes z = x-y.
func (z *Int) Sub(x, y *Int) *Int {
	if x.neg != y.neg {
		// x - (-y) == x + y
		// (-x) - y == -(x + y)
		z.neg = x.neg
		z.abs = z.abs.add(x.abs, y.abs)
	} else {
		// x - y == x - y == -(y - x)
		// (-x) - (-y) == y - x == -(x - y)
		if x.abs.cmp(y.abs) >= 0 {
			z.neg = x.neg
			z.abs = z.abs.sub(x.abs, y.abs)
		} else {
			z.neg = !x.neg
			z.abs = z.abs.sub(y.abs, x.abs)
		}
	}
	if len(z.abs) == 0 {
		z.neg = false // 0 has no sign
	}
	return z
}


// Mul computes z = x*y.
func (z *Int) Mul(x, y *Int) *Int {
	// x * y == x * y
	// x * (-y) == -(x * y)
	// (-x) * y == -(x * y)
	// (-x) * (-y) == x * y
	z.abs = z.abs.mul(x.abs, y.abs)
	z.neg = len(z.abs) > 0 && x.neg != y.neg // 0 has no sign
	return z
}


// Div calculates q = (x-r)/y and sets z = q.
func (z *Int) Div(x, y *Int) *Int {
	r := new(Int)
	div(z, r, x, y)
	return z
}


// Mod calculates q = (x-r)/y and sets z = r.
func (z *Int) Mod(x, y *Int) *Int {
	q := new(Int)
	div(q, z, x, y)
	return z
}


// DivMod calculates q = (x-r)/y and sets z = q.  (It returns z, r.)
func (z *Int) DivMod(x, y, r *Int) (*Int, *Int) {
	div(z, r, x, y)
	return z, r
}


func div(q, r, x, y *Int) {
	q.neg = x.neg != y.neg
	r.neg = x.neg
	q.abs, r.abs = q.abs.div(r.abs, x.abs, y.abs)
	return
}


// Neg computes z = -x.
func (z *Int) Neg(x *Int) *Int {
	z.abs = z.abs.set(x.abs)
	z.neg = len(z.abs) > 0 && !x.neg // 0 has no sign
	return z
}


// Cmp compares x and y. The result is
//
//   -1 if x <  y
//    0 if x == y
//   +1 if x >  y
//
func (x *Int) Cmp(y *Int) (r int) {
	// x cmp y == x cmp y
	// x cmp (-y) == x
	// (-x) cmp y == y
	// (-x) cmp (-y) == -(x cmp y)
	switch {
	case x.neg == y.neg:
		r = x.abs.cmp(y.abs)
		if x.neg {
			r = -r
		}
	case x.neg:
		r = -1
	default:
		r = 1
	}
	return
}


func (z *Int) String() string {
	s := ""
	if z.neg {
		s = "-"
	}
	return s + z.abs.string(10)
}


// Int64 returns the int64 representation of z.
// If z cannot be represented in an int64, the result is undefined.
func (z *Int) Int64() int64 {
	if len(z.abs) == 0 {
		return 0
	}
	v := int64(z.abs[0])
	if _W == 32 && len(z.abs) > 1 {
		v |= int64(z.abs[1]) << 32
	}
	if z.neg {
		v = -v
	}
	return v
}


// SetString sets z to the value of s, interpreted in the given base.
// If base is 0 then SetString attempts to detect the base by at the prefix of
// s. '0x' implies base 16, '0' implies base 8. Otherwise base 10 is assumed.
func (z *Int) SetString(s string, base int) (*Int, bool) {
	var scanned int

	if base == 1 || base > 16 {
		goto Error
	}

	if len(s) == 0 {
		goto Error
	}

	if s[0] == '-' {
		z.neg = true
		s = s[1:]
	} else {
		z.neg = false
	}

	z.abs, _, scanned = z.abs.scan(s, base)
	if scanned != len(s) {
		goto Error
	}
	if len(z.abs) == 0 {
		z.neg = false // 0 has no sign
	}

	return z, true

Error:
	z.neg = false
	z.abs = nil
	return z, false
}


// SetBytes interprets b as the bytes of a big-endian, unsigned integer and
// sets z to that value.
func (z *Int) SetBytes(b []byte) *Int {
	s := int(_S)
	z.abs = z.abs.make((len(b) + s - 1) / s)
	z.neg = false

	j := 0
	for len(b) >= s {
		var w Word

		for i := s; i > 0; i-- {
			w <<= 8
			w |= Word(b[len(b)-i])
		}

		z.abs[j] = w
		j++
		b = b[0 : len(b)-s]
	}

	if len(b) > 0 {
		var w Word

		for i := len(b); i > 0; i-- {
			w <<= 8
			w |= Word(b[len(b)-i])
		}

		z.abs[j] = w
	}

	z.abs = z.abs.norm()

	return z
}


// Bytes returns the absolute value of x as a big-endian byte array.
func (z *Int) Bytes() []byte {
	s := int(_S)
	b := make([]byte, len(z.abs)*s)

	for i, w := range z.abs {
		wordBytes := b[(len(z.abs)-i-1)*s : (len(z.abs)-i)*s]
		for j := s - 1; j >= 0; j-- {
			wordBytes[j] = byte(w)
			w >>= 8
		}
	}

	i := 0
	for i < len(b) && b[i] == 0 {
		i++
	}

	return b[i:]
}


// Len returns the length of the absolute value of z in bits. Zero is
// considered to have a length of zero.
func (z *Int) Len() int {
	if len(z.abs) == 0 {
		return 0
	}

	return len(z.abs)*_W - int(leadingZeros(z.abs[len(z.abs)-1]))
}


// Exp sets z = x**y mod m. If m is nil, z = x**y.
// See Knuth, volume 2, section 4.6.3.
func (z *Int) Exp(x, y, m *Int) *Int {
	if y.neg || len(y.abs) == 0 {
		z.New(1)
		z.neg = x.neg
		return z
	}

	var mWords nat
	if m != nil {
		mWords = m.abs
	}

	z.abs = z.abs.expNN(x.abs, y.abs, mWords)
	z.neg = x.neg && y.abs[0]&1 == 1
	return z
}


// GcdInt sets d to the greatest common divisor of a and b, which must be
// positive numbers.
// If x and y are not nil, GcdInt sets x and y such that d = a*x + b*y.
// If either a or b is not positive, GcdInt sets d = x = y = 0.
func GcdInt(d, x, y, a, b *Int) {
	if a.neg || b.neg {
		d.New(0)
		if x != nil {
			x.New(0)
		}
		if y != nil {
			y.New(0)
		}
		return
	}

	A := new(Int).Set(a)
	B := new(Int).Set(b)

	X := new(Int)
	Y := new(Int).New(1)

	lastX := new(Int).New(1)
	lastY := new(Int)

	q := new(Int)
	temp := new(Int)

	for len(B.abs) > 0 {
		r := new(Int)
		q, r = q.DivMod(A, B, r)

		A, B = B, r

		temp.Set(X)
		X.Mul(X, q)
		X.neg = !X.neg
		X.Add(X, lastX)
		lastX.Set(temp)

		temp.Set(Y)
		Y.Mul(Y, q)
		Y.neg = !Y.neg
		Y.Add(Y, lastY)
		lastY.Set(temp)
	}

	if x != nil {
		*x = *lastX
	}

	if y != nil {
		*y = *lastY
	}

	*d = *A
}


// ProbablyPrime performs n Miller-Rabin tests to check whether z is prime.
// If it returns true, z is prime with probability 1 - 1/4^n.
// If it returns false, z is not prime.
func ProbablyPrime(z *Int, n int) bool { return !z.neg && z.abs.probablyPrime(n) }


// Lsh sets z = x << n and returns z.
func (z *Int) Lsh(x *Int, n uint) *Int {
	z.neg = x.neg
	z.abs = z.abs.shl(x.abs, n)
	return z
}


// Rsh sets z = x >> n and returns z.
func (z *Int) Rsh(x *Int, n uint) *Int {
	if x.neg {
		// (-x) >> s == ^(x-1) >> s == ^((x-1) >> s) == -(((x-1) >> s) + 1)
		z.neg = true
		t := z.abs.sub(x.abs, natOne) // no underflow because |x| > 0
		t = t.shr(t, n)
		z.abs = t.add(t, natOne)
		return z
	}

	z.neg = false
	z.abs = z.abs.shr(x.abs, n)
	return z
}
