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


var intOne = &Int{false, natOne}


// SetInt64 sets z to x and returns z.
func (z *Int) SetInt64(x int64) *Int {
	z.neg = false
	if x < 0 {
		z.neg = true
		x = -x
	}
	z.abs = z.abs.new(uint64(x))
	return z
}


// NewInt allocates and returns a new Int set to x.
func NewInt(x int64) *Int {
	return new(Int).SetInt64(x)
}


// Set sets z to x.
func (z *Int) Set(x *Int) *Int {
	z.neg = x.neg
	z.abs = z.abs.set(x.abs)
	return z
}


// Add sets z to the sum x+y and returns z.
func (z *Int) Add(x, y *Int) *Int {
	neg := x.neg
	if x.neg == y.neg {
		// x + y == x + y
		// (-x) + (-y) == -(x + y)
		z.abs = z.abs.add(x.abs, y.abs)
	} else {
		// x + (-y) == x - y == -(y - x)
		// (-x) + y == y - x == -(x - y)
		if x.abs.cmp(y.abs) >= 0 {
			z.abs = z.abs.sub(x.abs, y.abs)
		} else {
			neg = !neg
			z.abs = z.abs.sub(y.abs, x.abs)
		}
	}
	z.neg = len(z.abs) > 0 && neg // 0 has no sign
	return z
}


// Sub sets z to the difference x-y and returns z.
func (z *Int) Sub(x, y *Int) *Int {
	neg := x.neg
	if x.neg != y.neg {
		// x - (-y) == x + y
		// (-x) - y == -(x + y)
		z.abs = z.abs.add(x.abs, y.abs)
	} else {
		// x - y == x - y == -(y - x)
		// (-x) - (-y) == y - x == -(x - y)
		if x.abs.cmp(y.abs) >= 0 {
			z.abs = z.abs.sub(x.abs, y.abs)
		} else {
			neg = !neg
			z.abs = z.abs.sub(y.abs, x.abs)
		}
	}
	z.neg = len(z.abs) > 0 && neg // 0 has no sign
	return z
}


// Mul sets z to the product x*y and returns z.
func (z *Int) Mul(x, y *Int) *Int {
	// x * y == x * y
	// x * (-y) == -(x * y)
	// (-x) * y == -(x * y)
	// (-x) * (-y) == x * y
	z.abs = z.abs.mul(x.abs, y.abs)
	z.neg = len(z.abs) > 0 && x.neg != y.neg // 0 has no sign
	return z
}


// Quo sets z to the quotient x/y for y != 0 and returns z.
// If y == 0, a division-by-zero run-time panic occurs.
// See QuoRem for more details.
func (z *Int) Quo(x, y *Int) *Int {
	z.abs, _ = z.abs.div(nil, x.abs, y.abs)
	z.neg = len(z.abs) > 0 && x.neg != y.neg // 0 has no sign
	return z
}


// Rem sets z to the remainder x%y for y != 0 and returns z.
// If y == 0, a division-by-zero run-time panic occurs.
// See QuoRem for more details.
func (z *Int) Rem(x, y *Int) *Int {
	_, z.abs = nat(nil).div(z.abs, x.abs, y.abs)
	z.neg = len(z.abs) > 0 && x.neg // 0 has no sign
	return z
}


// QuoRem sets z to the quotient x/y and r to the remainder x%y
// and returns the pair (z, r) for y != 0.
// If y == 0, a division-by-zero run-time panic occurs.
//
// QuoRem implements T-division and modulus (like Go):
//
//	q = x/y      with the result truncated to zero
//	r = x - y*q
//
// (See Daan Leijen, ``Division and Modulus for Computer Scientists''.)
//
func (z *Int) QuoRem(x, y, r *Int) (*Int, *Int) {
	z.abs, r.abs = z.abs.div(r.abs, x.abs, y.abs)
	z.neg, r.neg = len(z.abs) > 0 && x.neg != y.neg, len(r.abs) > 0 && x.neg // 0 has no sign
	return z, r
}


// Div sets z to the quotient x/y for y != 0 and returns z.
// If y == 0, a division-by-zero run-time panic occurs.
// See DivMod for more details.
func (z *Int) Div(x, y *Int) *Int {
	y_neg := y.neg // z may be an alias for y
	var r Int
	z.QuoRem(x, y, &r)
	if r.neg {
		if y_neg {
			z.Add(z, intOne)
		} else {
			z.Sub(z, intOne)
		}
	}
	return z
}


// Mod sets z to the modulus x%y for y != 0 and returns z.
// If y == 0, a division-by-zero run-time panic occurs.
// See DivMod for more details.
func (z *Int) Mod(x, y *Int) *Int {
	y0 := y // save y
	if z == y || alias(z.abs, y.abs) {
		y0 = new(Int).Set(y)
	}
	var q Int
	q.QuoRem(x, y, z)
	if z.neg {
		if y0.neg {
			z.Sub(z, y0)
		} else {
			z.Add(z, y0)
		}
	}
	return z
}


// DivMod sets z to the quotient x div y and m to the modulus x mod y
// and returns the pair (z, m) for y != 0.
// If y == 0, a division-by-zero run-time panic occurs.
//
// DivMod implements Euclidian division and modulus (unlike Go):
//
//	q = x div y  such that
//	m = x - y*q  with 0 <= m < |q|
//
// (See Raymond T. Boute, ``The Euclidian definition of the functions
// div and mod''. ACM Transactions on Programming Languages and
// Systems (TOPLAS), 14(2):127-144, New York, NY, USA, 4/1992.
// ACM press.)
//
func (z *Int) DivMod(x, y, m *Int) (*Int, *Int) {
	y0 := y // save y
	if z == y || alias(z.abs, y.abs) {
		y0 = new(Int).Set(y)
	}
	z.QuoRem(x, y, m)
	if m.neg {
		if y0.neg {
			z.Add(z, intOne)
			m.Sub(m, y0)
		} else {
			z.Sub(z, intOne)
			m.Add(m, y0)
		}
	}
	return z, m
}


// Neg computes the negation z = -x.
func (z *Int) Neg(x *Int) *Int {
	z.abs = z.abs.set(x.abs)
	z.neg = len(z.abs) > 0 && !x.neg // 0 has no sign
	return z
}


// Cmp compares x and y and returns:
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

	neg := false
	if s[0] == '-' {
		neg = true
		s = s[1:]
	}

	z.abs, _, scanned = z.abs.scan(s, base)
	if scanned != len(s) {
		goto Error
	}
	z.neg = len(z.abs) > 0 && neg // 0 has no sign

	return z, true

Error:
	z.abs = nil
	z.neg = false
	return z, false
}


// SetBytes interprets b as the bytes of a big-endian, unsigned integer and
// sets z to that value.
func (z *Int) SetBytes(b []byte) *Int {
	s := int(_S)
	z.abs = z.abs.make((len(b) + s - 1) / s)

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
	z.neg = false
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


// BitLen returns the length of the absolute value of z in bits.
// The bit length of 0 is 0.
func (z *Int) BitLen() int {
	return z.abs.bitLen()
}


// Exp sets z = x**y mod m. If m is nil, z = x**y.
// See Knuth, volume 2, section 4.6.3.
func (z *Int) Exp(x, y, m *Int) *Int {
	if y.neg || len(y.abs) == 0 {
		z.SetInt64(1)
		z.neg = x.neg
		return z
	}

	var mWords nat
	if m != nil {
		mWords = m.abs
	}

	z.abs = z.abs.expNN(x.abs, y.abs, mWords)
	z.neg = len(z.abs) > 0 && x.neg && y.abs[0]&1 == 1 // 0 has no sign
	return z
}


// GcdInt sets d to the greatest common divisor of a and b, which must be
// positive numbers.
// If x and y are not nil, GcdInt sets x and y such that d = a*x + b*y.
// If either a or b is not positive, GcdInt sets d = x = y = 0.
func GcdInt(d, x, y, a, b *Int) {
	if a.neg || b.neg {
		d.SetInt64(0)
		if x != nil {
			x.SetInt64(0)
		}
		if y != nil {
			y.SetInt64(0)
		}
		return
	}

	A := new(Int).Set(a)
	B := new(Int).Set(b)

	X := new(Int)
	Y := new(Int).SetInt64(1)

	lastX := new(Int).SetInt64(1)
	lastY := new(Int)

	q := new(Int)
	temp := new(Int)

	for len(B.abs) > 0 {
		r := new(Int)
		q, r = q.QuoRem(A, B, r)

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
func ProbablyPrime(z *Int, n int) bool {
	return !z.neg && z.abs.probablyPrime(n)
}


// Lsh sets z = x << n and returns z.
func (z *Int) Lsh(x *Int, n uint) *Int {
	z.abs = z.abs.shl(x.abs, n)
	z.neg = x.neg
	return z
}


// Rsh sets z = x >> n and returns z.
func (z *Int) Rsh(x *Int, n uint) *Int {
	if x.neg {
		// (-x) >> s == ^(x-1) >> s == ^((x-1) >> s) == -(((x-1) >> s) + 1)
		t := z.abs.sub(x.abs, natOne) // no underflow because |x| > 0
		t = t.shr(t, n)
		z.abs = t.add(t, natOne)
		z.neg = true // z cannot be zero if x is negative
		return z
	}

	z.abs = z.abs.shr(x.abs, n)
	z.neg = false
	return z
}


// And sets z = x & y and returns z.
func (z *Int) And(x, y *Int) *Int {
	if x.neg == y.neg {
		if x.neg {
			// (-x) & (-y) == ^(x-1) & ^(y-1) == ^((x-1) | (y-1)) == -(((x-1) | (y-1)) + 1)
			x1 := nat{}.sub(x.abs, natOne)
			y1 := z.abs.sub(y.abs, natOne)
			z.abs = z.abs.add(z.abs.or(x1, y1), natOne)
			z.neg = true // z cannot be zero if x and y are negative
			return z
		}

		// x & y == x & y
		z.abs = z.abs.and(x.abs, y.abs)
		z.neg = false
		return z
	}

	// x.neg != y.neg
	if x.neg {
		x, y = y, x // & is symmetric
	}

	// x & (-y) == x & ^(y-1) == x &^ (y-1)
	y1 := z.abs.sub(y.abs, natOne)
	z.abs = z.abs.andNot(x.abs, y1)
	z.neg = false
	return z
}


// AndNot sets z = x &^ y and returns z.
func (z *Int) AndNot(x, y *Int) *Int {
	if x.neg == y.neg {
		if x.neg {
			// (-x) &^ (-y) == ^(x-1) &^ ^(y-1) == ^(x-1) & (y-1) == (y-1) &^ (x-1)
			x1 := nat{}.sub(x.abs, natOne)
			y1 := z.abs.sub(y.abs, natOne)
			z.abs = z.abs.andNot(y1, x1)
			z.neg = false
			return z
		}

		// x &^ y == x &^ y
		z.abs = z.abs.andNot(x.abs, y.abs)
		z.neg = false
		return z
	}

	if x.neg {
		// (-x) &^ y == ^(x-1) &^ y == ^(x-1) & ^y == ^((x-1) | y) == -(((x-1) | y) + 1)
		x1 := z.abs.sub(x.abs, natOne)
		z.abs = z.abs.add(z.abs.or(x1, y.abs), natOne)
		z.neg = true // z cannot be zero if x is negative and y is positive
		return z
	}

	// x &^ (-y) == x &^ ^(y-1) == x & (y-1)
	y1 := z.abs.add(y.abs, natOne)
	z.abs = z.abs.and(x.abs, y1)
	z.neg = false
	return z
}


// Or sets z = x | y and returns z.
func (z *Int) Or(x, y *Int) *Int {
	if x.neg == y.neg {
		if x.neg {
			// (-x) | (-y) == ^(x-1) | ^(y-1) == ^((x-1) & (y-1)) == -(((x-1) & (y-1)) + 1)
			x1 := nat{}.sub(x.abs, natOne)
			y1 := z.abs.sub(y.abs, natOne)
			z.abs = z.abs.add(z.abs.and(x1, y1), natOne)
			z.neg = true // z cannot be zero if x and y are negative
			return z
		}

		// x | y == x | y
		z.abs = z.abs.or(x.abs, y.abs)
		z.neg = false
		return z
	}

	// x.neg != y.neg
	if x.neg {
		x, y = y, x // | is symmetric
	}

	// x | (-y) == x | ^(y-1) == ^((y-1) &^ x) == -(^((y-1) &^ x) + 1)
	y1 := z.abs.sub(y.abs, natOne)
	z.abs = z.abs.add(z.abs.andNot(y1, x.abs), natOne)
	z.neg = true // z cannot be zero if one of x or y is negative
	return z
}


// Xor sets z = x ^ y and returns z.
func (z *Int) Xor(x, y *Int) *Int {
	if x.neg == y.neg {
		if x.neg {
			// (-x) ^ (-y) == ^(x-1) ^ ^(y-1) == (x-1) ^ (y-1)
			x1 := nat{}.sub(x.abs, natOne)
			y1 := z.abs.sub(y.abs, natOne)
			z.abs = z.abs.xor(x1, y1)
			z.neg = false
			return z
		}

		// x ^ y == x ^ y
		z.abs = z.abs.xor(x.abs, y.abs)
		z.neg = false
		return z
	}

	// x.neg != y.neg
	if x.neg {
		x, y = y, x // ^ is symmetric
	}

	// x ^ (-y) == x ^ ^(y-1) == ^(x ^ (y-1)) == -((x ^ (y-1)) + 1)
	y1 := z.abs.sub(y.abs, natOne)
	z.abs = z.abs.add(z.abs.xor(x.abs, y1), natOne)
	z.neg = true // z cannot be zero if only one of x or y is negative
	return z
}


// Not sets z = ^x and returns z.
func (z *Int) Not(x *Int) *Int {
	if x.neg {
		// ^(-x) == ^(^(x-1)) == x-1
		z.abs = z.abs.sub(x.abs, natOne)
		z.neg = false
		return z
	}

	// ^x == -x-1 == -(x+1)
	z.abs = z.abs.add(x.abs, natOne)
	z.neg = true // z cannot be zero if x is positive
	return z
}
