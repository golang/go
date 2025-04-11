// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements multi-precision rational numbers.

package big

import (
	"fmt"
	"math"
)

// A Rat represents a quotient a/b of arbitrary precision.
// The zero value for a Rat represents the value 0.
//
// Operations always take pointer arguments (*Rat) rather
// than Rat values, and each unique Rat value requires
// its own unique *Rat pointer. To "copy" a Rat value,
// an existing (or newly allocated) Rat must be set to
// a new value using the [Rat.Set] method; shallow copies
// of Rats are not supported and may lead to errors.
type Rat struct {
	// To make zero values for Rat work w/o initialization,
	// a zero value of b (len(b) == 0) acts like b == 1. At
	// the earliest opportunity (when an assignment to the Rat
	// is made), such uninitialized denominators are set to 1.
	// a.neg determines the sign of the Rat, b.neg is ignored.
	a, b Int
}

// NewRat creates a new [Rat] with numerator a and denominator b.
func NewRat(a, b int64) *Rat {
	return new(Rat).SetFrac64(a, b)
}

// SetFloat64 sets z to exactly f and returns z.
// If f is not finite, SetFloat returns nil.
func (z *Rat) SetFloat64(f float64) *Rat {
	const expMask = 1<<11 - 1
	bits := math.Float64bits(f)
	mantissa := bits & (1<<52 - 1)
	exp := int((bits >> 52) & expMask)
	switch exp {
	case expMask: // non-finite
		return nil
	case 0: // denormal
		exp -= 1022
	default: // normal
		mantissa |= 1 << 52
		exp -= 1023
	}

	shift := 52 - exp

	// Optimization (?): partially pre-normalise.
	for mantissa&1 == 0 && shift > 0 {
		mantissa >>= 1
		shift--
	}

	z.a.SetUint64(mantissa)
	z.a.neg = f < 0
	z.b.Set(intOne)
	if shift > 0 {
		z.b.Lsh(&z.b, uint(shift))
	} else {
		z.a.Lsh(&z.a, uint(-shift))
	}
	return z.norm()
}

// quotToFloat32 returns the non-negative float32 value
// nearest to the quotient a/b, using round-to-even in
// halfway cases. It does not mutate its arguments.
// Preconditions: b is non-zero; a and b have no common factors.
func quotToFloat32(stk *stack, a, b nat) (f float32, exact bool) {
	const (
		// float size in bits
		Fsize = 32

		// mantissa
		Msize  = 23
		Msize1 = Msize + 1 // incl. implicit 1
		Msize2 = Msize1 + 1

		// exponent
		Esize = Fsize - Msize1
		Ebias = 1<<(Esize-1) - 1
		Emin  = 1 - Ebias
		Emax  = Ebias
	)

	// TODO(adonovan): specialize common degenerate cases: 1.0, integers.
	alen := a.bitLen()
	if alen == 0 {
		return 0, true
	}
	blen := b.bitLen()
	if blen == 0 {
		panic("division by zero")
	}

	// 1. Left-shift A or B such that quotient A/B is in [1<<Msize1, 1<<(Msize2+1)
	// (Msize2 bits if A < B when they are left-aligned, Msize2+1 bits if A >= B).
	// This is 2 or 3 more than the float32 mantissa field width of Msize:
	// - the optional extra bit is shifted away in step 3 below.
	// - the high-order 1 is omitted in "normal" representation;
	// - the low-order 1 will be used during rounding then discarded.
	exp := alen - blen
	var a2, b2 nat
	a2 = a2.set(a)
	b2 = b2.set(b)
	if shift := Msize2 - exp; shift > 0 {
		a2 = a2.lsh(a2, uint(shift))
	} else if shift < 0 {
		b2 = b2.lsh(b2, uint(-shift))
	}

	// 2. Compute quotient and remainder (q, r).  NB: due to the
	// extra shift, the low-order bit of q is logically the
	// high-order bit of r.
	var q nat
	q, r := q.div(stk, a2, a2, b2) // (recycle a2)
	mantissa := low32(q)
	haveRem := len(r) > 0 // mantissa&1 && !haveRem => remainder is exactly half

	// 3. If quotient didn't fit in Msize2 bits, redo division by b2<<1
	// (in effect---we accomplish this incrementally).
	if mantissa>>Msize2 == 1 {
		if mantissa&1 == 1 {
			haveRem = true
		}
		mantissa >>= 1
		exp++
	}
	if mantissa>>Msize1 != 1 {
		panic(fmt.Sprintf("expected exactly %d bits of result", Msize2))
	}

	// 4. Rounding.
	if Emin-Msize <= exp && exp <= Emin {
		// Denormal case; lose 'shift' bits of precision.
		shift := uint(Emin - (exp - 1)) // [1..Esize1)
		lostbits := mantissa & (1<<shift - 1)
		haveRem = haveRem || lostbits != 0
		mantissa >>= shift
		exp = 2 - Ebias // == exp + shift
	}
	// Round q using round-half-to-even.
	exact = !haveRem
	if mantissa&1 != 0 {
		exact = false
		if haveRem || mantissa&2 != 0 {
			if mantissa++; mantissa >= 1<<Msize2 {
				// Complete rollover 11...1 => 100...0, so shift is safe
				mantissa >>= 1
				exp++
			}
		}
	}
	mantissa >>= 1 // discard rounding bit.  Mantissa now scaled by 1<<Msize1.

	f = float32(math.Ldexp(float64(mantissa), exp-Msize1))
	if math.IsInf(float64(f), 0) {
		exact = false
	}
	return
}

// quotToFloat64 returns the non-negative float64 value
// nearest to the quotient a/b, using round-to-even in
// halfway cases. It does not mutate its arguments.
// Preconditions: b is non-zero; a and b have no common factors.
func quotToFloat64(stk *stack, a, b nat) (f float64, exact bool) {
	const (
		// float size in bits
		Fsize = 64

		// mantissa
		Msize  = 52
		Msize1 = Msize + 1 // incl. implicit 1
		Msize2 = Msize1 + 1

		// exponent
		Esize = Fsize - Msize1
		Ebias = 1<<(Esize-1) - 1
		Emin  = 1 - Ebias
		Emax  = Ebias
	)

	// TODO(adonovan): specialize common degenerate cases: 1.0, integers.
	alen := a.bitLen()
	if alen == 0 {
		return 0, true
	}
	blen := b.bitLen()
	if blen == 0 {
		panic("division by zero")
	}

	// 1. Left-shift A or B such that quotient A/B is in [1<<Msize1, 1<<(Msize2+1)
	// (Msize2 bits if A < B when they are left-aligned, Msize2+1 bits if A >= B).
	// This is 2 or 3 more than the float64 mantissa field width of Msize:
	// - the optional extra bit is shifted away in step 3 below.
	// - the high-order 1 is omitted in "normal" representation;
	// - the low-order 1 will be used during rounding then discarded.
	exp := alen - blen
	var a2, b2 nat
	a2 = a2.set(a)
	b2 = b2.set(b)
	if shift := Msize2 - exp; shift > 0 {
		a2 = a2.lsh(a2, uint(shift))
	} else if shift < 0 {
		b2 = b2.lsh(b2, uint(-shift))
	}

	// 2. Compute quotient and remainder (q, r).  NB: due to the
	// extra shift, the low-order bit of q is logically the
	// high-order bit of r.
	var q nat
	q, r := q.div(stk, a2, a2, b2) // (recycle a2)
	mantissa := low64(q)
	haveRem := len(r) > 0 // mantissa&1 && !haveRem => remainder is exactly half

	// 3. If quotient didn't fit in Msize2 bits, redo division by b2<<1
	// (in effect---we accomplish this incrementally).
	if mantissa>>Msize2 == 1 {
		if mantissa&1 == 1 {
			haveRem = true
		}
		mantissa >>= 1
		exp++
	}
	if mantissa>>Msize1 != 1 {
		panic(fmt.Sprintf("expected exactly %d bits of result", Msize2))
	}

	// 4. Rounding.
	if Emin-Msize <= exp && exp <= Emin {
		// Denormal case; lose 'shift' bits of precision.
		shift := uint(Emin - (exp - 1)) // [1..Esize1)
		lostbits := mantissa & (1<<shift - 1)
		haveRem = haveRem || lostbits != 0
		mantissa >>= shift
		exp = 2 - Ebias // == exp + shift
	}
	// Round q using round-half-to-even.
	exact = !haveRem
	if mantissa&1 != 0 {
		exact = false
		if haveRem || mantissa&2 != 0 {
			if mantissa++; mantissa >= 1<<Msize2 {
				// Complete rollover 11...1 => 100...0, so shift is safe
				mantissa >>= 1
				exp++
			}
		}
	}
	mantissa >>= 1 // discard rounding bit.  Mantissa now scaled by 1<<Msize1.

	f = math.Ldexp(float64(mantissa), exp-Msize1)
	if math.IsInf(f, 0) {
		exact = false
	}
	return
}

// Float32 returns the nearest float32 value for x and a bool indicating
// whether f represents x exactly. If the magnitude of x is too large to
// be represented by a float32, f is an infinity and exact is false.
// The sign of f always matches the sign of x, even if f == 0.
func (x *Rat) Float32() (f float32, exact bool) {
	b := x.b.abs
	if len(b) == 0 {
		b = natOne
	}
	stk := getStack()
	defer stk.free()
	f, exact = quotToFloat32(stk, x.a.abs, b)
	if x.a.neg {
		f = -f
	}
	return
}

// Float64 returns the nearest float64 value for x and a bool indicating
// whether f represents x exactly. If the magnitude of x is too large to
// be represented by a float64, f is an infinity and exact is false.
// The sign of f always matches the sign of x, even if f == 0.
func (x *Rat) Float64() (f float64, exact bool) {
	b := x.b.abs
	if len(b) == 0 {
		b = natOne
	}
	stk := getStack()
	defer stk.free()
	f, exact = quotToFloat64(stk, x.a.abs, b)
	if x.a.neg {
		f = -f
	}
	return
}

// SetFrac sets z to a/b and returns z.
// If b == 0, SetFrac panics.
func (z *Rat) SetFrac(a, b *Int) *Rat {
	z.a.neg = a.neg != b.neg
	babs := b.abs
	if len(babs) == 0 {
		panic("division by zero")
	}
	if &z.a == b || alias(z.a.abs, babs) {
		babs = nat(nil).set(babs) // make a copy
	}
	z.a.abs = z.a.abs.set(a.abs)
	z.b.abs = z.b.abs.set(babs)
	return z.norm()
}

// SetFrac64 sets z to a/b and returns z.
// If b == 0, SetFrac64 panics.
func (z *Rat) SetFrac64(a, b int64) *Rat {
	if b == 0 {
		panic("division by zero")
	}
	z.a.SetInt64(a)
	if b < 0 {
		b = -b
		z.a.neg = !z.a.neg
	}
	z.b.abs = z.b.abs.setUint64(uint64(b))
	return z.norm()
}

// SetInt sets z to x (by making a copy of x) and returns z.
func (z *Rat) SetInt(x *Int) *Rat {
	z.a.Set(x)
	z.b.abs = z.b.abs.setWord(1)
	return z
}

// SetInt64 sets z to x and returns z.
func (z *Rat) SetInt64(x int64) *Rat {
	z.a.SetInt64(x)
	z.b.abs = z.b.abs.setWord(1)
	return z
}

// SetUint64 sets z to x and returns z.
func (z *Rat) SetUint64(x uint64) *Rat {
	z.a.SetUint64(x)
	z.b.abs = z.b.abs.setWord(1)
	return z
}

// Set sets z to x (by making a copy of x) and returns z.
func (z *Rat) Set(x *Rat) *Rat {
	if z != x {
		z.a.Set(&x.a)
		z.b.Set(&x.b)
	}
	if len(z.b.abs) == 0 {
		z.b.abs = z.b.abs.setWord(1)
	}
	return z
}

// Abs sets z to |x| (the absolute value of x) and returns z.
func (z *Rat) Abs(x *Rat) *Rat {
	z.Set(x)
	z.a.neg = false
	return z
}

// Neg sets z to -x and returns z.
func (z *Rat) Neg(x *Rat) *Rat {
	z.Set(x)
	z.a.neg = len(z.a.abs) > 0 && !z.a.neg // 0 has no sign
	return z
}

// Inv sets z to 1/x and returns z.
// If x == 0, Inv panics.
func (z *Rat) Inv(x *Rat) *Rat {
	if len(x.a.abs) == 0 {
		panic("division by zero")
	}
	z.Set(x)
	z.a.abs, z.b.abs = z.b.abs, z.a.abs
	return z
}

// Sign returns:
//   - -1 if x < 0;
//   - 0 if x == 0;
//   - +1 if x > 0.
func (x *Rat) Sign() int {
	return x.a.Sign()
}

// IsInt reports whether the denominator of x is 1.
func (x *Rat) IsInt() bool {
	return len(x.b.abs) == 0 || x.b.abs.cmp(natOne) == 0
}

// Num returns the numerator of x; it may be <= 0.
// The result is a reference to x's numerator; it
// may change if a new value is assigned to x, and vice versa.
// The sign of the numerator corresponds to the sign of x.
func (x *Rat) Num() *Int {
	return &x.a
}

// Denom returns the denominator of x; it is always > 0.
// The result is a reference to x's denominator, unless
// x is an uninitialized (zero value) [Rat], in which case
// the result is a new [Int] of value 1. (To initialize x,
// any operation that sets x will do, including x.Set(x).)
// If the result is a reference to x's denominator it
// may change if a new value is assigned to x, and vice versa.
func (x *Rat) Denom() *Int {
	// Note that x.b.neg is guaranteed false.
	if len(x.b.abs) == 0 {
		// Note: If this proves problematic, we could
		//       panic instead and require the Rat to
		//       be explicitly initialized.
		return &Int{abs: nat{1}}
	}
	return &x.b
}

func (z *Rat) norm() *Rat {
	switch {
	case len(z.a.abs) == 0:
		// z == 0; normalize sign and denominator
		z.a.neg = false
		fallthrough
	case len(z.b.abs) == 0:
		// z is integer; normalize denominator
		z.b.abs = z.b.abs.setWord(1)
	default:
		// z is fraction; normalize numerator and denominator
		stk := getStack()
		defer stk.free()
		neg := z.a.neg
		z.a.neg = false
		z.b.neg = false
		if f := NewInt(0).lehmerGCD(nil, nil, &z.a, &z.b); f.Cmp(intOne) != 0 {
			z.a.abs, _ = z.a.abs.div(stk, nil, z.a.abs, f.abs)
			z.b.abs, _ = z.b.abs.div(stk, nil, z.b.abs, f.abs)
		}
		z.a.neg = neg
	}
	return z
}

// mulDenom sets z to the denominator product x*y (by taking into
// account that 0 values for x or y must be interpreted as 1) and
// returns z.
func mulDenom(stk *stack, z, x, y nat) nat {
	switch {
	case len(x) == 0 && len(y) == 0:
		return z.setWord(1)
	case len(x) == 0:
		return z.set(y)
	case len(y) == 0:
		return z.set(x)
	}
	return z.mul(stk, x, y)
}

// scaleDenom sets z to the product x*f.
// If f == 0 (zero value of denominator), z is set to (a copy of) x.
func (z *Int) scaleDenom(stk *stack, x *Int, f nat) {
	if len(f) == 0 {
		z.Set(x)
		return
	}
	z.abs = z.abs.mul(stk, x.abs, f)
	z.neg = x.neg
}

// Cmp compares x and y and returns:
//   - -1 if x < y;
//   - 0 if x == y;
//   - +1 if x > y.
func (x *Rat) Cmp(y *Rat) int {
	var a, b Int
	stk := getStack()
	defer stk.free()
	a.scaleDenom(stk, &x.a, y.b.abs)
	b.scaleDenom(stk, &y.a, x.b.abs)
	return a.Cmp(&b)
}

// Add sets z to the sum x+y and returns z.
func (z *Rat) Add(x, y *Rat) *Rat {
	stk := getStack()
	defer stk.free()

	var a1, a2 Int
	a1.scaleDenom(stk, &x.a, y.b.abs)
	a2.scaleDenom(stk, &y.a, x.b.abs)
	z.a.Add(&a1, &a2)
	z.b.abs = mulDenom(stk, z.b.abs, x.b.abs, y.b.abs)
	return z.norm()
}

// Sub sets z to the difference x-y and returns z.
func (z *Rat) Sub(x, y *Rat) *Rat {
	stk := getStack()
	defer stk.free()

	var a1, a2 Int
	a1.scaleDenom(stk, &x.a, y.b.abs)
	a2.scaleDenom(stk, &y.a, x.b.abs)
	z.a.Sub(&a1, &a2)
	z.b.abs = mulDenom(stk, z.b.abs, x.b.abs, y.b.abs)
	return z.norm()
}

// Mul sets z to the product x*y and returns z.
func (z *Rat) Mul(x, y *Rat) *Rat {
	stk := getStack()
	defer stk.free()

	if x == y {
		// a squared Rat is positive and can't be reduced (no need to call norm())
		z.a.neg = false
		z.a.abs = z.a.abs.sqr(stk, x.a.abs)
		if len(x.b.abs) == 0 {
			z.b.abs = z.b.abs.setWord(1)
		} else {
			z.b.abs = z.b.abs.sqr(stk, x.b.abs)
		}
		return z
	}

	z.a.mul(stk, &x.a, &y.a)
	z.b.abs = mulDenom(stk, z.b.abs, x.b.abs, y.b.abs)
	return z.norm()
}

// Quo sets z to the quotient x/y and returns z.
// If y == 0, Quo panics.
func (z *Rat) Quo(x, y *Rat) *Rat {
	stk := getStack()
	defer stk.free()

	if len(y.a.abs) == 0 {
		panic("division by zero")
	}
	var a, b Int
	a.scaleDenom(stk, &x.a, y.b.abs)
	b.scaleDenom(stk, &y.a, x.b.abs)
	z.a.abs = a.abs
	z.b.abs = b.abs
	z.a.neg = a.neg != b.neg
	return z.norm()
}
