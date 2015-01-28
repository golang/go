// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements multi-precision rational numbers.

package big

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

// A Rat represents a quotient a/b of arbitrary precision.
// The zero value for a Rat represents the value 0.
type Rat struct {
	// To make zero values for Rat work w/o initialization,
	// a zero value of b (len(b) == 0) acts like b == 1.
	// a.neg determines the sign of the Rat, b.neg is ignored.
	a, b Int
}

// NewRat creates a new Rat with numerator a and denominator b.
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
// halfway cases.  It does not mutate its arguments.
// Preconditions: b is non-zero; a and b have no common factors.
func quotToFloat32(a, b nat) (f float32, exact bool) {
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
		a2 = a2.shl(a2, uint(shift))
	} else if shift < 0 {
		b2 = b2.shl(b2, uint(-shift))
	}

	// 2. Compute quotient and remainder (q, r).  NB: due to the
	// extra shift, the low-order bit of q is logically the
	// high-order bit of r.
	var q nat
	q, r := q.div(a2, a2, b2) // (recycle a2)
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
// halfway cases.  It does not mutate its arguments.
// Preconditions: b is non-zero; a and b have no common factors.
func quotToFloat64(a, b nat) (f float64, exact bool) {
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
		a2 = a2.shl(a2, uint(shift))
	} else if shift < 0 {
		b2 = b2.shl(b2, uint(-shift))
	}

	// 2. Compute quotient and remainder (q, r).  NB: due to the
	// extra shift, the low-order bit of q is logically the
	// high-order bit of r.
	var q nat
	q, r := q.div(a2, a2, b2) // (recycle a2)
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
		b = b.set(natOne) // materialize denominator
	}
	f, exact = quotToFloat32(x.a.abs, b)
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
		b = b.set(natOne) // materialize denominator
	}
	f, exact = quotToFloat64(x.a.abs, b)
	if x.a.neg {
		f = -f
	}
	return
}

// SetFrac sets z to a/b and returns z.
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
func (z *Rat) SetFrac64(a, b int64) *Rat {
	z.a.SetInt64(a)
	if b == 0 {
		panic("division by zero")
	}
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
	z.b.abs = z.b.abs[:0]
	return z
}

// SetInt64 sets z to x and returns z.
func (z *Rat) SetInt64(x int64) *Rat {
	z.a.SetInt64(x)
	z.b.abs = z.b.abs[:0]
	return z
}

// Set sets z to x (by making a copy of x) and returns z.
func (z *Rat) Set(x *Rat) *Rat {
	if z != x {
		z.a.Set(&x.a)
		z.b.Set(&x.b)
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
func (z *Rat) Inv(x *Rat) *Rat {
	if len(x.a.abs) == 0 {
		panic("division by zero")
	}
	z.Set(x)
	a := z.b.abs
	if len(a) == 0 {
		a = a.set(natOne) // materialize numerator
	}
	b := z.a.abs
	if b.cmp(natOne) == 0 {
		b = b[:0] // normalize denominator
	}
	z.a.abs, z.b.abs = a, b // sign doesn't change
	return z
}

// Sign returns:
//
//	-1 if x <  0
//	 0 if x == 0
//	+1 if x >  0
//
func (x *Rat) Sign() int {
	return x.a.Sign()
}

// IsInt returns true if the denominator of x is 1.
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
// The result is a reference to x's denominator; it
// may change if a new value is assigned to x, and vice versa.
func (x *Rat) Denom() *Int {
	x.b.neg = false // the result is always >= 0
	if len(x.b.abs) == 0 {
		x.b.abs = x.b.abs.set(natOne) // materialize denominator
	}
	return &x.b
}

func (z *Rat) norm() *Rat {
	switch {
	case len(z.a.abs) == 0:
		// z == 0 - normalize sign and denominator
		z.a.neg = false
		z.b.abs = z.b.abs[:0]
	case len(z.b.abs) == 0:
		// z is normalized int - nothing to do
	case z.b.abs.cmp(natOne) == 0:
		// z is int - normalize denominator
		z.b.abs = z.b.abs[:0]
	default:
		neg := z.a.neg
		z.a.neg = false
		z.b.neg = false
		if f := NewInt(0).binaryGCD(&z.a, &z.b); f.Cmp(intOne) != 0 {
			z.a.abs, _ = z.a.abs.div(nil, z.a.abs, f.abs)
			z.b.abs, _ = z.b.abs.div(nil, z.b.abs, f.abs)
			if z.b.abs.cmp(natOne) == 0 {
				// z is int - normalize denominator
				z.b.abs = z.b.abs[:0]
			}
		}
		z.a.neg = neg
	}
	return z
}

// mulDenom sets z to the denominator product x*y (by taking into
// account that 0 values for x or y must be interpreted as 1) and
// returns z.
func mulDenom(z, x, y nat) nat {
	switch {
	case len(x) == 0:
		return z.set(y)
	case len(y) == 0:
		return z.set(x)
	}
	return z.mul(x, y)
}

// scaleDenom computes x*f.
// If f == 0 (zero value of denominator), the result is (a copy of) x.
func scaleDenom(x *Int, f nat) *Int {
	var z Int
	if len(f) == 0 {
		return z.Set(x)
	}
	z.abs = z.abs.mul(x.abs, f)
	z.neg = x.neg
	return &z
}

// Cmp compares x and y and returns:
//
//   -1 if x <  y
//    0 if x == y
//   +1 if x >  y
//
func (x *Rat) Cmp(y *Rat) int {
	return scaleDenom(&x.a, y.b.abs).Cmp(scaleDenom(&y.a, x.b.abs))
}

// Add sets z to the sum x+y and returns z.
func (z *Rat) Add(x, y *Rat) *Rat {
	a1 := scaleDenom(&x.a, y.b.abs)
	a2 := scaleDenom(&y.a, x.b.abs)
	z.a.Add(a1, a2)
	z.b.abs = mulDenom(z.b.abs, x.b.abs, y.b.abs)
	return z.norm()
}

// Sub sets z to the difference x-y and returns z.
func (z *Rat) Sub(x, y *Rat) *Rat {
	a1 := scaleDenom(&x.a, y.b.abs)
	a2 := scaleDenom(&y.a, x.b.abs)
	z.a.Sub(a1, a2)
	z.b.abs = mulDenom(z.b.abs, x.b.abs, y.b.abs)
	return z.norm()
}

// Mul sets z to the product x*y and returns z.
func (z *Rat) Mul(x, y *Rat) *Rat {
	z.a.Mul(&x.a, &y.a)
	z.b.abs = mulDenom(z.b.abs, x.b.abs, y.b.abs)
	return z.norm()
}

// Quo sets z to the quotient x/y and returns z.
// If y == 0, a division-by-zero run-time panic occurs.
func (z *Rat) Quo(x, y *Rat) *Rat {
	if len(y.a.abs) == 0 {
		panic("division by zero")
	}
	a := scaleDenom(&x.a, y.b.abs)
	b := scaleDenom(&y.a, x.b.abs)
	z.a.abs = a.abs
	z.b.abs = b.abs
	z.a.neg = a.neg != b.neg
	return z.norm()
}

func ratTok(ch rune) bool {
	return strings.IndexRune("+-/0123456789.eE", ch) >= 0
}

// Scan is a support routine for fmt.Scanner. It accepts the formats
// 'e', 'E', 'f', 'F', 'g', 'G', and 'v'. All formats are equivalent.
func (z *Rat) Scan(s fmt.ScanState, ch rune) error {
	tok, err := s.Token(true, ratTok)
	if err != nil {
		return err
	}
	if strings.IndexRune("efgEFGv", ch) < 0 {
		return errors.New("Rat.Scan: invalid verb")
	}
	if _, ok := z.SetString(string(tok)); !ok {
		return errors.New("Rat.Scan: invalid syntax")
	}
	return nil
}

// SetString sets z to the value of s and returns z and a boolean indicating
// success. s can be given as a fraction "a/b" or as a floating-point number
// optionally followed by an exponent. If the operation failed, the value of
// z is undefined but the returned value is nil.
func (z *Rat) SetString(s string) (*Rat, bool) {
	if len(s) == 0 {
		return nil, false
	}
	// len(s) > 0

	// parse fraction a/b, if any
	if sep := strings.Index(s, "/"); sep >= 0 {
		if _, ok := z.a.SetString(s[:sep], 10); !ok {
			return nil, false
		}
		s = s[sep+1:]
		var err error
		if z.b.abs, _, _, err = z.b.abs.scan(strings.NewReader(s), 10); err != nil {
			return nil, false
		}
		if len(z.b.abs) == 0 {
			return nil, false
		}
		return z.norm(), true
	}

	// parse floating-point number
	r := strings.NewReader(s)

	// sign
	neg, err := scanSign(r)
	if err != nil {
		return nil, false
	}

	// mantissa
	var ecorr int
	z.a.abs, _, ecorr, err = z.a.abs.scan(r, 1)
	if err != nil {
		return nil, false
	}

	// exponent
	var exp int64
	var ebase int
	exp, ebase, err = scanExponent(r)
	if ebase == 2 || err != nil {
		return nil, false
	}

	// there should be no unread characters left
	if _, err = r.ReadByte(); err != io.EOF {
		return nil, false
	}

	// correct exponent
	if ecorr < 0 {
		exp += int64(ecorr)
	}

	// compute exponent power
	expabs := exp
	if expabs < 0 {
		expabs = -expabs
	}
	powTen := nat(nil).expNN(natTen, nat(nil).setWord(Word(expabs)), nil)

	// complete fraction
	if exp < 0 {
		z.b.abs = powTen
		z.norm()
	} else {
		z.a.abs = z.a.abs.mul(z.a.abs, powTen)
		z.b.abs = z.b.abs[:0]
	}

	z.a.neg = neg && len(z.a.abs) > 0 // 0 has no sign

	return z, true
}

func scanExponent(r io.ByteScanner) (exp int64, base int, err error) {
	base = 10

	var ch byte
	if ch, err = r.ReadByte(); err != nil {
		if err == io.EOF {
			err = nil // no exponent; same as e0
		}
		return
	}

	switch ch {
	case 'e', 'E':
		// ok
	case 'p':
		base = 2
	default:
		r.UnreadByte()
		return // no exponent; same as e0
	}

	var neg bool
	if neg, err = scanSign(r); err != nil {
		return
	}

	var digits []byte
	if neg {
		digits = append(digits, '-')
	}

	// no need to use nat.scan for exponent digits
	// since we only care about int64 values - the
	// from-scratch scan is easy enough and faster
	for i := 0; ; i++ {
		if ch, err = r.ReadByte(); err != nil {
			if err != io.EOF || i == 0 {
				return
			}
			err = nil
			break // i > 0
		}
		if ch < '0' || '9' < ch {
			if i == 0 {
				r.UnreadByte()
				err = fmt.Errorf("invalid exponent (missing digits)")
				return
			}
			break // i > 0
		}
		digits = append(digits, byte(ch))
	}
	// i > 0 => we have at least one digit

	exp, err = strconv.ParseInt(string(digits), 10, 64)
	return
}

// String returns a string representation of x in the form "a/b" (even if b == 1).
func (x *Rat) String() string {
	s := "/1"
	if len(x.b.abs) != 0 {
		s = "/" + x.b.abs.decimalString()
	}
	return x.a.String() + s
}

// RatString returns a string representation of x in the form "a/b" if b != 1,
// and in the form "a" if b == 1.
func (x *Rat) RatString() string {
	if x.IsInt() {
		return x.a.String()
	}
	return x.String()
}

// FloatString returns a string representation of x in decimal form with prec
// digits of precision after the decimal point and the last digit rounded.
func (x *Rat) FloatString(prec int) string {
	if x.IsInt() {
		s := x.a.String()
		if prec > 0 {
			s += "." + strings.Repeat("0", prec)
		}
		return s
	}
	// x.b.abs != 0

	q, r := nat(nil).div(nat(nil), x.a.abs, x.b.abs)

	p := natOne
	if prec > 0 {
		p = nat(nil).expNN(natTen, nat(nil).setUint64(uint64(prec)), nil)
	}

	r = r.mul(r, p)
	r, r2 := r.div(nat(nil), r, x.b.abs)

	// see if we need to round up
	r2 = r2.add(r2, r2)
	if x.b.abs.cmp(r2) <= 0 {
		r = r.add(r, natOne)
		if r.cmp(p) >= 0 {
			q = nat(nil).add(q, natOne)
			r = nat(nil).sub(r, p)
		}
	}

	s := q.decimalString()
	if x.a.neg {
		s = "-" + s
	}

	if prec > 0 {
		rs := r.decimalString()
		leadingZeros := prec - len(rs)
		s += "." + strings.Repeat("0", leadingZeros) + rs
	}

	return s
}

// Gob codec version. Permits backward-compatible changes to the encoding.
const ratGobVersion byte = 1

// GobEncode implements the gob.GobEncoder interface.
func (x *Rat) GobEncode() ([]byte, error) {
	if x == nil {
		return nil, nil
	}
	buf := make([]byte, 1+4+(len(x.a.abs)+len(x.b.abs))*_S) // extra bytes for version and sign bit (1), and numerator length (4)
	i := x.b.abs.bytes(buf)
	j := x.a.abs.bytes(buf[:i])
	n := i - j
	if int(uint32(n)) != n {
		// this should never happen
		return nil, errors.New("Rat.GobEncode: numerator too large")
	}
	binary.BigEndian.PutUint32(buf[j-4:j], uint32(n))
	j -= 1 + 4
	b := ratGobVersion << 1 // make space for sign bit
	if x.a.neg {
		b |= 1
	}
	buf[j] = b
	return buf[j:], nil
}

// GobDecode implements the gob.GobDecoder interface.
func (z *Rat) GobDecode(buf []byte) error {
	if len(buf) == 0 {
		// Other side sent a nil or default value.
		*z = Rat{}
		return nil
	}
	b := buf[0]
	if b>>1 != ratGobVersion {
		return errors.New(fmt.Sprintf("Rat.GobDecode: encoding version %d not supported", b>>1))
	}
	const j = 1 + 4
	i := j + binary.BigEndian.Uint32(buf[j-4:j])
	z.a.neg = b&1 != 0
	z.a.abs = z.a.abs.setBytes(buf[j:i])
	z.b.abs = z.b.abs.setBytes(buf[i:])
	return nil
}

// MarshalText implements the encoding.TextMarshaler interface.
func (r *Rat) MarshalText() (text []byte, err error) {
	return []byte(r.RatString()), nil
}

// UnmarshalText implements the encoding.TextUnmarshaler interface.
func (r *Rat) UnmarshalText(text []byte) error {
	if _, ok := r.SetString(string(text)); !ok {
		return fmt.Errorf("math/big: cannot unmarshal %q into a *big.Rat", text)
	}
	return nil
}
