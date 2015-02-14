// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements multi-precision floating-point numbers.
// Like in the GNU MPFR library (http://www.mpfr.org/), operands
// can be of mixed precision. Unlike MPFR, the rounding mode is
// not specified with each operation, but with each operand. The
// rounding mode of the result operand determines the rounding
// mode of an operation. This is a from-scratch implementation.

// CAUTION: WORK IN PROGRESS - USE AT YOUR OWN RISK.

package big

import (
	"fmt"
	"math"
)

const debugFloat = true // enable for debugging

// A Float represents a multi-precision floating point number of the form
//
//   sign × mantissa × 2**exponent
//
// with 0.5 <= mantissa < 1.0, and MinExp <= exponent <= MaxExp (with the
// exception of 0 and Inf which have a 0 mantissa and special exponents).
//
// Each Float value also has a precision, rounding mode, and accuracy.
//
// The precision is the maximum number of mantissa bits available to
// represent the value. The rounding mode specifies how a result should
// be rounded to fit into the mantissa bits, and accuracy describes the
// rounding error with respect to the exact result.
//
// All operations, including setters, that specify a *Float for the result,
// usually via the receiver, round their result to the result's precision
// and according to its rounding mode, unless specified otherwise. If the
// result precision is 0 (see below), it is set to the precision of the
// argument with the largest precision value before any rounding takes
// place. The rounding mode remains unchanged, thus uninitialized Floats
// provided as result arguments will "inherit" a reasonble precision from
// the incoming arguments and their mode is the zero value for RoundingMode
// (ToNearestEven).
//
// By setting the desired precision to 24 or 53 and using ToNearestEven
// rounding, Float operations produce the same results as the corresponding
// float32 or float64 IEEE-754 arithmetic for normalized operands (no NaNs
// or denormalized numbers). Additionally, positive and negative zeros and
// infinities are fully supported.
//
// The zero (uninitialized) value for a Float is ready to use and represents
// the number +0.0 exactly, with precision 0 and rounding mode ToNearestEven.
//
type Float struct {
	mode RoundingMode
	acc  Accuracy
	neg  bool
	mant nat
	exp  int32
	prec uint // TODO(gri) make this a 32bit field
}

// Internal representation: The mantissa bits x.mant of a Float x are stored
// in a nat slice long enough to hold up to x.prec bits; the slice may (but
// doesn't have to) be shorter if the mantissa contains trailing 0 bits.
// Unless x is a zero or an infinity, x.mant is normalized such that the
// msb of x.mant == 1 (i.e., the msb is shifted all the way "to the left").
// Thus, if the mantissa has trailing 0 bits or x.prec is not a multiple
// of the the Word size _W, x.mant[0] has trailing zero bits. Zero and Inf
// values have an empty mantissa and a 0 or infExp exponent, respectively.

const (
	MaxExp = math.MaxInt32 // largest supported exponent magnitude
	infExp = -MaxExp - 1   // exponent for Inf values
)

// NewInf returns a new infinite Float value with value +Inf (sign >= 0),
// or -Inf (sign < 0).
func NewInf(sign int) *Float {
	return &Float{neg: sign < 0, exp: infExp}
}

// Accuracy describes the rounding error produced by the most recent
// operation that generated a Float value, relative to the exact value:
//
//  -1: below exact value
//   0: exact value
//  +1: above exact value
//
type Accuracy int8

// Constants describing the Accuracy of a Float.
const (
	Below Accuracy = -1
	Exact Accuracy = 0
	Above Accuracy = +1
)

func (a Accuracy) String() string {
	switch {
	case a < 0:
		return "below"
	default:
		return "exact"
	case a > 0:
		return "above"
	}
}

// RoundingMode determines how a Float value is rounded to the
// desired precision. Rounding may change the Float value; the
// rounding error is described by the Float's Accuracy.
type RoundingMode uint8

// The following rounding modes are supported.
const (
	ToNearestEven RoundingMode = iota // == IEEE 754-2008 roundTiesToEven
	ToNearestAway                     // == IEEE 754-2008 roundTiesToAway
	ToZero                            // == IEEE 754-2008 roundTowardZero
	AwayFromZero                      // no IEEE 754-2008 equivalent
	ToNegativeInf                     // == IEEE 754-2008 roundTowardNegative
	ToPositiveInf                     // == IEEE 754-2008 roundTowardPositive
)

func (mode RoundingMode) String() string {
	switch mode {
	case ToNearestEven:
		return "ToNearestEven"
	case ToNearestAway:
		return "ToNearestAway"
	case ToZero:
		return "ToZero"
	case AwayFromZero:
		return "AwayFromZero"
	case ToNegativeInf:
		return "ToNegativeInf"
	case ToPositiveInf:
		return "ToPositiveInf"
	}
	panic("unreachable")
}

// SetPrec sets z's precision to prec and returns the (possibly) rounded
// value of z. Rounding occurs according to z's rounding mode if the mantissa
// cannot be represented in prec bits without loss of precision.
func (z *Float) SetPrec(prec uint) *Float {
	old := z.prec
	z.acc = Exact
	z.prec = prec
	if prec < old {
		z.round(0)
	}
	return z
}

// SetMode sets z's rounding mode to mode and returns an exact z.
// z remains unchanged otherwise.
func (z *Float) SetMode(mode RoundingMode) *Float {
	z.acc = Exact
	z.mode = mode
	return z
}

// Prec returns the mantissa precision of x in bits.
// The result may be 0 for |x| == 0 or |x| == Inf.
func (x *Float) Prec() uint {
	return uint(x.prec)
}

// Acc returns the accuracy of x produced by the most recent operation.
func (x *Float) Acc() Accuracy {
	return x.acc
}

// Mode returns the rounding mode of x.
func (x *Float) Mode() RoundingMode {
	return x.mode
}

// Sign returns:
//
//	-1 if x <  0
//	 0 if x == 0 or x == -0
//	+1 if x >  0
//
func (x *Float) Sign() int {
	s := 0
	if len(x.mant) != 0 || x.exp == infExp {
		s = 1 // non-zero x
	}
	if x.neg {
		s = -s
	}
	return s
}

// MantExp breaks x into its mantissa and exponent components.
// It returns mant and exp satisfying x == mant × 2**exp, with
// the absolute value of mant satisfying 0.5 <= |mant| < 1.0.
// mant has the same precision and rounding mode as x.
//
// Special cases are:
//
//	(  ±0).MantExp() =   ±0, 0
//	(±Inf).MantExp() = ±Inf, 0
//
// MantExp does not modify x; the result mant is a new Float.
func (x *Float) MantExp() (mant *Float, exp int) {
	mant = new(Float).Copy(x)
	if x.exp != infExp {
		mant.exp = 0
		exp = int(x.exp)
	}
	return
}

// SetMantExp is the inverse of MantExp. It sets z to mant × 2**exp and
// and returns z. The result z has the same precision and rounding mode
// as mant.
//
// Special cases are:
//
//	z.SetMantExp(  ±0, exp) =   ±0
//	z.SetMantExp(±Inf, exp) = ±Inf
//
// The result is ±Inf if the magnitude of exp is > MaxExp.
func (z *Float) SetMantExp(mant *Float, exp int) *Float {
	z.Copy(mant)
	if len(z.mant) == 0 || z.exp == infExp {
		return z
	}
	z.setExp(int64(exp))
	return z
}

// IsInt reports whether x is an integer.
// ±Inf are not considered integers.
func (x *Float) IsInt() bool {
	if debugFloat {
		validate(x)
	}
	// pick off easy cases
	if x.exp <= 0 {
		// |x| < 1 || |x| == Inf
		return len(x.mant) == 0 && x.exp != infExp
	}
	// x.exp > 0
	return x.prec <= uint(x.exp) || x.minPrec() <= uint(x.exp) // not enough bits for fractional mantissa
}

// IsInf reports whether x is an infinity, according to sign.
// If sign > 0, IsInf reports whether x is positive infinity.
// If sign < 0, IsInf reports whether x is negative infinity.
// If sign == 0, IsInf reports whether x is either infinity.
func (x *Float) IsInf(sign int) bool {
	return x.exp == infExp && (sign == 0 || x.neg == (sign < 0))
}

// setExp sets the exponent for z.
// If the exponent's magnitude is too large, z becomes ±Inf.
func (z *Float) setExp(e int64) {
	if -MaxExp <= e && e <= MaxExp {
		if len(z.mant) == 0 {
			e = 0
		}
		z.exp = int32(e)
		return
	}
	// Inf
	z.mant = z.mant[:0]
	z.exp = infExp
}

// debugging support
func validate(args ...*Float) {
	for i, x := range args {
		const msb = 1 << (_W - 1)
		m := len(x.mant)
		if m == 0 {
			// 0.0 or Inf
			if x.exp != 0 && x.exp != infExp {
				panic(fmt.Sprintf("#%d: %empty matissa with invalid exponent %d", i, x.exp))
			}
			continue
		}
		if x.mant[m-1]&msb == 0 {
			panic(fmt.Sprintf("#%d: msb not set in last word %#x of %s", i, x.mant[m-1], x.Format('p', 0)))
		}
		if x.prec <= 0 {
			panic(fmt.Sprintf("#%d: invalid precision %d", i, x.prec))
		}
	}
}

// round rounds z according to z.mode to z.prec bits and sets z.acc accordingly.
// sbit must be 0 or 1 and summarizes any "sticky bit" information one might
// have before calling round. z's mantissa must be normalized (with the msb set)
// or empty.
func (z *Float) round(sbit uint) {
	if debugFloat {
		validate(z)
	}

	z.acc = Exact

	// handle zero and Inf
	m := uint(len(z.mant)) // present mantissa length in words
	if m == 0 {
		if z.exp != infExp {
			z.exp = 0
		}
		return
	}
	// m > 0 implies z.prec > 0 (checked by validate)

	bits := m * _W // present mantissa bits
	if bits <= z.prec {
		// mantissa fits => nothing to do
		return
	}
	// bits > z.prec

	n := (z.prec + (_W - 1)) / _W // mantissa length in words for desired precision

	// Rounding is based on two bits: the rounding bit (rbit) and the
	// sticky bit (sbit). The rbit is the bit immediately before the
	// z.prec leading mantissa bits (the "0.5"). The sbit is set if any
	// of the bits before the rbit are set (the "0.25", "0.125", etc.):
	//
	//   rbit  sbit  => "fractional part"
	//
	//   0     0        == 0
	//   0     1        >  0  , < 0.5
	//   1     0        == 0.5
	//   1     1        >  0.5, < 1.0

	// bits > z.prec: mantissa too large => round
	r := bits - z.prec - 1 // rounding bit position; r >= 0
	rbit := z.mant.bit(r)  // rounding bit
	if sbit == 0 {
		sbit = z.mant.sticky(r)
	}
	if debugFloat && sbit&^1 != 0 {
		panic(fmt.Sprintf("invalid sbit %#x", sbit))
	}

	// convert ToXInf rounding modes
	mode := z.mode
	switch mode {
	case ToNegativeInf:
		mode = ToZero
		if z.neg {
			mode = AwayFromZero
		}
	case ToPositiveInf:
		mode = AwayFromZero
		if z.neg {
			mode = ToZero
		}
	}

	// cut off extra words
	if m > n {
		copy(z.mant, z.mant[m-n:]) // move n last words to front
		z.mant = z.mant[:n]
	}

	// determine number of trailing zero bits t
	t := n*_W - z.prec // 0 <= t < _W
	lsb := Word(1) << t

	// make rounding decision
	// TODO(gri) This can be simplified (see roundBits in float_test.go).
	switch mode {
	case ToZero:
		// nothing to do
	case ToNearestEven, ToNearestAway:
		if rbit == 0 {
			// rounding bits == 0b0x
			mode = ToZero
		} else if sbit == 1 {
			// rounding bits == 0b11
			mode = AwayFromZero
		}
	case AwayFromZero:
		if rbit|sbit == 0 {
			mode = ToZero
		}
	default:
		// ToXInf modes have been converted to ToZero or AwayFromZero
		panic("unreachable")
	}

	// round and determine accuracy
	switch mode {
	case ToZero:
		if rbit|sbit != 0 {
			z.acc = Below
		}

	case ToNearestEven, ToNearestAway:
		if debugFloat && rbit != 1 {
			panic("internal error in rounding")
		}
		if mode == ToNearestEven && sbit == 0 && z.mant[0]&lsb == 0 {
			z.acc = Below
			break
		}
		// mode == ToNearestAway || sbit == 1 || z.mant[0]&lsb != 0
		fallthrough

	case AwayFromZero:
		// add 1 to mantissa
		if addVW(z.mant, z.mant, lsb) != 0 {
			// overflow => shift mantissa right by 1 and add msb
			shrVU(z.mant, z.mant, 1)
			z.mant[n-1] |= 1 << (_W - 1)
			// adjust exponent
			z.exp++
		}
		z.acc = Above
	}

	// zero out trailing bits in least-significant word
	z.mant[0] &^= lsb - 1

	// update accuracy
	if z.neg {
		z.acc = -z.acc
	}

	if debugFloat {
		validate(z)
	}

	return
}

// nlz returns the number of leading zero bits in x.
func nlz(x Word) uint {
	return _W - uint(bitLen(x))
}

func nlz64(x uint64) uint {
	// TODO(gri) this can be done more nicely
	if _W == 32 {
		if x>>32 == 0 {
			return 32 + nlz(Word(x))
		}
		return nlz(Word(x >> 32))
	}
	if _W == 64 {
		return nlz(Word(x))
	}
	panic("unreachable")
}

func (z *Float) setBits64(neg bool, x uint64) *Float {
	if z.prec == 0 {
		z.prec = 64
	}
	z.acc = Exact
	z.neg = neg
	if x == 0 {
		z.mant = z.mant[:0]
		z.exp = 0
		return z
	}
	// x != 0
	s := nlz64(x)
	z.mant = z.mant.setUint64(x << s)
	z.exp = int32(64 - s) // always fits
	if z.prec < 64 {
		z.round(0)
	}
	return z
}

// SetUint64 sets z to the (possibly rounded) value of x and returns z.
// If z's precision is 0, it is changed to 64 (and rounding will have
// no effect).
func (z *Float) SetUint64(x uint64) *Float {
	return z.setBits64(false, x)
}

// SetInt64 sets z to the (possibly rounded) value of x and returns z.
// If z's precision is 0, it is changed to 64 (and rounding will have
// no effect).
func (z *Float) SetInt64(x int64) *Float {
	u := x
	if u < 0 {
		u = -u
	}
	// We cannot simply call z.SetUint64(uint64(u)) and change
	// the sign afterwards because the sign affects rounding.
	return z.setBits64(x < 0, uint64(u))
}

// SetFloat64 sets z to the (possibly rounded) value of x and returns z.
// If z's precision is 0, it is changed to 53 (and rounding will have
// no effect).
// If x is denormalized or NaN, the result is unspecified.
// TODO(gri) should return nil in those cases
func (z *Float) SetFloat64(x float64) *Float {
	if z.prec == 0 {
		z.prec = 53
	}
	z.acc = Exact
	z.neg = math.Signbit(x) // handle -0 correctly
	if math.IsInf(x, 0) {
		z.mant = z.mant[:0]
		z.exp = infExp
		return z
	}
	if x == 0 {
		z.mant = z.mant[:0]
		z.exp = 0
		return z
	}
	// x != 0
	fmant, exp := math.Frexp(x) // get normalized mantissa
	z.mant = z.mant.setUint64(1<<63 | math.Float64bits(fmant)<<11)
	z.exp = int32(exp) // always fits
	if z.prec < 53 {
		z.round(0)
	}
	return z
}

// fnorm normalizes mantissa m by shifting it to the left
// such that the msb of the most-significant word (msw) is 1.
// It returns the shift amount. It assumes that len(m) != 0.
func fnorm(m nat) uint {
	if debugFloat && (len(m) == 0 || m[len(m)-1] == 0) {
		panic("msw of mantissa is 0")
	}
	s := nlz(m[len(m)-1])
	if s > 0 {
		c := shlVU(m, m, s)
		if debugFloat && c != 0 {
			panic("nlz or shlVU incorrect")
		}
	}
	return s
}

// SetInt sets z to the (possibly rounded) value of x and returns z.
// If z's precision is 0, it is changed to the larger of x.BitLen()
// or 64 (and rounding will have no effect).
func (z *Float) SetInt(x *Int) *Float {
	// TODO(gri) can be more efficient if z.prec > 0
	// but small compared to the size of x, or if there
	// are many trailing 0's.
	bits := uint(x.BitLen())
	if z.prec == 0 {
		z.prec = umax(bits, 64)
	}
	z.acc = Exact
	z.neg = x.neg
	if len(x.abs) == 0 {
		z.mant = z.mant[:0]
		z.exp = 0
		return z
	}
	// x != 0
	z.mant = z.mant.set(x.abs)
	fnorm(z.mant)
	z.setExp(int64(bits))
	if z.prec < bits {
		z.round(0)
	}
	return z
}

// SetRat sets z to the (possibly rounded) value of x and returns z.
// If z's precision is 0, it is changed to the largest of a.BitLen(),
// b.BitLen(), or 64; with x = a/b.
func (z *Float) SetRat(x *Rat) *Float {
	// TODO(gri) can be more efficient if x is an integer
	var a, b Float
	a.SetInt(x.Num())
	b.SetInt(x.Denom())
	if z.prec == 0 {
		z.prec = umax(a.prec, b.prec)
	}
	return z.Quo(&a, &b)
}

// Set sets z to the (possibly rounded) value of x and returns z.
// If z's precision is 0, it is changed to the precision of x
// before setting z (and rounding will have no effect).
// Rounding is performed according to z's precision and rounding
// mode; and z's accuracy reports the result error relative to the
// exact (not rounded) result.
func (z *Float) Set(x *Float) *Float {
	// TODO(gri) what about z.acc? should it be always Exact?
	if z != x {
		if z.prec == 0 {
			z.prec = x.prec
		}
		z.acc = Exact
		z.neg = x.neg
		z.exp = x.exp
		z.mant = z.mant.set(x.mant)
		if z.prec < x.prec {
			z.round(0)
		}
	}
	return z
}

// Copy sets z to x, with the same precision and rounding mode as x,
// and returns z.
func (z *Float) Copy(x *Float) *Float {
	// TODO(gri) what about z.acc? should it be always Exact?
	if z != x {
		z.acc = Exact
		z.neg = x.neg
		z.exp = x.exp
		z.mant = z.mant.set(x.mant)
		z.prec = x.prec
		z.mode = x.mode
	}
	return z
}

func high64(x nat) uint64 {
	i := len(x)
	if i == 0 {
		return 0
	}
	// i > 0
	v := uint64(x[i-1])
	if _W == 32 {
		v <<= 32
		if i > 1 {
			v |= uint64(x[i-2])
		}
	}
	return v
}

// minPrec returns the minimum precision required to represent
// x without loss of accuracy.
// TODO(gri) this might be useful to export, perhaps under a better name
func (x *Float) minPrec() uint {
	return uint(len(x.mant))*_W - x.mant.trailingZeroBits()
}

// Uint64 returns the unsigned integer resulting from truncating x
// towards zero. If 0 <= x <= math.MaxUint64, the result is Exact
// if x is an integer and Below otherwise.
// The result is (0, Above) for x < 0, and (math.MaxUint64, Below)
// for x > math.MaxUint64.
func (x *Float) Uint64() (uint64, Accuracy) {
	if debugFloat {
		validate(x)
	}
	switch x.ord() {
	case -2, -1:
		// x < 0
		return 0, Above
	case 0:
		// x == 0 || x == -0
		return 0, Exact
	case 1:
		// 0 < x < +Inf
		if x.exp <= 0 {
			// 0 < x < 1
			return 0, Below
		}
		// 1 <= x < +Inf
		if x.exp <= 64 {
			// u = trunc(x) fits into a uint64
			u := high64(x.mant) >> (64 - uint32(x.exp))
			if x.minPrec() <= 64 {
				return u, Exact
			}
			return u, Below // x truncated
		}
		fallthrough // x too large
	case 2:
		// x == +Inf
		return math.MaxUint64, Below
	}
	panic("unreachable")
}

// Int64 returns the integer resulting from truncating x towards zero.
// If math.MinInt64 <= x <= math.MaxInt64, the result is Exact if x is
// an integer, and Above (x < 0) or Below (x > 0) otherwise.
// The result is (math.MinInt64, Above) for x < math.MinInt64, and
// (math.MaxInt64, Below) for x > math.MaxInt64.
func (x *Float) Int64() (int64, Accuracy) {
	if debugFloat {
		validate(x)
	}

	switch x.ord() {
	case -2:
		// x == -Inf
		return math.MinInt64, Above
	case 0:
		// x == 0 || x == -0
		return 0, Exact
	case -1, 1:
		// 0 < |x| < +Inf
		acc := Below
		if x.neg {
			acc = Above
		}
		if x.exp <= 0 {
			// 0 < |x| < 1
			return 0, acc
		}
		// 1 <= |x| < +Inf
		if x.exp <= 63 {
			// i = trunc(x) fits into an int64 (excluding math.MinInt64)
			i := int64(high64(x.mant) >> (64 - uint32(x.exp)))
			if x.neg {
				i = -i
			}
			if x.minPrec() <= 63 {
				return i, Exact
			}
			return i, acc // x truncated
		}
		if x.neg {
			// check for special case x == math.MinInt64 (i.e., x == -(0.5 << 64))
			if x.exp == 64 && x.minPrec() == 1 {
				acc = Exact
			}
			return math.MinInt64, acc
		}
		fallthrough
	case 2:
		// x == +Inf
		return math.MaxInt64, Below
	}
	panic("unreachable")
}

// Float64 returns the closest float64 value of x
// by rounding to nearest with 53 bits precision.
// TODO(gri) implement/document error scenarios.
func (x *Float) Float64() (float64, Accuracy) {
	// x == ±Inf
	if x.exp == infExp {
		var sign int
		if x.neg {
			sign = -1
		}
		return math.Inf(sign), Exact
	}
	// x == 0
	if len(x.mant) == 0 {
		return 0, Exact
	}
	// x != 0
	var r Float
	r.prec = 53
	r.Set(x)
	var s uint64
	if r.neg {
		s = 1 << 63
	}
	e := uint64(1022+r.exp) & 0x7ff // TODO(gri) check for overflow
	m := high64(r.mant) >> 11 & (1<<52 - 1)
	return math.Float64frombits(s | e<<52 | m), r.acc
}

// Int returns the result of truncating x towards zero; or nil
// if x is an infinity. The result is Exact if x.IsInt();
// otherwise it is Below for x > 0, and Above for x < 0.
func (x *Float) Int() (res *Int, acc Accuracy) {
	if debugFloat {
		validate(x)
	}
	// accuracy for inexact results
	acc = Below // truncation
	if x.neg {
		acc = Above
	}
	// pick off easy cases
	if x.exp <= 0 {
		// |x| < 1 || |x| == Inf
		if x.exp == infExp {
			return nil, acc // ±Inf
		}
		if len(x.mant) == 0 {
			acc = Exact // ±0
		}
		return new(Int), acc // ±0.xxx
	}
	// x.exp > 0
	// x.mant[len(x.mant)-1] != 0
	// determine minimum required precision for x
	allBits := uint(len(x.mant)) * _W
	exp := uint(x.exp)
	if x.minPrec() <= exp {
		acc = Exact
	}
	// shift mantissa as needed
	res = &Int{neg: x.neg}
	// TODO(gri) should have a shift that takes positive and negative shift counts
	switch {
	case exp > allBits:
		res.abs = res.abs.shl(x.mant, exp-allBits)
	default:
		res.abs = res.abs.set(x.mant)
	case exp < allBits:
		res.abs = res.abs.shr(x.mant, allBits-exp)
	}
	return
}

// BUG(gri) Rat is not yet implemented
func (x *Float) Rat() *Rat {
	panic("unimplemented")
}

// Abs sets z to the (possibly rounded) value |x| (the absolute value of x)
// and returns z.
func (z *Float) Abs(x *Float) *Float {
	z.Set(x)
	z.neg = false
	return z
}

// Neg sets z to the (possibly rounded) value of x with its sign negated,
// and returns z.
func (z *Float) Neg(x *Float) *Float {
	z.Set(x)
	z.neg = !z.neg
	return z
}

// z = x + y, ignoring signs of x and y.
// x and y must not be 0 or an Inf.
func (z *Float) uadd(x, y *Float) {
	// Note: This implementation requires 2 shifts most of the
	// time. It is also inefficient if exponents or precisions
	// differ by wide margins. The following article describes
	// an efficient (but much more complicated) implementation
	// compatible with the internal representation used here:
	//
	// Vincent Lefèvre: "The Generic Multiple-Precision Floating-
	// Point Addition With Exact Rounding (as in the MPFR Library)"
	// http://www.vinc17.net/research/papers/rnc6.pdf

	if debugFloat && (len(x.mant) == 0 || len(y.mant) == 0) {
		panic("uadd called with 0 argument")
	}

	// compute exponents ex, ey for mantissa with "binary point"
	// on the right (mantissa.0) - use int64 to avoid overflow
	ex := int64(x.exp) - int64(len(x.mant))*_W
	ey := int64(y.exp) - int64(len(y.mant))*_W

	// TODO(gri) having a combined add-and-shift primitive
	//           could make this code significantly faster
	switch {
	case ex < ey:
		// cannot re-use z.mant w/o testing for aliasing
		t := nat(nil).shl(y.mant, uint(ey-ex))
		z.mant = z.mant.add(x.mant, t)
	default:
		// ex == ey, no shift needed
		z.mant = z.mant.add(x.mant, y.mant)
	case ex > ey:
		// cannot re-use z.mant w/o testing for aliasing
		t := nat(nil).shl(x.mant, uint(ex-ey))
		z.mant = z.mant.add(t, y.mant)
		ex = ey
	}
	// len(z.mant) > 0

	z.setExp(ex + int64(len(z.mant))*_W - int64(fnorm(z.mant)))
	z.round(0)
}

// z = x - y for x >= y, ignoring signs of x and y.
// x and y must not be 0 or an Inf.
func (z *Float) usub(x, y *Float) {
	// This code is symmetric to uadd.
	// We have not factored the common code out because
	// eventually uadd (and usub) should be optimized
	// by special-casing, and the code will diverge.

	if debugFloat && (len(x.mant) == 0 || len(y.mant) == 0) {
		panic("usub called with 0 argument")
	}

	ex := int64(x.exp) - int64(len(x.mant))*_W
	ey := int64(y.exp) - int64(len(y.mant))*_W

	switch {
	case ex < ey:
		// cannot re-use z.mant w/o testing for aliasing
		t := nat(nil).shl(y.mant, uint(ey-ex))
		z.mant = t.sub(x.mant, t)
	default:
		// ex == ey, no shift needed
		z.mant = z.mant.sub(x.mant, y.mant)
	case ex > ey:
		// cannot re-use z.mant w/o testing for aliasing
		t := nat(nil).shl(x.mant, uint(ex-ey))
		z.mant = t.sub(t, y.mant)
		ex = ey
	}

	// operands may have cancelled each other out
	if len(z.mant) == 0 {
		z.acc = Exact
		z.setExp(0)
		return
	}
	// len(z.mant) > 0

	z.setExp(ex + int64(len(z.mant))*_W - int64(fnorm(z.mant)))
	z.round(0)
}

// z = x * y, ignoring signs of x and y.
// x and y must not be 0 or an Inf.
func (z *Float) umul(x, y *Float) {
	if debugFloat && (len(x.mant) == 0 || len(y.mant) == 0) {
		panic("umul called with 0 argument")
	}

	// Note: This is doing too much work if the precision
	// of z is less than the sum of the precisions of x
	// and y which is often the case (e.g., if all floats
	// have the same precision).
	// TODO(gri) Optimize this for the common case.

	e := int64(x.exp) + int64(y.exp)
	z.mant = z.mant.mul(x.mant, y.mant)

	// normalize mantissa
	z.setExp(e - int64(fnorm(z.mant)))
	z.round(0)
}

// z = x / y, ignoring signs of x and y.
// x and y must not be 0 or an Inf.
func (z *Float) uquo(x, y *Float) {
	if debugFloat && (len(x.mant) == 0 || len(y.mant) == 0) {
		panic("uquo called with 0 argument")
	}

	// mantissa length in words for desired result precision + 1
	// (at least one extra bit so we get the rounding bit after
	// the division)
	n := int(z.prec/_W) + 1

	// compute adjusted x.mant such that we get enough result precision
	xadj := x.mant
	if d := n - len(x.mant) + len(y.mant); d > 0 {
		// d extra words needed => add d "0 digits" to x
		xadj = make(nat, len(x.mant)+d)
		copy(xadj[d:], x.mant)
	}
	// TODO(gri): If we have too many digits (d < 0), we should be able
	// to shorten x for faster division. But we must be extra careful
	// with rounding in that case.

	// Compute d before division since there may be aliasing of x.mant
	// (via xadj) or y.mant with z.mant.
	d := len(xadj) - len(y.mant)

	// divide
	var r nat
	z.mant, r = z.mant.div(nil, xadj, y.mant)

	// determine exponent
	e := int64(x.exp) - int64(y.exp) - int64(d-len(z.mant))*_W

	// normalize mantissa
	z.setExp(e - int64(fnorm(z.mant)))

	// The result is long enough to include (at least) the rounding bit.
	// If there's a non-zero remainder, the corresponding fractional part
	// (if it were computed), would have a non-zero sticky bit (if it were
	// zero, it couldn't have a non-zero remainder).
	var sbit uint
	if len(r) > 0 {
		sbit = 1
	}
	z.round(sbit)
}

// ucmp returns -1, 0, or 1, depending on whether x < y, x == y, or x > y,
// while ignoring the signs of x and y. x and y must not be 0 or an Inf.
func (x *Float) ucmp(y *Float) int {
	if debugFloat && (len(x.mant) == 0 || len(y.mant) == 0) {
		panic("ucmp called with 0 argument")
	}

	switch {
	case x.exp < y.exp:
		return -1
	case x.exp > y.exp:
		return 1
	}
	// x.exp == y.exp

	// compare mantissas
	i := len(x.mant)
	j := len(y.mant)
	for i > 0 || j > 0 {
		var xm, ym Word
		if i > 0 {
			i--
			xm = x.mant[i]
		}
		if j > 0 {
			j--
			ym = y.mant[j]
		}
		switch {
		case xm < ym:
			return -1
		case xm > ym:
			return 1
		}
	}

	return 0
}

// Handling of sign bit as defined by IEEE 754-2008,
// section 6.3 (note that there are no NaN Floats):
//
// When neither the inputs nor result are NaN, the sign of a product or
// quotient is the exclusive OR of the operands’ signs; the sign of a sum,
// or of a difference x−y regarded as a sum x+(−y), differs from at most
// one of the addends’ signs; and the sign of the result of conversions,
// the quantize operation, the roundToIntegral operations, and the
// roundToIntegralExact (see 5.3.1) is the sign of the first or only operand.
// These rules shall apply even when operands or results are zero or infinite.
//
// When the sum of two operands with opposite signs (or the difference of
// two operands with like signs) is exactly zero, the sign of that sum (or
// difference) shall be +0 in all rounding-direction attributes except
// roundTowardNegative; under that attribute, the sign of an exact zero
// sum (or difference) shall be −0. However, x+x = x−(−x) retains the same
// sign as x even when x is zero.

// Add sets z to the rounded sum x+y and returns z.
// If z's precision is 0, it is changed to the larger
// of x's or y's precision before the operation.
// Rounding is performed according to z's precision
// and rounding mode; and z's accuracy reports the
// result error relative to the exact (not rounded)
// result.
func (z *Float) Add(x, y *Float) *Float {
	if debugFloat {
		validate(x, y)
	}

	if z.prec == 0 {
		z.prec = umax(x.prec, y.prec)
	}

	// TODO(gri) what about -0?
	if len(y.mant) == 0 {
		// TODO(gri) handle Inf
		return z.Set(x)
	}
	if len(x.mant) == 0 {
		// TODO(gri) handle Inf
		return z.Set(y)
	}

	// x, y != 0
	neg := x.neg
	if x.neg == y.neg {
		// x + y == x + y
		// (-x) + (-y) == -(x + y)
		z.uadd(x, y)
	} else {
		// x + (-y) == x - y == -(y - x)
		// (-x) + y == y - x == -(x - y)
		if x.ucmp(y) >= 0 {
			z.usub(x, y)
		} else {
			neg = !neg
			z.usub(y, x)
		}
	}
	z.neg = neg
	return z
}

// Sub sets z to the rounded difference x-y and returns z.
// Precision, rounding, and accuracy reporting are as for Add.
func (z *Float) Sub(x, y *Float) *Float {
	if debugFloat {
		validate(x, y)
	}

	if z.prec == 0 {
		z.prec = umax(x.prec, y.prec)
	}

	// TODO(gri) what about -0?
	if len(y.mant) == 0 {
		// TODO(gri) handle Inf
		return z.Set(x)
	}
	if len(x.mant) == 0 {
		return z.Neg(y)
	}

	// x, y != 0
	neg := x.neg
	if x.neg != y.neg {
		// x - (-y) == x + y
		// (-x) - y == -(x + y)
		z.uadd(x, y)
	} else {
		// x - y == x - y == -(y - x)
		// (-x) - (-y) == y - x == -(x - y)
		if x.ucmp(y) >= 0 {
			z.usub(x, y)
		} else {
			neg = !neg
			z.usub(y, x)
		}
	}
	z.neg = neg
	return z
}

// Mul sets z to the rounded product x*y and returns z.
// Precision, rounding, and accuracy reporting are as for Add.
func (z *Float) Mul(x, y *Float) *Float {
	if debugFloat {
		validate(x, y)
	}

	if z.prec == 0 {
		z.prec = umax(x.prec, y.prec)
	}

	// TODO(gri) handle Inf

	// TODO(gri) what about -0?
	if len(x.mant) == 0 || len(y.mant) == 0 {
		z.neg = false
		z.mant = z.mant[:0]
		z.exp = 0
		z.acc = Exact
		return z
	}

	// x, y != 0
	z.umul(x, y)
	z.neg = x.neg != y.neg
	return z
}

// Quo sets z to the rounded quotient x/y and returns z.
// Precision, rounding, and accuracy reporting are as for Add.
func (z *Float) Quo(x, y *Float) *Float {
	if debugFloat {
		validate(x, y)
	}

	if z.prec == 0 {
		z.prec = umax(x.prec, y.prec)
	}

	// TODO(gri) handle Inf

	// TODO(gri) check that this is correct
	z.neg = x.neg != y.neg

	if len(y.mant) == 0 {
		z.setExp(infExp)
		return z
	}

	if len(x.mant) == 0 {
		z.mant = z.mant[:0]
		z.exp = 0
		z.acc = Exact
		return z
	}

	// x, y != 0
	z.uquo(x, y)
	return z
}

// Lsh sets z to the rounded x * (1<<s) and returns z.
// If z's precision is 0, it is changed to x's precision.
// Rounding is performed according to z's precision
// and rounding mode; and z's accuracy reports the
// result error relative to the exact (not rounded)
// result.
// BUG(gri) Lsh is not tested and may not work correctly.
func (z *Float) Lsh(x *Float, s uint) *Float {
	if debugFloat {
		validate(x)
	}

	if z.prec == 0 {
		z.prec = x.prec
	}

	// TODO(gri) handle Inf

	z.round(0)
	z.setExp(int64(z.exp) + int64(s))
	return z
}

// Rsh sets z to the rounded x / (1<<s) and returns z.
// Precision, rounding, and accuracy reporting are as for Lsh.
// BUG(gri) Rsh is not tested and may not work correctly.
func (z *Float) Rsh(x *Float, s uint) *Float {
	if debugFloat {
		validate(x)
	}

	if z.prec == 0 {
		z.prec = x.prec
	}

	// TODO(gri) handle Inf

	z.round(0)
	z.setExp(int64(z.exp) - int64(s))
	return z
}

// Cmp compares x and y and returns:
//
//   -1 if x <  y
//    0 if x == y (incl. -0 == 0)
//   +1 if x >  y
//
// Infinities with matching sign are equal.
func (x *Float) Cmp(y *Float) int {
	if debugFloat {
		validate(x, y)
	}

	mx := x.ord()
	my := y.ord()
	switch {
	case mx < my:
		return -1
	case mx > my:
		return +1
	}
	// mx == my

	// only if |mx| == 1 we have to compare the mantissae
	switch mx {
	case -1:
		return -x.ucmp(y)
	case +1:
		return +x.ucmp(y)
	}

	return 0
}

func umax(x, y uint) uint {
	if x > y {
		return x
	}
	return y
}

// ord classifies x and returns:
//
//	-2 if -Inf == x
//	-1 if -Inf < x < 0
//	 0 if x == 0 (signed or unsigned)
//	+1 if 0 < x < +Inf
//	+2 if x == +Inf
//
// TODO(gri) export (and remove IsInf)?
func (x *Float) ord() int {
	m := 1 // common case
	if len(x.mant) == 0 {
		m = 0
		if x.exp == infExp {
			m = 2
		}
	}
	if x.neg {
		m = -m
	}
	return m
}
