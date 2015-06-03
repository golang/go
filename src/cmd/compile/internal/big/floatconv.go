// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements string-to-Float conversion functions.

package big

import (
	"fmt"
	"io"
	"strings"
)

// SetString sets z to the value of s and returns z and a boolean indicating
// success. s must be a floating-point number of the same format as accepted
// by Parse, with base argument 0.
func (z *Float) SetString(s string) (*Float, bool) {
	if f, _, err := z.Parse(s, 0); err == nil {
		return f, true
	}
	return nil, false
}

// scan is like Parse but reads the longest possible prefix representing a valid
// floating point number from an io.ByteScanner rather than a string. It serves
// as the implementation of Parse. It does not recognize ±Inf and does not expect
// EOF at the end.
func (z *Float) scan(r io.ByteScanner, base int) (f *Float, b int, err error) {
	prec := z.prec
	if prec == 0 {
		prec = 64
	}

	// A reasonable value in case of an error.
	z.form = zero

	// sign
	z.neg, err = scanSign(r)
	if err != nil {
		return
	}

	// mantissa
	var fcount int // fractional digit count; valid if <= 0
	z.mant, b, fcount, err = z.mant.scan(r, base, true)
	if err != nil {
		return
	}

	// exponent
	var exp int64
	var ebase int
	exp, ebase, err = scanExponent(r, true)
	if err != nil {
		return
	}

	// special-case 0
	if len(z.mant) == 0 {
		z.prec = prec
		z.acc = Exact
		z.form = zero
		f = z
		return
	}
	// len(z.mant) > 0

	// The mantissa may have a decimal point (fcount <= 0) and there
	// may be a nonzero exponent exp. The decimal point amounts to a
	// division by b**(-fcount). An exponent means multiplication by
	// ebase**exp. Finally, mantissa normalization (shift left) requires
	// a correcting multiplication by 2**(-shiftcount). Multiplications
	// are commutative, so we can apply them in any order as long as there
	// is no loss of precision. We only have powers of 2 and 10; keep
	// track via separate exponents exp2 and exp10.

	// normalize mantissa and get initial binary exponent
	var exp2 = int64(len(z.mant))*_W - fnorm(z.mant)

	// determine binary or decimal exponent contribution of decimal point
	var exp10 int64
	if fcount < 0 {
		// The mantissa has a "decimal" point ddd.dddd; and
		// -fcount is the number of digits to the right of '.'.
		// Adjust relevant exponent accodingly.
		switch b {
		case 16:
			fcount *= 4 // hexadecimal digits are 4 bits each
			fallthrough
		case 2:
			exp2 += int64(fcount)
		default: // b == 10
			exp10 = int64(fcount)
		}
		// we don't need fcount anymore
	}

	// take actual exponent into account
	if ebase == 2 {
		exp2 += exp
	} else { // ebase == 10
		exp10 += exp
	}
	// we don't need exp anymore

	// apply 2**exp2
	if MinExp <= exp2 && exp2 <= MaxExp {
		z.prec = prec
		z.form = finite
		z.exp = int32(exp2)
		f = z
	} else {
		err = fmt.Errorf("exponent overflow")
		return
	}

	if exp10 == 0 {
		// no decimal exponent to consider
		z.round(0)
		return
	}
	// exp10 != 0

	// apply 10**exp10
	p := new(Float).SetPrec(z.Prec() + 64) // use more bits for p -- TODO(gri) what is the right number?
	if exp10 < 0 {
		z.uquo(z, p.pow10(-exp10))
	} else {
		z.umul(z, p.pow10(exp10))
	}

	return
}

// These powers of 10 can be represented exactly as a float64.
var pow10tab = [...]float64{
	1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
	1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
}

// pow10 sets z to 10**n and returns z.
// n must not be negative.
func (z *Float) pow10(n int64) *Float {
	if n < 0 {
		panic("pow10 called with negative argument")
	}

	const m = int64(len(pow10tab) - 1)
	if n <= m {
		return z.SetFloat64(pow10tab[n])
	}
	// n > m

	z.SetFloat64(pow10tab[m])
	n -= m

	// use more bits for f than for z
	// TODO(gri) what is the right number?
	f := new(Float).SetPrec(z.Prec() + 64).SetInt64(10)

	for n > 0 {
		if n&1 != 0 {
			z.Mul(z, f)
		}
		f.Mul(f, f)
		n >>= 1
	}

	return z
}

// Parse parses s which must contain a text representation of a floating-
// point number with a mantissa in the given conversion base (the exponent
// is always a decimal number), or a string representing an infinite value.
//
// It sets z to the (possibly rounded) value of the corresponding floating-
// point value, and returns z, the actual base b, and an error err, if any.
// If z's precision is 0, it is changed to 64 before rounding takes effect.
// The number must be of the form:
//
//	number   = [ sign ] [ prefix ] mantissa [ exponent ] | infinity .
//	sign     = "+" | "-" .
//      prefix   = "0" ( "x" | "X" | "b" | "B" ) .
//	mantissa = digits | digits "." [ digits ] | "." digits .
//	exponent = ( "E" | "e" | "p" ) [ sign ] digits .
//	digits   = digit { digit } .
//	digit    = "0" ... "9" | "a" ... "z" | "A" ... "Z" .
//      infinity = [ sign ] ( "inf" | "Inf" ) .
//
// The base argument must be 0, 2, 10, or 16. Providing an invalid base
// argument will lead to a run-time panic.
//
// For base 0, the number prefix determines the actual base: A prefix of
// "0x" or "0X" selects base 16, and a "0b" or "0B" prefix selects
// base 2; otherwise, the actual base is 10 and no prefix is accepted.
// The octal prefix "0" is not supported (a leading "0" is simply
// considered a "0").
//
// A "p" exponent indicates a binary (rather then decimal) exponent;
// for instance "0x1.fffffffffffffp1023" (using base 0) represents the
// maximum float64 value. For hexadecimal mantissae, the exponent must
// be binary, if present (an "e" or "E" exponent indicator cannot be
// distinguished from a mantissa digit).
//
// The returned *Float f is nil and the value of z is valid but not
// defined if an error is reported.
//
func (z *Float) Parse(s string, base int) (f *Float, b int, err error) {
	// scan doesn't handle ±Inf
	if len(s) == 3 && (s == "Inf" || s == "inf") {
		f = z.SetInf(false)
		return
	}
	if len(s) == 4 && (s[0] == '+' || s[0] == '-') && (s[1:] == "Inf" || s[1:] == "inf") {
		f = z.SetInf(s[0] == '-')
		return
	}

	r := strings.NewReader(s)
	if f, b, err = z.scan(r, base); err != nil {
		return
	}

	// entire string must have been consumed
	if ch, err2 := r.ReadByte(); err2 == nil {
		err = fmt.Errorf("expected end of string, found %q", ch)
	} else if err2 != io.EOF {
		err = err2
	}

	return
}

// ParseFloat is like f.Parse(s, base) with f set to the given precision
// and rounding mode.
func ParseFloat(s string, base int, prec uint, mode RoundingMode) (f *Float, b int, err error) {
	return new(Float).SetPrec(prec).SetMode(mode).Parse(s, base)
}
