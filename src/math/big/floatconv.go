// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements float-to-string conversion functions.

package big

import (
	"fmt"
	"io"
	"strconv"
	"strings"
)

// SetString sets z to the value of s and returns z and a boolean indicating
// success. s must be a floating-point number of the same format as accepted
// by Scan, with number prefixes permitted.
func (z *Float) SetString(s string) (*Float, bool) {
	r := strings.NewReader(s)

	f, _, err := z.Scan(r, 0)
	if err != nil {
		return nil, false
	}

	// there should be no unread characters left
	if _, err = r.ReadByte(); err != io.EOF {
		return nil, false
	}

	return f, true
}

// Scan scans the number corresponding to the longest possible prefix
// of r representing a floating-point number with a mantissa in the
// given conversion base (the exponent is always a decimal number).
// It sets z to the (possibly rounded) value of the corresponding
// floating-point number, and returns z, the actual base b, and an
// error err, if any. If z's precision is 0, it is changed to 64
// before rounding takes effect. The number must be of the form:
//
//	number   = [ sign ] [ prefix ] mantissa [ exponent ] .
//	sign     = "+" | "-" .
//      prefix   = "0" ( "x" | "X" | "b" | "B" ) .
//	mantissa = digits | digits "." [ digits ] | "." digits .
//	exponent = ( "E" | "e" | "p" ) [ sign ] digits .
//	digits   = digit { digit } .
//	digit    = "0" ... "9" | "a" ... "z" | "A" ... "Z" .
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
// BUG(gri) This signature conflicts with Scan(s fmt.ScanState, ch rune) error.
func (z *Float) Scan(r io.ByteScanner, base int) (f *Float, b int, err error) {
	if z.prec == 0 {
		z.prec = 64
	}

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

	// set result
	f = z

	// special-case 0
	if len(z.mant) == 0 {
		z.acc = Exact
		z.exp = 0
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
	z.setExp(exp2)

	if exp10 == 0 {
		// no decimal exponent to consider
		z.round(0)
		return
	}
	// exp10 != 0

	// compute decimal exponent power
	expabs := exp10
	if expabs < 0 {
		expabs = -expabs
	}
	powTen := nat(nil).expNN(natTen, nat(nil).setUint64(uint64(expabs)), nil)
	fpowTen := new(Float).SetInt(new(Int).SetBits(powTen))

	// apply 10**exp10
	// (uquo and umul do the rounding)
	if exp10 < 0 {
		z.uquo(z, fpowTen)
	} else {
		z.umul(z, fpowTen)
	}

	return
}

// Parse is like z.Scan(r, base), but instead of reading from an
// io.ByteScanner, it parses the string s. An error is returned if
// the string contains invalid or trailing bytes not belonging to
// the number.
func (z *Float) Parse(s string, base int) (f *Float, b int, err error) {
	r := strings.NewReader(s)

	if f, b, err = z.Scan(r, base); err != nil {
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

// ScanFloat is like f.Scan(r, base) with f set to the given precision
// and rounding mode.
func ScanFloat(r io.ByteScanner, base int, prec uint, mode RoundingMode) (f *Float, b int, err error) {
	return new(Float).SetPrec(prec).SetMode(mode).Scan(r, base)
}

// ParseFloat is like f.Parse(s, base) with f set to the given precision
// and rounding mode.
func ParseFloat(s string, base int, prec uint, mode RoundingMode) (f *Float, b int, err error) {
	return new(Float).SetPrec(prec).SetMode(mode).Parse(s, base)
}

// Format converts the floating-point number x to a string according
// to the given format and precision prec. The format is one of:
//
//	'e'	-d.dddde±dd, decimal exponent, at least two (possibly 0) exponent digits
//	'E'	-d.ddddE±dd, decimal exponent, at least two (possibly 0) exponent digits
//	'f'	-ddddd.dddd, no exponent
//	'g'	like 'e' for large exponents, like 'f' otherwise
//	'G'	like 'E' for large exponents, like 'f' otherwise
//	'b'	-ddddddp±dd, binary exponent
//	'p'	-0x.dddp±dd, binary exponent, hexadecimal mantissa
//
// For the binary exponent formats, the mantissa is printed in normalized form:
//
//	'b'	decimal integer mantissa using x.Prec() bits, or -0
//	'p'	hexadecimal fraction with 0.5 <= 0.mantissa < 1.0, or -0
//
// The precision prec controls the number of digits (excluding the exponent)
// printed by the 'e', 'E', 'f', 'g', and 'G' formats. For 'e', 'E', and 'f'
// it is the number of digits after the decimal point. For 'g' and 'G' it is
// the total number of digits. A negative precision selects the smallest
// number of digits necessary such that ParseFloat will return f exactly.
// The prec value is ignored for the 'b' or 'p' format.
//
// BUG(gri) Currently, Format does not accept negative precisions.
func (x *Float) Format(format byte, prec int) string {
	const extra = 10 // TODO(gri) determine a good/better value here
	return string(x.Append(make([]byte, 0, prec+extra), format, prec))
}

// Append appends the string form of the floating-point number x,
// as generated by x.Format, to buf and returns the extended buffer.
func (x *Float) Append(buf []byte, format byte, prec int) []byte {
	// TODO(gri) factor out handling of sign?

	// Inf
	if x.IsInf(0) {
		var ch byte = '+'
		if x.neg {
			ch = '-'
		}
		buf = append(buf, ch)
		return append(buf, "Inf"...)
	}

	// easy formats
	switch format {
	case 'b':
		return x.bstring(buf)
	case 'p':
		return x.pstring(buf)
	}

	return x.bigFtoa(buf, format, prec)
}

// BUG(gri): Currently, String uses x.Format('g', 10) rather than x.Format('g', -1).
func (x *Float) String() string {
	return x.Format('g', 10)
}

// bstring appends the string of x in the format ["-"] mantissa "p" exponent
// with a decimal mantissa and a binary exponent, or ["-"] "0" if x is zero,
// and returns the extended buffer.
// The mantissa is normalized such that is uses x.Prec() bits in binary
// representation.
func (x *Float) bstring(buf []byte) []byte {
	if x.neg {
		buf = append(buf, '-')
	}
	if len(x.mant) == 0 {
		return append(buf, '0')
	}
	// x != 0

	// adjust mantissa to use exactly x.prec bits
	m := x.mant
	switch w := uint(len(x.mant)) * _W; {
	case w < x.prec:
		m = nat(nil).shl(m, x.prec-w)
	case w > x.prec:
		m = nat(nil).shr(m, w-x.prec)
	}

	buf = append(buf, m.decimalString()...)
	buf = append(buf, 'p')
	e := int64(x.exp) - int64(x.prec)
	if e >= 0 {
		buf = append(buf, '+')
	}
	return strconv.AppendInt(buf, e, 10)
}

// pstring appends the string of x in the format ["-"] "0x." mantissa "p" exponent
// with a hexadecimal mantissa and a binary exponent, or ["-"] "0" if x is zero,
// ad returns the extended buffer.
// The mantissa is normalized such that 0.5 <= 0.mantissa < 1.0.
func (x *Float) pstring(buf []byte) []byte {
	if x.neg {
		buf = append(buf, '-')
	}
	if len(x.mant) == 0 {
		return append(buf, '0')
	}
	// x != 0

	// remove trailing 0 words early
	// (no need to convert to hex 0's and trim later)
	m := x.mant
	i := 0
	for i < len(m) && m[i] == 0 {
		i++
	}
	m = m[i:]

	buf = append(buf, "0x."...)
	buf = append(buf, strings.TrimRight(x.mant.hexString(), "0")...)
	buf = append(buf, 'p')
	return strconv.AppendInt(buf, int64(x.exp), 10)
}
