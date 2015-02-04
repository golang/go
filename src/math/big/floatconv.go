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
	if _, _, err = r.ReadRune(); err != io.EOF {
		return nil, false
	}

	return f, true
}

// Scan scans the number corresponding to the longest possible prefix
// of r representing a floating-point number with a mantissa in the
// given conversion base (the exponent is always a decimal number).
// It returns the corresponding Float f, the actual base b, and an
// error err, if any. The number must be of the form:
//
//	number   = [ sign ] [ prefix ] mantissa [ exponent ] .
//	sign     = "+" | "-" .
//      prefix   = "0" ( "x" | "X" | "b" | "B" ) .
//	mantissa = digits | digits "." [ digits ] | "." digits .
//	exponent = ( "E" | "e" | "p" ) [ sign ] digits .
//	digits   = digit { digit } .
//	digit    = "0" ... "9" | "a" ... "z" | "A" ... "Z" .
//
// The base argument must be 0 or a value between 2 through MaxBase.
//
// For base 0, the number prefix determines the actual base: A prefix of
// ``0x'' or ``0X'' selects base 16, and a ``0b'' or ``0B'' prefix selects
// base 2; otherwise, the actual base is 10 and no prefix is permitted.
// The octal prefix ``0'' is not supported.
//
// A "p" exponent indicates power of 2 for the exponent; for instance "1.2p3"
// with base 0 or 10 corresponds to the value 1.2 * 2**3.
//
// BUG(gri) This signature conflicts with Scan(s fmt.ScanState, ch rune) error.
func (z *Float) Scan(r io.ByteScanner, base int) (f *Float, b int, err error) {
	// sign
	z.neg, err = scanSign(r)
	if err != nil {
		return
	}

	// mantissa
	var ecorr int // decimal exponent correction; valid if <= 0
	z.mant, b, ecorr, err = z.mant.scan(r, base, true)
	if err != nil {
		return
	}

	// exponent
	var exp int64
	var ebase int
	exp, ebase, err = scanExponent(r)
	if err != nil {
		return
	}
	// special-case 0
	if len(z.mant) == 0 {
		z.exp = 0
		f = z
		return
	}
	// len(z.mant) > 0

	// determine binary (exp2) and decimal (exp) exponent
	exp2 := int64(len(z.mant)*_W - int(fnorm(z.mant)))
	if ebase == 2 {
		exp2 += exp
		exp = 0
	}
	if ecorr < 0 {
		exp += int64(ecorr)
	}

	z.setExp(exp2)
	if exp == 0 {
		// no decimal exponent
		z.round(0)
		f = z
		return
	}
	// exp != 0

	// compute decimal exponent power
	expabs := exp
	if expabs < 0 {
		expabs = -expabs
	}
	powTen := new(Float).SetInt(new(Int).SetBits(nat(nil).expNN(natTen, nat(nil).setWord(Word(expabs)), nil)))

	// correct result
	if exp < 0 {
		z.uquo(z, powTen)
	} else {
		z.umul(z, powTen)
	}

	f = z
	return
}

// Parse is like z.Scan(r, base), but instead of reading from an
// io.ByteScanner, it parses the string s. An error is returned if the
// string contains invalid or trailing characters not belonging to the
// number.
//
// TODO(gri) define possible errors more precisely
func (z *Float) Parse(s string, base int) (f *Float, b int, err error) {
	r := strings.NewReader(s)

	if f, b, err = z.Scan(r, base); err != nil {
		return
	}

	// entire string must have been consumed
	var ch byte
	if ch, err = r.ReadByte(); err != io.EOF {
		if err == nil {
			err = fmt.Errorf("expected end of string, found %q", ch)
		}
	}

	return
}

// ScanFloat is like f.Scan(r, base) with f set to the given precision
// and rounding mode.
func ScanFloat(r io.ByteScanner, base int, prec uint, mode RoundingMode) (f *Float, b int, err error) {
	return NewFloat(0, prec, mode).Scan(r, base)
}

// ParseFloat is like f.Parse(s, base) with f set to the given precision
// and rounding mode.
func ParseFloat(s string, base int, prec uint, mode RoundingMode) (f *Float, b int, err error) {
	return NewFloat(0, prec, mode).Parse(s, base)
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
//	'b'	decimal integer mantissa using x.Precision() bits, or -0
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
	// pick off simple cases
	switch format {
	case 'b':
		return x.bstring(buf)
	case 'p':
		return x.pstring(buf)
	}
	return x.bigFtoa(buf, format, prec)
}

// BUG(gri): Currently, String uses the 'p' (rather than 'g') format.
func (x *Float) String() string {
	return x.Format('p', 0)
}

// bstring appends the string of x in the format ["-"] mantissa "p" exponent
// with a decimal mantissa and a binary exponent, or ["-"] "0" if x is zero,
// and returns the extended buffer.
// The mantissa is normalized such that is uses x.Precision() bits in binary
// representation.
func (x *Float) bstring(buf []byte) []byte {
	// TODO(gri) handle Inf
	if x.neg {
		buf = append(buf, '-')
	}
	if len(x.mant) == 0 {
		return append(buf, '0')
	}
	// x != 0
	// normalize mantissa
	m := x.mant
	t := uint(len(x.mant)*_W) - x.prec // 0 <= t < _W
	if t > 0 {
		m = nat(nil).shr(m, t)
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
	// TODO(gri) handle Inf
	if x.neg {
		buf = append(buf, '-')
	}
	if len(x.mant) == 0 {
		return append(buf, '0')
	}
	// x != 0
	// mantissa is stored in normalized form
	buf = append(buf, "0x."...)
	buf = append(buf, strings.TrimRight(x.mant.hexString(), "0")...)
	buf = append(buf, 'p')
	return strconv.AppendInt(buf, int64(x.exp), 10)
}
