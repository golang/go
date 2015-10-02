// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements rat-to-string conversion functions.

package big

import (
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
)

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
		if _, ok := z.a.SetString(s[:sep], 0); !ok {
			return nil, false
		}
		s = s[sep+1:]
		var err error
		if z.b.abs, _, _, err = z.b.abs.scan(strings.NewReader(s), 0, false); err != nil {
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
	z.a.abs, _, ecorr, err = z.a.abs.scan(r, 10, true)
	if err != nil {
		return nil, false
	}

	// exponent
	var exp int64
	exp, _, err = scanExponent(r, false)
	if err != nil {
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

// scanExponent scans the longest possible prefix of r representing a decimal
// ('e', 'E') or binary ('p') exponent, if any. It returns the exponent, the
// exponent base (10 or 2), or a read or syntax error, if any.
//
//	exponent = ( "E" | "e" | "p" ) [ sign ] digits .
//	sign     = "+" | "-" .
//	digits   = digit { digit } .
//	digit    = "0" ... "9" .
//
// A binary exponent is only permitted if binExpOk is set.
func scanExponent(r io.ByteScanner, binExpOk bool) (exp int64, base int, err error) {
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
		if binExpOk {
			base = 2
			break // ok
		}
		fallthrough // binary exponent not permitted
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
// digits of precision after the decimal point. The last digit is rounded to
// nearest, with halves rounded away from zero.
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
