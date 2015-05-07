// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements int-to-string conversion functions.

package big

import (
	"errors"
	"fmt"
	"io"
)

func (x *Int) String() string {
	switch {
	case x == nil:
		return "<nil>"
	case x.neg:
		return "-" + x.abs.decimalString()
	}
	return x.abs.decimalString()
}

func charset(ch rune) string {
	switch ch {
	case 'b':
		return lowercaseDigits[0:2]
	case 'o':
		return lowercaseDigits[0:8]
	case 'd', 's', 'v':
		return lowercaseDigits[0:10]
	case 'x':
		return lowercaseDigits[0:16]
	case 'X':
		return uppercaseDigits[0:16]
	}
	return "" // unknown format
}

// write count copies of text to s
func writeMultiple(s fmt.State, text string, count int) {
	if len(text) > 0 {
		b := []byte(text)
		for ; count > 0; count-- {
			s.Write(b)
		}
	}
}

// Format is a support routine for fmt.Formatter. It accepts
// the formats 'b' (binary), 'o' (octal), 'd' (decimal), 'x'
// (lowercase hexadecimal), and 'X' (uppercase hexadecimal).
// Also supported are the full suite of package fmt's format
// verbs for integral types, including '+', '-', and ' '
// for sign control, '#' for leading zero in octal and for
// hexadecimal, a leading "0x" or "0X" for "%#x" and "%#X"
// respectively, specification of minimum digits precision,
// output field width, space or zero padding, and left or
// right justification.
//
func (x *Int) Format(s fmt.State, ch rune) {
	cs := charset(ch)

	// special cases
	switch {
	case cs == "":
		// unknown format
		fmt.Fprintf(s, "%%!%c(big.Int=%s)", ch, x.String())
		return
	case x == nil:
		fmt.Fprint(s, "<nil>")
		return
	}

	// determine sign character
	sign := ""
	switch {
	case x.neg:
		sign = "-"
	case s.Flag('+'): // supersedes ' ' when both specified
		sign = "+"
	case s.Flag(' '):
		sign = " "
	}

	// determine prefix characters for indicating output base
	prefix := ""
	if s.Flag('#') {
		switch ch {
		case 'o': // octal
			prefix = "0"
		case 'x': // hexadecimal
			prefix = "0x"
		case 'X':
			prefix = "0X"
		}
	}

	// determine digits with base set by len(cs) and digit characters from cs
	digits := x.abs.string(cs)

	// number of characters for the three classes of number padding
	var left int   // space characters to left of digits for right justification ("%8d")
	var zeroes int // zero characters (actually cs[0]) as left-most digits ("%.8d")
	var right int  // space characters to right of digits for left justification ("%-8d")

	// determine number padding from precision: the least number of digits to output
	precision, precisionSet := s.Precision()
	if precisionSet {
		switch {
		case len(digits) < precision:
			zeroes = precision - len(digits) // count of zero padding
		case digits == "0" && precision == 0:
			return // print nothing if zero value (x == 0) and zero precision ("." or ".0")
		}
	}

	// determine field pad from width: the least number of characters to output
	length := len(sign) + len(prefix) + zeroes + len(digits)
	if width, widthSet := s.Width(); widthSet && length < width { // pad as specified
		switch d := width - length; {
		case s.Flag('-'):
			// pad on the right with spaces; supersedes '0' when both specified
			right = d
		case s.Flag('0') && !precisionSet:
			// pad with zeroes unless precision also specified
			zeroes = d
		default:
			// pad on the left with spaces
			left = d
		}
	}

	// print number as [left pad][sign][prefix][zero pad][digits][right pad]
	writeMultiple(s, " ", left)
	writeMultiple(s, sign, 1)
	writeMultiple(s, prefix, 1)
	writeMultiple(s, "0", zeroes)
	writeMultiple(s, digits, 1)
	writeMultiple(s, " ", right)
}

// scan sets z to the integer value corresponding to the longest possible prefix
// read from r representing a signed integer number in a given conversion base.
// It returns z, the actual conversion base used, and an error, if any. In the
// error case, the value of z is undefined but the returned value is nil. The
// syntax follows the syntax of integer literals in Go.
//
// The base argument must be 0 or a value from 2 through MaxBase. If the base
// is 0, the string prefix determines the actual conversion base. A prefix of
// ``0x'' or ``0X'' selects base 16; the ``0'' prefix selects base 8, and a
// ``0b'' or ``0B'' prefix selects base 2. Otherwise the selected base is 10.
//
func (z *Int) scan(r io.ByteScanner, base int) (*Int, int, error) {
	// determine sign
	neg, err := scanSign(r)
	if err != nil {
		return nil, 0, err
	}

	// determine mantissa
	z.abs, base, _, err = z.abs.scan(r, base, false)
	if err != nil {
		return nil, base, err
	}
	z.neg = len(z.abs) > 0 && neg // 0 has no sign

	return z, base, nil
}

func scanSign(r io.ByteScanner) (neg bool, err error) {
	var ch byte
	if ch, err = r.ReadByte(); err != nil {
		return false, err
	}
	switch ch {
	case '-':
		neg = true
	case '+':
		// nothing to do
	default:
		r.UnreadByte()
	}
	return
}

// byteReader is a local wrapper around fmt.ScanState;
// it implements the ByteReader interface.
type byteReader struct {
	fmt.ScanState
}

func (r byteReader) ReadByte() (byte, error) {
	ch, size, err := r.ReadRune()
	if size != 1 && err == nil {
		err = fmt.Errorf("invalid rune %#U", ch)
	}
	return byte(ch), err
}

func (r byteReader) UnreadByte() error {
	return r.UnreadRune()
}

// Scan is a support routine for fmt.Scanner; it sets z to the value of
// the scanned number. It accepts the formats 'b' (binary), 'o' (octal),
// 'd' (decimal), 'x' (lowercase hexadecimal), and 'X' (uppercase hexadecimal).
func (z *Int) Scan(s fmt.ScanState, ch rune) error {
	s.SkipSpace() // skip leading space characters
	base := 0
	switch ch {
	case 'b':
		base = 2
	case 'o':
		base = 8
	case 'd':
		base = 10
	case 'x', 'X':
		base = 16
	case 's', 'v':
		// let scan determine the base
	default:
		return errors.New("Int.Scan: invalid verb")
	}
	_, _, err := z.scan(byteReader{s}, base)
	return err
}
