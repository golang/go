// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

import "os"

// ErrRange indicates that a value is out of range for the target type.
var ErrRange = os.NewError("value out of range")

// ErrSyntax indicates that a value does not have the right syntax for the target type.
var ErrSyntax = os.NewError("invalid syntax")

// A NumError records a failed conversion.
type NumError struct {
	Num   string   // the input
	Error os.Error // the reason the conversion failed (ErrRange, ErrSyntax)
}

func (e *NumError) String() string { return `parsing "` + e.Num + `": ` + e.Error.String() }

func computeIntsize() uint {
	siz := uint(8)
	for 1<<siz != 0 {
		siz *= 2
	}
	return siz
}

var IntSize = computeIntsize()

// Return the first number n such that n*base >= 1<<64.
func cutoff64(base int) uint64 {
	if base < 2 {
		return 0
	}
	return (1<<64-1)/uint64(base) + 1
}

// Btoui64 interprets a string s in an arbitrary base b (2 to 36)
// and returns the corresponding value n.  If b == 0, the base
// is taken from the string prefix: base 16 for "0x", base 8 for "0",
// and base 10 otherwise.
//
// The errors that Btoui64 returns have concrete type *NumError
// and include err.Num = s.  If s is empty or contains invalid
// digits, err.Error = ErrSyntax; if the value corresponding
// to s cannot be represented by a uint64, err.Error = ErrRange.
func Btoui64(s string, b int) (n uint64, err os.Error) {
	var cutoff uint64

	s0 := s
	switch {
	case len(s) < 1:
		err = ErrSyntax
		goto Error

	case 2 <= b && b <= 36:
		// valid base; nothing to do

	case b == 0:
		// Look for octal, hex prefix.
		switch {
		case s[0] == '0' && len(s) > 1 && (s[1] == 'x' || s[1] == 'X'):
			b = 16
			s = s[2:]
			if len(s) < 1 {
				err = ErrSyntax
				goto Error
			}
		case s[0] == '0':
			b = 8
		default:
			b = 10
		}

	default:
		err = os.NewError("invalid base " + Itoa(b))
		goto Error
	}

	n = 0
	cutoff = cutoff64(b)

	for i := 0; i < len(s); i++ {
		var v byte
		d := s[i]
		switch {
		case '0' <= d && d <= '9':
			v = d - '0'
		case 'a' <= d && d <= 'z':
			v = d - 'a' + 10
		case 'A' <= d && d <= 'Z':
			v = d - 'A' + 10
		default:
			n = 0
			err = ErrSyntax
			goto Error
		}
		if int(v) >= b {
			n = 0
			err = ErrSyntax
			goto Error
		}

		if n >= cutoff {
			// n*b overflows
			n = 1<<64 - 1
			err = ErrRange
			goto Error
		}
		n *= uint64(b)

		n1 := n + uint64(v)
		if n1 < n {
			// n+v overflows
			n = 1<<64 - 1
			err = ErrRange
			goto Error
		}
		n = n1
	}

	return n, nil

Error:
	return n, &NumError{s0, err}
}

// Atoui64 interprets a string s as a decimal number and
// returns the corresponding value n.
//
// Atoui64 returns err.Error = ErrSyntax if s is empty or contains invalid digits.
// It returns err.Error = ErrRange if s cannot be represented by a uint64.
func Atoui64(s string) (n uint64, err os.Error) {
	return Btoui64(s, 10)
}

// Btoi64 is like Btoui64 but allows signed numbers and
// returns its result in an int64.
func Btoi64(s string, base int) (i int64, err os.Error) {
	// Empty string bad.
	if len(s) == 0 {
		return 0, &NumError{s, ErrSyntax}
	}

	// Pick off leading sign.
	s0 := s
	neg := false
	if s[0] == '+' {
		s = s[1:]
	} else if s[0] == '-' {
		neg = true
		s = s[1:]
	}

	// Convert unsigned and check range.
	var un uint64
	un, err = Btoui64(s, base)
	if err != nil && err.(*NumError).Error != ErrRange {
		err.(*NumError).Num = s0
		return 0, err
	}
	if !neg && un >= 1<<63 {
		return 1<<63 - 1, &NumError{s0, ErrRange}
	}
	if neg && un > 1<<63 {
		return -1 << 63, &NumError{s0, ErrRange}
	}
	n := int64(un)
	if neg {
		n = -n
	}
	return n, nil
}

// Atoi64 is like Atoui64 but allows signed numbers and
// returns its result in an int64.
func Atoi64(s string) (i int64, err os.Error) { return Btoi64(s, 10) }

// Atoui is like Atoui64 but returns its result as a uint.
func Atoui(s string) (i uint, err os.Error) {
	i1, e1 := Atoui64(s)
	if e1 != nil && e1.(*NumError).Error != ErrRange {
		return 0, e1
	}
	i = uint(i1)
	if uint64(i) != i1 {
		return ^uint(0), &NumError{s, ErrRange}
	}
	return i, nil
}

// Atoi is like Atoi64 but returns its result as an int.
func Atoi(s string) (i int, err os.Error) {
	i1, e1 := Atoi64(s)
	if e1 != nil && e1.(*NumError).Error != ErrRange {
		return 0, e1
	}
	i = int(i1)
	if int64(i) != i1 {
		if i1 < 0 {
			return -1 << (IntSize - 1), &NumError{s, ErrRange}
		}
		return 1<<(IntSize-1) - 1, &NumError{s, ErrRange}
	}
	return i, nil
}
