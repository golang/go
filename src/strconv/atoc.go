// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

const fnParseComplex = "ParseComplex"

func convErr(err error, s string) error {
	if x, ok := err.(*NumError); ok {
		x.Func = fnParseComplex
		x.Num = s
	}
	return err
}

// ParseComplex converts the string s to a complex number
// with the precision specified by bitSize: 64 for complex64, or 128 for complex128.
// When bitSize=64, the result still has type complex128, but it will be
// convertible to complex64 without changing its value.
//
// The number represented by s may or may not be parenthesized and have the format
// (N+Ni) where N is a floating-point number. There must not be spaces between the real
// and imaginary components.
//
// ParseComplex accepts decimal and hexadecimal floating-point number syntax.
// If s is well-formed and near a valid floating-point number,
// ParseComplex returns the nearest floating-point number rounded
// using IEEE754 unbiased rounding.
// (Parsing a hexadecimal floating-point value only rounds when
// there are more bits in the hexadecimal representation than
// will fit in the mantissa.)
//
// The errors that ParseComplex returns have concrete type *NumError
// and include err.Num = s.
//
// If s is not syntactically well-formed, ParseComplex returns err.Err = ErrSyntax.
//
// If s is syntactically well-formed but is more than 1/2 ULP
// away from the largest floating point number of the given size,
// ParseComplex returns f = ±Inf, err.Err = ErrRange.
//
// ParseComplex recognizes the strings "NaN", "+Inf", and "-Inf" as their
// respective special floating point values for each component. "NaN+NaNi" is also
// recognized. It ignores case when matching.
func ParseComplex(s string, bitSize int) (complex128, error) {
	size := 128
	if bitSize == 64 {
		size = 32 // complex64 uses float32 parts
	}

	orig := s

	// Remove parentheses, if any.
	if len(s) >= 2 && s[0] == '(' && s[len(s)-1] == ')' {
		s = s[1 : len(s)-1]
	}

	// Read real part (possibly imaginary part if followed by 'i').
	re, n, err := parseFloatPrefix(s, size)
	if err != nil {
		return 0, convErr(err, orig)
	}
	s = s[n:]

	// If we have nothing left, we're done.
	if len(s) == 0 {
		return complex(re, 0), nil
	}

	// Otherwise, look at the next character.
	switch s[0] {
	case '+':
		// Consume the '+' to avoid an error if we have "+NaNi", but
		// do this only if we don't have a "++" (don't hide that error).
		if len(s) > 1 && s[1] != '+' {
			s = s[1:]
		}
	case '-':
		// ok
	case 'i':
		// If 'i' is the last character, we only have an imaginary part.
		if len(s) == 1 {
			return complex(0, re), nil
		}
		fallthrough
	default:
		return 0, syntaxError(fnParseComplex, orig)
	}

	// Read imaginary part.
	im, n, err := parseFloatPrefix(s, size)
	if err != nil {
		return 0, convErr(err, orig)
	}
	s = s[n:]
	if s != "i" {
		return 0, syntaxError(fnParseComplex, orig)
	}
	return complex(re, im), nil
}
