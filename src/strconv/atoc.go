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
	if len(s) == 0 {
		return 0, syntaxError(fnParseComplex, s)
	}

	orig := s

	// Translate bitSize for ParseFloat/parseFloatPrefix function
	if bitSize == 64 {
		bitSize = 32 // complex64 uses float32 internally
	} else {
		bitSize = 64
	}

	endCh := s[len(s)-1]

	// Remove parentheses
	if len(s) > 1 && s[0] == '(' && endCh == ')' {
		s = s[1 : len(s)-1]
		endCh = s[len(s)-1]
	}

	// Is last character an i?
	if endCh != 'i' {
		// The last character is not an i so there is only a real component.
		real, err := ParseFloat(s, bitSize)
		if err != nil {
			return 0, convErr(err, orig)
		}
		return complex(real, 0), nil
	} else if s == "i" {
		return complex(0, 1), nil
	}

	// Remove last char which is an i
	s = s[0 : len(s)-1]

	// Some input does not get interpreted by parseFloatPrefix correctly.
	// Namely: i, -i, +i, +NaNi
	// The "i" (no sign) case is taken care of above.
	// +NaNi is only acceptable if both a real and imag component exist.

	var posNaNFound bool

	if len(s) >= 1 {
		endCh := s[len(s)-1]
		if endCh == '+' || endCh == '-' {
			s = s + "1"
		}
	}

	if len(s) >= 4 {
		endChs := s[len(s)-4 : len(s)]
		if endChs == "+NaN" {
			posNaNFound = true
			s = s[0:len(s)-4] + "NaN" // remove sign before NaN
		}
	}

	floatsFound := []float64{}

	for {
		f, n, err := parseFloatPrefix(s, bitSize)
		if err != nil {
			return 0, convErr(err, orig)
		}

		floatsFound = append(floatsFound, f)
		s = s[n:]

		if len(s) == 0 {
			break
		}
	}

	// Check how many floats were found in s
	switch len(floatsFound) {
	case 1:
		// only imag component
		imaj := floatsFound[0]
		if posNaNFound && imaj != imaj {
			// Reject if +NaN found
			return 0, syntaxError(fnParseComplex, orig)
		}
		return complex(0, floatsFound[0]), nil
	case 2:
		// real and imag component
		return complex(floatsFound[0], floatsFound[1]), nil
	}

	// 0 floats found or too many components
	return 0, syntaxError(fnParseComplex, orig)
}
