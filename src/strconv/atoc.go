// Copyright 2009 The Go Authors. All rights reserved.
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

func parseComplexComponent(s, orig string, bitSize int) (float64, error) {
	if bitSize == 64 {
		f, err := ParseFloat(s, 32)
		if err != nil {
			return 0, convErr(err, orig)
		}
		return f, nil
	}

	f, err := ParseFloat(s, 64)
	if err != nil {
		return 0, convErr(err, orig)
	}
	return f, nil
}

// ParseComplex converts the string s to a complex number
// with the precision specified by bitSize: 64 for complex64, or 128 for complex128.
// When bitSize=64, the result still has type complex128, but it will be
// convertible to complex64 without changing its value.
//
// The number represented by s may or may not be parenthesized and have the format (N+Ni) where N is
// a floating-point number. There must not be spaces between the real and imaginary components.
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
// respective special floating point values for each component. It ignores case when matching.
func ParseComplex(s string, bitSize int) (complex128, error) {

	orig := s

	if len(s) == 0 {
		return 0, syntaxError(fnParseComplex, orig)
	}

	lastChar := s[len(s)-1 : len(s)]

	// Remove brackets
	if len(s) > 1 && s[0:1] == "(" && lastChar == ")" {
		s = s[1 : len(s)-1]
		lastChar = s[len(s)-1 : len(s)]
	}

	// Is last character an i?
	if lastChar != "i" {
		// The last character is not an i so there is only a real component.
		real, err := parseComplexComponent(s, orig, bitSize)
		if err != nil {
			return 0, err
		}
		return complex(real, 0), nil
	}

	// Remove last char which is an i
	s = s[0 : len(s)-1]

	// Count how many ± exist.
	pos := []int{}

	for idx, rune := range s {
		if rune == '+' || rune == '-' {
			pos = append(pos, idx)
		}
	}

	if len(pos) == 0 {
		// There is only an imaginary component

		if s == "" {
			s = s + "1"
		}

		imag, err := parseComplexComponent(s, orig, bitSize)
		if err != nil {
			return 0, err
		}
		return complex(0, imag), nil

	} else if len(pos) > 4 {
		// Too many ± exists for a valid complex number
		return 0, syntaxError(fnParseComplex, orig)
	}

	/* From here onwards, it is either a complex number with both a real and imaginary component OR a pure imaginary number in exponential form. */

	// Loop through pos from middle of slice, outwards
	mid := (len(pos) - 1) >> 1
	for j := 0; j < len(pos); j++ {
		var idx int
		if j%2 == 0 {
			idx = mid - j/2
		} else {
			idx = mid + (j/2 + 1)
		}

		left := s[0:pos[idx]]
		right := s[pos[idx]:]

		if left == "" {
			left = left + "0"
		}

		// Check if left and right are valid float64
		real, err := parseComplexComponent(left, orig, bitSize)
		if err != nil {
			continue
		}

		if right == "+" || right == "-" {
			right = right + "1"
		}

		imag, err := parseComplexComponent(right, orig, bitSize)
		if err != nil {
			continue
		}

		return complex(real, imag), nil
	}

	// Pure imaginary number in exponential form
	imag, err := parseComplexComponent(s, orig, bitSize)
	if err != nil {
		return 0, err
	}
	return complex(0, imag), nil
}
