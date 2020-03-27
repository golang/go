// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

// FormatComplex converts the complex number c to a string,
// according to the format fmt and precision prec. It rounds the
// result assuming that the original was obtained from a complex
// value of bitSize bits (128 for complex128, 64 for complex64).
//
// The format fmt is one of
// 'b' (-ddddp±ddd, a binary exponent),
// 'e' (-d.dddde±dd, a decimal exponent),
// 'E' (-d.ddddE±dd, a decimal exponent),
// 'f' (-ddd.dddd, no exponent),
// 'g' ('e' for large exponents, 'f' otherwise),
// 'G' ('E' for large exponents, 'f' otherwise),
// 'x' (-0xd.ddddp±ddd, a hexadecimal fraction and binary exponent), or
// 'X' (-0Xd.ddddP±ddd, a hexadecimal fraction and binary exponent).
//
// The precision prec controls the number of digits (excluding the exponent)
// printed by the 'e', 'E', 'f', 'g', 'G', 'x', and 'X' formats.
// For 'e', 'E', 'f', 'x', and 'X', it is the number of digits after the decimal point.
// For 'g' and 'G' it is the maximum number of significant digits (trailing
// zeros are removed).
// The special precision -1 uses the smallest number of digits
// necessary such that ParseComplex will return f exactly.
func FormatComplex(c complex128, fmt byte, prec, bitSize int) string {

	if bitSize == 64 {
		bitSize = 32
	} else {
		bitSize = 64
	}

	imag := FormatFloat(imag(c), fmt, prec, bitSize)
	if imag[0:1] != "+" && imag[0:1] != "-" {
		imag = "+" + imag
	}

	return "(" + FormatFloat(real(c), fmt, prec, bitSize) + imag + "i)"
}
