// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

import (
	"errors"
	"internal/strconv"
	"internal/stringslite"
)

// IntSize is the size in bits of an int or uint value.
const IntSize = strconv.IntSize

// ParseBool returns the boolean value represented by the string.
// It accepts 1, t, T, TRUE, true, True, 0, f, F, FALSE, false, False.
// Any other value returns an error.
func ParseBool(str string) (bool, error) {
	x, err := strconv.ParseBool(str)
	if err != nil {
		return x, toError("ParseBool", str, 0, 0, err)
	}
	return x, nil
}

// FormatBool returns "true" or "false" according to the value of b.
func FormatBool(b bool) string {
	return strconv.FormatBool(b)
}

// AppendBool appends "true" or "false", according to the value of b,
// to dst and returns the extended buffer.
func AppendBool(dst []byte, b bool) []byte {
	return strconv.AppendBool(dst, b)
}

// ParseComplex converts the string s to a complex number
// with the precision specified by bitSize: 64 for complex64, or 128 for complex128.
// When bitSize=64, the result still has type complex128, but it will be
// convertible to complex64 without changing its value.
//
// The number represented by s must be of the form N, Ni, or N±Ni, where N stands
// for a floating-point number as recognized by [ParseFloat], and i is the imaginary
// component. If the second N is unsigned, a + sign is required between the two components
// as indicated by the ±. If the second N is NaN, only a + sign is accepted.
// The form may be parenthesized and cannot contain any spaces.
// The resulting complex number consists of the two components converted by ParseFloat.
//
// The errors that ParseComplex returns have concrete type [*NumError]
// and include err.Num = s.
//
// If s is not syntactically well-formed, ParseComplex returns err.Err = ErrSyntax.
//
// If s is syntactically well-formed but either component is more than 1/2 ULP
// away from the largest floating point number of the given component's size,
// ParseComplex returns err.Err = ErrRange and c = ±Inf for the respective component.
func ParseComplex(s string, bitSize int) (complex128, error) {
	x, err := strconv.ParseComplex(s, bitSize)
	if err != nil {
		return x, toError("ParseComplex", s, 0, bitSize, err)
	}
	return x, nil
}

// ParseFloat converts the string s to a floating-point number
// with the precision specified by bitSize: 32 for float32, or 64 for float64.
// When bitSize=32, the result still has type float64, but it will be
// convertible to float32 without changing its value.
//
// ParseFloat accepts decimal and hexadecimal floating-point numbers
// as defined by the Go syntax for [floating-point literals].
// If s is well-formed and near a valid floating-point number,
// ParseFloat returns the nearest floating-point number rounded
// using IEEE754 unbiased rounding.
// (Parsing a hexadecimal floating-point value only rounds when
// there are more bits in the hexadecimal representation than
// will fit in the mantissa.)
//
// The errors that ParseFloat returns have concrete type *NumError
// and include err.Num = s.
//
// If s is not syntactically well-formed, ParseFloat returns err.Err = ErrSyntax.
//
// If s is syntactically well-formed but is more than 1/2 ULP
// away from the largest floating point number of the given size,
// ParseFloat returns f = ±Inf, err.Err = ErrRange.
//
// ParseFloat recognizes the string "NaN", and the (possibly signed) strings "Inf" and "Infinity"
// as their respective special floating point values. It ignores case when matching.
//
// [floating-point literals]: https://go.dev/ref/spec#Floating-point_literals
func ParseFloat(s string, bitSize int) (float64, error) {
	x, err := strconv.ParseFloat(s, bitSize)
	if err != nil {
		return x, toError("ParseFloat", s, 0, bitSize, err)
	}
	return x, nil
}

// ParseUint is like [ParseInt] but for unsigned numbers.
//
// A sign prefix is not permitted.
func ParseUint(s string, base int, bitSize int) (uint64, error) {
	x, err := strconv.ParseUint(s, base, bitSize)
	if err != nil {
		return x, toError("ParseUint", s, base, bitSize, err)
	}
	return x, nil
}

// ParseInt interprets a string s in the given base (0, 2 to 36) and
// bit size (0 to 64) and returns the corresponding value i.
//
// The string may begin with a leading sign: "+" or "-".
//
// If the base argument is 0, the true base is implied by the string's
// prefix following the sign (if present): 2 for "0b", 8 for "0" or "0o",
// 16 for "0x", and 10 otherwise. Also, for argument base 0 only,
// underscore characters are permitted as defined by the Go syntax for
// [integer literals].
//
// The bitSize argument specifies the integer type
// that the result must fit into. Bit sizes 0, 8, 16, 32, and 64
// correspond to int, int8, int16, int32, and int64.
// If bitSize is below 0 or above 64, an error is returned.
//
// The errors that ParseInt returns have concrete type [*NumError]
// and include err.Num = s. If s is empty or contains invalid
// digits, err.Err = [ErrSyntax] and the returned value is 0;
// if the value corresponding to s cannot be represented by a
// signed integer of the given size, err.Err = [ErrRange] and the
// returned value is the maximum magnitude integer of the
// appropriate bitSize and sign.
//
// [integer literals]: https://go.dev/ref/spec#Integer_literals
func ParseInt(s string, base int, bitSize int) (i int64, err error) {
	x, err := strconv.ParseInt(s, base, bitSize)
	if err != nil {
		return x, toError("ParseInt", s, base, bitSize, err)
	}
	return x, nil
}

// Atoi is equivalent to ParseInt(s, 10, 0), converted to type int.
func Atoi(s string) (int, error) {
	x, err := strconv.Atoi(s)
	if err != nil {
		return x, toError("Atoi", s, 0, 0, err)
	}
	return strconv.Atoi(s)
}

// FormatComplex converts the complex number c to a string of the
// form (a+bi) where a and b are the real and imaginary parts,
// formatted according to the format fmt and precision prec.
//
// The format fmt and precision prec have the same meaning as in [FormatFloat].
// It rounds the result assuming that the original was obtained from a complex
// value of bitSize bits, which must be 64 for complex64 and 128 for complex128.
func FormatComplex(c complex128, fmt byte, prec, bitSize int) string {
	return strconv.FormatComplex(c, fmt, prec, bitSize)
}

// FormatFloat converts the floating-point number f to a string,
// according to the format fmt and precision prec. It rounds the
// result assuming that the original was obtained from a floating-point
// value of bitSize bits (32 for float32, 64 for float64).
//
// The format fmt is one of
//   - 'b' (-ddddp±ddd, a binary exponent),
//   - 'e' (-d.dddde±dd, a decimal exponent),
//   - 'E' (-d.ddddE±dd, a decimal exponent),
//   - 'f' (-ddd.dddd, no exponent),
//   - 'g' ('e' for large exponents, 'f' otherwise),
//   - 'G' ('E' for large exponents, 'f' otherwise),
//   - 'x' (-0xd.ddddp±ddd, a hexadecimal fraction and binary exponent), or
//   - 'X' (-0Xd.ddddP±ddd, a hexadecimal fraction and binary exponent).
//
// The precision prec controls the number of digits (excluding the exponent)
// printed by the 'e', 'E', 'f', 'g', 'G', 'x', and 'X' formats.
// For 'e', 'E', 'f', 'x', and 'X', it is the number of digits after the decimal point.
// For 'g' and 'G' it is the maximum number of significant digits (trailing
// zeros are removed).
// The special precision -1 uses the smallest number of digits
// necessary such that ParseFloat will return f exactly.
// The exponent is written as a decimal integer;
// for all formats other than 'b', it will be at least two digits.
func FormatFloat(f float64, fmt byte, prec, bitSize int) string {
	return strconv.FormatFloat(f, fmt, prec, bitSize)
}

// AppendFloat appends the string form of the floating-point number f,
// as generated by [FormatFloat], to dst and returns the extended buffer.
func AppendFloat(dst []byte, f float64, fmt byte, prec, bitSize int) []byte {
	return strconv.AppendFloat(dst, f, fmt, prec, bitSize)
}

// FormatUint returns the string representation of i in the given base,
// for 2 <= base <= 36. The result uses the lower-case letters 'a' to 'z'
// for digit values >= 10.
func FormatUint(i uint64, base int) string {
	return strconv.FormatUint(i, base)
}

// FormatInt returns the string representation of i in the given base,
// for 2 <= base <= 36. The result uses the lower-case letters 'a' to 'z'
// for digit values >= 10.
func FormatInt(i int64, base int) string {
	return strconv.FormatInt(i, base)
}

// Itoa is equivalent to [FormatInt](int64(i), 10).
func Itoa(i int) string {
	return strconv.Itoa(i)
}

// AppendInt appends the string form of the integer i,
// as generated by [FormatInt], to dst and returns the extended buffer.
func AppendInt(dst []byte, i int64, base int) []byte {
	return strconv.AppendInt(dst, i, base)
}

// AppendUint appends the string form of the unsigned integer i,
// as generated by [FormatUint], to dst and returns the extended buffer.
func AppendUint(dst []byte, i uint64, base int) []byte {
	return strconv.AppendUint(dst, i, base)
}

// toError converts from internal/strconv.Error to the error guaranteed by this package's APIs.
func toError(fn, s string, base, bitSize int, err error) error {
	switch err {
	case strconv.ErrSyntax:
		return syntaxError(fn, s)
	case strconv.ErrRange:
		return rangeError(fn, s)
	case strconv.ErrBase:
		return baseError(fn, s, base)
	case strconv.ErrBitSize:
		return bitSizeError(fn, s, bitSize)
	}
	return err
}

// ErrRange indicates that a value is out of range for the target type.
var ErrRange = errors.New("value out of range")

// ErrSyntax indicates that a value does not have the right syntax for the target type.
var ErrSyntax = errors.New("invalid syntax")

// A NumError records a failed conversion.
type NumError struct {
	Func string // the failing function (ParseBool, ParseInt, ParseUint, ParseFloat, ParseComplex)
	Num  string // the input
	Err  error  // the reason the conversion failed (e.g. ErrRange, ErrSyntax, etc.)
}

func (e *NumError) Error() string {
	return "strconv." + e.Func + ": " + "parsing " + Quote(e.Num) + ": " + e.Err.Error()
}

func (e *NumError) Unwrap() error { return e.Err }

// All ParseXXX functions allow the input string to escape to the error value.
// This hurts strconv.ParseXXX(string(b)) calls where b is []byte since
// the conversion from []byte must allocate a string on the heap.
// If we assume errors are infrequent, then we can avoid escaping the input
// back to the output by copying it first. This allows the compiler to call
// strconv.ParseXXX without a heap allocation for most []byte to string
// conversions, since it can now prove that the string cannot escape Parse.

func syntaxError(fn, str string) *NumError {
	return &NumError{fn, stringslite.Clone(str), ErrSyntax}
}

func rangeError(fn, str string) *NumError {
	return &NumError{fn, stringslite.Clone(str), ErrRange}
}

func baseError(fn, str string, base int) *NumError {
	return &NumError{fn, stringslite.Clone(str), errors.New("invalid base " + Itoa(base))}
}

func bitSizeError(fn, str string, bitSize int) *NumError {
	return &NumError{fn, stringslite.Clone(str), errors.New("invalid bit size " + Itoa(bitSize))}
}
