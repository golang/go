// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	"errors"
	"reflect"
	. "strconv"
	"testing"
)

type atoui64Test struct {
	in  string
	out uint64
	err error
}

var atoui64tests = []atoui64Test{
	{"", 0, ErrSyntax},
	{"0", 0, nil},
	{"1", 1, nil},
	{"12345", 12345, nil},
	{"012345", 12345, nil},
	{"12345x", 0, ErrSyntax},
	{"98765432100", 98765432100, nil},
	{"18446744073709551615", 1<<64 - 1, nil},
	{"18446744073709551616", 1<<64 - 1, ErrRange},
	{"18446744073709551620", 1<<64 - 1, ErrRange},
}

var btoui64tests = []atoui64Test{
	{"", 0, ErrSyntax},
	{"0", 0, nil},
	{"0x", 0, ErrSyntax},
	{"0X", 0, ErrSyntax},
	{"1", 1, nil},
	{"12345", 12345, nil},
	{"012345", 012345, nil},
	{"0x12345", 0x12345, nil},
	{"0X12345", 0x12345, nil},
	{"12345x", 0, ErrSyntax},
	{"0xabcdefg123", 0, ErrSyntax},
	{"123456789abc", 0, ErrSyntax},
	{"98765432100", 98765432100, nil},
	{"18446744073709551615", 1<<64 - 1, nil},
	{"18446744073709551616", 1<<64 - 1, ErrRange},
	{"18446744073709551620", 1<<64 - 1, ErrRange},
	{"0xFFFFFFFFFFFFFFFF", 1<<64 - 1, nil},
	{"0x10000000000000000", 1<<64 - 1, ErrRange},
	{"01777777777777777777777", 1<<64 - 1, nil},
	{"01777777777777777777778", 0, ErrSyntax},
	{"02000000000000000000000", 1<<64 - 1, ErrRange},
	{"0200000000000000000000", 1 << 61, nil},
}

type atoi64Test struct {
	in  string
	out int64
	err error
}

var atoi64tests = []atoi64Test{
	{"", 0, ErrSyntax},
	{"0", 0, nil},
	{"-0", 0, nil},
	{"1", 1, nil},
	{"-1", -1, nil},
	{"12345", 12345, nil},
	{"-12345", -12345, nil},
	{"012345", 12345, nil},
	{"-012345", -12345, nil},
	{"98765432100", 98765432100, nil},
	{"-98765432100", -98765432100, nil},
	{"9223372036854775807", 1<<63 - 1, nil},
	{"-9223372036854775807", -(1<<63 - 1), nil},
	{"9223372036854775808", 1<<63 - 1, ErrRange},
	{"-9223372036854775808", -1 << 63, nil},
	{"9223372036854775809", 1<<63 - 1, ErrRange},
	{"-9223372036854775809", -1 << 63, ErrRange},
}

type btoi64Test struct {
	in   string
	base int
	out  int64
	err  error
}

var btoi64tests = []btoi64Test{
	{"", 0, 0, ErrSyntax},
	{"0", 0, 0, nil},
	{"-0", 0, 0, nil},
	{"1", 0, 1, nil},
	{"-1", 0, -1, nil},
	{"12345", 0, 12345, nil},
	{"-12345", 0, -12345, nil},
	{"012345", 0, 012345, nil},
	{"-012345", 0, -012345, nil},
	{"0x12345", 0, 0x12345, nil},
	{"-0X12345", 0, -0x12345, nil},
	{"12345x", 0, 0, ErrSyntax},
	{"-12345x", 0, 0, ErrSyntax},
	{"98765432100", 0, 98765432100, nil},
	{"-98765432100", 0, -98765432100, nil},
	{"9223372036854775807", 0, 1<<63 - 1, nil},
	{"-9223372036854775807", 0, -(1<<63 - 1), nil},
	{"9223372036854775808", 0, 1<<63 - 1, ErrRange},
	{"-9223372036854775808", 0, -1 << 63, nil},
	{"9223372036854775809", 0, 1<<63 - 1, ErrRange},
	{"-9223372036854775809", 0, -1 << 63, ErrRange},

	// other bases
	{"g", 17, 16, nil},
	{"10", 25, 25, nil},
	{"holycow", 35, (((((17*35+24)*35+21)*35+34)*35+12)*35+24)*35 + 32, nil},
	{"holycow", 36, (((((17*36+24)*36+21)*36+34)*36+12)*36+24)*36 + 32, nil},

	// base 2
	{"0", 2, 0, nil},
	{"-1", 2, -1, nil},
	{"1010", 2, 10, nil},
	{"1000000000000000", 2, 1 << 15, nil},
	{"111111111111111111111111111111111111111111111111111111111111111", 2, 1<<63 - 1, nil},
	{"1000000000000000000000000000000000000000000000000000000000000000", 2, 1<<63 - 1, ErrRange},
	{"-1000000000000000000000000000000000000000000000000000000000000000", 2, -1 << 63, nil},
	{"-1000000000000000000000000000000000000000000000000000000000000001", 2, -1 << 63, ErrRange},

	// base 8
	{"-10", 8, -8, nil},
	{"57635436545", 8, 057635436545, nil},
	{"100000000", 8, 1 << 24, nil},

	// base 16
	{"10", 16, 16, nil},
	{"-123456789abcdef", 16, -0x123456789abcdef, nil},
	{"7fffffffffffffff", 16, 1<<63 - 1, nil},
}

type atoui32Test struct {
	in  string
	out uint32
	err error
}

var atoui32tests = []atoui32Test{
	{"", 0, ErrSyntax},
	{"0", 0, nil},
	{"1", 1, nil},
	{"12345", 12345, nil},
	{"012345", 12345, nil},
	{"12345x", 0, ErrSyntax},
	{"987654321", 987654321, nil},
	{"4294967295", 1<<32 - 1, nil},
	{"4294967296", 1<<32 - 1, ErrRange},
}

type atoi32Test struct {
	in  string
	out int32
	err error
}

var atoi32tests = []atoi32Test{
	{"", 0, ErrSyntax},
	{"0", 0, nil},
	{"-0", 0, nil},
	{"1", 1, nil},
	{"-1", -1, nil},
	{"12345", 12345, nil},
	{"-12345", -12345, nil},
	{"012345", 12345, nil},
	{"-012345", -12345, nil},
	{"12345x", 0, ErrSyntax},
	{"-12345x", 0, ErrSyntax},
	{"987654321", 987654321, nil},
	{"-987654321", -987654321, nil},
	{"2147483647", 1<<31 - 1, nil},
	{"-2147483647", -(1<<31 - 1), nil},
	{"2147483648", 1<<31 - 1, ErrRange},
	{"-2147483648", -1 << 31, nil},
	{"2147483649", 1<<31 - 1, ErrRange},
	{"-2147483649", -1 << 31, ErrRange},
}

type numErrorTest struct {
	num, want string
}

var numErrorTests = []numErrorTest{
	{"0", `strconv.ParseFloat: parsing "0": failed`},
	{"`", "strconv.ParseFloat: parsing \"`\": failed"},
	{"1\x00.2", `strconv.ParseFloat: parsing "1\x00.2": failed`},
}

func init() {
	// The atoi routines return NumErrors wrapping
	// the error and the string. Convert the tables above.
	for i := range atoui64tests {
		test := &atoui64tests[i]
		if test.err != nil {
			test.err = &NumError{"ParseUint", test.in, test.err}
		}
	}
	for i := range btoui64tests {
		test := &btoui64tests[i]
		if test.err != nil {
			test.err = &NumError{"ParseUint", test.in, test.err}
		}
	}
	for i := range atoi64tests {
		test := &atoi64tests[i]
		if test.err != nil {
			test.err = &NumError{"ParseInt", test.in, test.err}
		}
	}
	for i := range btoi64tests {
		test := &btoi64tests[i]
		if test.err != nil {
			test.err = &NumError{"ParseInt", test.in, test.err}
		}
	}
	for i := range atoui32tests {
		test := &atoui32tests[i]
		if test.err != nil {
			test.err = &NumError{"ParseUint", test.in, test.err}
		}
	}
	for i := range atoi32tests {
		test := &atoi32tests[i]
		if test.err != nil {
			test.err = &NumError{"ParseInt", test.in, test.err}
		}
	}
}

func TestParseUint64(t *testing.T) {
	for i := range atoui64tests {
		test := &atoui64tests[i]
		out, err := ParseUint(test.in, 10, 64)
		if test.out != out || !reflect.DeepEqual(test.err, err) {
			t.Errorf("Atoui64(%q) = %v, %v want %v, %v",
				test.in, out, err, test.out, test.err)
		}
	}
}

func TestParseUint64Base(t *testing.T) {
	for i := range btoui64tests {
		test := &btoui64tests[i]
		out, err := ParseUint(test.in, 0, 64)
		if test.out != out || !reflect.DeepEqual(test.err, err) {
			t.Errorf("ParseUint(%q) = %v, %v want %v, %v",
				test.in, out, err, test.out, test.err)
		}
	}
}

func TestParseInt64(t *testing.T) {
	for i := range atoi64tests {
		test := &atoi64tests[i]
		out, err := ParseInt(test.in, 10, 64)
		if test.out != out || !reflect.DeepEqual(test.err, err) {
			t.Errorf("Atoi64(%q) = %v, %v want %v, %v",
				test.in, out, err, test.out, test.err)
		}
	}
}

func TestParseInt64Base(t *testing.T) {
	for i := range btoi64tests {
		test := &btoi64tests[i]
		out, err := ParseInt(test.in, test.base, 64)
		if test.out != out || !reflect.DeepEqual(test.err, err) {
			t.Errorf("ParseInt(%q) = %v, %v want %v, %v",
				test.in, out, err, test.out, test.err)
		}
	}
}

func TestParseUint(t *testing.T) {
	switch IntSize {
	case 32:
		for i := range atoui32tests {
			test := &atoui32tests[i]
			out, err := ParseUint(test.in, 10, 0)
			if test.out != uint32(out) || !reflect.DeepEqual(test.err, err) {
				t.Errorf("Atoui(%q) = %v, %v want %v, %v",
					test.in, out, err, test.out, test.err)
			}
		}
	case 64:
		for i := range atoui64tests {
			test := &atoui64tests[i]
			out, err := ParseUint(test.in, 10, 0)
			if test.out != uint64(out) || !reflect.DeepEqual(test.err, err) {
				t.Errorf("Atoui(%q) = %v, %v want %v, %v",
					test.in, out, err, test.out, test.err)
			}
		}
	}
}

func TestParseInt(t *testing.T) {
	switch IntSize {
	case 32:
		for i := range atoi32tests {
			test := &atoi32tests[i]
			out, err := ParseInt(test.in, 10, 0)
			if test.out != int32(out) || !reflect.DeepEqual(test.err, err) {
				t.Errorf("Atoi(%q) = %v, %v want %v, %v",
					test.in, out, err, test.out, test.err)
			}
		}
	case 64:
		for i := range atoi64tests {
			test := &atoi64tests[i]
			out, err := ParseInt(test.in, 10, 0)
			if test.out != int64(out) || !reflect.DeepEqual(test.err, err) {
				t.Errorf("Atoi(%q) = %v, %v want %v, %v",
					test.in, out, err, test.out, test.err)
			}
		}
	}
}

func TestNumError(t *testing.T) {
	for _, test := range numErrorTests {
		err := &NumError{
			Func: "ParseFloat",
			Num:  test.num,
			Err:  errors.New("failed"),
		}
		if got := err.Error(); got != test.want {
			t.Errorf(`(&NumError{"ParseFloat", %q, "failed"}).Error() = %v, want %v`, test.num, got, test.want)
		}
	}
}

func BenchmarkAtoi(b *testing.B) {
	for i := 0; i < b.N; i++ {
		ParseInt("12345678", 10, 0)
	}
}

func BenchmarkAtoiNeg(b *testing.B) {
	for i := 0; i < b.N; i++ {
		ParseInt("-12345678", 10, 0)
	}
}

func BenchmarkAtoi64(b *testing.B) {
	for i := 0; i < b.N; i++ {
		ParseInt("12345678901234", 10, 64)
	}
}

func BenchmarkAtoi64Neg(b *testing.B) {
	for i := 0; i < b.N; i++ {
		ParseInt("-12345678901234", 10, 64)
	}
}
