// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	"fmt"
	. "internal/strconv"
	"reflect"
	"testing"
)

type parseUint64Test struct {
	in  string
	out uint64
	err error
}

var parseUint64Tests = []parseUint64Test{
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
	{"1_2_3_4_5", 0, ErrSyntax}, // base=10 so no underscores allowed
	{"_12345", 0, ErrSyntax},
	{"1__2345", 0, ErrSyntax},
	{"12345_", 0, ErrSyntax},
	{"-0", 0, ErrSyntax},
	{"-1", 0, ErrSyntax},
	{"+1", 0, ErrSyntax},
}

type parseUint64BaseTest struct {
	in   string
	base int
	out  uint64
	err  error
}

var parseUint64BaseTests = []parseUint64BaseTest{
	{"", 0, 0, ErrSyntax},
	{"0", 0, 0, nil},
	{"0x", 0, 0, ErrSyntax},
	{"0X", 0, 0, ErrSyntax},
	{"1", 0, 1, nil},
	{"12345", 0, 12345, nil},
	{"012345", 0, 012345, nil},
	{"0x12345", 0, 0x12345, nil},
	{"0X12345", 0, 0x12345, nil},
	{"12345x", 0, 0, ErrSyntax},
	{"0xabcdefg123", 0, 0, ErrSyntax},
	{"123456789abc", 0, 0, ErrSyntax},
	{"98765432100", 0, 98765432100, nil},
	{"18446744073709551615", 0, 1<<64 - 1, nil},
	{"18446744073709551616", 0, 1<<64 - 1, ErrRange},
	{"18446744073709551620", 0, 1<<64 - 1, ErrRange},
	{"0xFFFFFFFFFFFFFFFF", 0, 1<<64 - 1, nil},
	{"0x10000000000000000", 0, 1<<64 - 1, ErrRange},
	{"01777777777777777777777", 0, 1<<64 - 1, nil},
	{"01777777777777777777778", 0, 0, ErrSyntax},
	{"02000000000000000000000", 0, 1<<64 - 1, ErrRange},
	{"0200000000000000000000", 0, 1 << 61, nil},
	{"0b", 0, 0, ErrSyntax},
	{"0B", 0, 0, ErrSyntax},
	{"0b101", 0, 5, nil},
	{"0B101", 0, 5, nil},
	{"0o", 0, 0, ErrSyntax},
	{"0O", 0, 0, ErrSyntax},
	{"0o377", 0, 255, nil},
	{"0O377", 0, 255, nil},

	// underscores allowed with base == 0 only
	{"1_2_3_4_5", 0, 12345, nil}, // base 0 => 10
	{"_12345", 0, 0, ErrSyntax},
	{"1__2345", 0, 0, ErrSyntax},
	{"12345_", 0, 0, ErrSyntax},

	{"1_2_3_4_5", 10, 0, ErrSyntax}, // base 10
	{"_12345", 10, 0, ErrSyntax},
	{"1__2345", 10, 0, ErrSyntax},
	{"12345_", 10, 0, ErrSyntax},

	{"0x_1_2_3_4_5", 0, 0x12345, nil}, // base 0 => 16
	{"_0x12345", 0, 0, ErrSyntax},
	{"0x__12345", 0, 0, ErrSyntax},
	{"0x1__2345", 0, 0, ErrSyntax},
	{"0x1234__5", 0, 0, ErrSyntax},
	{"0x12345_", 0, 0, ErrSyntax},

	{"1_2_3_4_5", 16, 0, ErrSyntax}, // base 16
	{"_12345", 16, 0, ErrSyntax},
	{"1__2345", 16, 0, ErrSyntax},
	{"1234__5", 16, 0, ErrSyntax},
	{"12345_", 16, 0, ErrSyntax},

	{"0_1_2_3_4_5", 0, 012345, nil}, // base 0 => 8 (0377)
	{"_012345", 0, 0, ErrSyntax},
	{"0__12345", 0, 0, ErrSyntax},
	{"01234__5", 0, 0, ErrSyntax},
	{"012345_", 0, 0, ErrSyntax},

	{"0o_1_2_3_4_5", 0, 012345, nil}, // base 0 => 8 (0o377)
	{"_0o12345", 0, 0, ErrSyntax},
	{"0o__12345", 0, 0, ErrSyntax},
	{"0o1234__5", 0, 0, ErrSyntax},
	{"0o12345_", 0, 0, ErrSyntax},

	{"0_1_2_3_4_5", 8, 0, ErrSyntax}, // base 8
	{"_012345", 8, 0, ErrSyntax},
	{"0__12345", 8, 0, ErrSyntax},
	{"01234__5", 8, 0, ErrSyntax},
	{"012345_", 8, 0, ErrSyntax},

	{"0b_1_0_1", 0, 5, nil}, // base 0 => 2 (0b101)
	{"_0b101", 0, 0, ErrSyntax},
	{"0b__101", 0, 0, ErrSyntax},
	{"0b1__01", 0, 0, ErrSyntax},
	{"0b10__1", 0, 0, ErrSyntax},
	{"0b101_", 0, 0, ErrSyntax},

	{"1_0_1", 2, 0, ErrSyntax}, // base 2
	{"_101", 2, 0, ErrSyntax},
	{"1_01", 2, 0, ErrSyntax},
	{"10_1", 2, 0, ErrSyntax},
	{"101_", 2, 0, ErrSyntax},
}

type parseInt64Test struct {
	in  string
	out int64
	err error
}

var parseInt64Tests = []parseInt64Test{
	{"", 0, ErrSyntax},
	{"0", 0, nil},
	{"-0", 0, nil},
	{"+0", 0, nil},
	{"1", 1, nil},
	{"-1", -1, nil},
	{"+1", 1, nil},
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
	{"-1_2_3_4_5", 0, ErrSyntax}, // base=10 so no underscores allowed
	{"-_12345", 0, ErrSyntax},
	{"_12345", 0, ErrSyntax},
	{"1__2345", 0, ErrSyntax},
	{"12345_", 0, ErrSyntax},
	{"123%45", 0, ErrSyntax},
}

type parseInt64BaseTest struct {
	in   string
	base int
	out  int64
	err  error
}

var parseInt64BaseTests = []parseInt64BaseTest{
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

	// underscores
	{"-0x_1_2_3_4_5", 0, -0x12345, nil},
	{"0x_1_2_3_4_5", 0, 0x12345, nil},
	{"-_0x12345", 0, 0, ErrSyntax},
	{"_-0x12345", 0, 0, ErrSyntax},
	{"_0x12345", 0, 0, ErrSyntax},
	{"0x__12345", 0, 0, ErrSyntax},
	{"0x1__2345", 0, 0, ErrSyntax},
	{"0x1234__5", 0, 0, ErrSyntax},
	{"0x12345_", 0, 0, ErrSyntax},

	{"-0_1_2_3_4_5", 0, -012345, nil}, // octal
	{"0_1_2_3_4_5", 0, 012345, nil},   // octal
	{"-_012345", 0, 0, ErrSyntax},
	{"_-012345", 0, 0, ErrSyntax},
	{"_012345", 0, 0, ErrSyntax},
	{"0__12345", 0, 0, ErrSyntax},
	{"01234__5", 0, 0, ErrSyntax},
	{"012345_", 0, 0, ErrSyntax},

	{"+0xf", 0, 0xf, nil},
	{"-0xf", 0, -0xf, nil},
	{"0x+f", 0, 0, ErrSyntax},
	{"0x-f", 0, 0, ErrSyntax},
}

type parseUint32Test struct {
	in  string
	out uint32
	err error
}

var parseUint32Tests = []parseUint32Test{
	{"", 0, ErrSyntax},
	{"0", 0, nil},
	{"1", 1, nil},
	{"12345", 12345, nil},
	{"012345", 12345, nil},
	{"12345x", 0, ErrSyntax},
	{"987654321", 987654321, nil},
	{"4294967295", 1<<32 - 1, nil},
	{"4294967296", 1<<32 - 1, ErrRange},
	{"1_2_3_4_5", 0, ErrSyntax}, // base=10 so no underscores allowed
	{"_12345", 0, ErrSyntax},
	{"_12345", 0, ErrSyntax},
	{"1__2345", 0, ErrSyntax},
	{"12345_", 0, ErrSyntax},
}

type parseInt32Test struct {
	in  string
	out int32
	err error
}

var parseInt32Tests = []parseInt32Test{
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
	{"-1_2_3_4_5", 0, ErrSyntax}, // base=10 so no underscores allowed
	{"-_12345", 0, ErrSyntax},
	{"_12345", 0, ErrSyntax},
	{"1__2345", 0, ErrSyntax},
	{"12345_", 0, ErrSyntax},
	{"123%45", 0, ErrSyntax},
}

type numErrorTest struct {
	num, want string
}

var numErrorTests = []numErrorTest{
	{"0", `strconv.ParseFloat: parsing "0": failed`},
	{"`", "strconv.ParseFloat: parsing \"`\": failed"},
	{"1\x00.2", `strconv.ParseFloat: parsing "1\x00.2": failed`},
}

func TestParseUint32(t *testing.T) {
	for i := range parseUint32Tests {
		test := &parseUint32Tests[i]
		out, err := ParseUint(test.in, 10, 32)
		if uint64(test.out) != out || !reflect.DeepEqual(test.err, err) {
			t.Errorf("ParseUint(%q, 10, 32) = %v, %v want %v, %v",
				test.in, out, err, test.out, test.err)
		}
	}
}

func TestParseUint64(t *testing.T) {
	for i := range parseUint64Tests {
		test := &parseUint64Tests[i]
		out, err := ParseUint(test.in, 10, 64)
		if test.out != out || !reflect.DeepEqual(test.err, err) {
			t.Errorf("ParseUint(%q, 10, 64) = %v, %v want %v, %v",
				test.in, out, err, test.out, test.err)
		}
	}
}

func TestParseUint64Base(t *testing.T) {
	for i := range parseUint64BaseTests {
		test := &parseUint64BaseTests[i]
		out, err := ParseUint(test.in, test.base, 64)
		if test.out != out || !reflect.DeepEqual(test.err, err) {
			t.Errorf("ParseUint(%q, %v, 64) = %v, %v want %v, %v",
				test.in, test.base, out, err, test.out, test.err)
		}
	}
}

func TestParseInt32(t *testing.T) {
	for i := range parseInt32Tests {
		test := &parseInt32Tests[i]
		out, err := ParseInt(test.in, 10, 32)
		if int64(test.out) != out || !reflect.DeepEqual(test.err, err) {
			t.Errorf("ParseInt(%q, 10 ,32) = %v, %v want %v, %v",
				test.in, out, err, test.out, test.err)
		}
	}
}

func TestParseInt64(t *testing.T) {
	for i := range parseInt64Tests {
		test := &parseInt64Tests[i]
		out, err := ParseInt(test.in, 10, 64)
		if test.out != out || !reflect.DeepEqual(test.err, err) {
			t.Errorf("ParseInt(%q, 10, 64) = %v, %v want %v, %v",
				test.in, out, err, test.out, test.err)
		}
	}
}

func TestParseInt64Base(t *testing.T) {
	for i := range parseInt64BaseTests {
		test := &parseInt64BaseTests[i]
		out, err := ParseInt(test.in, test.base, 64)
		if test.out != out || !reflect.DeepEqual(test.err, err) {
			t.Errorf("ParseInt(%q, %v, 64) = %v, %v want %v, %v",
				test.in, test.base, out, err, test.out, test.err)
		}
	}
}

func TestParseUint(t *testing.T) {
	switch IntSize {
	case 32:
		for i := range parseUint32Tests {
			test := &parseUint32Tests[i]
			out, err := ParseUint(test.in, 10, 0)
			if uint64(test.out) != out || !reflect.DeepEqual(test.err, err) {
				t.Errorf("ParseUint(%q, 10, 0) = %v, %v want %v, %v",
					test.in, out, err, test.out, test.err)
			}
		}
	case 64:
		for i := range parseUint64Tests {
			test := &parseUint64Tests[i]
			out, err := ParseUint(test.in, 10, 0)
			if test.out != out || !reflect.DeepEqual(test.err, err) {
				t.Errorf("ParseUint(%q, 10, 0) = %v, %v want %v, %v",
					test.in, out, err, test.out, test.err)
			}
		}
	}
}

func TestParseInt(t *testing.T) {
	switch IntSize {
	case 32:
		for i := range parseInt32Tests {
			test := &parseInt32Tests[i]
			out, err := ParseInt(test.in, 10, 0)
			if int64(test.out) != out || !reflect.DeepEqual(test.err, err) {
				t.Errorf("ParseInt(%q, 10, 0) = %v, %v want %v, %v",
					test.in, out, err, test.out, test.err)
			}
		}
	case 64:
		for i := range parseInt64Tests {
			test := &parseInt64Tests[i]
			out, err := ParseInt(test.in, 10, 0)
			if test.out != out || !reflect.DeepEqual(test.err, err) {
				t.Errorf("ParseInt(%q, 10, 0) = %v, %v want %v, %v",
					test.in, out, err, test.out, test.err)
			}
		}
	}
}

func TestAtoi(t *testing.T) {
	switch IntSize {
	case 32:
		for i := range parseInt32Tests {
			test := &parseInt32Tests[i]
			out, err := Atoi(test.in)
			if out != int(test.out) || err != test.err {
				t.Errorf("Atoi(%q) = %v, %v, want %v, %v", test.in, out, err, test.out, test.err)
			}
		}
	case 64:
		for i := range parseInt64Tests {
			test := &parseInt64Tests[i]
			out, err := Atoi(test.in)
			if int64(out) != test.out || err != test.err {
				t.Errorf("Atoi(%q) = %v, %v, want %v, %v", test.in, out, err, test.out, test.err)
			}
		}
	}
}

type parseErrorTest struct {
	arg int
	err error
}

var parseBitSizeTests = []parseErrorTest{
	{-1, ErrBitSize},
	{0, nil},
	{64, nil},
	{65, ErrBitSize},
}

var parseBaseTests = []parseErrorTest{
	{-1, ErrBase},
	{0, nil},
	{1, ErrBase},
	{2, nil},
	{36, nil},
	{37, ErrBase},
}

func equalError(a, b error) bool {
	if a == nil {
		return b == nil
	}
	if b == nil {
		return a == nil
	}
	return a.Error() == b.Error()
}

func TestParseIntBitSize(t *testing.T) {
	for i := range parseBitSizeTests {
		test := &parseBitSizeTests[i]
		_, err := ParseInt("0", 0, test.arg)
		if err != test.err {
			t.Errorf("ParseInt(\"0\", 0, %v) = 0, %v want 0, %v",
				test.arg, err, test.err)
		}
	}
}

func TestParseUintBitSize(t *testing.T) {
	for i := range parseBitSizeTests {
		test := &parseBitSizeTests[i]
		_, err := ParseUint("0", 0, test.arg)
		if err != test.err {
			t.Errorf("ParseUint(\"0\", 0, %v) = 0, %v want 0, %v",
				test.arg, err, test.err)
		}
	}
}

func TestParseIntBase(t *testing.T) {
	for i := range parseBaseTests {
		test := &parseBaseTests[i]
		_, err := ParseInt("0", test.arg, 0)
		if err != test.err {
			t.Errorf("ParseInt(\"0\", %v, 0) = 0, %v want 0, %v",
				test.arg, err, test.err)
		}
	}
}

func TestParseUintBase(t *testing.T) {
	for i := range parseBaseTests {
		test := &parseBaseTests[i]
		_, err := ParseUint("0", test.arg, 0)
		if err != test.err {
			t.Errorf("ParseUint(\"0\", %v, 0) = 0, %v want 0, %v",
				test.arg, err, test.err)
		}
	}
}

func BenchmarkParseInt(b *testing.B) {
	b.Run("Pos", func(b *testing.B) {
		benchmarkParseInt(b, 1)
	})
	b.Run("Neg", func(b *testing.B) {
		benchmarkParseInt(b, -1)
	})
}

type benchCase struct {
	name string
	num  int64
}

func benchmarkParseInt(b *testing.B, neg int) {
	cases := []benchCase{
		{"7bit", 1<<7 - 1},
		{"26bit", 1<<26 - 1},
		{"31bit", 1<<31 - 1},
		{"56bit", 1<<56 - 1},
		{"63bit", 1<<63 - 1},
	}
	for _, cs := range cases {
		b.Run(cs.name, func(b *testing.B) {
			s := fmt.Sprintf("%d", cs.num*int64(neg))
			for i := 0; i < b.N; i++ {
				out, _ := ParseInt(s, 10, 64)
				BenchSink += int(out)
			}
		})
	}
}

func BenchmarkAtoi(b *testing.B) {
	b.Run("Pos", func(b *testing.B) {
		benchmarkAtoi(b, 1)
	})
	b.Run("Neg", func(b *testing.B) {
		benchmarkAtoi(b, -1)
	})
}

func benchmarkAtoi(b *testing.B, neg int) {
	cases := []benchCase{
		{"7bit", 1<<7 - 1},
		{"26bit", 1<<26 - 1},
		{"31bit", 1<<31 - 1},
	}
	if IntSize == 64 {
		cases = append(cases, []benchCase{
			{"56bit", 1<<56 - 1},
			{"63bit", 1<<63 - 1},
		}...)
	}
	for _, cs := range cases {
		b.Run(cs.name, func(b *testing.B) {
			s := fmt.Sprintf("%d", cs.num*int64(neg))
			for i := 0; i < b.N; i++ {
				out, _ := Atoi(s)
				BenchSink += out
			}
		})
	}
}
