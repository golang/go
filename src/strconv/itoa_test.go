// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	. "strconv"
	"testing"
)

type itob64Test struct {
	in   int64
	base int
	out  string
}

var itob64tests = []itob64Test{
	{0, 10, "0"},
	{1, 10, "1"},
	{-1, 10, "-1"},
	{12345678, 10, "12345678"},
	{-987654321, 10, "-987654321"},
	{1<<31 - 1, 10, "2147483647"},
	{-1<<31 + 1, 10, "-2147483647"},
	{1 << 31, 10, "2147483648"},
	{-1 << 31, 10, "-2147483648"},
	{1<<31 + 1, 10, "2147483649"},
	{-1<<31 - 1, 10, "-2147483649"},
	{1<<32 - 1, 10, "4294967295"},
	{-1<<32 + 1, 10, "-4294967295"},
	{1 << 32, 10, "4294967296"},
	{-1 << 32, 10, "-4294967296"},
	{1<<32 + 1, 10, "4294967297"},
	{-1<<32 - 1, 10, "-4294967297"},
	{1 << 50, 10, "1125899906842624"},
	{1<<63 - 1, 10, "9223372036854775807"},
	{-1<<63 + 1, 10, "-9223372036854775807"},
	{-1 << 63, 10, "-9223372036854775808"},

	{0, 2, "0"},
	{10, 2, "1010"},
	{-1, 2, "-1"},
	{1 << 15, 2, "1000000000000000"},

	{-8, 8, "-10"},
	{057635436545, 8, "57635436545"},
	{1 << 24, 8, "100000000"},

	{16, 16, "10"},
	{-0x123456789abcdef, 16, "-123456789abcdef"},
	{1<<63 - 1, 16, "7fffffffffffffff"},
	{1<<63 - 1, 2, "111111111111111111111111111111111111111111111111111111111111111"},
	{-1 << 63, 2, "-1000000000000000000000000000000000000000000000000000000000000000"},

	{16, 17, "g"},
	{25, 25, "10"},
	{(((((17*35+24)*35+21)*35+34)*35+12)*35+24)*35 + 32, 35, "holycow"},
	{(((((17*36+24)*36+21)*36+34)*36+12)*36+24)*36 + 32, 36, "holycow"},
}

func TestItoa(t *testing.T) {
	for _, test := range itob64tests {
		s := FormatInt(test.in, test.base)
		if s != test.out {
			t.Errorf("FormatInt(%v, %v) = %v want %v",
				test.in, test.base, s, test.out)
		}
		x := AppendInt([]byte("abc"), test.in, test.base)
		if string(x) != "abc"+test.out {
			t.Errorf("AppendInt(%q, %v, %v) = %q want %v",
				"abc", test.in, test.base, x, test.out)
		}

		if test.in >= 0 {
			s := FormatUint(uint64(test.in), test.base)
			if s != test.out {
				t.Errorf("FormatUint(%v, %v) = %v want %v",
					test.in, test.base, s, test.out)
			}
			x := AppendUint(nil, uint64(test.in), test.base)
			if string(x) != test.out {
				t.Errorf("AppendUint(%q, %v, %v) = %q want %v",
					"abc", uint64(test.in), test.base, x, test.out)
			}
		}

		if test.base == 10 && int64(int(test.in)) == test.in {
			s := Itoa(int(test.in))
			if s != test.out {
				t.Errorf("Itoa(%v) = %v want %v",
					test.in, s, test.out)
			}
		}
	}
}

type uitob64Test struct {
	in   uint64
	base int
	out  string
}

var uitob64tests = []uitob64Test{
	{1<<63 - 1, 10, "9223372036854775807"},
	{1 << 63, 10, "9223372036854775808"},
	{1<<63 + 1, 10, "9223372036854775809"},
	{1<<64 - 2, 10, "18446744073709551614"},
	{1<<64 - 1, 10, "18446744073709551615"},
	{1<<64 - 1, 2, "1111111111111111111111111111111111111111111111111111111111111111"},
}

func TestUitoa(t *testing.T) {
	for _, test := range uitob64tests {
		s := FormatUint(test.in, test.base)
		if s != test.out {
			t.Errorf("FormatUint(%v, %v) = %v want %v",
				test.in, test.base, s, test.out)
		}
		x := AppendUint([]byte("abc"), test.in, test.base)
		if string(x) != "abc"+test.out {
			t.Errorf("AppendUint(%q, %v, %v) = %q want %v",
				"abc", test.in, test.base, x, test.out)
		}

	}
}

var varlenUints = []struct {
	in  uint64
	out string
}{
	{1, "1"},
	{12, "12"},
	{123, "123"},
	{1234, "1234"},
	{12345, "12345"},
	{123456, "123456"},
	{1234567, "1234567"},
	{12345678, "12345678"},
	{123456789, "123456789"},
	{1234567890, "1234567890"},
	{12345678901, "12345678901"},
	{123456789012, "123456789012"},
	{1234567890123, "1234567890123"},
	{12345678901234, "12345678901234"},
	{123456789012345, "123456789012345"},
	{1234567890123456, "1234567890123456"},
	{12345678901234567, "12345678901234567"},
	{123456789012345678, "123456789012345678"},
	{1234567890123456789, "1234567890123456789"},
	{12345678901234567890, "12345678901234567890"},
}

func TestFormatUintVarlen(t *testing.T) {
	for _, test := range varlenUints {
		s := FormatUint(test.in, 10)
		if s != test.out {
			t.Errorf("FormatUint(%v, 10) = %v want %v", test.in, s, test.out)
		}
	}
}

func BenchmarkFormatInt(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for _, test := range itob64tests {
			s := FormatInt(test.in, test.base)
			BenchSink += len(s)
		}
	}
}

func BenchmarkAppendInt(b *testing.B) {
	dst := make([]byte, 0, 30)
	for i := 0; i < b.N; i++ {
		for _, test := range itob64tests {
			dst = AppendInt(dst[:0], test.in, test.base)
			BenchSink += len(dst)
		}
	}
}

func BenchmarkFormatUint(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for _, test := range uitob64tests {
			s := FormatUint(test.in, test.base)
			BenchSink += len(s)
		}
	}
}

func BenchmarkAppendUint(b *testing.B) {
	dst := make([]byte, 0, 30)
	for i := 0; i < b.N; i++ {
		for _, test := range uitob64tests {
			dst = AppendUint(dst[:0], test.in, test.base)
			BenchSink += len(dst)
		}
	}
}

func BenchmarkFormatIntSmall(b *testing.B) {
	smallInts := []int64{7, 42}
	for _, smallInt := range smallInts {
		b.Run(Itoa(int(smallInt)), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				s := FormatInt(smallInt, 10)
				BenchSink += len(s)
			}
		})
	}
}

func BenchmarkAppendIntSmall(b *testing.B) {
	dst := make([]byte, 0, 30)
	const smallInt = 42
	for i := 0; i < b.N; i++ {
		dst = AppendInt(dst[:0], smallInt, 10)
		BenchSink += len(dst)
	}
}

func BenchmarkAppendUintVarlen(b *testing.B) {
	for _, test := range varlenUints {
		b.Run(test.out, func(b *testing.B) {
			dst := make([]byte, 0, 30)
			for j := 0; j < b.N; j++ {
				dst = AppendUint(dst[:0], test.in, 10)
				BenchSink += len(dst)
			}
		})
	}
}

var BenchSink int // make sure compiler cannot optimize away benchmarks
