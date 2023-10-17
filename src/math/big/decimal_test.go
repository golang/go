// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"fmt"
	"testing"
)

func TestDecimalString(t *testing.T) {
	for _, test := range []struct {
		x    decimal
		want string
	}{
		{want: "0"},
		{decimal{nil, 1000}, "0"}, // exponent of 0 is ignored
		{decimal{[]byte("12345"), 0}, "0.12345"},
		{decimal{[]byte("12345"), -3}, "0.00012345"},
		{decimal{[]byte("12345"), +3}, "123.45"},
		{decimal{[]byte("12345"), +10}, "1234500000"},
	} {
		if got := test.x.String(); got != test.want {
			t.Errorf("%v == %s; want %s", test.x, got, test.want)
		}
	}
}

func TestDecimalInit(t *testing.T) {
	for _, test := range []struct {
		x     Word
		shift int
		want  string
	}{
		{0, 0, "0"},
		{0, -100, "0"},
		{0, 100, "0"},
		{1, 0, "1"},
		{1, 10, "1024"},
		{1, 100, "1267650600228229401496703205376"},
		{1, -100, "0.0000000000000000000000000000007888609052210118054117285652827862296732064351090230047702789306640625"},
		{12345678, 8, "3160493568"},
		{12345678, -8, "48225.3046875"},
		{195312, 9, "99999744"},
		{1953125, 9, "1000000000"},
	} {
		var d decimal
		d.init(nat{test.x}.norm(), test.shift)
		if got := d.String(); got != test.want {
			t.Errorf("%d << %d == %s; want %s", test.x, test.shift, got, test.want)
		}
	}
}

func TestDecimalRounding(t *testing.T) {
	for _, test := range []struct {
		x              uint64
		n              int
		down, even, up string
	}{
		{0, 0, "0", "0", "0"},
		{0, 1, "0", "0", "0"},

		{1, 0, "0", "0", "10"},
		{5, 0, "0", "0", "10"},
		{9, 0, "0", "10", "10"},

		{15, 1, "10", "20", "20"},
		{45, 1, "40", "40", "50"},
		{95, 1, "90", "100", "100"},

		{12344999, 4, "12340000", "12340000", "12350000"},
		{12345000, 4, "12340000", "12340000", "12350000"},
		{12345001, 4, "12340000", "12350000", "12350000"},
		{23454999, 4, "23450000", "23450000", "23460000"},
		{23455000, 4, "23450000", "23460000", "23460000"},
		{23455001, 4, "23450000", "23460000", "23460000"},

		{99994999, 4, "99990000", "99990000", "100000000"},
		{99995000, 4, "99990000", "100000000", "100000000"},
		{99999999, 4, "99990000", "100000000", "100000000"},

		{12994999, 4, "12990000", "12990000", "13000000"},
		{12995000, 4, "12990000", "13000000", "13000000"},
		{12999999, 4, "12990000", "13000000", "13000000"},
	} {
		x := nat(nil).setUint64(test.x)

		var d decimal
		d.init(x, 0)
		d.roundDown(test.n)
		if got := d.String(); got != test.down {
			t.Errorf("roundDown(%d, %d) = %s; want %s", test.x, test.n, got, test.down)
		}

		d.init(x, 0)
		d.round(test.n)
		if got := d.String(); got != test.even {
			t.Errorf("round(%d, %d) = %s; want %s", test.x, test.n, got, test.even)
		}

		d.init(x, 0)
		d.roundUp(test.n)
		if got := d.String(); got != test.up {
			t.Errorf("roundUp(%d, %d) = %s; want %s", test.x, test.n, got, test.up)
		}
	}
}

var sink string

func BenchmarkDecimalConversion(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for shift := -100; shift <= +100; shift++ {
			var d decimal
			d.init(natOne, shift)
			sink = d.String()
		}
	}
}

func BenchmarkFloatString(b *testing.B) {
	x := new(Float)
	for _, prec := range []uint{1e2, 1e3, 1e4, 1e5} {
		x.SetPrec(prec).SetRat(NewRat(1, 3))
		b.Run(fmt.Sprintf("%v", prec), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				sink = x.String()
			}
		})
	}
}
