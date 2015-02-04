// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"strconv"
	"testing"
)

func TestFloatSetFloat64String(t *testing.T) {
	for _, test := range []struct {
		s string
		x float64
	}{
		{"0", 0},
		{"-0", -0},
		{"+0", 0},
		{"1", 1},
		{"-1", -1},
		{"+1", 1},
		{"1.234", 1.234},
		{"-1.234", -1.234},
		{"+1.234", 1.234},
		{".1", 0.1},
		{"1.", 1},
		{"+1.", 1},

		{"0e100", 0},
		{"-0e+100", 0},
		{"+0e-100", 0},
		{"0E100", 0},
		{"-0E+100", 0},
		{"+0E-100", 0},
		{"0p100", 0},
		{"-0p+100", 0},
		{"+0p-100", 0},

		{"1.e10", 1e10},
		{"1e+10", 1e10},
		{"+1e-10", 1e-10},
		{"1E10", 1e10},
		{"1.E+10", 1e10},
		{"+1E-10", 1e-10},
		{"1p10", 1 << 10},
		{"1p+10", 1 << 10},
		{"+1.p-10", 1.0 / (1 << 10)},

		{"-687436.79457e-245", -687436.79457e-245},
		{"-687436.79457E245", -687436.79457e245},
		{"1024.p-12", 0.25},
		{"-1.p10", -1024},
		{"0.25p2", 1},

		{".0000000000000000000000000000000000000001", 1e-40},
		{"+10000000000000000000000000000000000000000e-0", 1e40},
	} {
		var x Float
		x.prec = 53 // TODO(gri) find better solution
		_, ok := x.SetString(test.s)
		if !ok {
			t.Errorf("%s: parse error", test.s)
			continue
		}
		f, _ := x.Float64()
		want := new(Float).SetFloat64(test.x)
		if x.Cmp(want) != 0 {
			t.Errorf("%s: got %s (%v); want %v", test.s, &x, f, test.x)
		}
	}
}

func TestFloatFormat(t *testing.T) {
	for _, test := range []struct {
		x      string
		format byte
		prec   int
		want   string
	}{
		{"0", 'b', 0, "0"},
		{"-0", 'b', 0, "-0"},
		{"1.0", 'b', 0, "4503599627370496p-52"},
		{"-1.0", 'b', 0, "-4503599627370496p-52"},
		{"4503599627370496", 'b', 0, "4503599627370496p+0"},

		{"0", 'p', 0, "0"},
		{"-0", 'p', 0, "-0"},
		{"1024.0", 'p', 0, "0x.8p11"},
		{"-1024.0", 'p', 0, "-0x.8p11"},
	} {
		f64, err := strconv.ParseFloat(test.x, 64)
		if err != nil {
			t.Error(err)
			continue
		}

		f := new(Float).SetFloat64(f64)
		got := f.Format(test.format, test.prec)
		if got != test.want {
			t.Errorf("%v: got %s; want %s", test, got, test.want)
		}

		if test.format == 'b' && f64 == 0 {
			continue // 'b' format in strconv.Float requires knowledge of bias for 0.0
		}
		if test.format == 'p' {
			continue // 'p' format not supported in strconv.Format
		}

		// verify that Float format matches strconv format
		want := strconv.FormatFloat(f64, test.format, test.prec, 64)
		if got != want {
			t.Errorf("%v: got %s; want %s", test, got, want)
		}
	}
}
