// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gover

import (
	"reflect"
	"testing"
)

func TestCompare(t *testing.T) { test2(t, compareTests, "Compare", Compare) }

var compareTests = []testCase2[string, string, int]{
	{"", "", 0},
	{"x", "x", 0},
	{"", "x", 0},
	{"1", "1.1", -1},
	{"1.5", "1.6", -1},
	{"1.5", "1.10", -1},
	{"1.6", "1.6.1", -1},
	{"1.19", "1.19.0", 0},
	{"1.19rc1", "1.19", -1},
	{"1.20", "1.20.0", 0},
	{"1.20rc1", "1.20", -1},
	{"1.21", "1.21.0", -1},
	{"1.21", "1.21rc1", -1},
	{"1.21rc1", "1.21.0", -1},
	{"1.6", "1.19", -1},
	{"1.19", "1.19.1", -1},
	{"1.19rc1", "1.19", -1},
	{"1.19rc1", "1.19.1", -1},
	{"1.19rc1", "1.19rc2", -1},
	{"1.19.0", "1.19.1", -1},
	{"1.19rc1", "1.19.0", -1},
	{"1.19alpha3", "1.19beta2", -1},
	{"1.19beta2", "1.19rc1", -1},
	{"1.9.2rc2", "1.9.2", -1},
	{"1.9.2rc2", "1.9.2rc3", -1},
	{"1.9.2beta2", "1.9.2rc3", -1},
	{"1.9.2alpha1", "1.9.2beta2", -1},
	{"1.1", "1.99999999999999998", -1},
	{"1.99999999999999998", "1.99999999999999999", -1},
}

func TestParse(t *testing.T) { test1(t, parseTests, "Parse", Parse) }

var parseTests = []testCase1[string, Version]{
	{"1", Version{"1", "0", "0", "", ""}},
	{"1.2", Version{"1", "2", "0", "", ""}},
	{"1.2.3", Version{"1", "2", "3", "", ""}},
	{"1.2rc3", Version{"1", "2", "", "rc", "3"}},
	{"1.20", Version{"1", "20", "0", "", ""}},
	{"1.21", Version{"1", "21", "", "", ""}},
	{"1.21rc3", Version{"1", "21", "", "rc", "3"}},
	{"1.21.0", Version{"1", "21", "0", "", ""}},
	{"1.24", Version{"1", "24", "", "", ""}},
	{"1.24rc3", Version{"1", "24", "", "rc", "3"}},
	{"1.24.0", Version{"1", "24", "0", "", ""}},
	{"1.9.2rc2", Version{"1", "9", "2", "rc", "2"}},
	{"1.8.5rc4", Version{"1", "8", "5", "rc", "4"}},
	{"1.8.2beta2", Version{"1", "8", "2", "beta", "2"}},
	{"1.999testmod", Version{"1", "999", "", "testmod", ""}},
	{"1.99999999999999999", Version{"1", "99999999999999999", "", "", ""}},
}

func TestLang(t *testing.T) { test1(t, langTests, "Lang", Lang) }

var langTests = []testCase1[string, string]{
	{"1.2rc3", "1.2"},
	{"1.2.3", "1.2"},
	{"1.2", "1.2"},
	{"1", "1"},
	{"1.9.2rc2", "1.9"},
	{"1.8.5rc4", "1.8"},
	{"1.999testmod", "1.999"},
}

func TestIsLang(t *testing.T) { test1(t, isLangTests, "IsLang", IsLang) }

var isLangTests = []testCase1[string, bool]{
	{"1.2rc3", false},
	{"1.2.3", false},
	{"1.9.2rc2", false},
	{"1.8.5rc5", false},
	{"1.999testmod", false},
	{"1.22", true},
	{"1.21", true},
	{"1.20", false}, // == 1.20.0
	{"1.19", false}, // == 1.20.0
	{"1.3", false},  // == 1.3.0
	{"1.2", false},  // == 1.2.0
	{"1", false},    // == 1.0.0
}

func TestIsValid(t *testing.T) { test1(t, isValidTests, "IsValid", IsValid) }

var isValidTests = []testCase1[string, bool]{
	{"1.2rc3", true},
	{"1.2.3", true},
	{"1.999testmod", true},
	{"1.600+auto", false},
	{"1.22", true},
	{"1.21.0", true},
	{"1.21rc2", true},
	{"1.21", true},
	{"1.20.0", true},
	{"1.20", true},
	{"1.19", true},
	{"1.8.5rc5", true},
	{"1.9.2rc2", true},
	{"1.9.2beta1", true},
	{"1.9.2alpha", true},
	{"1.3", true},
	{"1.2", true},
	{"1", true},
}

type testCase1[In, Out any] struct {
	in  In
	out Out
}

type testCase2[In1, In2, Out any] struct {
	in1 In1
	in2 In2
	out Out
}

type testCase3[In1, In2, In3, Out any] struct {
	in1 In1
	in2 In2
	in3 In3
	out Out
}

func test1[In, Out any](t *testing.T, tests []testCase1[In, Out], name string, f func(In) Out) {
	t.Helper()
	for _, tt := range tests {
		if out := f(tt.in); !reflect.DeepEqual(out, tt.out) {
			t.Errorf("%s(%v) = %v, want %v", name, tt.in, out, tt.out)
		}
	}
}

func test2[In1, In2, Out any](t *testing.T, tests []testCase2[In1, In2, Out], name string, f func(In1, In2) Out) {
	t.Helper()
	for _, tt := range tests {
		if out := f(tt.in1, tt.in2); !reflect.DeepEqual(out, tt.out) {
			t.Errorf("%s(%+v, %+v) = %+v, want %+v", name, tt.in1, tt.in2, out, tt.out)
		}
	}
}
