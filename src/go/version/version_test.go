// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package version

import (
	"reflect"
	"testing"
)

func TestCompare(t *testing.T) { test2(t, compareTests, "Compare", Compare) }

var compareTests = []testCase2[string, string, int]{
	{"", "", 0},
	{"x", "x", 0},
	{"", "x", 0},
	{"1", "1.1", 0},
	{"go1", "go1.1", -1},
	{"go1.5", "go1.6", -1},
	{"go1.5", "go1.10", -1},
	{"go1.6", "go1.6.1", -1},
	{"go1.19", "go1.19.0", 0},
	{"go1.19rc1", "go1.19", -1},
	{"go1.20", "go1.20.0", 0},
	{"go1.20", "go1.20.0-bigcorp", 0},
	{"go1.20rc1", "go1.20", -1},
	{"go1.21", "go1.21.0", -1},
	{"go1.21", "go1.21.0-bigcorp", -1},
	{"go1.21", "go1.21rc1", -1},
	{"go1.21rc1", "go1.21.0", -1},
	{"go1.6", "go1.19", -1},
	{"go1.19", "go1.19.1", -1},
	{"go1.19rc1", "go1.19", -1},
	{"go1.19rc1", "go1.19", -1},
	{"go1.19rc1", "go1.19.1", -1},
	{"go1.19rc1", "go1.19rc2", -1},
	{"go1.19.0", "go1.19.1", -1},
	{"go1.19rc1", "go1.19.0", -1},
	{"go1.19alpha3", "go1.19beta2", -1},
	{"go1.19beta2", "go1.19rc1", -1},
	{"go1.1", "go1.99999999999999998", -1},
	{"go1.99999999999999998", "go1.99999999999999999", -1},
}

func TestLang(t *testing.T) { test1(t, langTests, "Lang", Lang) }

var langTests = []testCase1[string, string]{
	{"bad", ""},
	{"go1.2rc3", "go1.2"},
	{"go1.2.3", "go1.2"},
	{"go1.2", "go1.2"},
	{"go1", "go1"},
	{"go222", "go222.0"},
	{"go1.999testmod", "go1.999"},
}

func TestIsValid(t *testing.T) { test1(t, isValidTests, "IsValid", IsValid) }

var isValidTests = []testCase1[string, bool]{
	{"", false},
	{"1.2.3", false},
	{"go1.2rc3", true},
	{"go1.2.3", true},
	{"go1.999testmod", true},
	{"go1.600+auto", false},
	{"go1.22", true},
	{"go1.21.0", true},
	{"go1.21rc2", true},
	{"go1.21", true},
	{"go1.20.0", true},
	{"go1.20", true},
	{"go1.19", true},
	{"go1.3", true},
	{"go1.2", true},
	{"go1", true},
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
