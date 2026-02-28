// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// string_ssa.go tests string operations.
package main

import "testing"

//go:noinline
func testStringSlice1_ssa(a string, i, j int) string {
	return a[i:]
}

//go:noinline
func testStringSlice2_ssa(a string, i, j int) string {
	return a[:j]
}

//go:noinline
func testStringSlice12_ssa(a string, i, j int) string {
	return a[i:j]
}

func testStringSlice(t *testing.T) {
	tests := [...]struct {
		fn        func(string, int, int) string
		s         string
		low, high int
		want      string
	}{
		// -1 means the value is not used.
		{testStringSlice1_ssa, "foobar", 0, -1, "foobar"},
		{testStringSlice1_ssa, "foobar", 3, -1, "bar"},
		{testStringSlice1_ssa, "foobar", 6, -1, ""},
		{testStringSlice2_ssa, "foobar", -1, 0, ""},
		{testStringSlice2_ssa, "foobar", -1, 3, "foo"},
		{testStringSlice2_ssa, "foobar", -1, 6, "foobar"},
		{testStringSlice12_ssa, "foobar", 0, 6, "foobar"},
		{testStringSlice12_ssa, "foobar", 0, 0, ""},
		{testStringSlice12_ssa, "foobar", 6, 6, ""},
		{testStringSlice12_ssa, "foobar", 1, 5, "ooba"},
		{testStringSlice12_ssa, "foobar", 3, 3, ""},
		{testStringSlice12_ssa, "", 0, 0, ""},
	}

	for i, test := range tests {
		if got := test.fn(test.s, test.low, test.high); test.want != got {
			t.Errorf("#%d %s[%d,%d] = %s, want %s", i, test.s, test.low, test.high, got, test.want)
		}
	}
}

type prefix struct {
	prefix string
}

func (p *prefix) slice_ssa() {
	p.prefix = p.prefix[:3]
}

//go:noinline
func testStructSlice(t *testing.T) {
	p := &prefix{"prefix"}
	p.slice_ssa()
	if "pre" != p.prefix {
		t.Errorf("wrong field slice: wanted %s got %s", "pre", p.prefix)
	}
}

func testStringSlicePanic(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			//println("panicked as expected")
		}
	}()

	str := "foobar"
	t.Errorf("got %s and expected to panic, but didn't", testStringSlice12_ssa(str, 3, 9))
}

const _Accuracy_name = "BelowExactAbove"

var _Accuracy_index = [...]uint8{0, 5, 10, 15}

//go:noinline
func testSmallIndexType_ssa(i int) string {
	return _Accuracy_name[_Accuracy_index[i]:_Accuracy_index[i+1]]
}

func testSmallIndexType(t *testing.T) {
	tests := []struct {
		i    int
		want string
	}{
		{0, "Below"},
		{1, "Exact"},
		{2, "Above"},
	}

	for i, test := range tests {
		if got := testSmallIndexType_ssa(test.i); got != test.want {
			t.Errorf("#%d got %s wanted %s", i, got, test.want)
		}
	}
}

//go:noinline
func testInt64Index_ssa(s string, i int64) byte {
	return s[i]
}

//go:noinline
func testInt64Slice_ssa(s string, i, j int64) string {
	return s[i:j]
}

func testInt64Index(t *testing.T) {
	tests := []struct {
		i int64
		j int64
		b byte
		s string
	}{
		{0, 5, 'B', "Below"},
		{5, 10, 'E', "Exact"},
		{10, 15, 'A', "Above"},
	}

	str := "BelowExactAbove"
	for i, test := range tests {
		if got := testInt64Index_ssa(str, test.i); got != test.b {
			t.Errorf("#%d got %d wanted %d", i, got, test.b)
		}
		if got := testInt64Slice_ssa(str, test.i, test.j); got != test.s {
			t.Errorf("#%d got %s wanted %s", i, got, test.s)
		}
	}
}

func testInt64IndexPanic(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			//println("panicked as expected")
		}
	}()

	str := "foobar"
	t.Errorf("got %d and expected to panic, but didn't", testInt64Index_ssa(str, 1<<32+1))
}

func testInt64SlicePanic(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			//println("panicked as expected")
		}
	}()

	str := "foobar"
	t.Errorf("got %s and expected to panic, but didn't", testInt64Slice_ssa(str, 1<<32, 1<<32+1))
}

//go:noinline
func testStringElem_ssa(s string, i int) byte {
	return s[i]
}

func testStringElem(t *testing.T) {
	tests := []struct {
		s string
		i int
		n byte
	}{
		{"foobar", 3, 98},
		{"foobar", 0, 102},
		{"foobar", 5, 114},
	}
	for _, test := range tests {
		if got := testStringElem_ssa(test.s, test.i); got != test.n {
			t.Errorf("testStringElem \"%s\"[%d] = %d, wanted %d", test.s, test.i, got, test.n)
		}
	}
}

//go:noinline
func testStringElemConst_ssa(i int) byte {
	s := "foobar"
	return s[i]
}

func testStringElemConst(t *testing.T) {
	if got := testStringElemConst_ssa(3); got != 98 {
		t.Errorf("testStringElemConst= %d, wanted 98", got)
	}
}

func TestString(t *testing.T) {
	testStringSlice(t)
	testStringSlicePanic(t)
	testStructSlice(t)
	testSmallIndexType(t)
	testStringElem(t)
	testStringElemConst(t)
	testInt64Index(t)
	testInt64IndexPanic(t)
	testInt64SlicePanic(t)
}
