// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// string_ssa.go tests string operations.
package main

var failed = false

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

func testStringSlice() {
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

	for i, t := range tests {
		if got := t.fn(t.s, t.low, t.high); t.want != got {
			println("#", i, " ", t.s, "[", t.low, ":", t.high, "] = ", got, " want ", t.want)
			failed = true
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
func testStructSlice() {
	p := &prefix{"prefix"}
	p.slice_ssa()
	if "pre" != p.prefix {
		println("wrong field slice: wanted %s got %s", "pre", p.prefix)
		failed = true
	}
}

func testStringSlicePanic() {
	defer func() {
		if r := recover(); r != nil {
			println("paniced as expected")
		}
	}()

	str := "foobar"
	println("got ", testStringSlice12_ssa(str, 3, 9))
	println("expected to panic, but didn't")
	failed = true
}

const _Accuracy_name = "BelowExactAbove"

var _Accuracy_index = [...]uint8{0, 5, 10, 15}

//go:noinline
func testSmallIndexType_ssa(i int) string {
	return _Accuracy_name[_Accuracy_index[i]:_Accuracy_index[i+1]]
}

func testSmallIndexType() {
	tests := []struct {
		i    int
		want string
	}{
		{0, "Below"},
		{1, "Exact"},
		{2, "Above"},
	}

	for i, t := range tests {
		if got := testSmallIndexType_ssa(t.i); got != t.want {
			println("#", i, "got ", got, ", wanted", t.want)
			failed = true
		}
	}
}

//go:noinline
func testStringElem_ssa(s string, i int) byte {
	return s[i]
}

func testStringElem() {
	tests := []struct {
		s string
		i int
		n byte
	}{
		{"foobar", 3, 98},
		{"foobar", 0, 102},
		{"foobar", 5, 114},
	}
	for _, t := range tests {
		if got := testStringElem_ssa(t.s, t.i); got != t.n {
			print("testStringElem \"", t.s, "\"[", t.i, "]=", got, ", wanted ", t.n, "\n")
			failed = true
		}
	}
}

//go:noinline
func testStringElemConst_ssa(i int) byte {
	s := "foobar"
	return s[i]
}

func testStringElemConst() {
	if got := testStringElemConst_ssa(3); got != 98 {
		println("testStringElemConst=", got, ", wanted 98")
		failed = true
	}
}

func main() {
	testStringSlice()
	testStringSlicePanic()
	testStructSlice()
	testSmallIndexType()
	testStringElem()
	testStringElemConst()

	if failed {
		panic("failed")
	}
}
