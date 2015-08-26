// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// string_ssa.go tests string operations.
package main

var failed = false

func testStringSlice1_ssa(a string, i, j int) string {
	switch { // prevent inlining
	}
	return a[i:]
}

func testStringSlice2_ssa(a string, i, j int) string {
	switch { // prevent inlining
	}
	return a[:j]
}

func testStringSlice12_ssa(a string, i, j int) string {
	switch { // prevent inlining
	}
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

func testStructSlice() {
	switch {
	}
	p := &prefix{"prefix"}
	p.slice_ssa()
	if "pre" != p.prefix {
		println("wrong field slice: wanted %s got %s", "pre", p.prefix)
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

func main() {
	testStringSlice()
	testStringSlicePanic()

	if failed {
		panic("failed")
	}
}
