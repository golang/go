// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godoc

import (
	"testing"
)

func TestPkgLinkFunc(t *testing.T) {
	for _, tc := range []struct {
		path string
		want string
	}{
		{"/src/fmt", "pkg/fmt"},
		{"src/fmt", "pkg/fmt"},
		{"/fmt", "pkg/fmt"},
		{"fmt", "pkg/fmt"},
	} {
		if got := pkgLinkFunc(tc.path); got != tc.want {
			t.Errorf("pkgLinkFunc(%v) = %v; want %v", tc.path, got, tc.want)
		}
	}
}

func TestSrcPosLinkFunc(t *testing.T) {
	for _, tc := range []struct {
		src  string
		line int
		low  int
		high int
		want string
	}{
		{"/src/fmt/print.go", 42, 30, 50, "/src/fmt/print.go?s=30:50#L32"},
		{"/src/fmt/print.go", 2, 1, 5, "/src/fmt/print.go?s=1:5#L1"},
		{"/src/fmt/print.go", 2, 0, 0, "/src/fmt/print.go#L2"},
		{"/src/fmt/print.go", 0, 0, 0, "/src/fmt/print.go"},
		{"/src/fmt/print.go", 0, 1, 5, "/src/fmt/print.go?s=1:5#L1"},
		{"fmt/print.go", 0, 0, 0, "/src/fmt/print.go"},
		{"fmt/print.go", 0, 1, 5, "/src/fmt/print.go?s=1:5#L1"},
	} {
		if got := srcPosLinkFunc(tc.src, tc.line, tc.low, tc.high); got != tc.want {
			t.Errorf("srcLinkFunc(%v, %v, %v, %v) = %v; want %v", tc.src, tc.line, tc.low, tc.high, got, tc.want)
		}
	}
}

func TestSrcLinkFunc(t *testing.T) {
	for _, tc := range []struct {
		src  string
		want string
	}{
		{"/src/fmt/print.go", "/src/fmt/print.go"},
		{"src/fmt/print.go", "/src/fmt/print.go"},
		{"/fmt/print.go", "/src/fmt/print.go"},
		{"fmt/print.go", "/src/fmt/print.go"},
	} {
		if got := srcLinkFunc(tc.src); got != tc.want {
			t.Errorf("srcLinkFunc(%v) = %v; want %v", tc.src, got, tc.want)
		}
	}
}

func TestQueryLinkFunc(t *testing.T) {
	for _, tc := range []struct {
		src   string
		query string
		line  int
		want  string
	}{
		{"/src/fmt/print.go", "Sprintf", 33, "/src/fmt/print.go?h=Sprintf#L33"},
		{"/src/fmt/print.go", "Sprintf", 0, "/src/fmt/print.go?h=Sprintf"},
		{"src/fmt/print.go", "EOF", 33, "/src/fmt/print.go?h=EOF#L33"},
		{"src/fmt/print.go", "a%3f+%26b", 1, "/src/fmt/print.go?h=a%3f+%26b#L1"},
	} {
		if got := queryLinkFunc(tc.src, tc.query, tc.line); got != tc.want {
			t.Errorf("queryLinkFunc(%v, %v, %v) = %v; want %v", tc.src, tc.query, tc.line, got, tc.want)
		}
	}
}

func TestDocLinkFunc(t *testing.T) {
	for _, tc := range []struct {
		src   string
		ident string
		want  string
	}{
		{"fmt", "Sprintf", "/pkg/fmt/#Sprintf"},
		{"fmt", "EOF", "/pkg/fmt/#EOF"},
	} {
		if got := docLinkFunc(tc.src, tc.ident); got != tc.want {
			t.Errorf("docLinkFunc(%v, %v) = %v; want %v", tc.src, tc.ident, got, tc.want)
		}
	}
}

func TestSanitizeFunc(t *testing.T) {
	for _, tc := range []struct {
		src  string
		want string
	}{
		{},
		{"foo", "foo"},
		{"func   f()", "func f()"},
		{"func f(a int,)", "func f(a int)"},
		{"func f(a int,\n)", "func f(a int)"},
		{"func f(\n\ta int,\n\tb int,\n\tc int,\n)", "func f(a int, b int, c int)"},
		{"  (   a,   b,  c  )  ", "(a, b, c)"},
		{"(  a,  b, c    int, foo   bar  ,  )", "(a, b, c int, foo bar)"},
		{"{   a,   b}", "{a, b}"},
		{"[   a,   b]", "[a, b]"},
	} {
		if got := sanitizeFunc(tc.src); got != tc.want {
			t.Errorf("sanitizeFunc(%v) = %v; want %v", tc.src, got, tc.want)
		}
	}
}
