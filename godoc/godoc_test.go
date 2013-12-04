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
		{"/src/pkg/fmt", "pkg/fmt"},
		{"/fmt", "pkg/fmt"},
	} {
		if got, want := pkgLinkFunc(tc.path), tc.want; got != want {
			t.Errorf("pkgLinkFunc(%v) = %v; want %v", tc.path, got, want)
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
		{"/src/pkg/fmt/print.go", 42, 30, 50, "/src/pkg/fmt/print.go?s=30:50#L32"},
		{"/src/pkg/fmt/print.go", 2, 1, 5, "/src/pkg/fmt/print.go?s=1:5#L1"},
		{"/src/pkg/fmt/print.go", 2, 0, 0, "/src/pkg/fmt/print.go#L2"},
		{"/src/pkg/fmt/print.go", 0, 0, 0, "/src/pkg/fmt/print.go"},
		{"/src/pkg/fmt/print.go", 0, 1, 5, "/src/pkg/fmt/print.go?s=1:5#L1"},
	} {
		if got, want := srcPosLinkFunc(tc.src, tc.line, tc.low, tc.high), tc.want; got != want {
			t.Errorf("srcLinkFunc(%v, %v, %v, %v) = %v; want %v", tc.src, tc.line, tc.low, tc.high, got, want)
		}
	}
}

func TestSrcLinkFunc(t *testing.T) {
	for _, tc := range []struct {
		src  string
		want string
	}{
		{"/src/pkg/fmt/print.go", "/src/pkg/fmt/print.go"},
		{"src/pkg/fmt/print.go", "/src/pkg/fmt/print.go"},
	} {
		if got, want := srcLinkFunc(tc.src), tc.want; got != want {
			t.Errorf("srcLinkFunc(%v) = %v; want %v", tc.src, got, want)
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
		{"/src/pkg/fmt/print.go", "Sprintf", 33, "/src/pkg/fmt/print.go?h=Sprintf#L33"},
		{"/src/pkg/fmt/print.go", "Sprintf", 0, "/src/pkg/fmt/print.go?h=Sprintf"},
		{"src/pkg/fmt/print.go", "EOF", 33, "/src/pkg/fmt/print.go?h=EOF#L33"},
		{"src/pkg/fmt/print.go", "a%3f+%26b", 1, "/src/pkg/fmt/print.go?h=a%3f+%26b#L1"},
	} {
		if got, want := queryLinkFunc(tc.src, tc.query, tc.line), tc.want; got != want {
			t.Errorf("queryLinkFunc(%v, %v, %v) = %v; want %v", tc.src, tc.query, tc.line, got, want)
		}
	}
}

func TestDocLinkFunc(t *testing.T) {
	for _, tc := range []struct {
		src   string
		ident string
		want  string
	}{
		{"/src/pkg/fmt", "Sprintf", "/pkg/fmt/#Sprintf"},
		{"/src/pkg/fmt", "EOF", "/pkg/fmt/#EOF"},
	} {
		if got, want := docLinkFunc(tc.src, tc.ident), tc.want; got != want {
			t.Errorf("docLinkFunc(%v, %v) = %v; want %v", tc.src, tc.ident, got, want)
		}
	}
}
