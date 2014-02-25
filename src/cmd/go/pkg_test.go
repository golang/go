// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"reflect"
	"strings"
	"testing"
)

var foldDupTests = []struct {
	list   []string
	f1, f2 string
}{
	{stringList("math/rand", "math/big"), "", ""},
	{stringList("math", "strings"), "", ""},
	{stringList("strings"), "", ""},
	{stringList("strings", "strings"), "strings", "strings"},
	{stringList("Rand", "rand", "math", "math/rand", "math/Rand"), "Rand", "rand"},
}

func TestFoldDup(t *testing.T) {
	for _, tt := range foldDupTests {
		f1, f2 := foldDup(tt.list)
		if f1 != tt.f1 || f2 != tt.f2 {
			t.Errorf("foldDup(%q) = %q, %q, want %q, %q", tt.list, f1, f2, tt.f1, tt.f2)
		}
	}
}

var parseMetaGoImportsTests = []struct {
	in  string
	out []metaImport
}{
	{
		`<meta name="go-import" content="foo/bar git https://github.com/rsc/foo/bar">`,
		[]metaImport{{"foo/bar", "git", "https://github.com/rsc/foo/bar"}},
	},
	{
		`<meta name="go-import" content="foo/bar git https://github.com/rsc/foo/bar">
		<meta name="go-import" content="baz/quux git http://github.com/rsc/baz/quux">`,
		[]metaImport{
			{"foo/bar", "git", "https://github.com/rsc/foo/bar"},
			{"baz/quux", "git", "http://github.com/rsc/baz/quux"},
		},
	},
	{
		`<head>
		<meta name="go-import" content="foo/bar git https://github.com/rsc/foo/bar">
		</head>`,
		[]metaImport{{"foo/bar", "git", "https://github.com/rsc/foo/bar"}},
	},
	{
		`<meta name="go-import" content="foo/bar git https://github.com/rsc/foo/bar">
		<body>`,
		[]metaImport{{"foo/bar", "git", "https://github.com/rsc/foo/bar"}},
	},
}

func TestParseMetaGoImports(t *testing.T) {
	for i, tt := range parseMetaGoImportsTests {
		out, err := parseMetaGoImports(strings.NewReader(tt.in))
		if err != nil {
			t.Errorf("test#%d: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(out, tt.out) {
			t.Errorf("test#%d:\n\thave %q\n\twant %q", i, out, tt.out)
		}
	}
}
