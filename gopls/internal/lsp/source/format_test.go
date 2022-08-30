// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"strings"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/tests/compare"
)

func TestImportPrefix(t *testing.T) {
	for i, tt := range []struct {
		input, want string
	}{
		{"package foo", "package foo"},
		{"package foo\n", "package foo\n"},
		{"package foo\n\nfunc f(){}\n", "package foo\n"},
		{"package foo\n\nimport \"fmt\"\n", "package foo\n\nimport \"fmt\""},
		{"package foo\nimport (\n\"fmt\"\n)\n", "package foo\nimport (\n\"fmt\"\n)"},
		{"\n\n\npackage foo\n", "\n\n\npackage foo\n"},
		{"// hi \n\npackage foo //xx\nfunc _(){}\n", "// hi \n\npackage foo //xx\n"},
		{"package foo //hi\n", "package foo //hi\n"},
		{"//hi\npackage foo\n//a\n\n//b\n", "//hi\npackage foo\n//a\n\n//b\n"},
		{
			"package a\n\nimport (\n  \"fmt\"\n)\n//hi\n",
			"package a\n\nimport (\n  \"fmt\"\n)\n//hi\n",
		},
		{`package a /*hi*/`, `package a /*hi*/`},
		{"package main\r\n\r\nimport \"go/types\"\r\n\r\n/*\r\n\r\n */\r\n", "package main\r\n\r\nimport \"go/types\"\r\n\r\n/*\r\n\r\n */\r\n"},
		{"package x; import \"os\"; func f() {}\n\n", "package x; import \"os\""},
		{"package x; func f() {fmt.Println()}\n\n", "package x"},
	} {
		got, err := importPrefix([]byte(tt.input))
		if err != nil {
			t.Fatal(err)
		}
		if d := compare.Text(tt.want, got); d != "" {
			t.Errorf("%d: failed for %q:\n%s", i, tt.input, d)
		}
	}
}

func TestCRLFFile(t *testing.T) {
	for i, tt := range []struct {
		input, want string
	}{
		{
			input: `package main

/*
Hi description
*/
func Hi() {
}
`,
			want: `package main

/*
Hi description
*/`,
		},
	} {
		got, err := importPrefix([]byte(strings.ReplaceAll(tt.input, "\n", "\r\n")))
		if err != nil {
			t.Fatal(err)
		}
		want := strings.ReplaceAll(tt.want, "\n", "\r\n")
		if d := compare.Text(want, got); d != "" {
			t.Errorf("%d: failed for %q:\n%s", i, tt.input, d)
		}
	}
}
