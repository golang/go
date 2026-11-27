// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ast

import (
	"testing"
)

var commentsText = []struct {
	list []string
	text string
}{
	{[]string{"//"}, ""},
	{[]string{"//   "}, ""},
	{[]string{"//", "//", "//   "}, ""},
	{[]string{"// foo   "}, "foo\n"},
	{[]string{"//", "//", "// foo"}, "foo\n"},
	{[]string{"// foo  bar  "}, "foo  bar\n"},
	{[]string{"// foo", "// bar"}, "foo\nbar\n"},
	{[]string{"// foo", "//", "//", "//", "// bar"}, "foo\n\nbar\n"},
	{[]string{"// foo", "/* bar */"}, "foo\n bar\n"},
	{[]string{"//", "//", "//", "// foo", "//", "//", "//"}, "foo\n"},

	{[]string{"/**/"}, ""},
	{[]string{"/*   */"}, ""},
	{[]string{"/**/", "/**/", "/*   */"}, ""},
	{[]string{"/* Foo   */"}, " Foo\n"},
	{[]string{"/* Foo  Bar  */"}, " Foo  Bar\n"},
	{[]string{"/* Foo*/", "/* Bar*/"}, " Foo\n Bar\n"},
	{[]string{"/* Foo*/", "/**/", "/**/", "/**/", "// Bar"}, " Foo\n\nBar\n"},
	{[]string{"/* Foo*/", "/*\n*/", "//", "/*\n*/", "// Bar"}, " Foo\n\nBar\n"},
	{[]string{"/* Foo*/", "// Bar"}, " Foo\nBar\n"},
	{[]string{"/* Foo\n Bar*/"}, " Foo\n Bar\n"},

	{[]string{"// foo", "//go:noinline", "// bar", "//:baz"}, "foo\nbar\n:baz\n"},
	{[]string{"// foo", "//lint123:ignore", "// bar"}, "foo\nbar\n"},
}

func TestCommentText(t *testing.T) {
	for i, c := range commentsText {
		list := make([]*Comment, len(c.list))
		for i, s := range c.list {
			list[i] = &Comment{Text: s}
		}

		text := (&CommentGroup{list}).Text()
		if text != c.text {
			t.Errorf("case %d: got %q; expected %q", i, text, c.text)
		}
	}
}

var commentsRaw = []struct {
	list []string
	raw  string
}{
	{[]string{"//"}, "//\n"},
	{[]string{"//   "}, "//   \n"},
	{[]string{"//", "//", "//   "}, "//\n//\n//   \n"},
	{[]string{"// foo   "}, "// foo   \n"},
	{[]string{"//", "//", "// foo"}, "//\n//\n// foo\n"},
	{[]string{"// foo  bar  "}, "// foo  bar  \n"},
	{[]string{"// foo", "// bar"}, "// foo\n// bar\n"},
	{[]string{"// foo", "//", "//", "//", "// bar"},
		"// foo\n//\n//\n//\n// bar\n"},
	{[]string{"// foo", "/* bar */"}, "// foo\n/* bar */\n"},
	{[]string{"//", "//", "//", "// foo", "//", "//", "//"},
		"//\n//\n//\n// foo\n//\n//\n//\n"},

	{[]string{"/**/"}, "/**/\n"},
	{[]string{"/*   */"}, "/*   */\n"},
	{[]string{"/**/", "/**/", "/*   */"},
		"/**/\n/**/\n/*   */\n"},
	{[]string{"/* Foo   */"}, "/* Foo   */\n"},
	{[]string{"/* Foo  Bar  */"}, "/* Foo  Bar  */\n"},
	{[]string{"/* Foo*/", "/* Bar*/"}, "/* Foo*/\n/* Bar*/\n"},
	{[]string{"/* Foo*/", "/**/", "/**/", "/**/", "// Bar"},
		"/* Foo*/\n/**/\n/**/\n/**/\n// Bar\n"},
	{[]string{"/* Foo*/", "/*\n*/", "//", "/*\n*/", "// Bar"},
		"/* Foo*/\n/*\n*/\n//\n/*\n*/\n// Bar\n"},
	{[]string{"/* Foo*/", "// Bar"}, "/* Foo*/\n// Bar\n"},
	{[]string{"/* Foo\n Bar*/"}, "/* Foo\n Bar*/\n"},

	{[]string{"// foo", "//go:noinline", "// bar", "//:baz"},
		"// foo\n//go:noinline\n// bar\n//:baz\n"},
	{[]string{"// foo", "//lint123:ignore", "// bar"},
		"// foo\n//lint123:ignore\n// bar\n"},
}

func TestCommentRaw(t *testing.T) {
	for i, c := range commentsRaw {
		list := make([]*Comment, len(c.list))
		for i, s := range c.list {
			list[i] = &Comment{Text: s}
		}

		text := (&CommentGroup{list}).Raw()
		if text != c.raw {
			t.Errorf("case %d: got %q; expected %q", i, text, c.raw)
		}
	}
}

var isDirectiveTests = []struct {
	in string
	ok bool
}{
	{"abc", false},
	{"go:inline", true},
	{"Go:inline", false},
	{"go:Inline", false},
	{":inline", false},
	{"lint:ignore", true},
	{"lint:1234", true},
	{"1234:lint", true},
	{"go: inline", false},
	{"go:", false},
	{"go:*", false},
	{"go:x*", true},
	{"export foo", true},
	{"extern foo", true},
	{"expert foo", false},
}

func TestIsDirective(t *testing.T) {
	for _, tt := range isDirectiveTests {
		if ok := isDirective(tt.in); ok != tt.ok {
			t.Errorf("isDirective(%q) = %v, want %v", tt.in, ok, tt.ok)
		}
	}
}
