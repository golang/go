// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ast

import (
	"go/token"
	"reflect"
	"strings"
	"testing"
)

func TestParseDirectiveMatchesIsDirective(t *testing.T) {
	for _, tt := range isDirectiveTests {
		want := tt.ok
		if strings.HasPrefix(tt.in, "extern ") || strings.HasPrefix(tt.in, "export ") {
			// ParseDirective does NOT support extern or export, unlike
			// isDirective.
			want = false
		}

		if _, ok := ParseDirective(0, "//"+tt.in); ok != want {
			t.Errorf("ParseDirective(0, %q) = %v, want %v", "// "+tt.in, ok, want)
		}
	}
}

func TestParseDirective(t *testing.T) {
	for _, test := range []struct {
		name   string
		in     string
		pos    token.Pos
		want   Directive
		wantOK bool
	}{
		{
			name: "valid",
			in:   "//go:generate stringer -type Op -trimprefix Op",
			pos:  10,
			want: Directive{
				Tool:    "go",
				Name:    "generate",
				Args:    "stringer -type Op -trimprefix Op",
				Slash:   10,
				ArgsPos: token.Pos(10 + len("//go:generate ")),
			},
			wantOK: true,
		},
		{
			name: "no args",
			in:   "//go:build ignore",
			pos:  20,
			want: Directive{
				Tool:    "go",
				Name:    "build",
				Args:    "ignore",
				Slash:   20,
				ArgsPos: token.Pos(20 + len("//go:build ")),
			},
			wantOK: true,
		},
		{
			name:   "not a directive",
			in:     "// not a directive",
			pos:    30,
			wantOK: false,
		},
		{
			name:   "not a comment",
			in:     "go:generate",
			pos:    40,
			wantOK: false,
		},
		{
			name:   "empty",
			in:     "",
			pos:    50,
			wantOK: false,
		},
		{
			name:   "just slashes",
			in:     "//",
			pos:    60,
			wantOK: false,
		},
		{
			name:   "no name",
			in:     "//go:",
			pos:    70,
			wantOK: false,
		},
		{
			name:   "no tool",
			in:     "//:generate",
			pos:    80,
			wantOK: false,
		},
		{
			name: "multiple spaces",
			in:   "//go:build  foo bar",
			pos:  90,
			want: Directive{
				Tool:    "go",
				Name:    "build",
				Args:    "foo bar",
				Slash:   90,
				ArgsPos: token.Pos(90 + len("//go:build  ")),
			},
			wantOK: true,
		},
		{
			name: "trailing space",
			in:   "//go:build foo ",
			pos:  100,
			want: Directive{
				Tool:    "go",
				Name:    "build",
				Args:    "foo",
				Slash:   100,
				ArgsPos: token.Pos(100 + len("//go:build ")),
			},
			wantOK: true,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, gotOK := ParseDirective(test.pos, test.in)
			if gotOK != test.wantOK {
				t.Fatalf("ParseDirective(%q) ok = %v, want %v", test.in, gotOK, test.wantOK)
			}
			if !reflect.DeepEqual(got, test.want) {
				t.Errorf("ParseDirective(%q) = %+v, want %+v", test.in, got, test.want)
			}
		})
	}
}

func TestParseArgs(t *testing.T) {
	for _, test := range []struct {
		name    string
		in      Directive
		want    []DirectiveArg
		wantErr bool
	}{
		{
			name: "simple",
			in: Directive{
				Tool:    "go",
				Name:    "generate",
				Args:    "stringer -type Op",
				ArgsPos: 10,
			},
			want: []DirectiveArg{
				{"stringer", 10},
				{"-type", token.Pos(10 + len("stringer "))},
				{"Op", token.Pos(10 + len("stringer -type "))},
			},
		},
		{
			name: "quoted",
			in: Directive{
				Tool:    "go",
				Name:    "generate",
				Args:    "\"foo bar\" baz",
				ArgsPos: 10,
			},
			want: []DirectiveArg{
				{"foo bar", 10},
				{"baz", token.Pos(10 + len("\"foo bar\" "))},
			},
		},
		{
			name: "raw quoted",
			in: Directive{
				Tool:    "go",
				Name:    "generate",
				Args:    "`foo bar` baz",
				ArgsPos: 10,
			},
			want: []DirectiveArg{
				{"foo bar", 10},
				{"baz", token.Pos(10 + len("`foo bar` "))},
			},
		},
		{
			name: "escapes",
			in: Directive{
				Tool:    "go",
				Name:    "generate",
				Args:    "\"foo\\U0001F60Abar\" `a\\tb`",
				ArgsPos: 10,
			},
			want: []DirectiveArg{
				{"foo😊bar", 10},
				{"a\\tb", token.Pos(10 + len("\"foo\\U0001F60Abar\" "))},
			},
		},
		{
			name: "empty args",
			in: Directive{
				Tool:    "go",
				Name:    "build",
				Args:    "",
				ArgsPos: 10,
			},
			want: []DirectiveArg{},
		},
		{
			name: "spaces",
			in: Directive{
				Tool:    "go",
				Name:    "build",
				Args:    "  foo   bar  ",
				ArgsPos: 10,
			},
			want: []DirectiveArg{
				{"foo", token.Pos(10 + len("  "))},
				{"bar", token.Pos(10 + len("  foo   "))},
			},
		},
		{
			name: "unterminated quote",
			in: Directive{
				Tool: "go",
				Name: "generate",
				Args: "`foo",
			},
			wantErr: true,
		},
		{
			name: "no space after quote",
			in: Directive{
				Tool: "go",
				Name: "generate",
				Args: `"foo"bar`,
			},
			wantErr: true,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := test.in.ParseArgs()
			if err != nil && !test.wantErr {
				t.Errorf("got ParseArgs(%+v) = error %s; want %+v", test.in, err, test.want)
			} else if err == nil && test.wantErr {
				t.Errorf("got ParseArgs(%+v) = %+v; want error", test.in, got)
			} else if err == nil && !reflect.DeepEqual(got, test.want) {
				t.Errorf("got ParseArgs(%+v) = %+v; want %+v", test.in, got, test.want)
			}
		})
	}
}
