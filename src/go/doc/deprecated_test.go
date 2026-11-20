// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

func TestDeprecated(t *testing.T) {
	for _, test := range []struct {
		name string
		doc  string
		want string
		ok   bool
	}{
		{
			name: "first paragraph",
			doc:  "Deprecated: don't use this.\n",
			want: "don't use this.",
			ok:   true,
		},
		{
			name: "later paragraph",
			doc:  "F is a function.\n\nDeprecated: don't use this, use E instead.\n",
			want: "don't use this, use E instead.",
			ok:   true,
		},
		{
			name: "multi-line notice keeps line breaks",
			doc:  "F is a function.\n\nDeprecated: don't use this.\nIt will be removed.\n\nMore text.\n",
			want: "don't use this.\nIt will be removed.",
			ok:   true,
		},
		{
			name: "no trailing newline",
			doc:  "F is a function.\n\nDeprecated: don't use this.",
			want: "don't use this.",
			ok:   true,
		},
		{
			name: "first of two notices wins",
			doc:  "Deprecated: first.\n\nDeprecated: second.\n",
			want: "first.",
			ok:   true,
		},
		{
			name: "empty",
			doc:  "",
			want: "",
			ok:   false,
		},
		{
			name: "no marker",
			doc:  "F is a function.\n",
			want: "",
			ok:   false,
		},
		{
			name: "marker mid-paragraph does not match",
			doc:  "F is a function. Deprecated: don't use this.\n",
			want: "",
			ok:   false,
		},
		{
			name: "marker on later line of paragraph does not match",
			doc:  "F is a function.\nDeprecated on Tuesdays: don't use this.\n",
			want: "",
			ok:   false,
		},
		{
			name: "marker without trailing space does not match",
			doc:  "F is a function.\n\nDeprecated:no space here.\n",
			want: "",
			ok:   false,
		},
		{
			name: "bare marker paragraph does not match",
			doc:  "F is a function.\n\nDeprecated:\n",
			want: "",
			ok:   false,
		},
		{
			name: "lowercase marker does not match",
			doc:  "F is a function.\n\ndeprecated: don't use this.\n",
			want: "",
			ok:   false,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got, ok := Deprecated(test.doc); got != test.want || ok != test.ok {
				t.Errorf("Deprecated(%q) = %q, %v; want %q, %v",
					test.doc, got, ok, test.want, test.ok)
			}
		})
	}
}

func TestDeprecatedFuncDoc(t *testing.T) {
	// Deprecated operates on Doc strings as produced by this package:
	// exercise it end to end through NewFromFiles.
	const src = `package p

// F is a function.
//
// Deprecated: don't use this, use E instead.
func F() {}

// G is a function.
func G() {}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "p.go", src, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	p, err := NewFromFiles(fset, []*ast.File{f}, "p", AllDecls)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Funcs) != 2 {
		t.Fatalf("len(Funcs) = %d, want 2", len(p.Funcs))
	}
	if got, ok := Deprecated(p.Funcs[0].Doc); !ok || got != "don't use this, use E instead." {
		t.Errorf("Deprecated(F.Doc) = %q, %v; want %q, true",
			got, ok, "don't use this, use E instead.")
	}
	if got, ok := Deprecated(p.Funcs[1].Doc); ok || got != "" {
		t.Errorf("Deprecated(G.Doc) = %q, %v; want %q, false", got, ok, "")
	}
}
