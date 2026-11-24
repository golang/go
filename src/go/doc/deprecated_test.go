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
	tests := []struct {
		doc  string
		want string
		ok   bool
	}{
		{"Deprecated: don't use this.\n", "don't use this.", true},
		{"F is a function.\n\nDeprecated: don't use this, use E instead.\n", "don't use this, use E instead.", true},
		{"F is a function.\n\nDeprecated: don't use this.\nIt will be removed.\n\nMore text.\n", "don't use this.\nIt will be removed.", true},
		{"F is a function.\n\nDeprecated: don't use this.", "don't use this.", true},
		{"Deprecated: first.\n\nDeprecated: second.\n", "first.", true},
		{"\nDeprecated: leading newline.\n", "leading newline.", true},
		{"F is a function.\n\n\nDeprecated: extra blank line.\n", "extra blank line.", true},
		{" Deprecated: single-line block comment.\n", "single-line block comment.", true},
		{"\tF is a function.\n\n\tDeprecated: indented block.\n\tSecond line.\n", "indented block.\nSecond line.", true},
		{"Deprecated: \n", "", true},
		{"F is a function.\n\tcode\nDeprecated: after code block.\n", "after code block.", true},
		{"Deprecated: before code block.\n\tcode\n", "before code block.", true},

		{"", "", false},
		{"F is a function.\n", "", false},
		{"F is a function. Deprecated: don't use this.\n", "", false},
		{"F is a function.\nDeprecated: don't use this.\n", "", false},
		{"F is a function.\n\nDeprecated:no space here.\n", "", false},
		{"F is a function.\n\nDeprecated:\n", "", false},
		{"F is a function.\n\ndeprecated: don't use this.\n", "", false},
		{"F is a function.\n\n\tDeprecated: indented code block.\n", "", false},
		{"F is a function.\n\n    Deprecated: space-indented code block.\n", "", false},
		{"\tDeprecated: code block before text.\n\nF is a function.\n", "", false},
	}
	for _, tt := range tests {
		if got, ok := Deprecated(tt.doc); got != tt.want || ok != tt.ok {
			t.Errorf("Deprecated(%q) = %q, %v; want %q, %v", tt.doc, got, ok, tt.want, tt.ok)
		}
	}
}

func TestDeprecatedNewFromFiles(t *testing.T) {
	files := map[string]string{
		"a.go": `// Package p does things.
package p

// F is a function.
//
// Deprecated: use G instead.
func F() {}

/* Deprecated: use G instead. */
func Block() {}

/*
	IndentedBlock is a function.

	Deprecated: use G instead.
	It will be removed.
*/
func IndentedBlock() {}

// CodeBlock shows an example.
//
//	Deprecated: not a deprecation notice.
func CodeBlock() {}

type (
	/*
		Grouped is a type.

		Deprecated: use G instead.
	*/
	Grouped int
)
`,
		// Directive-only doc comment between package comments causes
		// reader.readDoc to insert an extra blank line ("\n\n\n").
		"b.go": `//go:build unix
package p
`,
		"c.go": `// Deprecated: use package q instead.
package p
`,
	}

	fset := token.NewFileSet()
	var parsed []*ast.File
	for _, name := range []string{"a.go", "b.go", "c.go"} {
		f, err := parser.ParseFile(fset, name, files[name], parser.ParseComments)
		if err != nil {
			t.Fatal(err)
		}
		parsed = append(parsed, f)
	}
	p, err := NewFromFiles(fset, parsed, "p", AllDecls)
	if err != nil {
		t.Fatal(err)
	}

	if got, ok := Deprecated(p.Doc); !ok || got != "use package q instead." {
		t.Errorf("Deprecated(Package.Doc) = %q, %v; want %q, true", got, ok, "use package q instead.")
	}

	wantFuncs := map[string]struct {
		text string
		ok   bool
	}{
		"F":             {"use G instead.", true},
		"Block":         {"use G instead.", true},
		"IndentedBlock": {"use G instead.\nIt will be removed.", true},
		"CodeBlock":     {"", false},
	}
	for _, fn := range p.Funcs {
		want, found := wantFuncs[fn.Name]
		if !found {
			t.Fatalf("unexpected func %q", fn.Name)
		}
		delete(wantFuncs, fn.Name)
		if got, ok := Deprecated(fn.Doc); got != want.text || ok != want.ok {
			t.Errorf("Deprecated(%s.Doc) = %q, %v; want %q, %v", fn.Name, got, ok, want.text, want.ok)
		}
	}
	if len(wantFuncs) > 0 {
		t.Fatalf("missing funcs: %v", wantFuncs)
	}

	// gofmt does not unindent comments inside a grouped declaration.
	if len(p.Types) != 1 {
		t.Fatalf("got %d types, want 1", len(p.Types))
	}
	if got, ok := Deprecated(p.Types[0].Doc); !ok || got != "use G instead." {
		t.Errorf("Deprecated(Grouped.Doc) = %q, %v; want %q, true", got, ok, "use G instead.")
	}
}
