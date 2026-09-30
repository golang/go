// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specdoc

import (
	"go/ast"
	"go/parser"
	"go/token"
	"simd/archsimd/_gen/specgen"
	"testing"
)

func TestParseCommentAndFormat(t *testing.T) {
	doc := `// Add adds elements of
// x and y.
//
//	z[i] = x[i] + y[i]
//
// Asm: VPADDD, CPU Feature: AVX
//
// Deprecated: use Add2 instead.
//
//go:noescape
//go:noinline
`
	src := "package p\n\n" + doc + "func Add()"

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "test.go", src, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}

	fd := f.Decls[0].(*ast.FuncDecl)
	paras := parseCommentGroup(fd.Doc)

	expected := []struct {
		kind specgen.ParagraphKind
		text string
	}{
		{specgen.SpecOwned, "Add adds elements of\nx and y."},
		{specgen.SpecOwned, "\tz[i] = x[i] + y[i]"},
		{specgen.ImplementationNote, "Asm: VPADDD, CPU Feature: AVX"},
		{specgen.ImplementationNote, "Deprecated: use Add2 instead."},
		{specgen.DirectiveComment, "go:noescape"},
		{specgen.DirectiveComment, "go:noinline"},
	}
	if len(paras) != len(expected) {
		t.Fatalf("expected %d paragraphs, got %d", len(expected), len(paras))
	}
	for i, want := range expected {
		if paras[i].Kind != want.kind {
			t.Errorf("para %d: got kind %v, want %v", i, paras[i].Kind, want.kind)
		}
		if paras[i].Text != want.text {
			t.Errorf("para %d: got text %q, want %q", i, paras[i].Text, want.text)
		}
	}

	// Test FormatComment round-trip
	formatted, err := specgen.FormatComment(paras)
	if err != nil {
		t.Fatalf("unexpected error formatting comment: %v", err)
	}
	if formatted != doc {
		t.Errorf("FormatComment mismatch:\ngot:\n%s\nwant:\n%s", formatted, doc)
	}
}
