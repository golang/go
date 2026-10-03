// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specdoc

import (
	"go/ast"
	"go/parser"
	"go/token"
	"reflect"
	"simd/archsimd/_gen/specgen"
	"testing"
)

func TestEmulationDependencies(t *testing.T) {
	const src = `package p

import "math/bits"

// Emulated, CPU Feature: AVX
func (x Int8x16) GreaterEqual(y Int8x16) Mask8x16 {
	_ = bits.TrailingZeros64(0)
	_ = X86.AVX2()
	return y.Greater(x).ToInt8x16().Not().asMask()
}
`
	f, err := parser.ParseFile(token.NewFileSet(), "test.go", src, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	fd := f.Decls[1].(*ast.FuncDecl)
	imports := importNames(f)
	if got, want := emulationDependencies(fd, imports), []string{"Greater", "Not"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("emulationDependencies() = %v, want %v", got, want)
	}

	doc := fillEmulationNote(parseCommentGroup(fd.Doc), fd, imports)
	if got, want := doc[0].Text, "Emulated: Greater, Not"; got != want {
		t.Fatalf("emulation note = %q, want %q", got, want)
	}
	if got, want := doc[1].Text, "CPU Feature: AVX"; got != want {
		t.Fatalf("CPU feature note = %q, want %q", got, want)
	}

	rewritten, err := Fill([]byte(src), specgen.NewIndex(nil), Options{NoFillDoc: true})
	if err != nil {
		t.Fatal(err)
	}
	want := `package p

import "math/bits"

// Emulated: Greater, Not
//
// CPU Feature: AVX
func (x Int8x16) GreaterEqual(y Int8x16) Mask8x16 {
	_ = bits.TrailingZeros64(0)
	_ = X86.AVX2()
	return y.Greater(x).ToInt8x16().Not().asMask()
}
`
	if string(rewritten) != want {
		t.Fatalf("Fill() =\n%s\nwant:\n%s", rewritten, want)
	}
	rewrittenAgain, err := Fill(rewritten, specgen.NewIndex(nil), Options{NoFillDoc: true})
	if err != nil {
		t.Fatal(err)
	}
	if string(rewrittenAgain) != want {
		t.Fatalf("second Fill() =\n%s\nwant:\n%s", rewrittenAgain, want)
	}
}

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
