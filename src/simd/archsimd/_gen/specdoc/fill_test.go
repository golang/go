// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specdoc

import (
	"errors"
	"testing"

	"simd/archsimd/_gen/specgen"
	"simd/archsimd/_gen/specgen/specexpr"
)

func typeDecl(t *testing.T, name, doc string) *specgen.TypeDecl {
	shape, err := specexpr.ParseTypeName(name)
	if err != nil {
		t.Fatal(err)
	}
	return &specgen.TypeDecl{
		Name:  name,
		Shape: shape.(specexpr.Vector),
		Doc:   []specgen.Paragraph{{Kind: specgen.SpecOwned, Text: doc}},
	}
}

func TestFillTypeDocs(t *testing.T) {
	idx := specgen.NewIndex(nil, []*specgen.TypeDecl{
		typeDecl(t, "Int32x4", "Int32x4 is a vector."),
		typeDecl(t, "Mask32x4", "Mask32x4 is a mask."),
		typeDecl(t, "Float64s", "Float64s is a scalable vector."),
	})

	src := `package p

// Int32x4 has an old doc.
//
// CPU Feature: AVX
type Int32x4 struct{}

type (
	// Mask32x4 has an old doc.
	Mask32x4 struct{}

	Float64s struct{}
)

// X86Features is not described by spec.
type X86Features struct{}

// int32x4 is unexported.
type int32x4 struct{}
`
	want := `package p

// Int32x4 is a vector.
//
// CPU Feature: AVX
type Int32x4 struct{}

type (
	// Mask32x4 is a mask.
	Mask32x4 struct{}

	// Float64s is a scalable vector.
	Float64s struct{}
)

// X86Features is not described by spec.
type X86Features struct{}

// int32x4 is unexported.
type int32x4 struct{}
`
	got, err := Fill([]byte(src), idx, Options{AllowDocRewrite: true})
	if err != nil {
		t.Fatalf("Fill: %v", err)
	}
	if string(got) != want {
		t.Errorf("Fill:\n%s\nwant:\n%s", got, want)
	}

	// Filling again is idempotent.
	got2, err := Fill(got, idx, Options{AllowDocRewrite: true})
	if err != nil {
		t.Fatalf("second Fill: %v", err)
	}
	if string(got2) != string(got) {
		t.Errorf("second Fill changed source:\n%s", got2)
	}

	// Without AllowDocRewrite, existing spec-owned type docs are reported.
	_, err = Fill([]byte(src), idx, Options{})
	var rep *Report
	if !errors.As(err, &rep) || len(rep.UnexpectedDocs) != 2 {
		t.Errorf("want 2 UnexpectedDocs, got %v", err)
	}

	// With RejectUnknown, a generally available archsimd file can't declare
	// vector types missing from spec, but can declare CPU feature types.
	_, err = Fill([]byte("//go:build goexperiment.simdstable && amd64\n\npackage archsimd\n\ntype Int8x16 struct{}\n\ntype X86Features struct{}\n"), idx, Options{RejectUnknown: true})
	if !errors.As(err, &rep) || len(rep.UnknownDecls) != 1 || rep.UnknownDecls[0].D.Name != "Int8x16" {
		t.Errorf("want UnknownDecl for Int8x16, got %v", err)
	}
}
