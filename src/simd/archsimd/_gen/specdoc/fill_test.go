// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specdoc

import (
	"errors"
	"simd/archsimd/_gen/specgen"
	"simd/archsimd/_gen/specgen/specexpr"
	"strings"
	"testing"
)

func testFill(t *testing.T, src string, opts ...Options) (string, *Report) {
	t.Helper()

	int32x4 := specexpr.Vector{Elem: specexpr.Basic{Base: "int", Bits: 32}, Width: specexpr.Int(128)}

	specFunc := &specgen.Func{
		Name: "Add",
		Recv: specgen.Arg{Name: "x", Type: int32x4},
		In:   []specgen.Arg{{Name: "y", Type: int32x4}},
		Out:  []specgen.Arg{{Name: "z", Type: int32x4}},
		Doc: []specgen.Paragraph{
			{Kind: specgen.SpecOwned, Text: "Add returns x + y."},
		},
	}
	specIdx := specgen.NewIndex([]*specgen.Func{specFunc})

	var opt Options
	if len(opts) > 0 {
		opt = opts[0]
	}
	out, err := Fill([]byte(src), specIdx, opt)
	var rep *Report
	if errors.As(err, &rep) {
		return string(out), rep
	}
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	return string(out), nil
}

func TestFillClean(t *testing.T) {
	// Clean match (when AllowDocRewrite is true)
	cleanSrc := `package archsimd

// Add adds corresponding elements.
//
// Asm: VPADDD, CPU Feature: AVX
func (x Int32x4) Add(y Int32x4) (z Int32x4)
`
	_, rep := testFill(t, cleanSrc, Options{AllowDocRewrite: true})
	if rep != nil {
		t.Errorf("expected clean report, got: %s", rep)
	}

	// Clean match with notes and directives (default AllowDocRewrite: false)
	cleanNotesSrc := `package archsimd

// Asm: VPADDD, CPU Feature: AVX
//
//go:noescape
func (x Int32x4) Add(y Int32x4) (z Int32x4)
`
	_, rep = testFill(t, cleanNotesSrc)
	if rep != nil {
		t.Errorf("expected clean report for notes and directives, got: %s", rep)
	}

	// Clean match without doc comments (default AllowDocRewrite: false)
	cleanNoDocSrc := `package archsimd

func (x Int32x4) Add(y Int32x4) (z Int32x4)
`
	_, rep = testFill(t, cleanNoDocSrc)
	if rep != nil {
		t.Errorf("expected clean report for declaration without doc, got: %s", rep)
	}
}

func TestFillTypeMismatch(t *testing.T) {
	typeMismatchSrc := `package archsimd

func (x Int32x4) Add(y Int32x8) (z Int32x4)
`
	_, rep := testFill(t, typeMismatchSrc)
	if rep == nil {
		t.Fatalf("expected 1 type mismatch, got nil report")
	}
	if len(rep.TypeMismatches) != 1 || len(rep.Findings()) != 1 {
		t.Errorf("expected 1 type mismatch and no other findings, got: %s", rep)
	}
}

func TestFillNameMismatch(t *testing.T) {
	// Name mismatch (parameter and result)
	nameMismatchSrc := `package archsimd

func (x Int32x4) Add(b Int32x4) Int32x4
`
	_, rep := testFill(t, nameMismatchSrc)
	if rep == nil {
		t.Fatalf("expected 1 name mismatch, got nil report")
	}
	if len(rep.NameMismatches) != 1 || len(rep.Findings()) != 1 {
		t.Errorf("expected 1 name mismatch and no other findings, got: %s", rep)
	}

	// Name mismatch tolerated when AllowNameMismatches is true
	_, rep = testFill(t, nameMismatchSrc, Options{AllowNameMismatches: true})
	if rep != nil {
		t.Errorf("expected clean report with AllowNameMismatches, got: %s", rep)
	}
}

func TestFillDocOrder(t *testing.T) {
	// Doc order violation
	docOrderSrc := `package archsimd

// Asm: VPADDD, CPU Feature: AVX
//
// Add adds corresponding elements.
func (x Int32x4) Add(y Int32x4) (z Int32x4)
`
	_, rep := testFill(t, docOrderSrc, Options{AllowDocRewrite: true})
	if rep == nil {
		t.Fatalf("expected 1 doc order violation, got nil report")
	}
	if len(rep.DocOrderViolations) != 1 || len(rep.Findings()) != 1 {
		t.Errorf("expected 1 doc order violation and no other findings, got: %s", rep)
	}
}

func TestFillRejectUnknown(t *testing.T) {
	unknownSrc := `package archsimd

func (x Int32x4) UnknownOp()
`
	_, rep := testFill(t, unknownSrc, Options{RejectUnknown: true})
	if rep == nil {
		t.Fatalf("expected 1 unknown decl error, got nil report")
	}
	if len(rep.UnknownDecls) != 1 || len(rep.Findings()) != 1 {
		t.Errorf("expected 1 unknown decl error and no other findings, got: %s", rep)
	}
}

func TestFillUnexpectedDoc(t *testing.T) {
	srcWithSpecDoc := `package archsimd

// Add adds corresponding elements.
//
// Asm: VPADDD, CPU Feature: AVX
func (x Int32x4) Add(y Int32x4) (z Int32x4)
`
	_, rep := testFill(t, srcWithSpecDoc, Options{Filename: "archsimd/add.go"})
	if rep == nil {
		t.Fatalf("expected 1 unexpected doc finding, got nil report")
	}
	if len(rep.UnexpectedDocs) != 1 || len(rep.Findings()) != 1 {
		t.Fatalf("expected 1 unexpected doc finding and no other findings, got %d findings: %s", len(rep.Findings()), rep)
	}
	errStr := rep.Error()
	if !strings.Contains(errStr, "declaration contains unexpected doc comment") {
		t.Errorf("expected 'declaration contains unexpected doc comment' in error, got: %v", errStr)
	}
	if !strings.Contains(errStr, "archsimd/add.go") {
		t.Errorf("expected filename in error, got: %v", errStr)
	}
	if !strings.Contains(errStr, "(Int32x4) Add") {
		t.Errorf("expected declaration in error, got: %v", errStr)
	}
}

func TestFillWeirdTypes(t *testing.T) {
	// Declarations not in the spec may have receivers or parameter/result
	// types that cannot be parsed as spec types (e.g. struct types or any).
	// Spec lookup must happen before type parsing so these declarations
	// are ignored without attempting to parse their types.
	src := `package archsimd

type X86Features struct{}

func (X86Features) AVX() bool { return true }
func (x Int32x4) ToArch() any { return nil }
`
	_, rep := testFill(t, src)
	if rep != nil {
		t.Errorf("expected clean report, got: %s", rep)
	}
}

func TestCompareFuncs(t *testing.T) {
	int32x4 := specexpr.Vector{Elem: specexpr.Basic{Base: "int", Bits: 32}, Width: specexpr.Int(128)}
	int32x8 := specexpr.Vector{Elem: specexpr.Basic{Base: "int", Bits: 32}, Width: specexpr.Int(256)}
	intType := specexpr.Basic{Base: "int", Bits: 0}

	specFn := &specgen.Func{
		Name: "Add",
		Recv: specgen.Arg{Name: "x", Type: int32x4},
		In:   []specgen.Arg{{Name: "y", Type: int32x4}},
		Out:  []specgen.Arg{{Name: "z", Type: int32x4}},
	}

	// Identical
	declClean := &specgen.Func{
		Name: "Add",
		Recv: specgen.Arg{Name: "x", Type: int32x4},
		In:   []specgen.Arg{{Name: "y", Type: int32x4}},
		Out:  []specgen.Arg{{Name: "z", Type: int32x4}},
	}
	if nameDiffs, typeDiffs := compareFuncs(declClean, specFn); len(nameDiffs) != 0 || len(typeDiffs) != 0 {
		t.Errorf("expected clean, got nameDiffs: %v, typeDiffs: %v", nameDiffs, typeDiffs)
	}

	// Name mismatch
	declNameDiff := &specgen.Func{
		Name: "Add",
		Recv: specgen.Arg{Name: "x", Type: int32x4},
		In:   []specgen.Arg{{Name: "b", Type: int32x4}},
		Out:  []specgen.Arg{{Name: "z", Type: int32x4}},
	}
	if nameDiffs, typeDiffs := compareFuncs(declNameDiff, specFn); len(nameDiffs) != 1 || len(typeDiffs) != 0 || !strings.Contains(nameDiffs[0], "API name \"b\" != spec name \"y\"") {
		t.Errorf("expected name mismatch, got nameDiffs: %v, typeDiffs: %v", nameDiffs, typeDiffs)
	}

	// Receiver mismatch
	declRecvDiff := &specgen.Func{
		Name: "Add",
		Recv: specgen.Arg{Name: "x", Type: int32x8},
		In:   []specgen.Arg{{Name: "y", Type: int32x4}},
		Out:  []specgen.Arg{{Name: "z", Type: int32x4}},
	}
	if nameDiffs, typeDiffs := compareFuncs(declRecvDiff, specFn); len(typeDiffs) != 1 || !strings.Contains(typeDiffs[0], "receiver") {
		t.Errorf("expected receiver mismatch, got nameDiffs: %v, typeDiffs: %v", nameDiffs, typeDiffs)
	}

	// Param count mismatch
	declParamCount := &specgen.Func{
		Name: "Add",
		Recv: specgen.Arg{Name: "x", Type: int32x4},
		In:   []specgen.Arg{{Name: "y", Type: int32x4}, {Name: "w", Type: intType}},
		Out:  []specgen.Arg{{Name: "z", Type: int32x4}},
	}
	if nameDiffs, typeDiffs := compareFuncs(declParamCount, specFn); len(typeDiffs) != 1 || !strings.Contains(typeDiffs[0], "parameter count mismatch") {
		t.Errorf("expected parameter count mismatch, got nameDiffs: %v, typeDiffs: %v", nameDiffs, typeDiffs)
	}

	// Result type mismatch
	declResultType := &specgen.Func{
		Name: "Add",
		Recv: specgen.Arg{Name: "x", Type: int32x4},
		In:   []specgen.Arg{{Name: "y", Type: int32x4}},
		Out:  []specgen.Arg{{Name: "z", Type: intType}},
	}
	if nameDiffs, typeDiffs := compareFuncs(declResultType, specFn); len(typeDiffs) != 1 || !strings.Contains(typeDiffs[0], "result 0") {
		t.Errorf("expected result type mismatch, got nameDiffs: %v, typeDiffs: %v", nameDiffs, typeDiffs)
	}

	// Spec with unnamed result ignores API result name
	specUnnamedResult := &specgen.Func{
		Name: "StorePart",
		Recv: specgen.Arg{Name: "x", Type: int32x4},
		In:   []specgen.Arg{{Name: "s", Type: specexpr.Slice{Elem: intType}}},
		Out:  []specgen.Arg{{Name: "", Type: intType}},
	}
	declNamedResult := &specgen.Func{
		Name: "StorePart",
		Recv: specgen.Arg{Name: "x", Type: int32x4},
		In:   []specgen.Arg{{Name: "s", Type: specexpr.Slice{Elem: intType}}},
		Out:  []specgen.Arg{{Name: "n", Type: intType}},
	}
	if nameDiffs, typeDiffs := compareFuncs(declNamedResult, specUnnamedResult); len(nameDiffs) != 0 || len(typeDiffs) != 0 {
		t.Errorf("expected clean when spec has unnamed result, got nameDiffs: %v, typeDiffs: %v", nameDiffs, typeDiffs)
	}
}
