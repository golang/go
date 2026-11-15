// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import (
	"math"
	"strings"
	"testing"
)

type constSpec struct {
	src    string
	ptypes []*Type
	rtypes []*Type
	name   string
	call   string
}

func evalConstFunc(t *testing.T, cfg Config, spec constSpec) (Value, Stats, error) {
	t.Helper()
	h := newFakeHost(t)
	fn := parseFunc(t, spec.src, true, spec.ptypes, spec.rtypes)
	h.funcs[spec.name] = fn
	call := parseExpr(t, spec.call)
	return Eval(h, cfg, call)
}

func mustInt(t *testing.T, v Value, want int64, what string) {
	t.Helper()
	if v.Int64() != want {
		t.Fatalf("%s = %v, want %d", what, v.Int64(), want)
	}
}

func TestPR3ShiftConstKeepsUntyped(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	want := int64(1 << 8)
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc ShiftConst() int { return 1 << uint8(8) }", rtypes: []*Type{BasicType("int")}, name: "ShiftConst", call: `ShiftConst()`})
	if err != nil {
		t.Fatalf("ShiftConst: %v, want %d", err, want)
	}
	mustInt(t, v, want, "1<<uint8(8)")
}

func TestPR3ShiftVarCount256(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc ShiftVar(c uint8) int { return 1 << c }", ptypes: []*Type{BasicType("uint8")}, rtypes: []*Type{BasicType("int")}, name: "ShiftVar", call: `ShiftVar(8)`})
	if err != nil {
		t.Fatalf("ShiftVar(8): %v", err)
	}
	mustInt(t, v, 256, "1<<var uint8(8)")
}

func TestPR3RawStringStripsCR(t *testing.T) {
	h := newFakeHost(t)
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	e := parseExpr(t, "`a\r\nb`")
	v, _, err := Eval(h, cfg, e)
	if err != nil {
		t.Fatalf("raw literal: %v", err)
	}
	if v.Str() != "a\nb" {
		t.Fatalf("raw sval = %q, want %q", v.Str(), "a\nb")
	}
	if len(v.Str()) != 3 {
		t.Fatalf("raw len = %d, want 3", len(v.Str()))
	}
}

func TestPR3InterpretedStringKeepsCRLen(t *testing.T) {
	h := newFakeHost(t)
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	e := parseExpr(t, `"a\r\nb"`)
	v, _, err := Eval(h, cfg, e)
	if err != nil {
		t.Fatalf("interpreted literal: %v", err)
	}
	if len(v.Str()) != 4 {
		t.Fatalf("interpreted len = %d, want 4", len(v.Str()))
	}
}

func TestPR3NaNOrderedComparisons(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	cases := []struct {
		name string
		src  string
	}{
		{"NanLe", "package p\nfunc NanLe() int { z := float64(0); n := z / z; if n <= 0 { return 1 }; return 0 }"},
		{"NanGe", "package p\nfunc NanGe() int { z := float64(0); n := z / z; if n >= 0 { return 1 }; return 0 }"},
		{"NanLt", "package p\nfunc NanLt() int { z := float64(0); n := z / z; if n < 0 { return 1 }; return 0 }"},
		{"NanGt", "package p\nfunc NanGt() int { z := float64(0); n := z / z; if n > 0 { return 1 }; return 0 }"},
	}
	for _, tc := range cases {
		v, _, err := evalConstFunc(t, cfg, constSpec{src: tc.src, rtypes: []*Type{BasicType("int")}, name: tc.name, call: tc.name + `()`})
		if err != nil {
			t.Fatalf("%s: %v, want 0", tc.name, err)
		}
		mustInt(t, v, 0, tc.name)
	}
	nan := math.NaN()
	if nan <= 0 || nan >= 0 || nan < 0 || nan > 0 {
		t.Fatalf("Go oracle violated")
	}
}

func TestPR3NaNNotEqualSelf(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc NanNe() int { z := float64(0); n := z / z; if n != n { return 1 }; return 0 }", rtypes: []*Type{BasicType("int")}, name: "NanNe", call: `NanNe()`})
	if err != nil {
		t.Fatalf("NanNe: %v", err)
	}
	mustInt(t, v, 1, "n != n")
}

func TestPR3ConstFloatExactEquality(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc FloatExact() int { if 1.00000000000000000001 == 1.0 { return 1 }; return 0 }", rtypes: []*Type{BasicType("int")}, name: "FloatExact", call: `FloatExact()`})
	if err != nil {
		t.Fatalf("FloatExact: %v", err)
	}
	mustInt(t, v, 0, "1.000...01 == 1.0")
}

func TestPR3ConstFloatEqualControl(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc FloatSame() int { if 1.0 == 1.0 { return 1 }; return 0 }", rtypes: []*Type{BasicType("int")}, name: "FloatSame", call: `FloatSame()`})
	if err != nil {
		t.Fatalf("FloatSame: %v", err)
	}
	mustInt(t, v, 1, "1.0 == 1.0")
}

func TestPR3MixedIntFloatEquality(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc MixedEq() int { if 1 == 1.0 { return 1 }; return 0 }", rtypes: []*Type{BasicType("int")}, name: "MixedEq", call: `MixedEq()`})
	if err != nil {
		t.Fatalf("MixedEq: %v, want 1", err)
	}
	mustInt(t, v, 1, "1 == 1.0")
	if !(1 == 1.0) {
		t.Fatalf("Go oracle violated")
	}
}

func TestPR3MixedVarFloatIntEquality(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc MixedVar(x float64) int { if x == 1 { return 1 }; return 0 }", ptypes: []*Type{BasicType("float64")}, rtypes: []*Type{BasicType("int")}, name: "MixedVar", call: `MixedVar(1)`})
	if err != nil {
		t.Fatalf("MixedVar: %v, want 1", err)
	}
	mustInt(t, v, 1, "float64(1) == 1")
}

func TestPR3MixedPreciseInequality(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc MixedPrecise() int { if 1 == 1.00000000000000000001 { return 1 }; return 0 }", rtypes: []*Type{BasicType("int")}, name: "MixedPrecise", call: `MixedPrecise()`})
	if err != nil {
		t.Fatalf("MixedPrecise: %v, want 0", err)
	}
	mustInt(t, v, 0, "1 == 1.000...01")
}

func TestPR3IntEqualityControl(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc IntNe() int { if 1 == 2 { return 1 }; return 0 }", rtypes: []*Type{BasicType("int")}, name: "IntNe", call: `IntNe()`})
	if err != nil {
		t.Fatalf("IntNe: %v", err)
	}
	mustInt(t, v, 0, "1 == 2")
}

func TestPR3ArraySliceAliases(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc ArrAlias() int { a := [4]int{1, 2, 3, 4}; s := a[1:]; s[0] = 9; return a[1] }", rtypes: []*Type{BasicType("int")}, name: "ArrAlias", call: `ArrAlias()`})
	if err != nil {
		t.Fatalf("ArrAlias: %v, want 9", err)
	}
	mustInt(t, v, 9, "a[1] after s[0]=9")
}

func TestPR3ArraySliceCap(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc ArrCap() int { a := [4]int{1, 2, 3, 4}; s := a[1:]; return cap(s) }", rtypes: []*Type{BasicType("int")}, name: "ArrCap", call: `ArrCap()`})
	if err != nil {
		t.Fatalf("ArrCap: %v, want 3", err)
	}
	mustInt(t, v, 3, "cap(a[1:])")
}

func TestPR3SliceHiBeyondLen(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc SliceHi() int { s := make([]int, 2, 4); t := s[:3]; return len(t) }", rtypes: []*Type{BasicType("int")}, name: "SliceHi", call: `SliceHi()`})
	if err != nil {
		t.Fatalf("SliceHi: %v, want 3", err)
	}
	mustInt(t, v, 3, "len(s[:3])")
}

func TestPR3ResliceCapAbsolute(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc ResliceCap() int { s := make([]int, 4, 4); t := s[1:3]; u := t[:1]; return cap(u) }", rtypes: []*Type{BasicType("int")}, name: "ResliceCap", call: `ResliceCap()`})
	if err != nil {
		t.Fatalf("ResliceCap: %v, want 3", err)
	}
	mustInt(t, v, 3, "cap(t[:1])")
}

func TestPR3CopyOverlapSafe(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc CopyOverlap() int { s := []int{1, 2, 3}; copy(s[1:], s[:2]); return s[2] }", rtypes: []*Type{BasicType("int")}, name: "CopyOverlap", call: `CopyOverlap()`})
	if err != nil {
		t.Fatalf("CopyOverlap: %v, want 2", err)
	}
	g := []int{1, 2, 3}
	copy(g[1:], g[:2])
	if g[2] != 2 {
		t.Fatalf("Go oracle violated: g[2]=%d", g[2])
	}
	mustInt(t, v, 2, "s[2] after overlapping copy")
}

func TestPR3TypedNilLenCap(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	for _, tc := range []struct{ name, src, call string }{
		{"NilSliceLen", "package p\nfunc NilSliceLen() int { var s []int; return len(s) }", `NilSliceLen()`},
		{"NilSliceCap", "package p\nfunc NilSliceCap() int { var s []int; return cap(s) }", `NilSliceCap()`},
		{"NilMapLen", "package p\nfunc NilMapLen() int { var m map[int]int; return len(m) }", `NilMapLen()`},
	} {
		v, _, err := evalConstFunc(t, cfg, constSpec{src: tc.src, rtypes: []*Type{BasicType("int")}, name: tc.name, call: tc.call})
		if err != nil {
			t.Fatalf("%s: %v, want 0", tc.name, err)
		}
		mustInt(t, v, 0, tc.name)
	}
}

func TestPR3NilMapCapStillFails(t *testing.T) {
	h := newFakeHost(t)
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	fn := parseFunc(t, "package p\nfunc NilMapCap() int { var m map[int]int; return cap(m) }", true, nil, []*Type{BasicType("int")})
	h.funcs["NilMapCap"] = fn
	call := parseExpr(t, `NilMapCap()`)
	_, _, err := Eval(h, cfg, call)
	if err == nil {
		t.Fatalf("cap(nil map) succeeded, want bad cap call")
	}
	if !strings.Contains(err.Error(), "cap") {
		t.Fatalf("cap(nil map) err = %v, want cap complaint", err)
	}
}

func TestPR3MapMakeHintDivZero(t *testing.T) {
	h := newFakeHost(t)
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	fn := parseFunc(t, "package p\nfunc MapHint() int { m := make(map[int]int, 1/0); return len(m) }", true, nil, []*Type{BasicType("int")})
	h.funcs["MapHint"] = fn
	call := parseExpr(t, `MapHint()`)
	_, _, err := Eval(h, cfg, call)
	if err == nil {
		t.Fatalf("make hint 1/0 succeeded, want divide by zero")
	}
	if !strings.Contains(err.Error(), "divide by zero") {
		t.Fatalf("hint err = %v, want divide by zero", err)
	}
}

func TestPR3MapMakeHintNegativeVariable(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc MapNegVar() int { n := -1; m := make(map[int]int, n); return len(m) }", rtypes: []*Type{BasicType("int")}, name: "MapNegVar", call: `MapNegVar()`})
	if err != nil {
		t.Fatalf("MapNegVar: %v, want 0", err)
	}
	mustInt(t, v, 0, "variable negative map hint len")
}

func TestPR3RuneToStringReplacement(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	for _, tc := range []struct{ name, src, call string }{
		{"RuneNeg", "package p\nfunc RuneNeg() string { return string(-1) }", `RuneNeg()`},
		{"RuneSurrogate", "package p\nfunc RuneSurrogate() string { return string(55296) }", `RuneSurrogate()`},
	} {
		h := newFakeHost(t)
		fn := parseFunc(t, tc.src, true, nil, []*Type{BasicType("string")})
		h.funcs[tc.name] = fn
		call := parseExpr(t, tc.call)
		v, _, err := Eval(h, cfg, call)
		if err != nil {
			t.Fatalf("%s: %v, want U+FFFD", tc.name, err)
		}
		if v.Str() != "\uFFFD" {
			t.Fatalf("%s = %q, want U+FFFD", tc.name, v.Str())
		}
		if !v.isConstant() {
			t.Fatalf("%s constant = false, want true", tc.name)
		}
	}
}

func TestPR3ArrayCopyOnBind(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc ArrCopy() int { a := [2]int{1, 2}; b := a; b[0] = 9; return a[0] }", rtypes: []*Type{BasicType("int")}, name: "ArrCopy", call: `ArrCopy()`})
	if err != nil {
		t.Fatalf("ArrCopy: %v", err)
	}
	mustInt(t, v, 1, "a[0] after b[0]=9")
}

func TestPR3SliceAliasControl(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc SliceAlias() int { s := []int{1}; t := s; t[0] = 9; return s[0] }", rtypes: []*Type{BasicType("int")}, name: "SliceAlias", call: `SliceAlias()`})
	if err != nil {
		t.Fatalf("SliceAlias: %v", err)
	}
	mustInt(t, v, 9, "s[0] after t[0]=9")
}

func TestPR3AssignPhaseOrder(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc PhaseOrder() int { i := 0; a := []int{10, 20}; i, a[i] = 1, 99; return a[0] }", rtypes: []*Type{BasicType("int")}, name: "PhaseOrder", call: `PhaseOrder()`})
	if err != nil {
		t.Fatalf("PhaseOrder: %v", err)
	}
	mustInt(t, v, 99, "a[0]")
}

func TestPR3AssignPhaseOrderSecond(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc PhaseOrder1() int { i := 0; a := []int{10, 20}; i, a[i] = 1, 99; return a[1] }", rtypes: []*Type{BasicType("int")}, name: "PhaseOrder1", call: `PhaseOrder1()`})
	if err != nil {
		t.Fatalf("PhaseOrder1: %v", err)
	}
	i, a := 0, []int{10, 20}
	i, a[i] = 1, 99
	_ = i
	if a[1] != 20 {
		t.Fatalf("Go oracle violated: a[1]=%d", a[1])
	}
	mustInt(t, v, 20, "a[1]")
}

func TestPR3LabeledContinueFor(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc LabCont() int { n := 0; Outer: for i := 0; i < 3; i++ { for j := 0; j < 1; j++ { n++; continue Outer } }; return n }", rtypes: []*Type{BasicType("int")}, name: "LabCont", call: `LabCont()`})
	if err != nil {
		t.Fatalf("LabCont: %v", err)
	}
	mustInt(t, v, 3, "labeled continue count")
}

func TestPR3LabeledContinueRange(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc LabContR() int { n := 0; Outer: for i := range 3 { for j := 0; j < 1; j++ { n++; continue Outer }; _ = i }; return n }", rtypes: []*Type{BasicType("int")}, name: "LabContR", call: `LabContR()`})
	if err != nil {
		t.Fatalf("LabContR: %v", err)
	}
	mustInt(t, v, 3, "range labeled continue count")
}

func TestPR3ExplicitReturnConverts(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	h := newFakeHost(t)
	hf := parseFunc(t, "package p\nfunc H() float64 { return 1 }", true, nil, []*Type{BasicType("float64")})
	h.funcs["H"] = hf
	gf := parseFunc(t, "package p\nfunc RetConv() int { if H()/2 == 0 { return 1 }; return 0 }", true, nil, []*Type{BasicType("int")})
	h.funcs["RetConv"] = gf
	call := parseExpr(t, `RetConv()`)
	v, _, err := Eval(h, cfg, call)
	if err != nil {
		t.Fatalf("RetConv: %v", err)
	}
	if float64(1)/2 == 0 {
		t.Fatalf("Go oracle violated")
	}
	mustInt(t, v, 0, "H()/2 == 0")
}

func TestPR3AssignConvertsToDeclared(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc AssignConv() int { var x float64; x = 1; if x/2 == 0 { return 1 }; return 0 }", rtypes: []*Type{BasicType("int")}, name: "AssignConv", call: `AssignConv()`})
	if err != nil {
		t.Fatalf("AssignConv: %v", err)
	}
	mustInt(t, v, 0, "x/2 == 0")
}

func TestPR3MapElemAssignConverts(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc MapConv() int { m := map[int]float64{}; m[0] = 1; if m[0]/2 == 0 { return 1 }; return 0 }", rtypes: []*Type{BasicType("int")}, name: "MapConv", call: `MapConv()`})
	if err != nil {
		t.Fatalf("MapConv: %v", err)
	}
	mustInt(t, v, 0, "m[0]/2 == 0")
}

func TestPR3BlankAssignDiscard(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc BlankDisc() int { _ = 1; return 2 }", rtypes: []*Type{BasicType("int")}, name: "BlankDisc", call: `BlankDisc()`})
	if err != nil {
		t.Fatalf("BlankDisc: %v", err)
	}
	mustInt(t, v, 2, "_ = 1")
}

func TestPR3BlankMultiAssign(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	h := newFakeHost(t)
	two := parseFunc(t, "package p\nfunc two() (int, int) { return 1, 2 }", true, nil, []*Type{BasicType("int"), BasicType("int")})
	h.funcs["two"] = two
	f := parseFunc(t, "package p\nfunc BlankMulti() int { a, _ := two(); _, b := two(); _ = a; return b }", true, nil, []*Type{BasicType("int")})
	h.funcs["BlankMulti"] = f
	call := parseExpr(t, `BlankMulti()`)
	v, _, err := Eval(h, cfg, call)
	if err != nil {
		t.Fatalf("BlankMulti: %v", err)
	}
	mustInt(t, v, 2, "blank multi")
}

func TestPR3RangeArraySnapshot(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc RangeSnap() int { a := [2]int{1, 2}; s := 0; for _, v := range a { if s == 0 { a[1] = 9 }; s += v }; return s }", rtypes: []*Type{BasicType("int")}, name: "RangeSnap", call: `RangeSnap()`})
	if err != nil {
		t.Fatalf("RangeSnap: %v, want 3", err)
	}
	mustInt(t, v, 3, "array range snapshot sum")
}

func TestPR3RangeArraySnapshotIndexed(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc RangeSnapI() int { a := [2]int{1, 2}; s := 0; for i, v := range a { if i == 0 { a[1] = 9 }; s += v }; return s }", rtypes: []*Type{BasicType("int")}, name: "RangeSnapI", call: `RangeSnapI()`})
	if err != nil {
		t.Fatalf("RangeSnapI: %v, want 3", err)
	}
	mustInt(t, v, 3, "indexed array range snapshot sum")
}

func TestPR3RangeSliceObservesWrite(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc RangeSliceLive() int { a := []int{1, 2}; s := 0; for _, v := range a { if s == 0 { a[1] = 9 }; s += v }; return s }", rtypes: []*Type{BasicType("int")}, name: "RangeSliceLive", call: `RangeSliceLive()`})
	if err != nil {
		t.Fatalf("RangeSliceLive: %v", err)
	}
	ga := []int{1, 2}
	gs := 0
	for _, gv := range ga {
		if gs == 0 {
			ga[1] = 9
		}
		gs += gv
	}
	mustInt(t, v, int64(gs), "slice range live sum")
}

func TestPR3RangeStringInvalidUTF8Index(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc StrIdx() int { n := 0; for i := range \"\\xffa\" { n += i }; return n }", rtypes: []*Type{BasicType("int")}, name: "StrIdx", call: `StrIdx()`})
	if err != nil {
		t.Fatalf("StrIdx: %v, want 1", err)
	}
	g := 0
	for gi := range "\xffa" {
		g += gi
	}
	if g != 1 {
		t.Fatalf("Go oracle violated: sum=%d", g)
	}
	mustInt(t, v, 1, "invalid UTF-8 index sum")
}

func TestPR3RangeStringInvalidRune(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc StrRune() int { for _, r := range \"\\xff\" { if r == 65533 { return 1 } }; return 0 }", rtypes: []*Type{BasicType("int")}, name: "StrRune", call: `StrRune()`})
	if err != nil {
		t.Fatalf("StrRune: %v", err)
	}
	mustInt(t, v, 1, "0xFF decodes to U+FFFD")
}

func TestPR3SwitchDefaultBeforeMatch(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc SwDef() int { switch 2 { default: return 0; case 2: return 1 }; return 9 }", rtypes: []*Type{BasicType("int")}, name: "SwDef", call: `SwDef()`})
	if err != nil {
		t.Fatalf("SwDef: %v, want 1", err)
	}
	mustInt(t, v, 1, "default before match")
}

func TestPR3SwitchDefaultControls(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc SwDefC() int { switch 1 { case 0: return 7; default: return 8 }; return 9 }", rtypes: []*Type{BasicType("int")}, name: "SwDefC", call: `SwDefC()`})
	if err != nil {
		t.Fatalf("SwDefC: %v", err)
	}
	mustInt(t, v, 8, "default taken")
	w, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc SwFall() int { switch 1 { case 1: fallthrough; default: return 4 }; return 9 }", rtypes: []*Type{BasicType("int")}, name: "SwFall", call: `SwFall()`})
	if err != nil {
		t.Fatalf("SwFall: %v", err)
	}
	mustInt(t, w, 4, "fallthrough into default")
}

func TestPR3RangeIntNoVars(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc RangeNoVars() int { c := 0; for range 3 { c++ }; return c }", rtypes: []*Type{BasicType("int")}, name: "RangeNoVars", call: `RangeNoVars()`})
	if err != nil {
		t.Fatalf("RangeNoVars: %v, want 3", err)
	}
	mustInt(t, v, 3, "for range 3 count")
}

func TestPR3RangeNegativeInt(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc RangeNeg() int { n := -1; for i := range n { return 1 }; return 0 }", rtypes: []*Type{BasicType("int")}, name: "RangeNeg", call: `RangeNeg()`})
	if err != nil {
		t.Fatalf("RangeNeg: %v, want 0", err)
	}
	mustInt(t, v, 0, "range -1 runs zero times")
}

func TestPR3RangeZeroControl(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc RangeZero() int { for i := range 0 { return 1 }; return 0 }", rtypes: []*Type{BasicType("int")}, name: "RangeZero", call: `RangeZero()`})
	if err != nil {
		t.Fatalf("RangeZero: %v", err)
	}
	mustInt(t, v, 0, "range 0 runs zero times")
}

func TestPR3VariadicConvertsToElem(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	h := newFakeHost(t)
	fn := parseFunc(t, "package p\nfunc VSum(xs ...float64) float64 { return xs[0] / 2 }", true, []*Type{{Kind: KSlice, Elem: BasicType("float64")}}, []*Type{BasicType("float64")})
	fn.Variadic = true
	h.funcs["VSum"] = fn
	call := parseExpr(t, `VSum(1)`)
	v, _, err := Eval(h, cfg, call)
	if err != nil {
		t.Fatalf("VSum(1): %v", err)
	}
	if v.Float64() != 0.5 {
		t.Fatalf("VSum(1) = %v, want 0.5", v.Float64())
	}
}

func TestPR3VariadicEmptyBinds(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	h := newFakeHost(t)
	fn := parseFunc(t, "package p\nfunc VLen(xs ...float64) int { return len(xs) }", true, []*Type{{Kind: KSlice, Elem: BasicType("float64")}}, []*Type{BasicType("int")})
	fn.Variadic = true
	h.funcs["VLen"] = fn
	call := parseExpr(t, `VLen()`)
	v, _, err := Eval(h, cfg, call)
	if err != nil {
		t.Fatalf("VLen(): %v", err)
	}
	mustInt(t, v, 0, "empty variadic len")
}

func TestPR3BudgetDoublingLoopFails(t *testing.T) {
	cfg := Config{MaxSteps: 1000000, MaxValueBytes: 64, WordSize: 8}
	h := newFakeHost(t)
	fn := parseFunc(t, "package p\nfunc Grow() int { s := make([]byte, 1); for i := 0; i < 8; i++ { s = append(s, s...) }; return len(s) }", true, nil, []*Type{BasicType("int")})
	h.funcs["Grow"] = fn
	call := parseExpr(t, `Grow()`)
	_, _, err := Eval(h, cfg, call)
	if err == nil {
		t.Fatalf("Grow with 64-byte cap succeeded, want CodeTooLarge")
	}
	cerr, ok := err.(*Error)
	if !ok {
		t.Fatalf("want *Error, got %T (%v)", err, err)
	}
	if cerr.Code != CodeTooLarge {
		t.Fatalf("code = %v, want CodeTooLarge", cerr.Code)
	}
}

func TestPR3BudgetSmallGrowthPasses(t *testing.T) {
	cfg := Config{MaxSteps: 1000000, MaxValueBytes: 64, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{src: "package p\nfunc GrowSmall() int { s := make([]byte, 1); for i := 0; i < 5; i++ { s = append(s, s...) }; return len(s) }", rtypes: []*Type{BasicType("int")}, name: "GrowSmall", call: `GrowSmall()`})
	if err != nil {
		t.Fatalf("GrowSmall: %v, want len 32", err)
	}
	mustInt(t, v, 32, "5 doublings from 1")
}
