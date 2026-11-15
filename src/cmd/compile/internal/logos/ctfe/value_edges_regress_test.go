// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import (
	"bytes"
	"cmd/compile/internal/syntax"
	"testing"
)

func TestReviewMixedUntypedFloatArithmetic(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	cases := []struct {
		name string
		src  string
	}{
		{"AddIntFloat", "package p\nfunc AddIntFloat() int { if 1 + 1.0 == 2.0 { return 1 }; return 0 }"},
		{"AddFloatInt", "package p\nfunc AddFloatInt() int { if 1.5 + 2 == 3.5 { return 1 }; return 0 }"},
		{"SubIntFloat", "package p\nfunc SubIntFloat() int { if 3 - 1.5 == 1.5 { return 1 }; return 0 }"},
		{"MulFloatInt", "package p\nfunc MulFloatInt() int { if 2.0 * 3 == 6.0 { return 1 }; return 0 }"},
		{"DivIntFloat", "package p\nfunc DivIntFloat() int { if 5 / 2.0 == 2.5 { return 1 }; return 0 }"},
		{"DivFloatInt", "package p\nfunc DivFloatInt() int { if 5.0 / 2 == 2.5 { return 1 }; return 0 }"},
		{"AddRuneFloat", "package p\nfunc AddRuneFloat() int { if 'a' + 1.0 == 98.0 { return 1 }; return 0 }"},
	}
	for _, tc := range cases {
		v, _, err := evalConstFunc(t, cfg, constSpec{
			src:    tc.src,
			rtypes: []*Type{BasicType("int")},
			name:   tc.name,
			call:   tc.name + "()",
		})
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		mustInt(t, v, 1, tc.name)
	}
}

func TestReviewMixedUntypedFloatOrdering(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	cases := []struct {
		name string
		src  string
		want int64
	}{
		{"IntLtFloatFalse", "package p\nfunc IntLtFloatFalse() int { if 2 < 1.5 { return 1 }; return 0 }", 0},
		{"FloatLtIntTrue", "package p\nfunc FloatLtIntTrue() int { if 1.5 < 2 { return 1 }; return 0 }", 1},
		{"IntLeFloatTrue", "package p\nfunc IntLeFloatTrue() int { if 2 <= 2.0 { return 1 }; return 0 }", 1},
		{"IntLeFloatFalse", "package p\nfunc IntLeFloatFalse() int { if 3 <= 2.0 { return 1 }; return 0 }", 0},
		{"FloatGtIntTrue", "package p\nfunc FloatGtIntTrue() int { if 2.5 > 2 { return 1 }; return 0 }", 1},
		{"FloatGeIntFalse", "package p\nfunc FloatGeIntFalse() int { if 1.5 >= 2 { return 1 }; return 0 }", 0},
		{"RuneGtFloatTrue", "package p\nfunc RuneGtFloatTrue() int { if 'b' > 97.5 { return 1 }; return 0 }", 1},
		{"RuneLtFloatFalse", "package p\nfunc RuneLtFloatFalse() int { if 'z' < 97.5 { return 1 }; return 0 }", 0},
	}
	for _, tc := range cases {
		v, _, err := evalConstFunc(t, cfg, constSpec{
			src:    tc.src,
			rtypes: []*Type{BasicType("int")},
			name:   tc.name,
			call:   tc.name + "()",
		})
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		mustInt(t, v, tc.want, tc.name)
	}
}

func TestReviewFloat64IncDec(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	cases := []struct {
		name string
		src  string
	}{
		{"Float64Inc", "package p\nfunc Float64Inc() int { x := float64(1); x++; if x == 2 { return 1 }; return 0 }"},
		{"Float64Dec", "package p\nfunc Float64Dec() int { x := float64(2); x--; if x == 1 { return 1 }; return 0 }"},
	}
	for _, tc := range cases {
		v, _, err := evalConstFunc(t, cfg, constSpec{
			src:    tc.src,
			rtypes: []*Type{BasicType("int")},
			name:   tc.name,
			call:   tc.name + "()",
		})
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		mustInt(t, v, 1, tc.name)
	}
}

func TestReviewFloat32IncDec(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	cases := []struct {
		name string
		src  string
	}{
		{"Float32Inc", "package p\nfunc Float32Inc() int { x := float32(1); x++; if x == 2 { return 1 }; return 0 }"},
		{"Float32Dec", "package p\nfunc Float32Dec() int { x := float32(2); x--; if x == 1 { return 1 }; return 0 }"},
	}
	for _, tc := range cases {
		v, _, err := evalConstFunc(t, cfg, constSpec{
			src:    tc.src,
			rtypes: []*Type{BasicType("int")},
			name:   tc.name,
			call:   tc.name + "()",
		})
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		mustInt(t, v, 1, tc.name)
	}
}

func TestReviewStandaloneNestedBlockScope(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	cases := []struct {
		name string
		src  string
		want int64
	}{
		{
			name: "SimpleBlock",
			src:  "package p\nfunc SimpleBlock() int { { n := 1; return n } }",
			want: 1,
		},
		{
			name: "NestedShadowing",
			src: `package p
func NestedShadowing() int {
	x := 10
	{
		x := 20
		{
			x := 30
			if x != 30 {
				return 0
			}
		}
		if x != 20 {
			return 0
		}
	}
	if x != 10 {
		return 0
	}
	return x
}`,
			want: 10,
		},
	}
	for _, tc := range cases {
		v, _, err := evalConstFunc(t, cfg, constSpec{
			src:    tc.src,
			rtypes: []*Type{BasicType("int")},
			name:   tc.name,
			call:   tc.name + "()",
		})
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		mustInt(t, v, tc.want, tc.name)
	}
}

func TestReviewStandaloneBlockControl(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	cases := []struct {
		name string
		src  string
		want int64
	}{
		{
			name: "DeepReturn",
			src: `package p
func DeepReturn() int {
	{
		{
			{
				return 42
			}
		}
	}
}`,
			want: 42,
		},
		{
			name: "BreakInBlock",
			src: `package p
func BreakInBlock() int {
	total := 0
	for i := 0; i < 10; i++ {
		{
			total += i
			if i == 2 {
				break
			}
		}
	}
	return total
}`,
			want: 3,
		},
		{
			name: "ContinueInBlock",
			src: `package p
func ContinueInBlock() int {
	total := 0
	for i := 0; i < 4; i++ {
		{
			if i == 2 {
				continue
			}
			total += i
		}
	}
	return total
}`,
			want: 4,
		},
	}
	for _, tc := range cases {
		v, _, err := evalConstFunc(t, cfg, constSpec{
			src:    tc.src,
			rtypes: []*Type{BasicType("int")},
			name:   tc.name,
			call:   tc.name + "()",
		})
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		mustInt(t, v, tc.want, tc.name)
	}
}

func TestReviewBackwardKeySliceLiteral(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{
		src: `package p
func BackwardKey() int {
	s := []int{2: 3, 0: 1, 2}
	if len(s) != 3 {
		return -1
	}
	if s[0] != 1 || s[1] != 2 || s[2] != 3 {
		return -2
	}
	return 1
}`,
		rtypes: []*Type{BasicType("int")},
		name:   "BackwardKey",
		call:   "BackwardKey()",
	})
	if err != nil {
		t.Fatalf("BackwardKey: %v", err)
	}
	mustInt(t, v, 1, "[]int{2: 3, 0: 1, 2}")

	h := newFakeHost(t)
	expr := parseExpr(t, `[]int{2: 3, 0: 1, 2}`)
	val, _, err := Eval(h, cfg, expr)
	if err != nil {
		t.Fatalf("Eval([]int{2: 3, 0: 1, 2}): %v", err)
	}
	elems := sliceElems(val)
	if len(elems) != 3 {
		t.Fatalf("slice length = %d, want 3", len(elems))
	}
	if elems[0].Int64() != 1 || elems[1].Int64() != 2 || elems[2].Int64() != 3 {
		t.Fatalf("elems = [%v, %v, %v], want [1, 2, 3]", elems[0].Int64(), elems[1].Int64(), elems[2].Int64())
	}
}

func TestReviewUnnamedSliceConversions(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	cases := []struct {
		name string
		src  string
		want int64
	}{
		{"ByteConvLen", `package p
func ByteConvLen() int { s := "abc"; return len([]byte(s)) }`, 3},
		{"ByteConvElem", `package p
func ByteConvElem() int { s := "abc"; return int([]byte(s)[0]) }`, 97},
		{"RuneConvLen", `package p
func RuneConvLen() int { return len([]rune("a✓")) }`, 2},
		{"RuneConvElem", `package p
func RuneConvElem() int { return int([]rune("a✓")[1]) }`, 10003},
		{"Uint8ConvLen", `package p
func Uint8ConvLen() int { return len([]uint8("ab")) }`, 2},
		{"ParenByteConvLen", `package p
func ParenByteConvLen() int { return len(([]byte)("zz")) }`, 2},
	}
	for _, tc := range cases {
		v, _, err := evalConstFunc(t, cfg, constSpec{
			src:    tc.src,
			rtypes: []*Type{BasicType("int")},
			name:   tc.name,
			call:   tc.name + "()",
		})
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		mustInt(t, v, tc.want, tc.name)
	}
}

func TestReviewNilSliceRenderingRoundTrip(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	h := newFakeHost(t)
	v, _, err := evalConstFunc(t, cfg, constSpec{
		src:    "package p\nfunc MakeNilSlice() []int { var s []int; return s }",
		rtypes: []*Type{&Type{Kind: KSlice, Elem: BasicType("int")}},
		name:   "MakeNilSlice",
		call:   "MakeNilSlice()",
	})
	if err != nil {
		t.Fatalf("MakeNilSlice: %v", err)
	}
	if !v.isNil {
		t.Fatalf("computed slice isNil = false, want true")
	}

	rendered, err := v.Expr(syntax.Pos{})
	if err != nil {
		t.Fatalf("v.Expr: %v", err)
	}

	v2, _, err := Eval(h, cfg, rendered)
	if err != nil {
		t.Fatalf("Eval(rendered): %v", err)
	}
	if !v2.isNil {
		t.Fatalf("round-tripped slice isNil = false, want true (rendered expr must evaluate to nil slice)")
	}

	h.vars["res"] = v2
	eqVal, _, err := Eval(h, cfg, parseExpr(t, `res == nil`))
	if err != nil {
		t.Fatalf("Eval(res == nil): %v", err)
	}
	if !eqVal.Bool() {
		t.Fatalf("res == nil is false, want true")
	}
}

func TestReviewNilMapRenderingRoundTrip(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	h := newFakeHost(t)
	v, _, err := evalConstFunc(t, cfg, constSpec{
		src:    "package p\nfunc MakeNilMap() map[string]int { var m map[string]int; return m }",
		rtypes: []*Type{&Type{Kind: KMap, Key: BasicType("string"), Elem: BasicType("int")}},
		name:   "MakeNilMap",
		call:   "MakeNilMap()",
	})
	if err != nil {
		t.Fatalf("MakeNilMap: %v", err)
	}
	if !v.isNil {
		t.Fatalf("computed map isNil = false, want true")
	}

	rendered, err := v.Expr(syntax.Pos{})
	if err != nil {
		t.Fatalf("v.Expr: %v", err)
	}

	v2, _, err := Eval(h, cfg, rendered)
	if err != nil {
		t.Fatalf("Eval(rendered): %v", err)
	}
	if !v2.isNil {
		t.Fatalf("round-tripped map isNil = false, want true (rendered expr must evaluate to nil map)")
	}

	h.vars["resMap"] = v2
	eqVal, _, err := Eval(h, cfg, parseExpr(t, `resMap == nil`))
	if err != nil {
		t.Fatalf("Eval(resMap == nil): %v", err)
	}
	if !eqVal.Bool() {
		t.Fatalf("resMap == nil is false, want true")
	}
}

func TestReviewNonNilEmptySliceMapRenderingControl(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	h := newFakeHost(t)

	vSlice, _, err := evalConstFunc(t, cfg, constSpec{
		src:    "package p\nfunc EmptySlice() []int { return []int{} }",
		rtypes: []*Type{&Type{Kind: KSlice, Elem: BasicType("int")}},
		name:   "EmptySlice",
		call:   "EmptySlice()",
	})
	if err != nil {
		t.Fatalf("EmptySlice: %v", err)
	}
	if vSlice.isNil {
		t.Fatalf("empty slice literal isNil = true, want false")
	}
	renderedSlice, err := vSlice.Expr(syntax.Pos{})
	if err != nil {
		t.Fatalf("vSlice.Expr: %v", err)
	}
	vSlice2, _, err := Eval(h, cfg, renderedSlice)
	if err != nil {
		t.Fatalf("Eval(renderedSlice): %v", err)
	}
	if vSlice2.isNil {
		t.Fatalf("round-tripped empty slice isNil = true, want false")
	}
	h.vars["emptySlice"] = vSlice2
	eqSlice, _, err := Eval(h, cfg, parseExpr(t, `emptySlice == nil`))
	if err != nil {
		t.Fatalf("Eval(emptySlice == nil): %v", err)
	}
	if eqSlice.Bool() {
		t.Fatalf("emptySlice == nil is true, want false")
	}

	vMap, _, err := evalConstFunc(t, cfg, constSpec{
		src:    "package p\nfunc EmptyMap() map[string]int { return map[string]int{} }",
		rtypes: []*Type{&Type{Kind: KMap, Key: BasicType("string"), Elem: BasicType("int")}},
		name:   "EmptyMap",
		call:   "EmptyMap()",
	})
	if err != nil {
		t.Fatalf("EmptyMap: %v", err)
	}
	if vMap.isNil {
		t.Fatalf("empty map literal isNil = true, want false")
	}
	renderedMap, err := vMap.Expr(syntax.Pos{})
	if err != nil {
		t.Fatalf("vMap.Expr: %v", err)
	}
	vMap2, _, err := Eval(h, cfg, renderedMap)
	if err != nil {
		t.Fatalf("Eval(renderedMap): %v", err)
	}
	if vMap2.isNil {
		t.Fatalf("round-tripped empty map isNil = true, want false")
	}
	h.vars["emptyMap"] = vMap2
	eqMap, _, err := Eval(h, cfg, parseExpr(t, `emptyMap == nil`))
	if err != nil {
		t.Fatalf("Eval(emptyMap == nil): %v", err)
	}
	if eqMap.Bool() {
		t.Fatalf("emptyMap == nil is true, want false")
	}
}

func TestReviewNilAppendDestination(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{
		src: `package p
func NilAppend() int {
	var s []int
	s = append(s, 10, 20, 30)
	if len(s) != 3 {
		return -1
	}
	if s[0] != 10 || s[1] != 20 || s[2] != 30 {
		return -2
	}
	return s[0] + s[1] + s[2]
}`,
		rtypes: []*Type{BasicType("int")},
		name:   "NilAppend",
		call:   "NilAppend()",
	})
	if err != nil {
		t.Fatalf("NilAppend: %v", err)
	}
	mustInt(t, v, 60, "append to nil slice")

	v2, _, err := evalConstFunc(t, cfg, constSpec{
		src: `package p
func NilAppendDirect() int {
	s := append([]int(nil), 42)
	return s[0]
}`,
		rtypes: []*Type{BasicType("int")},
		name:   "NilAppendDirect",
		call:   "NilAppendDirect()",
	})
	if err != nil {
		t.Fatalf("NilAppendDirect: %v", err)
	}
	mustInt(t, v2, 42, "append to []int(nil)")
}

func TestReviewMakeZeroLenExceedingCapBudget(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 10, WordSize: 8}
	h := newFakeHost(t)
	fn := parseFunc(t, `package p
func MakeCapOver() int {
	s := make([]byte, 0, 15)
	return len(s)
}`, true, nil, []*Type{BasicType("int")})
	h.funcs["MakeCapOver"] = fn
	call := parseExpr(t, `MakeCapOver()`)
	_, _, err := Eval(h, cfg, call)
	if err == nil {
		t.Fatalf("make([]byte, 0, 15) with 10-byte cap succeeded, want CodeTooLarge")
	}
	cerr, ok := err.(*Error)
	if !ok || cerr.Code != CodeTooLarge {
		t.Fatalf("want CodeTooLarge, got %v", err)
	}

	fnOk := parseFunc(t, `package p
func MakeCapWithin() int {
	s := make([]byte, 0, 8)
	return cap(s)
}`, true, nil, []*Type{BasicType("int")})
	h.funcs["MakeCapWithin"] = fnOk
	callOk := parseExpr(t, `MakeCapWithin()`)
	v, _, err := Eval(h, cfg, callOk)
	if err != nil {
		t.Fatalf("MakeCapWithin: %v", err)
	}
	mustInt(t, v, 8, "cap(make([]byte, 0, 8))")
}

func TestReviewUntypedStringLen(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{
		src: `package p
func UntypedStrLen() int {
	return len("hello world")
}`,
		rtypes: []*Type{BasicType("int")},
		name:   "UntypedStrLen",
		call:   "UntypedStrLen()",
	})
	if err != nil {
		t.Fatalf("UntypedStrLen: %v", err)
	}
	mustInt(t, v, 11, `len("hello world")`)

	h := newFakeHost(t)
	expr := parseExpr(t, `len("abc")`)
	v2, _, err := Eval(h, cfg, expr)
	if err != nil {
		t.Fatalf("Eval(len(\"abc\")): %v", err)
	}
	mustInt(t, v2, 3, `len("abc")`)
}

func TestReviewUntypedStringIndex(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	v, _, err := evalConstFunc(t, cfg, constSpec{
		src: `package p
func UntypedStrIndex() int {
	return int("hello"[1])
}`,
		rtypes: []*Type{BasicType("int")},
		name:   "UntypedStrIndex",
		call:   "UntypedStrIndex()",
	})
	if err != nil {
		t.Fatalf("UntypedStrIndex: %v", err)
	}
	mustInt(t, v, int64('e'), `"hello"[1]`)

	h := newFakeHost(t)
	expr := parseExpr(t, `"hello"[0]`)
	v2, _, err := Eval(h, cfg, expr)
	if err != nil {
		t.Fatalf("Eval(\"hello\"[0]): %v", err)
	}
	mustInt(t, v2, int64('h'), `"hello"[0]`)
}

func TestReviewRenderFloatLexicalKindPreservation(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	h := newFakeHost(t)

	v := FloatValue(2.0, BasicType("float64"))
	rendered, err := v.Expr(syntax.Pos{})
	if err != nil {
		t.Fatalf("v.Expr: %v", err)
	}

	var buf bytes.Buffer
	if _, err := syntax.Fprint(&buf, rendered, syntax.LineForm); err != nil {
		t.Fatalf("syntax.Fprint: %v", err)
	}

	reparsed := parseExpr(t, buf.String())
	rval, _, err := Eval(h, cfg, reparsed)
	if err != nil {
		t.Fatalf("Eval(reparsed): %v", err)
	}
	if !isFloat(rval.typ.Kind) {
		t.Fatalf("reparsed float evaluated to non-float kind %v", rval.typ.Kind)
	}
	if rval.Float64() != 2.0 {
		t.Fatalf("reparsed float evaluated to %v, want 2.0", rval.Float64())
	}

	h.vars["f"] = rval
	divExpr := parseExpr(t, `f / 4`)
	res, _, err := Eval(h, cfg, divExpr)
	if err != nil {
		t.Fatalf("Eval(f / 4): %v", err)
	}
	if !isFloat(res.typ.Kind) || res.Float64() != 0.5 {
		t.Fatalf("f / 4 = %v (kind %v), want 0.5", res.Float64(), res.typ.Kind)
	}
}
