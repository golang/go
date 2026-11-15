// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import (
	"math"
	"math/big"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
)

func matInt(t *testing.T, n int64) Value {
	t.Helper()
	return intValue(big.NewInt(n), &Type{Kind: KInt}, true)
}

func TestMaterializeSliceHeaderPlain(t *testing.T) {
	elems := []Value{matInt(t, 1), matInt(t, 2), matInt(t, 3)}
	v := sliceValue(&Type{Kind: KInt}, elems, 3)
	backing, lo, length, cap_, ok := v.SliceHeader()
	if !ok {
		t.Fatalf("SliceHeader reported false for a slice")
	}
	if lo != 0 || length != 3 || cap_ != 3 {
		t.Fatalf("window = [%d:%d:%d], want [0:3:3]", lo, length, cap_)
	}
	if len(backing) != 3 {
		t.Fatalf("backing length = %d, want 3", len(backing))
	}
	if v.SliceNeedsBacking() {
		t.Fatalf("plain literal slice must not need backing emission")
	}
	vis := v.sliceVisible()
	if len(vis) != 3 || vis[0].Int64() != 1 || vis[2].Int64() != 3 {
		t.Fatalf("visible elements mismatch")
	}
	if v.IsNilValue() {
		t.Fatalf("non-nil slice reported nil")
	}
}

func TestMaterializeSliceHeaderWindow(t *testing.T) {
	store := []Value{matInt(t, 1), matInt(t, 2), matInt(t, 3), matInt(t, 4), matInt(t, 5)}
	v := Value{
		typ:   &Type{Kind: KSlice, Elem: &Type{Kind: KInt}},
		slice: &sliceVal{backing: &store, lo: 1, length: 2, cap_: 4},
	}
	if !v.SliceNeedsBacking() {
		t.Fatalf("sliced window with spare capacity must need backing emission")
	}
	acc := v.sliceAccessible()
	if len(acc) != 3 || acc[0].Int64() != 2 || acc[2].Int64() != 4 {
		t.Fatalf("accessible window must preserve tail values, got %d elements", len(acc))
	}
	vis := v.sliceVisible()
	if len(vis) != 2 || vis[0].Int64() != 2 || vis[1].Int64() != 3 {
		t.Fatalf("visible window mismatch")
	}
	_, _, _, _, ok := Value{typ: &Type{Kind: KInt}}.SliceHeader()
	if ok {
		t.Fatalf("non-slice value must not report a header")
	}
}

func TestMaterializeSliceWindowBounds(t *testing.T) {
	store := []Value{matInt(t, 1)}
	v := Value{
		typ:   &Type{Kind: KSlice, Elem: &Type{Kind: KInt}},
		slice: &sliceVal{backing: &store, lo: 0, length: 1, cap_: 9},
	}
	if v.sliceAccessible() != nil {
		t.Fatalf("out-of-range accessible window must be nil")
	}
	if v.sliceVisible() == nil {
		t.Fatalf("in-range visible window must not be nil")
	}
}

func TestMaterializeNil(t *testing.T) {
	v := Value{typ: &Type{Kind: KSlice, Elem: &Type{Kind: KInt}}, isNil: true}
	if !v.IsNilValue() {
		t.Fatalf("typed nil must report nil")
	}
	if v.SliceNeedsBacking() {
		t.Fatalf("nil slice must not need backing emission")
	}
}

func TestRenderSliceWindowPreservesCapTail(t *testing.T) {
	f, err := syntax.Parse(syntax.NewFileBase("window"), strings.NewReader("package p\n"), nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	pos := f.Pos()
	mk := func(n int64) Value { return intValue(big.NewInt(n), &Type{Kind: KInt}, true) }
	store := []Value{mk(10), mk(20), mk(30), mk(40), mk(50)}
	v := Value{typ: &Type{Kind: KSlice, Elem: &Type{Kind: KInt}}, slice: &sliceVal{backing: &store, lo: 1, length: 2, cap_: 4}}
	x, err := v.Expr(pos)
	if err != nil {
		t.Fatal(err)
	}
	sl, ok := x.(*syntax.SliceExpr)
	if !ok || !sl.Full {
		t.Fatalf("windowed slice must render as a full slice expression, got %T", x)
	}
	for i, want := range []string{"0", "2", "3"} {
		b, ok := sl.Index[i].(*syntax.BasicLit)
		if !ok || b.Value != want || b.Kind != syntax.IntLit {
			t.Fatalf("slice bound %d must be integer literal %s", i, want)
		}
	}
	bl, ok := sl.X.(*syntax.CompositeLit)
	if !ok || len(bl.ElemList) != 3 {
		t.Fatalf("backing literal must spell exactly the accessible window")
	}
	for i, want := range []string{"20", "30", "40"} {
		b, ok := bl.ElemList[i].(*syntax.BasicLit)
		if !ok || b.Value != want {
			t.Fatalf("backing element %d must spell tail value %s", i, want)
		}
	}
	plain, err := sliceValue(&Type{Kind: KInt}, []Value{mk(1), mk(2), mk(3)}, 3).Expr(pos)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := plain.(*syntax.CompositeLit); !ok {
		t.Fatalf("plain slice must render as a composite literal, got %T", plain)
	}
	if _, ok := plain.(*syntax.SliceExpr); ok {
		t.Fatal("plain slice must not need a backing expression")
	}
}

func TestRenderNilAndFloatLexical(t *testing.T) {
	f, err := syntax.Parse(syntax.NewFileBase("lexical"), strings.NewReader("package p\n"), nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	pos := f.Pos()
	zn, err := Value{typ: &Type{Kind: KUntypedNil}, isNil: true}.Expr(pos)
	if err != nil {
		t.Fatal(err)
	}
	if nm, ok := zn.(*syntax.Name); !ok || nm.Value != "nil" {
		t.Fatalf("nil must still render as the nil name, got %v", zn)
	}
	nf, err := FloatValue(math.Copysign(0, -1), &Type{Kind: KFloat64}).Expr(pos)
	if err != nil {
		t.Fatal(err)
	}
	op, ok := nf.(*syntax.Operation)
	if !ok || op.Y != nil || op.Op != syntax.Sub {
		t.Fatalf("negative zero must keep the exact unary-minus spelling, got %T", nf)
	}
	if mag, ok := op.X.(*syntax.BasicLit); !ok || mag.Value != "0.0" || mag.Kind != syntax.FloatLit {
		t.Fatalf("negative zero magnitude must stay 0.0, got %v", op.X)
	}
	pf, err := FloatValue(1.5, &Type{Kind: KFloat64}).Expr(pos)
	if err != nil {
		t.Fatal(err)
	}
	if b, ok := pf.(*syntax.BasicLit); !ok || b.Value != "1.5" {
		t.Fatalf("float literal must keep round-trip spelling, got %v", pf)
	}
}
