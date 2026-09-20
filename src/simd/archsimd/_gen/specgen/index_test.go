// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specgen

import (
	"simd/archsimd/_gen/specgen/specexpr"
	"testing"
)

func TestIndex(t *testing.T) {
	vecType := specexpr.Vector{Elem: specexpr.Basic{Base: "int", Bits: 32}, Width: specexpr.Int(128)}
	f1 := &Func{
		Name: "Add",
		Recv: Arg{Name: "x", Type: vecType},
	}
	f2 := &Func{
		Name: "GlobalFunc",
	}

	idx := NewIndex([]*Func{f1, f2})
	if idx.Len() != 2 {
		t.Fatalf("expected Len 2, got %d", idx.Len())
	}

	if got := idx.Lookup("Int32x4", "Add"); got != f1 {
		t.Errorf("Lookup(Int32x4, Add) = %v, want %v", got, f1)
	}
	if got := idx.Lookup("", "GlobalFunc"); got != f2 {
		t.Errorf("Lookup('', GlobalFunc) = %v, want %v", got, f2)
	}
	if got := idx.Lookup("Int32x4", "Missing"); got != nil {
		t.Errorf("Lookup(Int32x4, Missing) = %v, want nil", got)
	}
}
