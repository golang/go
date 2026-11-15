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
	f3 := &Func{
		Name:       "Sub",
		Recv:       Arg{Name: "x", Type: vecType},
		Experiment: true,
	}

	idx := NewIndex([]*Func{f1, f2, f3}, nil)
	if idx.Len() != 3 {
		t.Fatalf("expected Len 3, got %d", idx.Len())
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

	for _, test := range []struct {
		recv, name string
		want       bool
	}{
		{"Int32x4", "Add", false},
		{"", "GlobalFunc", false},
		{"Int32x4", "Sub", true},
		{"Int32x4", "Missing", true},
	} {
		if got := idx.IsExperiment(test.recv, test.name); got != test.want {
			t.Errorf("IsExperiment(%q, %q) = %v, want %v", test.recv, test.name, got, test.want)
		}
	}
}
