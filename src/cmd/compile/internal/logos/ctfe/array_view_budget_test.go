// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import (
	"testing"
)

func TestArrayViewHostSharingAliasIsolation(t *testing.T) {
	intT := BasicType("int")
	arr2T := &Type{Kind: KArray, Len: 2, Elem: &Type{Kind: KSlice, Elem: intT}}
	big := Config{MaxSteps: 1 << 20, MaxValueBytes: 1 << 20, WordSize: 8}

	build := newFakeHost(t)
	buildFn := parseFunc(t, `package p
func Build() [2][]int {
	a := [4]int{1, 2, 3, 4}
	s1 := a[0:4]
	s2 := a[0:4]
	return [2][]int{s1, s2}
}`, true, nil, []*Type{arr2T})
	build.funcs[buildFn.Name] = buildFn
	shared, _, err := Eval(build, big, parseExpr(t, `Build()`))
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	h := newFakeHost(t)
	h.vars["H"] = shared
	cfg := Config{}

	if v, _ := evalOne(t, h, cfg, parseExpr(t, `len(H)`)); v.Int64() != 2 {
		t.Fatalf("len(H) = %d, want 2", v.Int64())
	}
	if v, _ := evalOne(t, h, cfg, parseExpr(t, `len(H[0])`)); v.Int64() != 4 {
		t.Fatalf("len(H[0]) = %d, want 4", v.Int64())
	}
	if v, _ := evalOne(t, h, cfg, parseExpr(t, `H[0][0]`)); v.Int64() != 1 {
		t.Fatalf("H[0][0] = %d, want 1", v.Int64())
	}
	if v, _ := evalOne(t, h, cfg, parseExpr(t, `H[1][3]`)); v.Int64() != 4 {
		t.Fatalf("H[1][3] = %d, want 4", v.Int64())
	}

	aliasFn := parseFunc(t, `package p
func CheckAlias() int {
	h := H
	h[0][0] = 99
	return h[1][0]
}`, true, nil, []*Type{intT})
	h.funcs[aliasFn.Name] = aliasFn
	if v, _ := evalOne(t, h, cfg, parseExpr(t, `CheckAlias()`)); v.Int64() != 99 {
		t.Fatalf("CheckAlias() = %d, want 99 (two array views share one backing)", v.Int64())
	}

	isoFn := parseFunc(t, `package p
func CheckIso() int {
	x := H
	y := H
	x[0][0] = 99
	return y[0][0]
}`, true, nil, []*Type{intT})
	h.funcs[isoFn.Name] = isoFn
	if v, _ := evalOne(t, h, cfg, parseExpr(t, `CheckIso()`)); v.Int64() != 1 {
		t.Fatalf("CheckIso() = %d, want 1 (two reads isolated)", v.Int64())
	}
	if v, _ := evalOne(t, h, cfg, parseExpr(t, `H[0][0]`)); v.Int64() != 1 {
		t.Fatalf("host H[0][0] after mutation = %d, want 1", v.Int64())
	}
}
