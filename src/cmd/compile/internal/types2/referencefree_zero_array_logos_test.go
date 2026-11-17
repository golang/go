// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import "testing"

func TestLogosReferenceFreeZeroArrays(t *testing.T) {
	references := []struct {
		name string
		typ  Type
	}{
		{"pointer", NewPointer(Typ[Int])},
		{"slice", NewSlice(Typ[Int])},
		{"map", NewMap(Typ[String], Typ[Int])},
		{"channel", NewChan(SendRecv, Typ[Int])},
		{"function", NewSignatureType(nil, nil, nil, nil, nil, false)},
		{"interface", NewInterfaceType(nil, nil)},
		{"unsafe-pointer", Typ[UnsafePointer]},
	}
	lengths := []struct {
		name string
		n    int64
		want bool
	}{
		{"unknown", -1, false},
		{"zero", 0, true},
		{"one", 1, false},
		{"two", 2, false},
	}
	for _, ref := range references {
		t.Run(ref.name, func(t *testing.T) {
			for _, length := range lengths {
				t.Run(length.name, func(t *testing.T) {
					typ := NewArray(ref.typ, length.n)
					if got := LogosReferenceFree(typ); got != length.want {
						t.Fatalf("LogosReferenceFree(%v) = %v, want %v", typ, got, length.want)
					}
				})
			}
		})
	}
}

func TestLogosReferenceFreeZeroArrayNesting(t *testing.T) {
	fn := NewSignatureType(nil, nil, nil, nil, nil, false)
	zero := NewArray(fn, 0)
	live := NewArray(fn, 1)
	marker := NewStruct([]*Var{NewField(nopos, nil, "_", zero, false)}, nil)
	value := NewStruct([]*Var{
		NewField(nopos, nil, "Marker", marker, false),
		NewField(nopos, nil, "Value", Typ[Int], false),
	}, nil)
	withPointer := NewStruct([]*Var{
		NewField(nopos, nil, "Marker", marker, false),
		NewField(nopos, nil, "Pointer", NewPointer(Typ[Int]), false),
	}, nil)
	cases := []struct {
		name string
		typ  Type
		want bool
	}{
		{"named-zero", NewNamed(NewTypeName(nopos, nil, "Zero", nil), zero, nil), true},
		{"alias-zero", NewAlias(NewTypeName(nopos, nil, "Alias", nil), zero), true},
		{"named-marker-value", NewNamed(NewTypeName(nopos, nil, "Value", nil), value, nil), true},
		{"zero-outer-live-inner", NewArray(live, 0), true},
		{"live-outer-zero-inner", NewArray(zero, 2), true},
		{"live-outer-live-inner", NewArray(live, 2), false},
		{"marker-only", marker, true},
		{"marker-and-scalar", value, true},
		{"marker-and-pointer", withPointer, false},
		{"pointer-to-zero", NewPointer(zero), false},
		{"slice-of-zero", NewSlice(zero), false},
		{"map-of-zero", NewMap(Typ[Int], zero), false},
		{"channel-of-zero", NewChan(SendRecv, zero), false},
		{"option-of-zero", &Option{Elem: zero}, true},
		{"option-of-live", &Option{Elem: live}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := LogosReferenceFree(tc.typ); got != tc.want {
				t.Fatalf("LogosReferenceFree(%v) = %v, want %v", tc.typ, got, tc.want)
			}
		})
	}
}

func TestLogosReferenceFreeZeroArrayTypeParams(t *testing.T) {
	typeParam := func(name string, types ...Type) *TypeParam {
		var embeds []Type
		if len(types) != 0 {
			terms := make([]*Term, len(types))
			for i, typ := range types {
				terms[i] = NewTerm(true, typ)
			}
			embeds = []Type{NewUnion(terms)}
		}
		constraint := NewInterfaceType(nil, embeds)
		return NewTypeParam(NewTypeName(nopos, nil, name, nil), constraint)
	}
	anyParam := typeParam("Any")
	refParam := typeParam("References", NewSlice(Typ[Int]), NewMap(Typ[String], Typ[Int]))
	zeroPointer := NewArray(NewPointer(Typ[Int]), 0)
	livePointer := NewArray(NewPointer(Typ[Int]), 1)
	fn := NewSignatureType(nil, nil, nil, nil, nil, false)
	zeroFunction := NewArray(fn, 0)
	cases := []struct {
		name string
		typ  Type
		want bool
	}{
		{"unconstrained", anyParam, false},
		{"zero-unconstrained", NewArray(anyParam, 0), true},
		{"one-unconstrained", NewArray(anyParam, 1), false},
		{"nested-zero-unconstrained", NewArray(NewArray(anyParam, 0), 2), true},
		{"reference-set", refParam, false},
		{"zero-reference-set", NewArray(refParam, 0), true},
		{"zero-only-set", typeParam("Zero", zeroPointer, zeroFunction), true},
		{"reference-free-set", typeParam("Free", zeroPointer, NewArray(Typ[Int], 2)), true},
		{"live-alternative", typeParam("Live", zeroPointer, livePointer), false},
		{"slice-alternative", typeParam("Slice", zeroPointer, NewSlice(Typ[Int])), false},
		{"function-alternative", typeParam("Function", zeroFunction, fn), false},
		{"scalar-set", typeParam("Scalars", Typ[Int], Typ[String]), true},
		{"slice-of-zero-param", NewSlice(NewArray(anyParam, 0)), false},
		{"pointer-to-zero-param", NewPointer(NewArray(anyParam, 0)), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := LogosReferenceFree(tc.typ); got != tc.want {
				t.Fatalf("LogosReferenceFree(%v) = %v, want %v", tc.typ, got, tc.want)
			}
		})
	}
}
