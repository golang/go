// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that StructuralType() calculates the correct value of structural type for
// unusual cases.

package types

import (
	"cmd/internal/src"
	"testing"
)

type test struct {
	typ            *Type
	structuralType *Type
}

func TestStructuralType(t *testing.T) {
	// These are the few constants that need to be initialized in order to use
	// the types package without using the typecheck package by calling
	// typecheck.InitUniverse() (the normal way to initialize the types package).
	PtrSize = 8
	RegSize = 8
	MaxWidth = 1 << 50

	// type intType = int
	intType := newType(TINT)
	// type structf = struct { f int }
	structf := NewStruct(nil, []*Field{
		NewField(src.NoXPos, LocalPkg.Lookup("f"), intType),
	})

	// type Sf structf
	Sf := newType(TFORW)
	Sf.sym = LocalPkg.Lookup("Sf")
	Sf.SetUnderlying(structf)

	// type A int
	A := newType(TFORW)
	A.sym = LocalPkg.Lookup("A")
	A.SetUnderlying(intType)

	// type B int
	B := newType(TFORW)
	B.sym = LocalPkg.Lookup("B")
	B.SetUnderlying(intType)

	emptyInterface := NewInterface(BuiltinPkg, []*Field{}, false)
	any := newType(TFORW)
	any.sym = LocalPkg.Lookup("any")
	any.SetUnderlying(emptyInterface)

	// The tests marked NONE have no structural type; all the others have a
	// structural type of structf - "struct { f int }"
	tests := []*test{
		{
			// interface { struct { f int } }
			embed(structf),
			structf,
		},
		{
			// interface { struct { f int }; any }
			embed(structf, any),
			structf,
		},
		{
			// interface { Sf }
			embed(Sf),
			structf,
		},
		{
			// interface { any | Sf }
			embed(any, Sf),
			structf,
		},
		{
			// interface { struct { f int }; Sf } - NONE
			embed(structf, Sf),
			nil,
		},
		{
			// interface { struct { f int } | ~struct { f int } }
			embed(NewUnion([]*Type{structf, structf}, []bool{false, true})),
			structf,
		},
		{
			// interface { ~struct { f int } ; Sf }
			embed(NewUnion([]*Type{structf}, []bool{true}), Sf),
			structf,
		},
		{
			// interface { struct { f int } ; Sf } - NONE
			embed(NewUnion([]*Type{structf}, []bool{false}), Sf),
			nil,
		},
		{
			// interface { Sf | A; B | Sf}
			embed(NewUnion([]*Type{Sf, A}, []bool{false, false}),
				NewUnion([]*Type{B, Sf}, []bool{false, false})),
			structf,
		},
		{
			// interface { Sf | A; A | Sf } - NONE
			embed(NewUnion([]*Type{Sf, A}, []bool{false, false}),
				NewUnion([]*Type{A, Sf}, []bool{false, false})),
			nil,
		},
		{
			// interface { Sf | any } - NONE
			embed(NewUnion([]*Type{Sf, any}, []bool{false, false})),
			nil,
		},
		{
			// interface { Sf | any; Sf }
			embed(NewUnion([]*Type{Sf, any}, []bool{false, false}), Sf),
			structf,
		},
	}
	for _, tst := range tests {
		if got, want := tst.typ.StructuralType(), tst.structuralType; got != want {
			t.Errorf("StructuralType(%v) = %v, wanted %v",
				tst.typ, got, want)
		}
	}
}

func embed(types ...*Type) *Type {
	fields := make([]*Field, len(types))
	for i, t := range types {
		fields[i] = NewField(src.NoXPos, nil, t)
	}
	return NewInterface(LocalPkg, fields, false)
}
