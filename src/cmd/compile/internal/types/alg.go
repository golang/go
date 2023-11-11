// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import "cmd/compile/internal/base"

// AlgKind describes the kind of algorithms used for comparing and
// hashing a Type.
type AlgKind int

//go:generate stringer -type AlgKind -trimprefix A alg.go

const (
	ANOEQ AlgKind = iota
	AMEM0
	AMEM8
	AMEM16
	AMEM32
	AMEM64
	AMEM128
	ASTRING
	AINTER
	ANILINTER
	AFLOAT32
	AFLOAT64
	ACPLX64
	ACPLX128

	// Type can be compared/hashed as regular memory.
	AMEM AlgKind = 100

	// Type needs special comparison/hashing functions.
	ASPECIAL AlgKind = -1
)

// AlgType returns the AlgKind used for comparing and hashing Type t.
// If it returns ANOEQ, it also returns the component type of t that
// makes it incomparable.
func AlgType(t *Type) (AlgKind, *Type) {
	if t.Noalg() {
		return ANOEQ, t
	}

	switch t.Kind() {
	case TANY, TFORW:
		// will be defined later.
		return ANOEQ, t

	case TINT8, TUINT8, TINT16, TUINT16,
		TINT32, TUINT32, TINT64, TUINT64,
		TINT, TUINT, TUINTPTR,
		TBOOL, TPTR,
		TCHAN, TUNSAFEPTR:
		return AMEM, nil

	case TFUNC, TMAP:
		return ANOEQ, t

	case TFLOAT32:
		return AFLOAT32, nil

	case TFLOAT64:
		return AFLOAT64, nil

	case TCOMPLEX64:
		return ACPLX64, nil

	case TCOMPLEX128:
		return ACPLX128, nil

	case TSTRING:
		return ASTRING, nil

	case TINTER:
		if t.IsEmptyInterface() {
			return ANILINTER, nil
		}
		return AINTER, nil

	case TSLICE:
		return ANOEQ, t

	case TARRAY:
		a, bad := AlgType(t.Elem())
		switch a {
		case AMEM:
			return AMEM, nil
		case ANOEQ:
			return ANOEQ, bad
		}

		switch t.NumElem() {
		case 0:
			// We checked above that the element type is comparable.
			return AMEM, nil
		case 1:
			// Single-element array is same as its lone element.
			return a, nil
		}

		return ASPECIAL, nil

	case TSTRUCT:
		fields := t.Fields()

		// One-field struct is same as that one field alone.
		if len(fields) == 1 && !fields[0].Sym.IsBlank() {
			return AlgType(fields[0].Type)
		}

		ret := AMEM
		for i, f := range fields {
			// All fields must be comparable.
			a, bad := AlgType(f.Type)
			if a == ANOEQ {
				return ANOEQ, bad
			}

			// Blank fields, padded fields, fields with non-memory
			// equality need special compare.
			if a != AMEM || f.Sym.IsBlank() || IsPaddedField(t, i) {
				ret = ASPECIAL
			}
		}

		return ret, nil
	}

	base.Fatalf("AlgType: unexpected type %v", t)
	return 0, nil
}

// TypeHasNoAlg reports whether t does not have any associated hash/eq
// algorithms because t, or some component of t, is marked Noalg.
func TypeHasNoAlg(t *Type) bool {
	a, bad := AlgType(t)
	return a == ANOEQ && bad.Noalg()
}

// IsComparable reports whether t is a comparable type.
func IsComparable(t *Type) bool {
	a, _ := AlgType(t)
	return a != ANOEQ
}

// IncomparableField returns an incomparable Field of struct Type t, if any.
func IncomparableField(t *Type) *Field {
	for _, f := range t.Fields() {
		if !IsComparable(f.Type) {
			return f
		}
	}
	return nil
}

// IsPaddedField reports whether the i'th field of struct type t is followed
// by padding.
func IsPaddedField(t *Type, i int) bool {
	if !t.IsStruct() {
		base.Fatalf("IsPaddedField called non-struct %v", t)
	}
	end := t.width
	if i+1 < t.NumFields() {
		end = t.Field(i + 1).Offset
	}
	return t.Field(i).End() != end
}
