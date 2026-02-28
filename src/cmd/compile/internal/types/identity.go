// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

const (
	identIgnoreTags = 1 << iota
	identStrict
)

// Identical reports whether t1 and t2 are identical types, following the spec rules.
// Receiver parameter types are ignored. Named (defined) types are only equal if they
// are pointer-equal - i.e. there must be a unique types.Type for each specific named
// type. Also, a type containing a shape type is considered identical to another type
// (shape or not) if their underlying types are the same, or they are both pointers.
func Identical(t1, t2 *Type) bool {
	return identical(t1, t2, 0, nil)
}

// IdenticalIgnoreTags is like Identical, but it ignores struct tags
// for struct identity.
func IdenticalIgnoreTags(t1, t2 *Type) bool {
	return identical(t1, t2, identIgnoreTags, nil)
}

// IdenticalStrict is like Identical, but matches types exactly, without the
// exception for shapes.
func IdenticalStrict(t1, t2 *Type) bool {
	return identical(t1, t2, identStrict, nil)
}

type typePair struct {
	t1 *Type
	t2 *Type
}

func identical(t1, t2 *Type, flags int, assumedEqual map[typePair]struct{}) bool {
	if t1 == t2 {
		return true
	}
	if t1 == nil || t2 == nil || t1.kind != t2.kind {
		return false
	}
	if t1.obj != nil || t2.obj != nil {
		if flags&identStrict == 0 && (t1.HasShape() || t2.HasShape()) {
			switch t1.kind {
			case TINT8, TUINT8, TINT16, TUINT16, TINT32, TUINT32, TINT64, TUINT64, TINT, TUINT, TUINTPTR, TCOMPLEX64, TCOMPLEX128, TFLOAT32, TFLOAT64, TBOOL, TSTRING, TPTR, TUNSAFEPTR:
				return true
			}
			// fall through to unnamed type comparison for complex types.
			goto cont
		}
		// Special case: we keep byte/uint8 and rune/int32
		// separate for error messages. Treat them as equal.
		switch t1.kind {
		case TUINT8:
			return (t1 == Types[TUINT8] || t1 == ByteType) && (t2 == Types[TUINT8] || t2 == ByteType)
		case TINT32:
			return (t1 == Types[TINT32] || t1 == RuneType) && (t2 == Types[TINT32] || t2 == RuneType)
		case TINTER:
			// Make sure named any type matches any unnamed empty interface
			// (but not a shape type, if identStrict).
			isUnnamedEface := func(t *Type) bool { return t.IsEmptyInterface() && t.Sym() == nil }
			if flags&identStrict != 0 {
				return t1 == AnyType && isUnnamedEface(t2) && !t2.HasShape() || t2 == AnyType && isUnnamedEface(t1) && !t1.HasShape()
			}
			return t1 == AnyType && isUnnamedEface(t2) || t2 == AnyType && isUnnamedEface(t1)
		default:
			return false
		}
	}
cont:

	// Any cyclic type must go through a named type, and if one is
	// named, it is only identical to the other if they are the
	// same pointer (t1 == t2), so there's no chance of chasing
	// cycles ad infinitum, so no need for a depth counter.
	if assumedEqual == nil {
		assumedEqual = make(map[typePair]struct{})
	} else if _, ok := assumedEqual[typePair{t1, t2}]; ok {
		return true
	}
	assumedEqual[typePair{t1, t2}] = struct{}{}

	switch t1.kind {
	case TIDEAL:
		// Historically, cmd/compile used a single "untyped
		// number" type, so all untyped number types were
		// identical. Match this behavior.
		// TODO(mdempsky): Revisit this.
		return true

	case TINTER:
		if len(t1.AllMethods()) != len(t2.AllMethods()) {
			return false
		}
		for i, f1 := range t1.AllMethods() {
			f2 := t2.AllMethods()[i]
			if f1.Sym != f2.Sym || !identical(f1.Type, f2.Type, flags, assumedEqual) {
				return false
			}
		}
		return true

	case TSTRUCT:
		if t1.NumFields() != t2.NumFields() {
			return false
		}
		for i, f1 := range t1.Fields() {
			f2 := t2.Field(i)
			if f1.Sym != f2.Sym || f1.Embedded != f2.Embedded || !identical(f1.Type, f2.Type, flags, assumedEqual) {
				return false
			}
			if (flags&identIgnoreTags) == 0 && f1.Note != f2.Note {
				return false
			}
		}
		return true

	case TFUNC:
		// Check parameters and result parameters for type equality.
		// We intentionally ignore receiver parameters for type
		// equality, because they're never relevant.
		if t1.NumParams() != t2.NumParams() ||
			t1.NumResults() != t2.NumResults() ||
			t1.IsVariadic() != t2.IsVariadic() {
			return false
		}

		fs1 := t1.ParamsResults()
		fs2 := t2.ParamsResults()
		for i, f1 := range fs1 {
			if !identical(f1.Type, fs2[i].Type, flags, assumedEqual) {
				return false
			}
		}
		return true

	case TARRAY:
		if t1.NumElem() != t2.NumElem() {
			return false
		}

	case TCHAN:
		if t1.ChanDir() != t2.ChanDir() {
			return false
		}

	case TMAP:
		if !identical(t1.Key(), t2.Key(), flags, assumedEqual) {
			return false
		}
	}

	return identical(t1.Elem(), t2.Elem(), flags, assumedEqual)
}
