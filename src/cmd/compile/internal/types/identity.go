// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

// Identical reports whether t1 and t2 are identical types, following
// the spec rules. Receiver parameter types are ignored.
func Identical(t1, t2 *Type) bool {
	return identical(t1, t2, true, nil)
}

// IdenticalIgnoreTags is like Identical, but it ignores struct tags
// for struct identity.
func IdenticalIgnoreTags(t1, t2 *Type) bool {
	return identical(t1, t2, false, nil)
}

type typePair struct {
	t1 *Type
	t2 *Type
}

func identical(t1, t2 *Type, cmpTags bool, assumedEqual map[typePair]struct{}) bool {
	if t1 == t2 {
		return true
	}
	if t1 == nil || t2 == nil || t1.Etype != t2.Etype || t1.Broke() || t2.Broke() {
		return false
	}
	if t1.Sym != nil || t2.Sym != nil {
		// Special case: we keep byte/uint8 and rune/int32
		// separate for error messages. Treat them as equal.
		switch t1.Etype {
		case TUINT8:
			return (t1 == Types[TUINT8] || t1 == Bytetype) && (t2 == Types[TUINT8] || t2 == Bytetype)
		case TINT32:
			return (t1 == Types[TINT32] || t1 == Runetype) && (t2 == Types[TINT32] || t2 == Runetype)
		default:
			return false
		}
	}

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

	switch t1.Etype {
	case TINTER:
		if t1.NumFields() != t2.NumFields() {
			return false
		}
		for i, f1 := range t1.FieldSlice() {
			f2 := t2.Field(i)
			if f1.Sym != f2.Sym || !identical(f1.Type, f2.Type, cmpTags, assumedEqual) {
				return false
			}
		}
		return true

	case TSTRUCT:
		if t1.NumFields() != t2.NumFields() {
			return false
		}
		for i, f1 := range t1.FieldSlice() {
			f2 := t2.Field(i)
			if f1.Sym != f2.Sym || f1.Embedded != f2.Embedded || !identical(f1.Type, f2.Type, cmpTags, assumedEqual) {
				return false
			}
			if cmpTags && f1.Note != f2.Note {
				return false
			}
		}
		return true

	case TFUNC:
		// Check parameters and result parameters for type equality.
		// We intentionally ignore receiver parameters for type
		// equality, because they're never relevant.
		for _, f := range ParamsResults {
			// Loop over fields in structs, ignoring argument names.
			fs1, fs2 := f(t1).FieldSlice(), f(t2).FieldSlice()
			if len(fs1) != len(fs2) {
				return false
			}
			for i, f1 := range fs1 {
				f2 := fs2[i]
				if f1.IsDDD() != f2.IsDDD() || !identical(f1.Type, f2.Type, cmpTags, assumedEqual) {
					return false
				}
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
		if !identical(t1.Key(), t2.Key(), cmpTags, assumedEqual) {
			return false
		}
	}

	return identical(t1.Elem(), t2.Elem(), cmpTags, assumedEqual)
}
