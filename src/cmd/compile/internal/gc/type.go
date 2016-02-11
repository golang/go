// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file provides methods that let us export a Type as an ../ssa:Type.
// We don't export this package's Type directly because it would lead
// to an import cycle with this package and ../ssa.
// TODO: move Type to its own package, then we don't need to dance around import cycles.

package gc

import (
	"cmd/compile/internal/ssa"
	"fmt"
)

func (t *Type) Size() int64 {
	dowidth(t)
	return t.Width
}

func (t *Type) Alignment() int64 {
	dowidth(t)
	return int64(t.Align)
}

func (t *Type) SimpleString() string {
	return Econv(t.Etype)
}

func (t *Type) Equal(u ssa.Type) bool {
	x, ok := u.(*Type)
	if !ok {
		return false
	}
	return Eqtype(t, x)
}

// Compare compares types for purposes of the SSA back
// end, returning an ssa.Cmp (one of CMPlt, CMPeq, CMPgt).
// The answers are correct for an optimizer
// or code generator, but not for Go source.
// For example, "type gcDrainFlags int" results in
// two Go-different types that Compare equal.
// The order chosen is also arbitrary, only division into
// equivalence classes (Types that compare CMPeq) matters.
func (t *Type) Compare(u ssa.Type) ssa.Cmp {
	x, ok := u.(*Type)
	// ssa.CompilerType is smaller than gc.Type
	// bare pointer equality is easy.
	if !ok {
		return ssa.CMPgt
	}
	if x == t {
		return ssa.CMPeq
	}
	return t.cmp(x)
}

func cmpForNe(x bool) ssa.Cmp {
	if x {
		return ssa.CMPlt
	}
	return ssa.CMPgt
}

func (r *Sym) cmpsym(s *Sym) ssa.Cmp {
	if r == s {
		return ssa.CMPeq
	}
	if r == nil {
		return ssa.CMPlt
	}
	if s == nil {
		return ssa.CMPgt
	}
	// Fast sort, not pretty sort
	if len(r.Name) != len(s.Name) {
		return cmpForNe(len(r.Name) < len(s.Name))
	}
	if r.Pkg != s.Pkg {
		if len(r.Pkg.Prefix) != len(s.Pkg.Prefix) {
			return cmpForNe(len(r.Pkg.Prefix) < len(s.Pkg.Prefix))
		}
		if r.Pkg.Prefix != s.Pkg.Prefix {
			return cmpForNe(r.Pkg.Prefix < s.Pkg.Prefix)
		}
	}
	if r.Name != s.Name {
		return cmpForNe(r.Name < s.Name)
	}
	return ssa.CMPeq
}

// cmp compares two *Types t and x, returning ssa.CMPlt,
// ssa.CMPeq, ssa.CMPgt as t<x, t==x, t>x, for an arbitrary
// and optimizer-centric notion of comparison.
func (t *Type) cmp(x *Type) ssa.Cmp {
	// This follows the structure of Eqtype in subr.go
	// with two exceptions.
	// 1. Symbols are compared more carefully because a <,=,> result is desired.
	// 2. Maps are treated specially to avoid endless recursion -- maps
	//    contain an internal data type not expressible in Go source code.
	if t == x {
		return ssa.CMPeq
	}
	if t == nil {
		return ssa.CMPlt
	}
	if x == nil {
		return ssa.CMPgt
	}

	if t.Etype != x.Etype {
		return cmpForNe(t.Etype < x.Etype)
	}

	if t.Sym != nil || x.Sym != nil {
		// Special case: we keep byte and uint8 separate
		// for error messages.  Treat them as equal.
		switch t.Etype {
		case TUINT8:
			if (t == Types[TUINT8] || t == bytetype) && (x == Types[TUINT8] || x == bytetype) {
				return ssa.CMPeq
			}

		case TINT32:
			if (t == Types[runetype.Etype] || t == runetype) && (x == Types[runetype.Etype] || x == runetype) {
				return ssa.CMPeq
			}
		}
	}

	csym := t.Sym.cmpsym(x.Sym)
	if csym != ssa.CMPeq {
		return csym
	}

	if x.Sym != nil {
		// Syms non-nil, if vargens match then equal.
		if t.Vargen == x.Vargen {
			return ssa.CMPeq
		}
		if t.Vargen < x.Vargen {
			return ssa.CMPlt
		}
		return ssa.CMPgt
	}
	// both syms nil, look at structure below.

	switch t.Etype {
	case TBOOL, TFLOAT32, TFLOAT64, TCOMPLEX64, TCOMPLEX128, TUNSAFEPTR, TUINTPTR,
		TINT8, TINT16, TINT32, TINT64, TINT, TUINT8, TUINT16, TUINT32, TUINT64, TUINT:
		return ssa.CMPeq
	}

	switch t.Etype {
	case TMAP, TFIELD:
		// No special cases for these two, they are handled
		// by the general code after the switch.

	case TPTR32, TPTR64:
		return t.Type.cmp(x.Type)

	case TSTRUCT:
		if t.Map == nil {
			if x.Map != nil {
				return ssa.CMPlt // nil < non-nil
			}
			// to the fallthrough
		} else if x.Map == nil {
			return ssa.CMPgt // nil > non-nil
		} else if t.Map.Bucket == t {
			// Both have non-nil Map
			// Special case for Maps which include a recursive type where the recursion is not broken with a named type
			if x.Map.Bucket != x {
				return ssa.CMPlt // bucket maps are least
			}
			return t.Map.cmp(x.Map)
		} // If t != t.Map.Bucket, fall through to general case

		fallthrough
	case TINTER:
		t1 := t.Type
		x1 := x.Type
		for ; t1 != nil && x1 != nil; t1, x1 = t1.Down, x1.Down {
			if t1.Embedded != x1.Embedded {
				if t1.Embedded < x1.Embedded {
					return ssa.CMPlt
				}
				return ssa.CMPgt
			}
			if t1.Note != x1.Note {
				if t1.Note == nil {
					return ssa.CMPlt
				}
				if x1.Note == nil {
					return ssa.CMPgt
				}
				if *t1.Note != *x1.Note {
					if *t1.Note < *x1.Note {
						return ssa.CMPlt
					}
					return ssa.CMPgt
				}
			}
			c := t1.Sym.cmpsym(x1.Sym)
			if c != ssa.CMPeq {
				return c
			}
			c = t1.Type.cmp(x1.Type)
			if c != ssa.CMPeq {
				return c
			}
		}
		if t1 == x1 {
			return ssa.CMPeq
		}
		if t1 == nil {
			return ssa.CMPlt
		}
		return ssa.CMPgt

	case TFUNC:
		t1 := t.Type
		t2 := x.Type
		for ; t1 != nil && t2 != nil; t1, t2 = t1.Down, t2.Down {
			// Loop over fields in structs, ignoring argument names.
			ta := t1.Type
			tb := t2.Type
			for ; ta != nil && tb != nil; ta, tb = ta.Down, tb.Down {
				if ta.Isddd != tb.Isddd {
					if ta.Isddd {
						return ssa.CMPgt
					}
					return ssa.CMPlt
				}
				c := ta.Type.cmp(tb.Type)
				if c != ssa.CMPeq {
					return c
				}
			}

			if ta != tb {
				if t1 == nil {
					return ssa.CMPlt
				}
				return ssa.CMPgt
			}
		}
		if t1 != t2 {
			if t1 == nil {
				return ssa.CMPlt
			}
			return ssa.CMPgt
		}
		return ssa.CMPeq

	case TARRAY:
		if t.Bound != x.Bound {
			return cmpForNe(t.Bound < x.Bound)
		}

	case TCHAN:
		if t.Chan != x.Chan {
			return cmpForNe(t.Chan < x.Chan)
		}

	default:
		e := fmt.Sprintf("Do not know how to compare %s with %s", t, x)
		panic(e)
	}

	c := t.Down.cmp(x.Down)
	if c != ssa.CMPeq {
		return c
	}
	return t.Type.cmp(x.Type)
}

func (t *Type) IsBoolean() bool {
	return t.Etype == TBOOL
}

func (t *Type) IsInteger() bool {
	switch t.Etype {
	case TINT8, TUINT8, TINT16, TUINT16, TINT32, TUINT32, TINT64, TUINT64, TINT, TUINT, TUINTPTR:
		return true
	}
	return false
}

func (t *Type) IsSigned() bool {
	switch t.Etype {
	case TINT8, TINT16, TINT32, TINT64, TINT:
		return true
	}
	return false
}

func (t *Type) IsFloat() bool {
	return t.Etype == TFLOAT32 || t.Etype == TFLOAT64
}

func (t *Type) IsComplex() bool {
	return t.Etype == TCOMPLEX64 || t.Etype == TCOMPLEX128
}

func (t *Type) IsPtr() bool {
	return t.Etype == TPTR32 || t.Etype == TPTR64 || t.Etype == TUNSAFEPTR ||
		t.Etype == TMAP || t.Etype == TCHAN || t.Etype == TFUNC
}

func (t *Type) IsString() bool {
	return t.Etype == TSTRING
}

func (t *Type) IsMap() bool {
	return t.Etype == TMAP
}

func (t *Type) IsChan() bool {
	return t.Etype == TCHAN
}

func (t *Type) IsSlice() bool {
	return t.Etype == TARRAY && t.Bound < 0
}

func (t *Type) IsArray() bool {
	return t.Etype == TARRAY && t.Bound >= 0
}

func (t *Type) IsStruct() bool {
	return t.Etype == TSTRUCT
}

func (t *Type) IsInterface() bool {
	return t.Etype == TINTER
}

func (t *Type) Elem() ssa.Type {
	return t.Type
}
func (t *Type) PtrTo() ssa.Type {
	return Ptrto(t)
}

func (t *Type) NumFields() int64 {
	return int64(countfield(t))
}
func (t *Type) FieldType(i int64) ssa.Type {
	// TODO: store fields in a slice so we can
	// look them up by index in constant time.
	for t1 := t.Type; t1 != nil; t1 = t1.Down {
		if t1.Etype != TFIELD {
			panic("non-TFIELD in a TSTRUCT")
		}
		if i == 0 {
			return t1.Type
		}
		i--
	}
	panic("not enough fields")
}
func (t *Type) FieldOff(i int64) int64 {
	for t1 := t.Type; t1 != nil; t1 = t1.Down {
		if t1.Etype != TFIELD {
			panic("non-TFIELD in a TSTRUCT")
		}
		if i == 0 {
			return t1.Width
		}
		i--
	}
	panic("not enough fields")
}

func (t *Type) NumElem() int64 {
	if t.Etype != TARRAY {
		panic("NumElem on non-TARRAY")
	}
	return int64(t.Bound)
}

func (t *Type) IsMemory() bool { return false }
func (t *Type) IsFlags() bool  { return false }
func (t *Type) IsVoid() bool   { return false }
