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
	return Econv(int(t.Etype), 0)
}

func (t *Type) Equal(u ssa.Type) bool {
	x, ok := u.(*Type)
	if !ok {
		return false
	}
	return Eqtype(t, x)
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
