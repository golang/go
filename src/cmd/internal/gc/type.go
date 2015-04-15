// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file provides methods that let us export a Type as an ../ssa:Type.
// We don't export this package's Type directly because it would lead
// to an import cycle with this package and ../ssa.
// TODO: move Type to its own package, then we don't need to dance around import cycles.

package gc

import (
	"cmd/internal/ssa"
)

func (t *Type) Size() int64 {
	dowidth(t)
	return t.Width
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

func (t *Type) IsPtr() bool {
	return t.Etype == TPTR32 || t.Etype == TPTR64 ||
		t.Etype == TMAP || t.Etype == TCHAN || t.Etype == TFUNC
}

func (t *Type) Elem() ssa.Type {
	return t.Type
}
func (t *Type) PtrTo() ssa.Type {
	return Ptrto(t)
}

func (t *Type) IsMemory() bool { return false }
func (t *Type) IsFlags() bool  { return false }

func (t *Type) String() string {
	return typefmt(t, 0)
}
