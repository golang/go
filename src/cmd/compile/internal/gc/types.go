// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/types"
	"fmt"
)

// convenience constants
const (
	Txxx = types.Txxx

	TINT8    = types.TINT8
	TUINT8   = types.TUINT8
	TINT16   = types.TINT16
	TUINT16  = types.TUINT16
	TINT32   = types.TINT32
	TUINT32  = types.TUINT32
	TINT64   = types.TINT64
	TUINT64  = types.TUINT64
	TINT     = types.TINT
	TUINT    = types.TUINT
	TUINTPTR = types.TUINTPTR

	TCOMPLEX64  = types.TCOMPLEX64
	TCOMPLEX128 = types.TCOMPLEX128

	TFLOAT32 = types.TFLOAT32
	TFLOAT64 = types.TFLOAT64

	TBOOL = types.TBOOL

	TPTR32 = types.TPTR32
	TPTR64 = types.TPTR64

	TFUNC      = types.TFUNC
	TSLICE     = types.TSLICE
	TARRAY     = types.TARRAY
	TSTRUCT    = types.TSTRUCT
	TCHAN      = types.TCHAN
	TMAP       = types.TMAP
	TINTER     = types.TINTER
	TFORW      = types.TFORW
	TANY       = types.TANY
	TSTRING    = types.TSTRING
	TUNSAFEPTR = types.TUNSAFEPTR

	// pseudo-types for literals
	TIDEAL = types.TIDEAL
	TNIL   = types.TNIL
	TBLANK = types.TBLANK

	// pseudo-types for frame layout
	TFUNCARGS = types.TFUNCARGS
	TCHANARGS = types.TCHANARGS

	// pseudo-types for import/export
	TDDDFIELD = types.TDDDFIELD // wrapper: contained type is a ... field

	NTYPE = types.NTYPE
)

func cmpForNe(x bool) ssa.Cmp {
	if x {
		return ssa.CMPlt
	}
	return ssa.CMPgt
}

func cmpsym(r, s *types.Sym) ssa.Cmp {
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

// cmptyp compares two *Types t and x, returning ssa.CMPlt,
// ssa.CMPeq, ssa.CMPgt as t<x, t==x, t>x, for an arbitrary
// and optimizer-centric notion of comparison.
func cmptyp(t, x *types.Type) ssa.Cmp {
	// This follows the structure of eqtype in subr.go
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
		// for error messages. Treat them as equal.
		switch t.Etype {
		case TUINT8:
			if (t == types.Types[TUINT8] || t == types.Bytetype) && (x == types.Types[TUINT8] || x == types.Bytetype) {
				return ssa.CMPeq
			}

		case TINT32:
			if (t == types.Types[types.Runetype.Etype] || t == types.Runetype) && (x == types.Types[types.Runetype.Etype] || x == types.Runetype) {
				return ssa.CMPeq
			}
		}
	}

	if c := cmpsym(t.Sym, x.Sym); c != ssa.CMPeq {
		return c
	}

	if x.Sym != nil {
		// Syms non-nil, if vargens match then equal.
		if t.Vargen != x.Vargen {
			return cmpForNe(t.Vargen < x.Vargen)
		}
		return ssa.CMPeq
	}
	// both syms nil, look at structure below.

	switch t.Etype {
	case TBOOL, TFLOAT32, TFLOAT64, TCOMPLEX64, TCOMPLEX128, TUNSAFEPTR, TUINTPTR,
		TINT8, TINT16, TINT32, TINT64, TINT, TUINT8, TUINT16, TUINT32, TUINT64, TUINT:
		return ssa.CMPeq
	}

	switch t.Etype {
	case TMAP:
		if c := cmptyp(t.Key(), x.Key()); c != ssa.CMPeq {
			return c
		}
		return cmptyp(t.Val(), x.Val())

	case TPTR32, TPTR64, TSLICE:
		// No special cases for these, they are handled
		// by the general code after the switch.

	case TSTRUCT:
		if t.StructType().Map == nil {
			if x.StructType().Map != nil {
				return ssa.CMPlt // nil < non-nil
			}
			// to the fallthrough
		} else if x.StructType().Map == nil {
			return ssa.CMPgt // nil > non-nil
		} else if t.StructType().Map.MapType().Bucket == t {
			// Both have non-nil Map
			// Special case for Maps which include a recursive type where the recursion is not broken with a named type
			if x.StructType().Map.MapType().Bucket != x {
				return ssa.CMPlt // bucket maps are least
			}
			return cmptyp(t.StructType().Map, x.StructType().Map)
		} else if x.StructType().Map.MapType().Bucket == x {
			return ssa.CMPgt // bucket maps are least
		} // If t != t.Map.Bucket, fall through to general case

		fallthrough
	case TINTER:
		t1, ti := types.IterFields(t)
		x1, xi := types.IterFields(x)
		for ; t1 != nil && x1 != nil; t1, x1 = ti.Next(), xi.Next() {
			if t1.Embedded != x1.Embedded {
				return cmpForNe(t1.Embedded < x1.Embedded)
			}
			if t1.Note != x1.Note {
				return cmpForNe(t1.Note < x1.Note)
			}
			if c := cmpsym(t1.Sym, x1.Sym); c != ssa.CMPeq {
				return c
			}
			if c := cmptyp(t1.Type, x1.Type); c != ssa.CMPeq {
				return c
			}
		}
		if t1 != x1 {
			return cmpForNe(t1 == nil)
		}
		return ssa.CMPeq

	case TFUNC:
		for _, f := range types.RecvsParamsResults {
			// Loop over fields in structs, ignoring argument names.
			ta, ia := types.IterFields(f(t))
			tb, ib := types.IterFields(f(x))
			for ; ta != nil && tb != nil; ta, tb = ia.Next(), ib.Next() {
				if ta.Isddd() != tb.Isddd() {
					return cmpForNe(!ta.Isddd())
				}
				if c := cmptyp(ta.Type, tb.Type); c != ssa.CMPeq {
					return c
				}
			}
			if ta != tb {
				return cmpForNe(ta == nil)
			}
		}
		return ssa.CMPeq

	case TARRAY:
		if t.NumElem() != x.NumElem() {
			return cmpForNe(t.NumElem() < x.NumElem())
		}

	case TCHAN:
		if t.ChanDir() != x.ChanDir() {
			return cmpForNe(t.ChanDir() < x.ChanDir())
		}

	default:
		e := fmt.Sprintf("Do not know how to compare %v with %v", t, x)
		panic(e)
	}

	// Common element type comparison for TARRAY, TCHAN, TPTR32, TPTR64, and TSLICE.
	return cmptyp(t.Elem(), x.Elem())
}
