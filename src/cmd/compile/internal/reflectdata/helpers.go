// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflectdata

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

func hasRType(n, rtype ir.Node, fieldName string) bool {
	if rtype != nil {
		return true
	}

	return false
}

// assertOp asserts that n is an op.
func assertOp(n ir.Node, op ir.Op) {
	base.AssertfAt(n.Op() == op, n.Pos(), "want %v, have %v", op, n)
}

// assertOp2 asserts that n is an op1 or op2.
func assertOp2(n ir.Node, op1, op2 ir.Op) {
	base.AssertfAt(n.Op() == op1 || n.Op() == op2, n.Pos(), "want %v or %v, have %v", op1, op2, n)
}

// kindRType asserts that typ has the given kind, and returns an
// expression that yields the *runtime._type value representing typ.
func kindRType(pos src.XPos, typ *types.Type, k types.Kind) ir.Node {
	base.AssertfAt(typ.Kind() == k, pos, "want %v type, have %v", k, typ)
	return TypePtrAt(pos, typ)
}

// mapRType asserts that typ is a map type, and returns an expression
// that yields the *runtime._type value representing typ.
func mapRType(pos src.XPos, typ *types.Type) ir.Node {
	return kindRType(pos, typ, types.TMAP)
}

// chanRType asserts that typ is a map type, and returns an expression
// that yields the *runtime._type value representing typ.
func chanRType(pos src.XPos, typ *types.Type) ir.Node {
	return kindRType(pos, typ, types.TCHAN)
}

// sliceElemRType asserts that typ is a slice type, and returns an
// expression that yields the *runtime._type value representing typ's
// element type.
func sliceElemRType(pos src.XPos, typ *types.Type) ir.Node {
	base.AssertfAt(typ.IsSlice(), pos, "want slice type, have %v", typ)
	return TypePtrAt(pos, typ.Elem())
}

// concreteRType asserts that typ is not an interface type, and
// returns an expression that yields the *runtime._type value
// representing typ.
func concreteRType(pos src.XPos, typ *types.Type) ir.Node {
	base.AssertfAt(!typ.IsInterface(), pos, "want non-interface type, have %v", typ)
	return TypePtrAt(pos, typ)
}

// AppendElemRType asserts that n is an "append" operation, and
// returns an expression that yields the *runtime._type value
// representing the result slice type's element type.
func AppendElemRType(pos src.XPos, n *ir.CallExpr) ir.Node {
	assertOp(n, ir.OAPPEND)
	if hasRType(n, n.RType, "RType") {
		return n.RType
	}
	return sliceElemRType(pos, n.Type())
}

// CompareRType asserts that n is a comparison (== or !=) operation
// between expressions of interface and non-interface type, and
// returns an expression that yields the *runtime._type value
// representing the non-interface type.
func CompareRType(pos src.XPos, n *ir.BinaryExpr) ir.Node {
	assertOp2(n, ir.OEQ, ir.ONE)
	base.AssertfAt(n.X.Type().IsInterface() != n.Y.Type().IsInterface(), n.Pos(), "expect mixed interface and non-interface, have %L and %L", n.X, n.Y)
	if hasRType(n, n.RType, "RType") {
		return n.RType
	}
	typ := n.X.Type()
	if typ.IsInterface() {
		typ = n.Y.Type()
	}
	return concreteRType(pos, typ)
}

// ConvIfaceTypeWord asserts that n is conversion to interface type,
// and returns an expression that yields the *runtime._type or
// *runtime.itab value necessary for implementing the conversion.
//
//   - *runtime._type for the destination type, for I2I conversions
//   - *runtime.itab, for T2I conversions
//   - *runtime._type for the source type, for T2E conversions
func ConvIfaceTypeWord(pos src.XPos, n *ir.ConvExpr) ir.Node {
	assertOp(n, ir.OCONVIFACE)
	src, dst := n.X.Type(), n.Type()
	base.AssertfAt(dst.IsInterface(), n.Pos(), "want interface type, have %L", n)
	if hasRType(n, n.TypeWord, "TypeWord") {
		return n.TypeWord
	}
	if dst.IsEmptyInterface() {
		return concreteRType(pos, src) // direct eface construction
	}
	if !src.IsInterface() {
		return ITabAddrAt(pos, src, dst) // direct iface construction
	}
	return TypePtrAt(pos, dst) // convI2I
}

// ConvIfaceSrcRType asserts that n is a conversion from
// non-interface type to interface type, and
// returns an expression that yields the *runtime._type for copying
// the convertee value to the heap.
func ConvIfaceSrcRType(pos src.XPos, n *ir.ConvExpr) ir.Node {
	assertOp(n, ir.OCONVIFACE)
	if hasRType(n, n.SrcRType, "SrcRType") {
		return n.SrcRType
	}
	return concreteRType(pos, n.X.Type())
}

// CopyElemRType asserts that n is a "copy" operation, and returns an
// expression that yields the *runtime._type value representing the
// destination slice type's element type.
func CopyElemRType(pos src.XPos, n *ir.BinaryExpr) ir.Node {
	assertOp(n, ir.OCOPY)
	if hasRType(n, n.RType, "RType") {
		return n.RType
	}
	return sliceElemRType(pos, n.X.Type())
}

// DeleteMapRType asserts that n is a "delete" operation, and returns
// an expression that yields the *runtime._type value representing the
// map type.
func DeleteMapRType(pos src.XPos, n *ir.CallExpr) ir.Node {
	assertOp(n, ir.ODELETE)
	if hasRType(n, n.RType, "RType") {
		return n.RType
	}
	return mapRType(pos, n.Args[0].Type())
}

// IndexMapRType asserts that n is a map index operation, and returns
// an expression that yields the *runtime._type value representing the
// map type.
func IndexMapRType(pos src.XPos, n *ir.IndexExpr) ir.Node {
	assertOp(n, ir.OINDEXMAP)
	if hasRType(n, n.RType, "RType") {
		return n.RType
	}
	return mapRType(pos, n.X.Type())
}

// MakeChanRType asserts that n is a "make" operation for a channel
// type, and returns an expression that yields the *runtime._type
// value representing that channel type.
func MakeChanRType(pos src.XPos, n *ir.MakeExpr) ir.Node {
	assertOp(n, ir.OMAKECHAN)
	if hasRType(n, n.RType, "RType") {
		return n.RType
	}
	return chanRType(pos, n.Type())
}

// MakeMapRType asserts that n is a "make" operation for a map type,
// and returns an expression that yields the *runtime._type value
// representing that map type.
func MakeMapRType(pos src.XPos, n *ir.MakeExpr) ir.Node {
	assertOp(n, ir.OMAKEMAP)
	if hasRType(n, n.RType, "RType") {
		return n.RType
	}
	return mapRType(pos, n.Type())
}

// MakeSliceElemRType asserts that n is a "make" operation for a slice
// type, and returns an expression that yields the *runtime._type
// value representing that slice type's element type.
func MakeSliceElemRType(pos src.XPos, n *ir.MakeExpr) ir.Node {
	assertOp2(n, ir.OMAKESLICE, ir.OMAKESLICECOPY)
	if hasRType(n, n.RType, "RType") {
		return n.RType
	}
	return sliceElemRType(pos, n.Type())
}

// RangeMapRType asserts that n is a "range" loop over a map value,
// and returns an expression that yields the *runtime._type value
// representing that map type.
func RangeMapRType(pos src.XPos, n *ir.RangeStmt) ir.Node {
	assertOp(n, ir.ORANGE)
	if hasRType(n, n.RType, "RType") {
		return n.RType
	}
	return mapRType(pos, n.X.Type())
}

// UnsafeSliceElemRType asserts that n is an "unsafe.Slice" operation,
// and returns an expression that yields the *runtime._type value
// representing the result slice type's element type.
func UnsafeSliceElemRType(pos src.XPos, n *ir.BinaryExpr) ir.Node {
	assertOp(n, ir.OUNSAFESLICE)
	if hasRType(n, n.RType, "RType") {
		return n.RType
	}
	return sliceElemRType(pos, n.Type())
}
