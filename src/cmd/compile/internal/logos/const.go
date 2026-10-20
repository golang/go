// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// This file erases M5-S1 const views during lowering (SPEC Sec 2.1,
// T-201): every `const T` node becomes its element type, and recorded
// slice expressions over const views gain a max bound of len(base) so
// a bound past the length panics at run time. The parent wires one
// call into lower.go, ahead of the other passes:
//
//	fl.eraseConstViews() // M5-S1: erase const views and bound recorded slices.
//
// The max-bound splice changes cap() of derived slices to len(base),
// which keeps every derived slice inside its view; a bound the
// checker already rejected never reaches here. Emission covers side
// effect-free (name) bases only; other recorded sites stay recorded
// for a later pass. Comments between `const` and its element do not
// survive the splice; comment-bearing programs still lower.

// eraseConstViews rewrites every const view type of one file to its
// element and bounds the recorded slice expressions. Every const view
// sits in a type-holding field below, so direct assignment keeps every
// replacement addressable.
func (fl *fileLower) eraseConstViews() {
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.Field:
			fl.eraseConstIn(&n.Type)
		case *syntax.TypeDecl:
			fl.eraseConstIn(&n.Type)
		case *syntax.VarDecl:
			fl.eraseConstIn(&n.Type)
		case *syntax.ConstDecl:
			fl.eraseConstIn(&n.Type)
		case *syntax.ArrayType:
			fl.eraseConstIn(&n.Elem)
		case *syntax.SliceType:
			fl.eraseConstIn(&n.Elem)
		case *syntax.MapType:
			fl.eraseConstIn(&n.Key)
			fl.eraseConstIn(&n.Value)
		case *syntax.ChanType:
			fl.eraseConstIn(&n.Elem)
		case *syntax.DotsType:
			fl.eraseConstIn(&n.Elem)
		case *syntax.Operation:
			if n.Op == syntax.Mul && n.Y == nil {
				fl.eraseConstIn(&n.X)
			}
		case *syntax.ParenExpr:
			fl.eraseConstIn(&n.X)
		case *syntax.IndexExpr:
			fl.eraseConstIn(&n.Index)
		case *syntax.ListExpr:
			for i := range n.ElemList {
				fl.eraseConstIn(&n.ElemList[i])
			}
		}
		return true
	})
	for e := range fl.pkg.info.LogosConstSlices {
		fl.boundConstSlice(e)
	}
}

// eraseConstIn replaces ConstType wrappers in *e with their elements,
// flattening chains.
func (fl *fileLower) eraseConstIn(e *syntax.Expr) {
	for {
		c, ok := (*e).(*syntax.ConstType)
		if !ok {
			return
		}
		fl.eraseConstNode(c)
		*e = c.Elem
	}
}

// eraseConstNode records the eject splice deleting the `const `
// prefix of one const view type.
func (fl *fileLower) eraseConstNode(c *syntax.ConstType) {
	start := fl.off(c.Pos())
	end := fl.off(c.Elem.Pos())
	if end > start {
		fl.add(start, end, "")
	}
}

// boundConstSlice gives one recorded slice expression a max bound of
// len(base), so a bound past the const length panics through the
// ordinary out-of-range failure. It runs only when the high bound
// needs a check, no max is present, and the base is a plain name.
func (fl *fileLower) boundConstSlice(e syntax.Expr) {
	s, ok := e.(*syntax.SliceExpr)
	if !ok || s.Index[2] != nil {
		return
	}
	base, ok := syntax.Unparen(s.X).(*syntax.Name)
	if !ok || s.Index[1] == nil {
		return
	}
	if !fl.constBoundNeedsCheck(s, s.Index[1]) {
		return
	}
	lenCall := &syntax.CallExpr{
		Fun:     nameOf(s.Pos(), "len"),
		ArgList: []syntax.Expr{nameOf(base.Pos(), base.Value)},
	}
	lenCall.SetPos(s.Pos())
	s.Index[2] = lenCall
	s.Full = true
	end := fl.off(syntax.EndPos(s.Index[1]))
	fl.add(end, end, ":len("+base.Value+")")
}

// constBoundNeedsCheck reports whether a recorded high bound still
// needs its runtime check: cap calls over the base already failed in
// the checker, len calls over the base and constant in-range array
// bounds are provably fine.
func (fl *fileLower) constBoundNeedsCheck(s *syntax.SliceExpr, hi syntax.Expr) bool {
	if fl.constIsLenOrCapOf(hi, s.X, "cap") {
		return false
	}
	if fl.constIsLenOrCapOf(hi, s.X, "len") {
		return false
	}
	if fl.constBoundValue(hi) {
		if c := constAsView(fl.pkg.info.TypeOf(syntax.Unparen(s.X))); c != nil {
			if _, ok := types2.Unalias(c.Elem).(*types2.Array); ok {
				return false
			}
		}
	}
	return true
}

// constIsLenOrCapOf reports whether bound is a call to the named
// built-in over the same variable as base.
func (fl *fileLower) constIsLenOrCapOf(bound, base syntax.Expr, name string) bool {
	call, ok := syntax.Unparen(bound).(*syntax.CallExpr)
	if !ok || len(call.ArgList) != 1 {
		return false
	}
	fn, ok := syntax.Unparen(call.Fun).(*syntax.Name)
	if !ok {
		return false
	}
	obj, ok := fl.pkg.info.Uses[fn].(*types2.Builtin)
	if !ok || obj == nil || obj.Name() != name {
		return false
	}
	bVar, ok := syntax.Unparen(base).(*syntax.Name)
	if !ok {
		return false
	}
	aVar, ok := syntax.Unparen(call.ArgList[0]).(*syntax.Name)
	if !ok {
		return false
	}
	bObj, bOk := fl.pkg.info.Uses[bVar]
	aObj, aOk := fl.pkg.info.Uses[aVar]
	return bOk && aOk && bObj == aObj
}

// constBoundValue reports whether bound is a constant.
func (fl *fileLower) constBoundValue(bound syntax.Expr) bool {
	un := syntax.Unparen(bound)
	if tv, ok := fl.pkg.info.Types[un]; ok && tv.Value != nil {
		return true
	}
	return un.GetTypeInfo().Value != nil
}

// infoHasConstView reports whether the first-check info holds a const
// view in a declared variable, an expression type, or an instantiation
// argument (SPEC T-201, M5-S6): the rt.Freeze call result is a const
// with no ConstType syntax node, so a file that uses only Freeze
// reports no rewrite and would hand const types to the export writer.
// A true result forces the plain recheck, which re-derives Go-facing
// types for the writer and the backend. files covers the production
// check, which stores types in the syntax tree and leaves info.Types
// nil, so an inline-only Freeze is visible only there.
func infoHasConstView(info *types2.Info, files []*syntax.File) bool {
	if info == nil {
		return false
	}
	for _, f := range files {
		found := false
		syntax.Inspect(f, func(n syntax.Node) bool {
			if e, ok := n.(syntax.Expr); ok && constAsView(info.TypeOf(e)) != nil {
				found = true
				return false
			}
			return true
		})
		if found {
			return true
		}
	}
	for _, obj := range info.Defs {
		if v, ok := obj.(*types2.Var); ok && constAsView(v.Type()) != nil {
			return true
		}
	}
	for _, obj := range info.Uses {
		if v, ok := obj.(*types2.Var); ok && constAsView(v.Type()) != nil {
			return true
		}
	}
	for _, tv := range info.Types {
		if constAsView(tv.Type) != nil {
			return true
		}
	}
	for _, inst := range info.Instances {
		if targs := inst.TypeArgs; targs != nil {
			for i := range targs.Len() {
				if constAsView(targs.At(i)) != nil {
					return true
				}
			}
		}
	}
	return false
}
