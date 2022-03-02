// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"fmt"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/compile/internal/types2"
	"cmd/internal/src"
)

func (g *irgen) expr(expr syntax.Expr) ir.Node {
	expr = unparen(expr) // skip parens; unneeded after parse+typecheck

	if expr == nil {
		return nil
	}

	if expr, ok := expr.(*syntax.Name); ok && expr.Value == "_" {
		return ir.BlankNode
	}

	tv, ok := g.info.Types[expr]
	if !ok {
		base.FatalfAt(g.pos(expr), "missing type for %v (%T)", expr, expr)
	}
	switch {
	case tv.IsBuiltin():
		// Qualified builtins, such as unsafe.Add and unsafe.Slice.
		if expr, ok := expr.(*syntax.SelectorExpr); ok {
			if name, ok := expr.X.(*syntax.Name); ok {
				if _, ok := g.info.Uses[name].(*types2.PkgName); ok {
					return g.use(expr.Sel)
				}
			}
		}
		return g.use(expr.(*syntax.Name))
	case tv.IsType():
		return ir.TypeNode(g.typ(tv.Type))
	case tv.IsValue(), tv.IsVoid():
		// ok
	default:
		base.FatalfAt(g.pos(expr), "unrecognized type-checker result")
	}

	base.Assert(g.exprStmtOK)

	// The gc backend expects all expressions to have a concrete type, and
	// types2 mostly satisfies this expectation already. But there are a few
	// cases where the Go spec doesn't require converting to concrete type,
	// and so types2 leaves them untyped. So we need to fix those up here.
	typ := tv.Type
	if basic, ok := typ.(*types2.Basic); ok && basic.Info()&types2.IsUntyped != 0 {
		switch basic.Kind() {
		case types2.UntypedNil:
			// ok; can appear in type switch case clauses
			// TODO(mdempsky): Handle as part of type switches instead?
		case types2.UntypedBool:
			typ = types2.Typ[types2.Bool] // expression in "if" or "for" condition
		case types2.UntypedString:
			typ = types2.Typ[types2.String] // argument to "append" or "copy" calls
		default:
			base.FatalfAt(g.pos(expr), "unexpected untyped type: %v", basic)
		}
	}

	// Constant expression.
	if tv.Value != nil {
		typ := g.typ(typ)
		value := FixValue(typ, tv.Value)
		return OrigConst(g.pos(expr), typ, value, constExprOp(expr), syntax.String(expr))
	}

	n := g.expr0(typ, expr)
	if n.Typecheck() != 1 && n.Typecheck() != 3 {
		base.FatalfAt(g.pos(expr), "missed typecheck: %+v", n)
	}
	if n.Op() != ir.OFUNCINST && !g.match(n.Type(), typ, tv.HasOk()) {
		base.FatalfAt(g.pos(expr), "expected %L to have type %v", n, typ)
	}
	return n
}

func (g *irgen) expr0(typ types2.Type, expr syntax.Expr) ir.Node {
	pos := g.pos(expr)
	assert(pos.IsKnown())

	// Set base.Pos for transformation code that still uses base.Pos, rather than
	// the pos of the node being converted.
	base.Pos = pos

	switch expr := expr.(type) {
	case *syntax.Name:
		if _, isNil := g.info.Uses[expr].(*types2.Nil); isNil {
			return Nil(pos, g.typ(typ))
		}
		return g.use(expr)

	case *syntax.CompositeLit:
		return g.compLit(typ, expr)

	case *syntax.FuncLit:
		return g.funcLit(typ, expr)

	case *syntax.AssertExpr:
		return Assert(pos, g.expr(expr.X), g.typeExpr(expr.Type))

	case *syntax.CallExpr:
		fun := g.expr(expr.Fun)
		return g.callExpr(pos, g.typ(typ), fun, g.exprs(expr.ArgList), expr.HasDots)

	case *syntax.IndexExpr:
		args := unpackListExpr(expr.Index)
		if len(args) == 1 {
			tv, ok := g.info.Types[args[0]]
			assert(ok)
			if tv.IsValue() {
				// This is just a normal index expression
				n := Index(pos, g.typ(typ), g.expr(expr.X), g.expr(args[0]))
				if !g.delayTransform() {
					// transformIndex will modify n.Type() for OINDEXMAP.
					transformIndex(n)
				}
				return n
			}
		}

		// expr.Index is a list of type args, so we ignore it, since types2 has
		// already provided this info with the Info.Instances map.
		return g.expr(expr.X)

	case *syntax.SelectorExpr:
		// Qualified identifier.
		if name, ok := expr.X.(*syntax.Name); ok {
			if _, ok := g.info.Uses[name].(*types2.PkgName); ok {
				return g.use(expr.Sel)
			}
		}
		return g.selectorExpr(pos, typ, expr)

	case *syntax.SliceExpr:
		n := Slice(pos, g.typ(typ), g.expr(expr.X), g.expr(expr.Index[0]), g.expr(expr.Index[1]), g.expr(expr.Index[2]))
		if !g.delayTransform() {
			transformSlice(n)
		}
		return n

	case *syntax.Operation:
		if expr.Y == nil {
			n := Unary(pos, g.typ(typ), g.op(expr.Op, unOps[:]), g.expr(expr.X))
			if n.Op() == ir.OADDR && !g.delayTransform() {
				transformAddr(n.(*ir.AddrExpr))
			}
			return n
		}
		switch op := g.op(expr.Op, binOps[:]); op {
		case ir.OEQ, ir.ONE, ir.OLT, ir.OLE, ir.OGT, ir.OGE:
			n := Compare(pos, g.typ(typ), op, g.expr(expr.X), g.expr(expr.Y))
			if !g.delayTransform() {
				transformCompare(n)
			}
			return n
		case ir.OANDAND, ir.OOROR:
			x := g.expr(expr.X)
			y := g.expr(expr.Y)
			return typed(x.Type(), ir.NewLogicalExpr(pos, op, x, y))
		default:
			n := Binary(pos, op, g.typ(typ), g.expr(expr.X), g.expr(expr.Y))
			if op == ir.OADD && !g.delayTransform() {
				return transformAdd(n)
			}
			return n
		}

	default:
		g.unhandled("expression", expr)
		panic("unreachable")
	}
}

// substType does a normal type substition, but tparams is in the form of a field
// list, and targs is in terms of a slice of type nodes. substType records any newly
// instantiated types into g.instTypeList.
func (g *irgen) substType(typ *types.Type, tparams *types.Type, targs []ir.Node) *types.Type {
	fields := tparams.FieldSlice()
	tparams1 := make([]*types.Type, len(fields))
	for i, f := range fields {
		tparams1[i] = f.Type
	}
	targs1 := make([]*types.Type, len(targs))
	for i, n := range targs {
		targs1[i] = n.Type()
	}
	ts := typecheck.Tsubster{
		Tparams: tparams1,
		Targs:   targs1,
	}
	newt := ts.Typ(typ)
	return newt
}

// callExpr creates a call expression (which might be a type conversion, built-in
// call, or a regular call) and does standard transforms, unless we are in a generic
// function.
func (g *irgen) callExpr(pos src.XPos, typ *types.Type, fun ir.Node, args []ir.Node, dots bool) ir.Node {
	n := ir.NewCallExpr(pos, ir.OCALL, fun, args)
	n.IsDDD = dots
	typed(typ, n)

	if fun.Op() == ir.OTYPE {
		// Actually a type conversion, not a function call.
		if !g.delayTransform() {
			return transformConvCall(n)
		}
		return n
	}

	if fun, ok := fun.(*ir.Name); ok && fun.BuiltinOp != 0 {
		if !g.delayTransform() {
			return transformBuiltin(n)
		}
		return n
	}

	// Add information, now that we know that fun is actually being called.
	switch fun := fun.(type) {
	case *ir.SelectorExpr:
		if fun.Op() == ir.OMETHVALUE {
			op := ir.ODOTMETH
			if fun.X.Type().IsInterface() {
				op = ir.ODOTINTER
			}
			fun.SetOp(op)
			// Set the type to include the receiver, since that's what
			// later parts of the compiler expect
			fun.SetType(fun.Selection.Type)
		}
	}

	// A function instantiation (even if fully concrete) shouldn't be
	// transformed yet, because we need to add the dictionary during the
	// transformation.
	if fun.Op() != ir.OFUNCINST && !g.delayTransform() {
		transformCall(n)
	}
	return n
}

// selectorExpr resolves the choice of ODOT, ODOTPTR, OMETHVALUE (eventually
// ODOTMETH & ODOTINTER), and OMETHEXPR and deals with embedded fields here rather
// than in typecheck.go.
func (g *irgen) selectorExpr(pos src.XPos, typ types2.Type, expr *syntax.SelectorExpr) ir.Node {
	x := g.expr(expr.X)
	if x.Type().HasTParam() {
		// Leave a method call on a type param as an OXDOT, since it can
		// only be fully transformed once it has an instantiated type.
		n := ir.NewSelectorExpr(pos, ir.OXDOT, x, typecheck.Lookup(expr.Sel.Value))
		typed(g.typ(typ), n)
		return n
	}

	selinfo := g.info.Selections[expr]
	// Everything up to the last selection is an implicit embedded field access,
	// and the last selection is determined by selinfo.Kind().
	index := selinfo.Index()
	embeds, last := index[:len(index)-1], index[len(index)-1]

	origx := x
	for _, ix := range embeds {
		x = Implicit(DotField(pos, x, ix))
	}

	kind := selinfo.Kind()
	if kind == types2.FieldVal {
		return DotField(pos, x, last)
	}

	var n ir.Node
	method2 := selinfo.Obj().(*types2.Func)

	if kind == types2.MethodExpr {
		// OMETHEXPR is unusual in using directly the node and type of the
		// original OTYPE node (origx) before passing through embedded
		// fields, even though the method is selected from the type
		// (x.Type()) reached after following the embedded fields. We will
		// actually drop any ODOT nodes we created due to the embedded
		// fields.
		n = MethodExpr(pos, origx, x.Type(), last)
	} else {
		// Add implicit addr/deref for method values, if needed.
		if x.Type().IsInterface() {
			n = DotMethod(pos, x, last)
		} else {
			recvType2 := method2.Type().(*types2.Signature).Recv().Type()
			_, wantPtr := recvType2.(*types2.Pointer)
			havePtr := x.Type().IsPtr()

			if havePtr != wantPtr {
				if havePtr {
					x = Implicit(Deref(pos, x.Type().Elem(), x))
				} else {
					x = Implicit(Addr(pos, x))
				}
			}
			recvType2Base := recvType2
			if wantPtr {
				recvType2Base = types2.AsPointer(recvType2).Elem()
			}
			if recvType2Base.(*types2.Named).TypeParams().Len() > 0 {
				// recvType2 is the original generic type that is
				// instantiated for this method call.
				// selinfo.Recv() is the instantiated type
				recvType2 = recvType2Base
				recvTypeSym := g.pkg(method2.Pkg()).Lookup(recvType2.(*types2.Named).Obj().Name())
				recvType := recvTypeSym.Def.(*ir.Name).Type()
				// method is the generic method associated with
				// the base generic type. The instantiated type may not
				// have method bodies filled in, if it was imported.
				method := recvType.Methods().Index(last).Nname.(*ir.Name)
				n = ir.NewSelectorExpr(pos, ir.OMETHVALUE, x, typecheck.Lookup(expr.Sel.Value))
				n.(*ir.SelectorExpr).Selection = types.NewField(pos, method.Sym(), method.Type())
				n.(*ir.SelectorExpr).Selection.Nname = method
				typed(method.Type(), n)

				xt := deref(x.Type())
				targs := make([]ir.Node, len(xt.RParams()))
				for i := range targs {
					targs[i] = ir.TypeNode(xt.RParams()[i])
				}

				// Create function instantiation with the type
				// args for the receiver type for the method call.
				n = ir.NewInstExpr(pos, ir.OFUNCINST, n, targs)
				typed(g.typ(typ), n)
				return n
			}

			if !g.match(x.Type(), recvType2, false) {
				base.FatalfAt(pos, "expected %L to have type %v", x, recvType2)
			} else {
				n = DotMethod(pos, x, last)
			}
		}
	}
	if have, want := n.Sym(), g.selector(method2); have != want {
		base.FatalfAt(pos, "bad Sym: have %v, want %v", have, want)
	}
	return n
}

func (g *irgen) exprList(expr syntax.Expr) []ir.Node {
	return g.exprs(unpackListExpr(expr))
}

func unpackListExpr(expr syntax.Expr) []syntax.Expr {
	switch expr := expr.(type) {
	case nil:
		return nil
	case *syntax.ListExpr:
		return expr.ElemList
	default:
		return []syntax.Expr{expr}
	}
}

func (g *irgen) exprs(exprs []syntax.Expr) []ir.Node {
	nodes := make([]ir.Node, len(exprs))
	for i, expr := range exprs {
		nodes[i] = g.expr(expr)
	}
	return nodes
}

func (g *irgen) compLit(typ types2.Type, lit *syntax.CompositeLit) ir.Node {
	if ptr, ok := types2.CoreType(typ).(*types2.Pointer); ok {
		n := ir.NewAddrExpr(g.pos(lit), g.compLit(ptr.Elem(), lit))
		n.SetOp(ir.OPTRLIT)
		return typed(g.typ(typ), n)
	}

	_, isStruct := types2.CoreType(typ).(*types2.Struct)

	exprs := make([]ir.Node, len(lit.ElemList))
	for i, elem := range lit.ElemList {
		switch elem := elem.(type) {
		case *syntax.KeyValueExpr:
			var key ir.Node
			if isStruct {
				key = ir.NewIdent(g.pos(elem.Key), g.name(elem.Key.(*syntax.Name)))
			} else {
				key = g.expr(elem.Key)
			}
			value := wrapname(g.pos(elem.Value), g.expr(elem.Value))
			if value.Op() == ir.OPAREN {
				// Make sure any PAREN node added by wrapper has a type
				typed(value.(*ir.ParenExpr).X.Type(), value)
			}
			exprs[i] = ir.NewKeyExpr(g.pos(elem), key, value)
		default:
			exprs[i] = wrapname(g.pos(elem), g.expr(elem))
			if exprs[i].Op() == ir.OPAREN {
				// Make sure any PAREN node added by wrapper has a type
				typed(exprs[i].(*ir.ParenExpr).X.Type(), exprs[i])
			}
		}
	}

	n := ir.NewCompLitExpr(g.pos(lit), ir.OCOMPLIT, nil, exprs)
	typed(g.typ(typ), n)
	var r ir.Node = n
	if !g.delayTransform() {
		r = transformCompLit(n)
	}
	return r
}

func (g *irgen) funcLit(typ2 types2.Type, expr *syntax.FuncLit) ir.Node {
	fn := ir.NewClosureFunc(g.pos(expr), ir.CurFunc != nil)
	ir.NameClosure(fn.OClosure, ir.CurFunc)

	typ := g.typ(typ2)
	typed(typ, fn.Nname)
	typed(typ, fn.OClosure)
	fn.SetTypecheck(1)

	g.funcBody(fn, nil, expr.Type, expr.Body)

	ir.FinishCaptureNames(fn.Pos(), ir.CurFunc, fn)

	// TODO(mdempsky): ir.CaptureName should probably handle
	// copying these fields from the canonical variable.
	for _, cv := range fn.ClosureVars {
		cv.SetType(cv.Canonical().Type())
		cv.SetTypecheck(1)
		cv.SetWalkdef(1)
	}

	if g.topFuncIsGeneric {
		// Don't add any closure inside a generic function/method to the
		// g.target.Decls list, even though it may not be generic itself.
		// See issue #47514.
		return ir.UseClosure(fn.OClosure, nil)
	} else {
		return ir.UseClosure(fn.OClosure, g.target)
	}
}

func (g *irgen) typeExpr(typ syntax.Expr) *types.Type {
	n := g.expr(typ)
	if n.Op() != ir.OTYPE {
		base.FatalfAt(g.pos(typ), "expected type: %L", n)
	}
	return n.Type()
}

// constExprOp returns an ir.Op that represents the outermost
// operation of the given constant expression. It's intended for use
// with ir.RawOrigExpr.
func constExprOp(expr syntax.Expr) ir.Op {
	switch expr := expr.(type) {
	default:
		panic(fmt.Sprintf("%s: unexpected expression: %T", expr.Pos(), expr))

	case *syntax.BasicLit:
		return ir.OLITERAL
	case *syntax.Name, *syntax.SelectorExpr:
		return ir.ONAME
	case *syntax.CallExpr:
		return ir.OCALL
	case *syntax.Operation:
		if expr.Y == nil {
			return unOps[expr.Op]
		}
		return binOps[expr.Op]
	}
}

func unparen(expr syntax.Expr) syntax.Expr {
	for {
		paren, ok := expr.(*syntax.ParenExpr)
		if !ok {
			return expr
		}
		expr = paren.X
	}
}
