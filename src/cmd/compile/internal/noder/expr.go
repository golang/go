// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/compile/internal/types2"
	"cmd/internal/src"
)

func (g *irgen) expr(expr syntax.Expr) ir.Node {
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
		return Const(g.pos(expr), g.typ(typ), tv.Value)
	}

	n := g.expr0(typ, expr)
	if n.Typecheck() != 1 && n.Typecheck() != 3 {
		base.FatalfAt(g.pos(expr), "missed typecheck: %+v", n)
	}
	if !g.match(n.Type(), typ, tv.HasOk()) {
		base.FatalfAt(g.pos(expr), "expected %L to have type %v", n, typ)
	}
	return n
}

func (g *irgen) expr0(typ types2.Type, expr syntax.Expr) ir.Node {
	pos := g.pos(expr)

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

		// The key for the Inferred map is the CallExpr (if inferring
		// types required the function arguments) or the IndexExpr below
		// (if types could be inferred without the function arguments).
		if inferred, ok := g.info.Inferred[expr]; ok && len(inferred.Targs) > 0 {
			// This is the case where inferring types required the
			// types of the function arguments.
			targs := make([]ir.Node, len(inferred.Targs))
			for i, targ := range inferred.Targs {
				targs[i] = ir.TypeNode(g.typ(targ))
			}
			if fun.Op() == ir.OFUNCINST {
				// Replace explicit type args with the full list that
				// includes the additional inferred type args
				fun.(*ir.InstExpr).Targs = targs
			} else {
				// Create a function instantiation here, given
				// there are only inferred type args (e.g.
				// min(5,6), where min is a generic function)
				inst := ir.NewInstExpr(pos, ir.OFUNCINST, fun, targs)
				typed(fun.Type(), inst)
				fun = inst
			}

		}
		return Call(pos, g.typ(typ), fun, g.exprs(expr.ArgList), expr.HasDots)

	case *syntax.IndexExpr:
		var targs []ir.Node

		if inferred, ok := g.info.Inferred[expr]; ok && len(inferred.Targs) > 0 {
			// This is the partial type inference case where the types
			// can be inferred from other type arguments without using
			// the types of the function arguments.
			targs = make([]ir.Node, len(inferred.Targs))
			for i, targ := range inferred.Targs {
				targs[i] = ir.TypeNode(g.typ(targ))
			}
		} else if _, ok := expr.Index.(*syntax.ListExpr); ok {
			targs = g.exprList(expr.Index)
		} else {
			index := g.expr(expr.Index)
			if index.Op() != ir.OTYPE {
				// This is just a normal index expression
				return Index(pos, g.typ(typ), g.expr(expr.X), index)
			}
			// This is generic function instantiation with a single type
			targs = []ir.Node{index}
		}
		// This is a generic function instantiation (e.g. min[int]).
		// Generic type instantiation is handled in the type
		// section of expr() above (using g.typ).
		x := g.expr(expr.X)
		if x.Op() != ir.ONAME || x.Type().Kind() != types.TFUNC {
			panic("Incorrect argument for generic func instantiation")
		}
		n := ir.NewInstExpr(pos, ir.OFUNCINST, x, targs)
		typed(g.typ(typ), n)
		return n

	case *syntax.ParenExpr:
		return g.expr(expr.X) // skip parens; unneeded after parse+typecheck

	case *syntax.SelectorExpr:
		// Qualified identifier.
		if name, ok := expr.X.(*syntax.Name); ok {
			if _, ok := g.info.Uses[name].(*types2.PkgName); ok {
				return g.use(expr.Sel)
			}
		}
		return g.selectorExpr(pos, typ, expr)

	case *syntax.SliceExpr:
		return Slice(pos, g.typ(typ), g.expr(expr.X), g.expr(expr.Index[0]), g.expr(expr.Index[1]), g.expr(expr.Index[2]))

	case *syntax.Operation:
		if expr.Y == nil {
			return Unary(pos, g.typ(typ), g.op(expr.Op, unOps[:]), g.expr(expr.X))
		}
		switch op := g.op(expr.Op, binOps[:]); op {
		case ir.OEQ, ir.ONE, ir.OLT, ir.OLE, ir.OGT, ir.OGE:
			return Compare(pos, g.typ(typ), op, g.expr(expr.X), g.expr(expr.Y))
		default:
			return Binary(pos, op, g.typ(typ), g.expr(expr.X), g.expr(expr.Y))
		}

	default:
		g.unhandled("expression", expr)
		panic("unreachable")
	}
}

// selectorExpr resolves the choice of ODOT, ODOTPTR, OCALLPART (eventually
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

	// TODO(danscales,mdempsky): Interface method sets are not sorted the
	// same between types and types2. In particular, using "last" here
	// without conversion will likely fail if an interface contains
	// unexported methods from two different packages (due to cross-package
	// interface embedding).

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
			if len(types2.AsNamed(recvType2Base).TParams()) > 0 {
				// recvType2 is the original generic type that is
				// instantiated for this method call.
				// selinfo.Recv() is the instantiated type
				recvType2 = recvType2Base
				// method is the generic method associated with the gen type
				method := g.obj(types2.AsNamed(recvType2).Method(last))
				n = ir.NewSelectorExpr(pos, ir.OCALLPART, x, method.Sym())
				n.(*ir.SelectorExpr).Selection = types.NewField(pos, method.Sym(), method.Type())
				n.(*ir.SelectorExpr).Selection.Nname = method
				typed(method.Type(), n)

				// selinfo.Targs() are the types used to
				// instantiate the type of receiver
				targs2 := getTargs(selinfo)
				targs := make([]ir.Node, len(targs2))
				for i, targ2 := range targs2 {
					targs[i] = ir.TypeNode(g.typ(targ2))
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

// getTargs gets the targs associated with the receiver of a selected method
func getTargs(selinfo *types2.Selection) []types2.Type {
	r := selinfo.Recv()
	if p := types2.AsPointer(r); p != nil {
		r = p.Elem()
	}
	n := types2.AsNamed(r)
	if n == nil {
		base.Fatalf("Incorrect type for selinfo %v", selinfo)
	}
	return n.TArgs()
}

func (g *irgen) exprList(expr syntax.Expr) []ir.Node {
	switch expr := expr.(type) {
	case nil:
		return nil
	case *syntax.ListExpr:
		return g.exprs(expr.ElemList)
	default:
		return []ir.Node{g.expr(expr)}
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
	if ptr, ok := typ.Underlying().(*types2.Pointer); ok {
		n := ir.NewAddrExpr(g.pos(lit), g.compLit(ptr.Elem(), lit))
		n.SetOp(ir.OPTRLIT)
		return typed(g.typ(typ), n)
	}

	_, isStruct := typ.Underlying().(*types2.Struct)

	exprs := make([]ir.Node, len(lit.ElemList))
	for i, elem := range lit.ElemList {
		switch elem := elem.(type) {
		case *syntax.KeyValueExpr:
			if isStruct {
				exprs[i] = ir.NewStructKeyExpr(g.pos(elem), g.name(elem.Key.(*syntax.Name)), g.expr(elem.Value))
			} else {
				exprs[i] = ir.NewKeyExpr(g.pos(elem), g.expr(elem.Key), g.expr(elem.Value))
			}
		default:
			exprs[i] = g.expr(elem)
		}
	}

	n := ir.NewCompLitExpr(g.pos(lit), ir.OCOMPLIT, nil, exprs)
	typed(g.typ(typ), n)
	return transformCompLit(n)
}

func (g *irgen) funcLit(typ2 types2.Type, expr *syntax.FuncLit) ir.Node {
	fn := ir.NewFunc(g.pos(expr))
	fn.SetIsHiddenClosure(ir.CurFunc != nil)

	fn.Nname = ir.NewNameAt(g.pos(expr), typecheck.ClosureName(ir.CurFunc))
	ir.MarkFunc(fn.Nname)
	typ := g.typ(typ2)
	fn.Nname.Func = fn
	fn.Nname.Defn = fn
	typed(typ, fn.Nname)
	fn.SetTypecheck(1)

	fn.OClosure = ir.NewClosureExpr(g.pos(expr), fn)
	typed(typ, fn.OClosure)

	g.funcBody(fn, nil, expr.Type, expr.Body)

	ir.FinishCaptureNames(fn.Pos(), ir.CurFunc, fn)

	// TODO(mdempsky): ir.CaptureName should probably handle
	// copying these fields from the canonical variable.
	for _, cv := range fn.ClosureVars {
		cv.SetType(cv.Canonical().Type())
		cv.SetTypecheck(1)
		cv.SetWalkdef(1)
	}

	g.target.Decls = append(g.target.Decls, fn)

	return fn.OClosure
}

func (g *irgen) typeExpr(typ syntax.Expr) *types.Type {
	n := g.expr(typ)
	if n.Op() != ir.OTYPE {
		base.FatalfAt(g.pos(typ), "expected type: %L", n)
	}
	return n.Type()
}
