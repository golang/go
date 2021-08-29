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

		// The key for the Inferred map is the CallExpr (if inferring
		// types required the function arguments) or the IndexExpr below
		// (if types could be inferred without the function arguments).
		if inferred, ok := g.info.Inferred[expr]; ok && inferred.TArgs.Len() > 0 {
			// This is the case where inferring types required the
			// types of the function arguments.
			targs := make([]ir.Node, inferred.TArgs.Len())
			for i := range targs {
				targs[i] = ir.TypeNode(g.typ(inferred.TArgs.At(i)))
			}
			if fun.Op() == ir.OFUNCINST {
				if len(fun.(*ir.InstExpr).Targs) < len(targs) {
					// Replace explicit type args with the full list that
					// includes the additional inferred type args.
					// Substitute the type args for the type params in
					// the generic function's type.
					fun.(*ir.InstExpr).Targs = targs
					newt := g.substType(fun.(*ir.InstExpr).X.Type(), fun.(*ir.InstExpr).X.Type().TParams(), targs)
					typed(newt, fun)
				}
			} else {
				// Create a function instantiation here, given there
				// are only inferred type args (e.g. min(5,6), where
				// min is a generic function). Substitute the type
				// args for the type params in the generic function's
				// type.
				inst := ir.NewInstExpr(pos, ir.OFUNCINST, fun, targs)
				newt := g.substType(fun.Type(), fun.Type().TParams(), targs)
				typed(newt, inst)
				fun = inst
			}

		}
		return Call(pos, g.typ(typ), fun, g.exprs(expr.ArgList), expr.HasDots)

	case *syntax.IndexExpr:
		var targs []ir.Node

		if inferred, ok := g.info.Inferred[expr]; ok && inferred.TArgs.Len() > 0 {
			// This is the partial type inference case where the types
			// can be inferred from other type arguments without using
			// the types of the function arguments.
			targs = make([]ir.Node, inferred.TArgs.Len())
			for i := range targs {
				targs[i] = ir.TypeNode(g.typ(inferred.TArgs.At(i)))
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
		newt := g.typ(typ)
		// Substitute the type args for the type params in the uninstantiated
		// function's type. If there aren't enough type args, then the rest
		// will be inferred at the call node, so don't try the substitution yet.
		if x.Type().TParams().NumFields() == len(targs) {
			newt = g.substType(g.typ(typ), x.Type().TParams(), targs)
		}
		typed(newt, n)
		return n

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
	g.instTypeList = append(g.instTypeList, ts.InstTypeList...)
	return newt
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

		// Fill in n.Selection for a generic method reference or a bound
		// interface method, even though we won't use it directly, since it
		// is useful for analysis. Specifically do not fill in for fields or
		// other interfaces methods (method call on an interface value), so
		// n.Selection being non-nil means a method reference for a generic
		// type or a method reference due to a bound.
		obj2 := g.info.Selections[expr].Obj()
		sig := types2.AsSignature(obj2.Type())
		if sig == nil || sig.Recv() == nil {
			return n
		}
		index := g.info.Selections[expr].Index()
		last := index[len(index)-1]
		// recvType is the receiver of the method being called.  Because of the
		// way methods are imported, g.obj(obj2) doesn't work across
		// packages, so we have to lookup the method via the receiver type.
		recvType := deref2(sig.Recv().Type())
		if types2.AsInterface(recvType.Underlying()) != nil {
			fieldType := n.X.Type()
			for _, ix := range index[:len(index)-1] {
				fieldType = deref(fieldType).Field(ix).Type
			}
			if fieldType.Kind() == types.TTYPEPARAM {
				n.Selection = fieldType.Bound().AllMethods().Index(last)
				//fmt.Printf(">>>>> %v: Bound call %v\n", base.FmtPos(pos), n.Sel)
			} else {
				assert(fieldType.Kind() == types.TINTER)
				//fmt.Printf(">>>>> %v: Interface call %v\n", base.FmtPos(pos), n.Sel)
			}
			return n
		}

		recvObj := types2.AsNamed(recvType).Obj()
		recv := g.pkg(recvObj.Pkg()).Lookup(recvObj.Name()).Def
		n.Selection = recv.Type().Methods().Index(last)
		//fmt.Printf(">>>>> %v: Method call %v\n", base.FmtPos(pos), n.Sel)

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
			if types2.AsNamed(recvType2Base).TParams().Len() > 0 {
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

				// selinfo.Targs() are the types used to
				// instantiate the type of receiver
				targs2 := getTargs(selinfo)
				targs := make([]ir.Node, targs2.Len())
				for i := range targs {
					targs[i] = ir.TypeNode(g.typ(targs2.At(i)))
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
func getTargs(selinfo *types2.Selection) *types2.TypeList {
	r := deref2(selinfo.Recv())
	n := types2.AsNamed(r)
	if n == nil {
		base.Fatalf("Incorrect type for selinfo %v", selinfo)
	}
	return n.TArgs()
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
			var key ir.Node
			if isStruct {
				key = ir.NewIdent(g.pos(elem.Key), g.name(elem.Key.(*syntax.Name)))
			} else {
				key = g.expr(elem.Key)
			}
			exprs[i] = ir.NewKeyExpr(g.pos(elem), key, g.expr(elem.Value))
		default:
			exprs[i] = g.expr(elem)
		}
	}

	n := ir.NewCompLitExpr(g.pos(lit), ir.OCOMPLIT, nil, exprs)
	typed(g.typ(typ), n)
	return transformCompLit(n)
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
