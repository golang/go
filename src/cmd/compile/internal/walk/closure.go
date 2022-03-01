// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package walk

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

// directClosureCall rewrites a direct call of a function literal into
// a normal function call with closure variables passed as arguments.
// This avoids allocation of a closure object.
//
// For illustration, the following call:
//
//	func(a int) {
//		println(byval)
//		byref++
//	}(42)
//
// becomes:
//
//	func(byval int, &byref *int, a int) {
//		println(byval)
//		(*&byref)++
//	}(byval, &byref, 42)
func directClosureCall(n *ir.CallExpr) {
	clo := n.X.(*ir.ClosureExpr)
	clofn := clo.Func

	if ir.IsTrivialClosure(clo) {
		return // leave for walkClosure to handle
	}

	// We are going to insert captured variables before input args.
	var params []*types.Field
	var decls []*ir.Name
	for _, v := range clofn.ClosureVars {
		if !v.Byval() {
			// If v of type T is captured by reference,
			// we introduce function param &v *T
			// and v remains PAUTOHEAP with &v heapaddr
			// (accesses will implicitly deref &v).

			addr := ir.NewNameAt(clofn.Pos(), typecheck.Lookup("&"+v.Sym().Name))
			addr.Curfn = clofn
			addr.SetType(types.NewPtr(v.Type()))
			v.Heapaddr = addr
			v = addr
		}

		v.Class = ir.PPARAM
		decls = append(decls, v)

		fld := types.NewField(src.NoXPos, v.Sym(), v.Type())
		fld.Nname = v
		params = append(params, fld)
	}

	// f is ONAME of the actual function.
	f := clofn.Nname
	typ := f.Type()

	// Create new function type with parameters prepended, and
	// then update type and declarations.
	typ = types.NewSignature(typ.Pkg(), nil, nil, append(params, typ.Params().FieldSlice()...), typ.Results().FieldSlice())
	f.SetType(typ)
	clofn.Dcl = append(decls, clofn.Dcl...)

	// Rewrite call.
	n.X = f
	n.Args.Prepend(closureArgs(clo)...)

	// Update the call expression's type. We need to do this
	// because typecheck gave it the result type of the OCLOSURE
	// node, but we only rewrote the ONAME node's type. Logically,
	// they're the same, but the stack offsets probably changed.
	if typ.NumResults() == 1 {
		n.SetType(typ.Results().Field(0).Type)
	} else {
		n.SetType(typ.Results())
	}

	// Add to Closures for enqueueFunc. It's no longer a proper
	// closure, but we may have already skipped over it in the
	// functions list as a non-trivial closure, so this just
	// ensures it's compiled.
	ir.CurFunc.Closures = append(ir.CurFunc.Closures, clofn)
}

func walkClosure(clo *ir.ClosureExpr, init *ir.Nodes) ir.Node {
	clofn := clo.Func

	// If no closure vars, don't bother wrapping.
	if ir.IsTrivialClosure(clo) {
		if base.Debug.Closure > 0 {
			base.WarnfAt(clo.Pos(), "closure converted to global")
		}
		return clofn.Nname
	}

	// The closure is not trivial or directly called, so it's going to stay a closure.
	ir.ClosureDebugRuntimeCheck(clo)
	clofn.SetNeedctxt(true)

	// The closure expression may be walked more than once if it appeared in composite
	// literal initialization (e.g, see issue #49029).
	//
	// Don't add the closure function to compilation queue more than once, since when
	// compiling a function twice would lead to an ICE.
	if !clofn.Walked() {
		clofn.SetWalked(true)
		ir.CurFunc.Closures = append(ir.CurFunc.Closures, clofn)
	}

	typ := typecheck.ClosureType(clo)

	clos := ir.NewCompLitExpr(base.Pos, ir.OCOMPLIT, ir.TypeNode(typ), nil)
	clos.SetEsc(clo.Esc())
	clos.List = append([]ir.Node{ir.NewUnaryExpr(base.Pos, ir.OCFUNC, clofn.Nname)}, closureArgs(clo)...)
	for i, value := range clos.List {
		clos.List[i] = ir.NewStructKeyExpr(base.Pos, typ.Field(i), value)
	}

	addr := typecheck.NodAddr(clos)
	addr.SetEsc(clo.Esc())

	// Force type conversion from *struct to the func type.
	cfn := typecheck.ConvNop(addr, clo.Type())

	// non-escaping temp to use, if any.
	if x := clo.Prealloc; x != nil {
		if !types.Identical(typ, x.Type()) {
			panic("closure type does not match order's assigned type")
		}
		addr.Prealloc = x
		clo.Prealloc = nil
	}

	return walkExpr(cfn, init)
}

// closureArgs returns a slice of expressions that an be used to
// initialize the given closure's free variables. These correspond
// one-to-one with the variables in clo.Func.ClosureVars, and will be
// either an ONAME node (if the variable is captured by value) or an
// OADDR-of-ONAME node (if not).
func closureArgs(clo *ir.ClosureExpr) []ir.Node {
	fn := clo.Func

	args := make([]ir.Node, len(fn.ClosureVars))
	for i, v := range fn.ClosureVars {
		var outer ir.Node
		outer = v.Outer
		if !v.Byval() {
			outer = typecheck.NodAddrAt(fn.Pos(), outer)
		}
		args[i] = typecheck.Expr(outer)
	}
	return args
}

func walkMethodValue(n *ir.SelectorExpr, init *ir.Nodes) ir.Node {
	// Create closure in the form of a composite literal.
	// For x.M with receiver (x) type T, the generated code looks like:
	//
	//	clos = &struct{F uintptr; R T}{T.M·f, x}
	//
	// Like walkClosure above.

	if n.X.Type().IsInterface() {
		// Trigger panic for method on nil interface now.
		// Otherwise it happens in the wrapper and is confusing.
		n.X = cheapExpr(n.X, init)
		n.X = walkExpr(n.X, nil)

		tab := ir.NewUnaryExpr(base.Pos, ir.OITAB, n.X)
		check := ir.NewUnaryExpr(base.Pos, ir.OCHECKNIL, tab)
		init.Append(typecheck.Stmt(check))
	}

	typ := typecheck.MethodValueType(n)

	clos := ir.NewCompLitExpr(base.Pos, ir.OCOMPLIT, ir.TypeNode(typ), nil)
	clos.SetEsc(n.Esc())
	clos.List = []ir.Node{ir.NewUnaryExpr(base.Pos, ir.OCFUNC, methodValueWrapper(n)), n.X}

	addr := typecheck.NodAddr(clos)
	addr.SetEsc(n.Esc())

	// Force type conversion from *struct to the func type.
	cfn := typecheck.ConvNop(addr, n.Type())

	// non-escaping temp to use, if any.
	if x := n.Prealloc; x != nil {
		if !types.Identical(typ, x.Type()) {
			panic("partial call type does not match order's assigned type")
		}
		addr.Prealloc = x
		n.Prealloc = nil
	}

	return walkExpr(cfn, init)
}

// methodValueWrapper returns the ONAME node representing the
// wrapper function (*-fm) needed for the given method value. If the
// wrapper function hasn't already been created yet, it's created and
// added to typecheck.Target.Decls.
func methodValueWrapper(dot *ir.SelectorExpr) *ir.Name {
	if dot.Op() != ir.OMETHVALUE {
		base.Fatalf("methodValueWrapper: unexpected %v (%v)", dot, dot.Op())
	}

	t0 := dot.Type()
	meth := dot.Sel
	rcvrtype := dot.X.Type()
	sym := ir.MethodSymSuffix(rcvrtype, meth, "-fm")

	if sym.Uniq() {
		return sym.Def.(*ir.Name)
	}
	sym.SetUniq(true)

	if base.Debug.Unified != 0 && base.Debug.UnifiedQuirks == 0 {
		base.FatalfAt(dot.Pos(), "missing wrapper for %v", meth)
	}

	savecurfn := ir.CurFunc
	saveLineNo := base.Pos
	ir.CurFunc = nil

	base.Pos = base.AutogeneratedPos

	tfn := ir.NewFuncType(base.Pos, nil,
		typecheck.NewFuncParams(t0.Params(), true),
		typecheck.NewFuncParams(t0.Results(), false))

	fn := typecheck.DeclFunc(sym, tfn)
	fn.SetDupok(true)
	fn.SetWrapper(true)

	// Declare and initialize variable holding receiver.
	ptr := ir.NewHiddenParam(base.Pos, fn, typecheck.Lookup(".this"), rcvrtype)

	call := ir.NewCallExpr(base.Pos, ir.OCALL, ir.NewSelectorExpr(base.Pos, ir.OXDOT, ptr, meth), nil)
	call.Args = ir.ParamNames(tfn.Type())
	call.IsDDD = tfn.Type().IsVariadic()

	var body ir.Node = call
	if t0.NumResults() != 0 {
		ret := ir.NewReturnStmt(base.Pos, nil)
		ret.Results = []ir.Node{call}
		body = ret
	}

	fn.Body = []ir.Node{body}
	typecheck.FinishFuncBody()

	typecheck.Func(fn)
	// Need to typecheck the body of the just-generated wrapper.
	// typecheckslice() requires that Curfn is set when processing an ORETURN.
	ir.CurFunc = fn
	typecheck.Stmts(fn.Body)
	sym.Def = fn.Nname
	typecheck.Target.Decls = append(typecheck.Target.Decls, fn)
	ir.CurFunc = savecurfn
	base.Pos = saveLineNo

	return fn.Nname
}
