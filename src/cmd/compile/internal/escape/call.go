// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

// call evaluates a call expressions, including builtin calls. ks
// should contain the holes representing where the function callee's
// results flows.
func (e *escape) call(ks []hole, call ir.Node) {
	var init ir.Nodes
	e.callCommon(ks, call, &init, nil)
	if len(init) != 0 {
		call.(*ir.CallExpr).PtrInit().Append(init...)
	}
}

func (e *escape) callCommon(ks []hole, call ir.Node, init *ir.Nodes, wrapper *ir.Func) {

	// argumentPragma handles escape analysis of argument *argp to the
	// given hole. If the function callee is known, pragma is the
	// function's pragma flags; otherwise 0.
	argumentFunc := func(fn *ir.Name, k hole, argp *ir.Node) {
		e.rewriteArgument(argp, init, call, fn, wrapper)

		e.expr(k.note(call, "call parameter"), *argp)
	}

	argument := func(k hole, argp *ir.Node) {
		argumentFunc(nil, k, argp)
	}

	switch call.Op() {
	default:
		ir.Dump("esc", call)
		base.Fatalf("unexpected call op: %v", call.Op())

	case ir.OCALLFUNC, ir.OCALLMETH, ir.OCALLINTER:
		call := call.(*ir.CallExpr)
		typecheck.FixVariadicCall(call)
		typecheck.FixMethodCall(call)

		// Pick out the function callee, if statically known.
		//
		// TODO(mdempsky): Change fn from *ir.Name to *ir.Func, but some
		// functions (e.g., runtime builtins, method wrappers, generated
		// eq/hash functions) don't have it set. Investigate whether
		// that's a concern.
		var fn *ir.Name
		switch call.Op() {
		case ir.OCALLFUNC:
			// If we have a direct call to a closure (not just one we were
			// able to statically resolve with ir.StaticValue), mark it as
			// such so batch.outlives can optimize the flow results.
			if call.X.Op() == ir.OCLOSURE {
				call.X.(*ir.ClosureExpr).Func.SetClosureCalled(true)
			}

			switch v := ir.StaticValue(call.X); v.Op() {
			case ir.ONAME:
				if v := v.(*ir.Name); v.Class == ir.PFUNC {
					fn = v
				}
			case ir.OCLOSURE:
				fn = v.(*ir.ClosureExpr).Func.Nname
			case ir.OMETHEXPR:
				fn = ir.MethodExprName(v)
			}
		case ir.OCALLMETH:
			base.FatalfAt(call.Pos(), "OCALLMETH missed by typecheck")
		}

		fntype := call.X.Type()
		if fn != nil {
			fntype = fn.Type()
		}

		if ks != nil && fn != nil && e.inMutualBatch(fn) {
			for i, result := range fn.Type().Results().FieldSlice() {
				e.expr(ks[i], ir.AsNode(result.Nname))
			}
		}

		var recvp *ir.Node
		if call.Op() == ir.OCALLFUNC {
			// Evaluate callee function expression.
			//
			// Note: We use argument and not argumentFunc, because while
			// call.X here may be an argument to runtime.{new,defer}proc,
			// it's not an argument to fn itself.
			argument(e.discardHole(), &call.X)
		} else {
			recvp = &call.X.(*ir.SelectorExpr).X
		}

		args := call.Args
		if recv := fntype.Recv(); recv != nil {
			if recvp == nil {
				// Function call using method expression. Recevier argument is
				// at the front of the regular arguments list.
				recvp = &args[0]
				args = args[1:]
			}

			argumentFunc(fn, e.tagHole(ks, fn, recv), recvp)
		}

		for i, param := range fntype.Params().FieldSlice() {
			argumentFunc(fn, e.tagHole(ks, fn, param), &args[i])
		}

	case ir.OINLCALL:
		call := call.(*ir.InlinedCallExpr)
		e.stmts(call.Body)
		for i, result := range call.ReturnVars {
			k := e.discardHole()
			if ks != nil {
				k = ks[i]
			}
			e.expr(k, result)
		}

	case ir.OAPPEND:
		call := call.(*ir.CallExpr)
		args := call.Args

		// Appendee slice may flow directly to the result, if
		// it has enough capacity. Alternatively, a new heap
		// slice might be allocated, and all slice elements
		// might flow to heap.
		appendeeK := ks[0]
		if args[0].Type().Elem().HasPointers() {
			appendeeK = e.teeHole(appendeeK, e.heapHole().deref(call, "appendee slice"))
		}
		argument(appendeeK, &args[0])

		if call.IsDDD {
			appendedK := e.discardHole()
			if args[1].Type().IsSlice() && args[1].Type().Elem().HasPointers() {
				appendedK = e.heapHole().deref(call, "appended slice...")
			}
			argument(appendedK, &args[1])
		} else {
			for i := 1; i < len(args); i++ {
				argument(e.heapHole(), &args[i])
			}
		}

	case ir.OCOPY:
		call := call.(*ir.BinaryExpr)
		argument(e.discardHole(), &call.X)

		copiedK := e.discardHole()
		if call.Y.Type().IsSlice() && call.Y.Type().Elem().HasPointers() {
			copiedK = e.heapHole().deref(call, "copied slice")
		}
		argument(copiedK, &call.Y)

	case ir.OPANIC:
		call := call.(*ir.UnaryExpr)
		argument(e.heapHole(), &call.X)

	case ir.OCOMPLEX:
		call := call.(*ir.BinaryExpr)
		argument(e.discardHole(), &call.X)
		argument(e.discardHole(), &call.Y)

	case ir.ODELETE, ir.OPRINT, ir.OPRINTN, ir.ORECOVER:
		call := call.(*ir.CallExpr)
		fixRecoverCall(call)
		for i := range call.Args {
			argument(e.discardHole(), &call.Args[i])
		}

	case ir.OLEN, ir.OCAP, ir.OREAL, ir.OIMAG, ir.OCLOSE:
		call := call.(*ir.UnaryExpr)
		argument(e.discardHole(), &call.X)

	case ir.OUNSAFEADD, ir.OUNSAFESLICE:
		call := call.(*ir.BinaryExpr)
		argument(ks[0], &call.X)
		argument(e.discardHole(), &call.Y)
	}
}

// goDeferStmt analyzes a "go" or "defer" statement.
//
// In the process, it also normalizes the statement to always use a
// simple function call with no arguments and no results. For example,
// it rewrites:
//
//	defer f(x, y)
//
// into:
//
//	x1, y1 := x, y
//	defer func() { f(x1, y1) }()
func (e *escape) goDeferStmt(n *ir.GoDeferStmt) {
	k := e.heapHole()
	if n.Op() == ir.ODEFER && e.loopDepth == 1 {
		// Top-level defer arguments don't escape to the heap,
		// but they do need to last until they're invoked.
		k = e.later(e.discardHole())

		// force stack allocation of defer record, unless
		// open-coded defers are used (see ssa.go)
		n.SetEsc(ir.EscNever)
	}

	call := n.Call

	init := n.PtrInit()
	init.Append(ir.TakeInit(call)...)
	e.stmts(*init)

	// If the function is already a zero argument/result function call,
	// just escape analyze it normally.
	if call, ok := call.(*ir.CallExpr); ok && call.Op() == ir.OCALLFUNC {
		if sig := call.X.Type(); sig.NumParams()+sig.NumResults() == 0 {
			if clo, ok := call.X.(*ir.ClosureExpr); ok && n.Op() == ir.OGO {
				clo.IsGoWrap = true
			}
			e.expr(k, call.X)
			return
		}
	}

	// Create a new no-argument function that we'll hand off to defer.
	fn := ir.NewClosureFunc(n.Pos(), true)
	fn.SetWrapper(true)
	fn.Nname.SetType(types.NewSignature(types.LocalPkg, nil, nil, nil, nil))
	fn.Body = []ir.Node{call}
	if call, ok := call.(*ir.CallExpr); ok && call.Op() == ir.OCALLFUNC {
		// If the callee is a named function, link to the original callee.
		x := call.X
		if x.Op() == ir.ONAME && x.(*ir.Name).Class == ir.PFUNC {
			fn.WrappedFunc = call.X.(*ir.Name).Func
		} else if x.Op() == ir.OMETHEXPR && ir.MethodExprFunc(x).Nname != nil {
			fn.WrappedFunc = ir.MethodExprName(x).Func
		}
	}

	clo := fn.OClosure
	if n.Op() == ir.OGO {
		clo.IsGoWrap = true
	}

	e.callCommon(nil, call, init, fn)
	e.closures = append(e.closures, closure{e.spill(k, clo), clo})

	// Create new top level call to closure.
	n.Call = ir.NewCallExpr(call.Pos(), ir.OCALL, clo, nil)
	ir.WithFunc(e.curfn, func() {
		typecheck.Stmt(n.Call)
	})
}

// rewriteArgument rewrites the argument *argp of the given call expression.
// fn is the static callee function, if known.
// wrapper is the go/defer wrapper function for call, if any.
func (e *escape) rewriteArgument(argp *ir.Node, init *ir.Nodes, call ir.Node, fn *ir.Name, wrapper *ir.Func) {
	var pragma ir.PragmaFlag
	if fn != nil && fn.Func != nil {
		pragma = fn.Func.Pragma
	}

	// unsafeUintptr rewrites "uintptr(ptr)" arguments to syscall-like
	// functions, so that ptr is kept alive and/or escaped as
	// appropriate. unsafeUintptr also reports whether it modified arg0.
	unsafeUintptr := func(arg0 ir.Node) bool {
		if pragma&(ir.UintptrKeepAlive|ir.UintptrEscapes) == 0 {
			return false
		}

		// If the argument is really a pointer being converted to uintptr,
		// arrange for the pointer to be kept alive until the call returns,
		// by copying it into a temp and marking that temp
		// still alive when we pop the temp stack.
		if arg0.Op() != ir.OCONVNOP || !arg0.Type().IsUintptr() {
			return false
		}
		arg := arg0.(*ir.ConvExpr)

		if !arg.X.Type().IsUnsafePtr() {
			return false
		}

		// Create and declare a new pointer-typed temp variable.
		tmp := e.wrapExpr(arg.Pos(), &arg.X, init, call, wrapper)

		if pragma&ir.UintptrEscapes != 0 {
			e.flow(e.heapHole().note(arg, "//go:uintptrescapes"), e.oldLoc(tmp))
		}

		if pragma&ir.UintptrKeepAlive != 0 {
			call := call.(*ir.CallExpr)

			// SSA implements CallExpr.KeepAlive using OpVarLive, which
			// doesn't support PAUTOHEAP variables. I tried changing it to
			// use OpKeepAlive, but that ran into issues of its own.
			// For now, the easy solution is to explicitly copy to (yet
			// another) new temporary variable.
			keep := tmp
			if keep.Class == ir.PAUTOHEAP {
				keep = e.copyExpr(arg.Pos(), tmp, call.PtrInit(), wrapper, false)
			}

			keep.SetAddrtaken(true) // ensure SSA keeps the tmp variable
			call.KeepAlive = append(call.KeepAlive, keep)
		}

		return true
	}

	visit := func(pos src.XPos, argp *ir.Node) {
		// Optimize a few common constant expressions. By leaving these
		// untouched in the call expression, we let the wrapper handle
		// evaluating them, rather than taking up closure context space.
		switch arg := *argp; arg.Op() {
		case ir.OLITERAL, ir.ONIL, ir.OMETHEXPR:
			return
		case ir.ONAME:
			if arg.(*ir.Name).Class == ir.PFUNC {
				return
			}
		}

		if unsafeUintptr(*argp) {
			return
		}

		if wrapper != nil {
			e.wrapExpr(pos, argp, init, call, wrapper)
		}
	}

	// Peel away any slice literals for better escape analyze
	// them. For example:
	//
	//     go F([]int{a, b})
	//
	// If F doesn't escape its arguments, then the slice can
	// be allocated on the new goroutine's stack.
	//
	// For variadic functions, the compiler has already rewritten:
	//
	//     f(a, b, c)
	//
	// to:
	//
	//     f([]T{a, b, c}...)
	//
	// So we need to look into slice elements to handle uintptr(ptr)
	// arguments to syscall-like functions correctly.
	if arg := *argp; arg.Op() == ir.OSLICELIT {
		list := arg.(*ir.CompLitExpr).List
		for i := range list {
			el := &list[i]
			if list[i].Op() == ir.OKEY {
				el = &list[i].(*ir.KeyExpr).Value
			}
			visit(arg.Pos(), el)
		}
	} else {
		visit(call.Pos(), argp)
	}
}

// wrapExpr replaces *exprp with a temporary variable copy. If wrapper
// is non-nil, the variable will be captured for use within that
// function.
func (e *escape) wrapExpr(pos src.XPos, exprp *ir.Node, init *ir.Nodes, call ir.Node, wrapper *ir.Func) *ir.Name {
	tmp := e.copyExpr(pos, *exprp, init, e.curfn, true)

	if wrapper != nil {
		// Currently for "defer i.M()" if i is nil it panics at the point
		// of defer statement, not when deferred function is called.  We
		// need to do the nil check outside of the wrapper.
		if call.Op() == ir.OCALLINTER && exprp == &call.(*ir.CallExpr).X.(*ir.SelectorExpr).X {
			check := ir.NewUnaryExpr(pos, ir.OCHECKNIL, ir.NewUnaryExpr(pos, ir.OITAB, tmp))
			init.Append(typecheck.Stmt(check))
		}

		e.oldLoc(tmp).captured = true

		tmp = ir.NewClosureVar(pos, wrapper, tmp)
	}

	*exprp = tmp
	return tmp
}

// copyExpr creates and returns a new temporary variable within fn;
// appends statements to init to declare and initialize it to expr;
// and escape analyzes the data flow if analyze is true.
func (e *escape) copyExpr(pos src.XPos, expr ir.Node, init *ir.Nodes, fn *ir.Func, analyze bool) *ir.Name {
	if ir.HasUniquePos(expr) {
		pos = expr.Pos()
	}

	tmp := typecheck.TempAt(pos, fn, expr.Type())

	stmts := []ir.Node{
		ir.NewDecl(pos, ir.ODCL, tmp),
		ir.NewAssignStmt(pos, tmp, expr),
	}
	typecheck.Stmts(stmts)
	init.Append(stmts...)

	if analyze {
		e.newLoc(tmp, false)
		e.stmts(stmts)
	}

	return tmp
}

// tagHole returns a hole for evaluating an argument passed to param.
// ks should contain the holes representing where the function
// callee's results flows. fn is the statically-known callee function,
// if any.
func (e *escape) tagHole(ks []hole, fn *ir.Name, param *types.Field) hole {
	// If this is a dynamic call, we can't rely on param.Note.
	if fn == nil {
		return e.heapHole()
	}

	if e.inMutualBatch(fn) {
		return e.addr(ir.AsNode(param.Nname))
	}

	// Call to previously tagged function.

	var tagKs []hole

	esc := parseLeaks(param.Note)
	if x := esc.Heap(); x >= 0 {
		tagKs = append(tagKs, e.heapHole().shift(x))
	}

	if ks != nil {
		for i := 0; i < numEscResults; i++ {
			if x := esc.Result(i); x >= 0 {
				tagKs = append(tagKs, ks[i].shift(x))
			}
		}
	}

	return e.teeHole(tagKs...)
}
