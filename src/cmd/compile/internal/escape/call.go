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
	"strings"
)

// call evaluates a call expressions, including builtin calls. ks
// should contain the holes representing where the function callee's
// results flows.
func (e *escape) call(ks []hole, call ir.Node) {
	argument := func(k hole, arg ir.Node) {
		// TODO(mdempsky): Should be "call argument".
		e.expr(k.note(call, "call parameter"), arg)
	}

	switch call.Op() {
	default:
		ir.Dump("esc", call)
		base.Fatalf("unexpected call op: %v", call.Op())

	case ir.OCALLFUNC, ir.OCALLINTER:
		call := call.(*ir.CallExpr)
		typecheck.AssertFixedCall(call)

		// Pick out the function callee, if statically known.
		//
		// TODO(mdempsky): Change fn from *ir.Name to *ir.Func, but some
		// functions (e.g., runtime builtins, method wrappers, generated
		// eq/hash functions) don't have it set. Investigate whether
		// that's a concern.
		var fn *ir.Name
		switch call.Op() {
		case ir.OCALLFUNC:
			v := ir.StaticValue(call.Fun)
			fn = ir.StaticCalleeName(v)
		}

		fntype := call.Fun.Type()
		if fn != nil {
			fntype = fn.Type()
		}

		if ks != nil && fn != nil && e.inMutualBatch(fn) {
			for i, result := range fn.Type().Results() {
				e.expr(ks[i], result.Nname.(*ir.Name))
			}
		}

		var recvArg ir.Node
		if call.Op() == ir.OCALLFUNC {
			// Evaluate callee function expression.
			calleeK := e.discardHole()
			if fn == nil { // unknown callee
				for _, k := range ks {
					if k.dst != &e.blankLoc {
						// The results flow somewhere, but we don't statically
						// know the callee function. If a closure flows here, we
						// need to conservatively assume its results might flow to
						// the heap.
						calleeK = e.calleeHole().note(call, "callee operand")
						break
					}
				}
			}
			e.expr(calleeK, call.Fun)
		} else {
			recvArg = call.Fun.(*ir.SelectorExpr).X
		}

		// argumentParam handles escape analysis of assigning a call
		// argument to its corresponding parameter.
		argumentParam := func(param *types.Field, arg ir.Node) {
			e.rewriteArgument(arg, call, fn)
			argument(e.tagHole(ks, fn, param), arg)
		}

		// hash/maphash.escapeForHash forces its argument to be on
		// the heap, if it contains a non-string pointer. We cannot
		// hash pointers to local variables, as the address of the
		// local variable might change on stack growth.
		// Strings are okay as the hash depends on only the content,
		// not the pointer.
		// The actual call we match is
		//   hash/maphash.escapeForHash[go.shape.T](dict, go.shape.T)
		if fn != nil && fn.Sym().Pkg.Path == "hash/maphash" && strings.HasPrefix(fn.Sym().Name, "escapeForHash[") {
			ps := fntype.Params()
			if len(ps) == 2 && ps[1].Type.IsShape() {
				if !hasNonStringPointers(ps[1].Type) {
					argumentParam = func(param *types.Field, arg ir.Node) {
						argument(e.discardHole(), arg)
					}
				} else {
					argumentParam = func(param *types.Field, arg ir.Node) {
						argument(e.heapHole(), arg)
					}
				}
			}
		}

		args := call.Args
		if recvParam := fntype.Recv(); recvParam != nil {
			if recvArg == nil {
				// Function call using method expression. Receiver argument is
				// at the front of the regular arguments list.
				recvArg, args = args[0], args[1:]
			}

			argumentParam(recvParam, recvArg)
		}

		for i, param := range fntype.Params() {
			argumentParam(param, args[i])
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
		appendeeK := e.teeHole(ks[0], e.mutatorHole())
		if args[0].Type().Elem().HasPointers() {
			appendeeK = e.teeHole(appendeeK, e.heapHole().deref(call, "appendee slice"))
		}
		argument(appendeeK, args[0])

		if call.IsDDD {
			appendedK := e.discardHole()
			if args[1].Type().IsSlice() && args[1].Type().Elem().HasPointers() {
				appendedK = e.heapHole().deref(call, "appended slice...")
			}
			argument(appendedK, args[1])
		} else {
			for i := 1; i < len(args); i++ {
				argument(e.heapHole(), args[i])
			}
		}
		e.discard(call.RType)

	case ir.OCOPY:
		call := call.(*ir.BinaryExpr)
		argument(e.mutatorHole(), call.X)

		copiedK := e.discardHole()
		if call.Y.Type().IsSlice() && call.Y.Type().Elem().HasPointers() {
			copiedK = e.heapHole().deref(call, "copied slice")
		}
		argument(copiedK, call.Y)
		e.discard(call.RType)

	case ir.OPANIC:
		call := call.(*ir.UnaryExpr)
		argument(e.heapHole(), call.X)

	case ir.OCOMPLEX:
		call := call.(*ir.BinaryExpr)
		e.discard(call.X)
		e.discard(call.Y)

	case ir.ODELETE, ir.OPRINT, ir.OPRINTLN, ir.ORECOVERFP:
		call := call.(*ir.CallExpr)
		for _, arg := range call.Args {
			e.discard(arg)
		}
		e.discard(call.RType)

	case ir.OMIN, ir.OMAX:
		call := call.(*ir.CallExpr)
		for _, arg := range call.Args {
			argument(ks[0], arg)
		}
		e.discard(call.RType)

	case ir.OLEN, ir.OCAP, ir.OREAL, ir.OIMAG, ir.OCLOSE:
		call := call.(*ir.UnaryExpr)
		e.discard(call.X)

	case ir.OCLEAR:
		call := call.(*ir.UnaryExpr)
		argument(e.mutatorHole(), call.X)

	case ir.OUNSAFESTRINGDATA, ir.OUNSAFESLICEDATA:
		call := call.(*ir.UnaryExpr)
		argument(ks[0], call.X)

	case ir.OUNSAFEADD, ir.OUNSAFESLICE, ir.OUNSAFESTRING:
		call := call.(*ir.BinaryExpr)
		argument(ks[0], call.X)
		e.discard(call.Y)
		e.discard(call.RType)
	}
}

// goDeferStmt analyzes a "go" or "defer" statement.
func (e *escape) goDeferStmt(n *ir.GoDeferStmt) {
	k := e.heapHole()
	if n.Op() == ir.ODEFER && e.loopDepth == 1 && n.DeferAt == nil {
		// Top-level defer arguments don't escape to the heap,
		// but they do need to last until they're invoked.
		k = e.later(e.discardHole())

		// force stack allocation of defer record, unless
		// open-coded defers are used (see ssa.go)
		n.SetEsc(ir.EscNever)
	}

	// If the function is already a zero argument/result function call,
	// just escape analyze it normally.
	//
	// Note that the runtime is aware of this optimization for
	// "go" statements that start in reflect.makeFuncStub or
	// reflect.methodValueCall.

	call, ok := n.Call.(*ir.CallExpr)
	if !ok || call.Op() != ir.OCALLFUNC {
		base.FatalfAt(n.Pos(), "expected function call: %v", n.Call)
	}
	if sig := call.Fun.Type(); sig.NumParams()+sig.NumResults() != 0 {
		base.FatalfAt(n.Pos(), "expected signature without parameters or results: %v", sig)
	}

	if clo, ok := call.Fun.(*ir.ClosureExpr); ok && n.Op() == ir.OGO {
		clo.IsGoWrap = true
	}

	e.expr(k, call.Fun)
}

// rewriteArgument rewrites the argument arg of the given call expression.
// fn is the static callee function, if known.
func (e *escape) rewriteArgument(arg ir.Node, call *ir.CallExpr, fn *ir.Name) {
	if fn == nil || fn.Func == nil {
		return
	}
	pragma := fn.Func.Pragma
	if pragma&(ir.UintptrKeepAlive|ir.UintptrEscapes) == 0 {
		return
	}

	// unsafeUintptr rewrites "uintptr(ptr)" arguments to syscall-like
	// functions, so that ptr is kept alive and/or escaped as
	// appropriate. unsafeUintptr also reports whether it modified arg0.
	unsafeUintptr := func(arg ir.Node) {
		// If the argument is really a pointer being converted to uintptr,
		// arrange for the pointer to be kept alive until the call
		// returns, by copying it into a temp and marking that temp still
		// alive when we pop the temp stack.
		conv, ok := arg.(*ir.ConvExpr)
		if !ok || conv.Op() != ir.OCONVNOP {
			return // not a conversion
		}
		if !conv.X.Type().IsUnsafePtr() || !conv.Type().IsUintptr() {
			return // not an unsafe.Pointer->uintptr conversion
		}

		// Create and declare a new pointer-typed temp variable.
		//
		// TODO(mdempsky): This potentially violates the Go spec's order
		// of evaluations, by evaluating arg.X before any other
		// operands.
		tmp := e.copyExpr(conv.Pos(), conv.X, call.PtrInit())
		conv.X = tmp

		k := e.mutatorHole()
		if pragma&ir.UintptrEscapes != 0 {
			k = e.heapHole().note(conv, "//go:uintptrescapes")
		}
		e.flow(k, e.oldLoc(tmp))

		if pragma&ir.UintptrKeepAlive != 0 {
			tmp.SetAddrtaken(true) // ensure SSA keeps the tmp variable
			call.KeepAlive = append(call.KeepAlive, tmp)
		}
	}

	// For variadic functions, the compiler has already rewritten:
	//
	//     f(a, b, c)
	//
	// to:
	//
	//     f([]T{a, b, c}...)
	//
	// So we need to look into slice elements to handle uintptr(ptr)
	// arguments to variadic syscall-like functions correctly.
	if arg.Op() == ir.OSLICELIT {
		list := arg.(*ir.CompLitExpr).List
		for _, el := range list {
			if el.Op() == ir.OKEY {
				el = el.(*ir.KeyExpr).Value
			}
			unsafeUintptr(el)
		}
	} else {
		unsafeUintptr(arg)
	}
}

// copyExpr creates and returns a new temporary variable within fn;
// appends statements to init to declare and initialize it to expr;
// and escape analyzes the data flow.
func (e *escape) copyExpr(pos src.XPos, expr ir.Node, init *ir.Nodes) *ir.Name {
	if ir.HasUniquePos(expr) {
		pos = expr.Pos()
	}

	tmp := typecheck.TempAt(pos, e.curfn, expr.Type())

	stmts := []ir.Node{
		ir.NewDecl(pos, ir.ODCL, tmp),
		ir.NewAssignStmt(pos, tmp, expr),
	}
	typecheck.Stmts(stmts)
	init.Append(stmts...)

	e.newLoc(tmp, true)
	e.stmts(stmts)

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
		if param.Nname == nil {
			return e.discardHole()
		}
		return e.addr(param.Nname.(*ir.Name))
	}

	// Call to previously tagged function.

	var tagKs []hole
	esc := parseLeaks(param.Note)

	if x := esc.Heap(); x >= 0 {
		tagKs = append(tagKs, e.heapHole().shift(x))
	}
	if x := esc.Mutator(); x >= 0 {
		tagKs = append(tagKs, e.mutatorHole().shift(x))
	}
	if x := esc.Callee(); x >= 0 {
		tagKs = append(tagKs, e.calleeHole().shift(x))
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

func hasNonStringPointers(t *types.Type) bool {
	if !t.HasPointers() {
		return false
	}
	switch t.Kind() {
	case types.TSTRING:
		return false
	case types.TSTRUCT:
		for _, f := range t.Fields() {
			if hasNonStringPointers(f.Type) {
				return true
			}
		}
		return false
	case types.TARRAY:
		return hasNonStringPointers(t.Elem())
	}
	return true
}
