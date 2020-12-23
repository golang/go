// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

func (p *noder) funcLit(expr *syntax.FuncLit) ir.Node {
	xtype := p.typeExpr(expr.Type)
	ntype := p.typeExpr(expr.Type)

	fn := ir.NewFunc(p.pos(expr))
	fn.SetIsHiddenClosure(ir.CurFunc != nil)
	fn.Nname = ir.NewFuncNameAt(p.pos(expr), ir.BlankNode.Sym(), fn) // filled in by typecheckclosure
	fn.Nname.Ntype = xtype
	fn.Nname.Defn = fn

	clo := ir.NewClosureExpr(p.pos(expr), fn)
	fn.ClosureType = ntype
	fn.OClosure = clo

	p.funcBody(fn, expr.Body)

	// closure-specific variables are hanging off the
	// ordinary ones in the symbol table; see oldname.
	// unhook them.
	// make the list of pointers for the closure call.
	for _, v := range fn.ClosureVars {
		// Unlink from v1; see comment in syntax.go type Param for these fields.
		v1 := v.Defn
		v1.Name().Innermost = v.Outer

		// If the closure usage of v is not dense,
		// we need to make it dense; now that we're out
		// of the function in which v appeared,
		// look up v.Sym in the enclosing function
		// and keep it around for use in the compiled code.
		//
		// That is, suppose we just finished parsing the innermost
		// closure f4 in this code:
		//
		//	func f() {
		//		v := 1
		//		func() { // f2
		//			use(v)
		//			func() { // f3
		//				func() { // f4
		//					use(v)
		//				}()
		//			}()
		//		}()
		//	}
		//
		// At this point v.Outer is f2's v; there is no f3's v.
		// To construct the closure f4 from within f3,
		// we need to use f3's v and in this case we need to create f3's v.
		// We are now in the context of f3, so calling oldname(v.Sym)
		// obtains f3's v, creating it if necessary (as it is in the example).
		//
		// capturevars will decide whether to use v directly or &v.
		v.Outer = oldname(v.Sym()).(*ir.Name)
	}

	return clo
}

// transformclosure is called in a separate phase after escape analysis.
// It transform closure bodies to properly reference captured variables.
func transformclosure(fn *ir.Func) {
	lno := base.Pos
	base.Pos = fn.Pos()

	if fn.ClosureCalled() {
		// If the closure is directly called, we transform it to a plain function call
		// with variables passed as args. This avoids allocation of a closure object.
		// Here we do only a part of the transformation. Walk of OCALLFUNC(OCLOSURE)
		// will complete the transformation later.
		// For illustration, the following closure:
		//	func(a int) {
		//		println(byval)
		//		byref++
		//	}(42)
		// becomes:
		//	func(byval int, &byref *int, a int) {
		//		println(byval)
		//		(*&byref)++
		//	}(byval, &byref, 42)

		// f is ONAME of the actual function.
		f := fn.Nname

		// We are going to insert captured variables before input args.
		var params []*types.Field
		var decls []*ir.Name
		for _, v := range fn.ClosureVars {
			if !v.Byval() {
				// If v of type T is captured by reference,
				// we introduce function param &v *T
				// and v remains PAUTOHEAP with &v heapaddr
				// (accesses will implicitly deref &v).
				addr := typecheck.NewName(typecheck.Lookup("&" + v.Sym().Name))
				addr.SetType(types.NewPtr(v.Type()))
				v.Heapaddr = addr
				v = addr
			}

			v.Class_ = ir.PPARAM
			decls = append(decls, v)

			fld := types.NewField(src.NoXPos, v.Sym(), v.Type())
			fld.Nname = v
			params = append(params, fld)
		}

		if len(params) > 0 {
			// Prepend params and decls.
			f.Type().Params().SetFields(append(params, f.Type().Params().FieldSlice()...))
			fn.Dcl = append(decls, fn.Dcl...)
		}

		types.CalcSize(f.Type())
		fn.SetType(f.Type()) // update type of ODCLFUNC
	} else {
		// The closure is not called, so it is going to stay as closure.
		var body []ir.Node
		offset := int64(types.PtrSize)
		for _, v := range fn.ClosureVars {
			// cv refers to the field inside of closure OSTRUCTLIT.
			typ := v.Type()
			if !v.Byval() {
				typ = types.NewPtr(typ)
			}
			offset = types.Rnd(offset, int64(typ.Align))
			cr := ir.NewClosureRead(typ, offset)
			offset += typ.Width

			if v.Byval() && v.Type().Width <= int64(2*types.PtrSize) {
				// If it is a small variable captured by value, downgrade it to PAUTO.
				v.Class_ = ir.PAUTO
				fn.Dcl = append(fn.Dcl, v)
				body = append(body, ir.NewAssignStmt(base.Pos, v, cr))
			} else {
				// Declare variable holding addresses taken from closure
				// and initialize in entry prologue.
				addr := typecheck.NewName(typecheck.Lookup("&" + v.Sym().Name))
				addr.SetType(types.NewPtr(v.Type()))
				addr.Class_ = ir.PAUTO
				addr.SetUsed(true)
				addr.Curfn = fn
				fn.Dcl = append(fn.Dcl, addr)
				v.Heapaddr = addr
				var src ir.Node = cr
				if v.Byval() {
					src = typecheck.NodAddr(cr)
				}
				body = append(body, ir.NewAssignStmt(base.Pos, addr, src))
			}
		}

		if len(body) > 0 {
			typecheck.Stmts(body)
			fn.Enter.Set(body)
			fn.SetNeedctxt(true)
		}
	}

	base.Pos = lno
}

// hasemptycvars reports whether closure clo has an
// empty list of captured vars.
func hasemptycvars(clo *ir.ClosureExpr) bool {
	return len(clo.Func.ClosureVars) == 0
}

// closuredebugruntimecheck applies boilerplate checks for debug flags
// and compiling runtime
func closuredebugruntimecheck(clo *ir.ClosureExpr) {
	if base.Debug.Closure > 0 {
		if clo.Esc() == ir.EscHeap {
			base.WarnfAt(clo.Pos(), "heap closure, captured vars = %v", clo.Func.ClosureVars)
		} else {
			base.WarnfAt(clo.Pos(), "stack closure, captured vars = %v", clo.Func.ClosureVars)
		}
	}
	if base.Flag.CompilingRuntime && clo.Esc() == ir.EscHeap {
		base.ErrorfAt(clo.Pos(), "heap-allocated closure, not allowed in runtime")
	}
}

func walkclosure(clo *ir.ClosureExpr, init *ir.Nodes) ir.Node {
	fn := clo.Func

	// If no closure vars, don't bother wrapping.
	if hasemptycvars(clo) {
		if base.Debug.Closure > 0 {
			base.WarnfAt(clo.Pos(), "closure converted to global")
		}
		return fn.Nname
	}
	closuredebugruntimecheck(clo)

	typ := typecheck.ClosureType(clo)

	clos := ir.NewCompLitExpr(base.Pos, ir.OCOMPLIT, ir.TypeNode(typ).(ir.Ntype), nil)
	clos.SetEsc(clo.Esc())
	clos.List.Set(append([]ir.Node{ir.NewUnaryExpr(base.Pos, ir.OCFUNC, fn.Nname)}, fn.ClosureEnter...))

	addr := typecheck.NodAddr(clos)
	addr.SetEsc(clo.Esc())

	// Force type conversion from *struct to the func type.
	cfn := typecheck.ConvNop(addr, clo.Type())

	// non-escaping temp to use, if any.
	if x := clo.Prealloc; x != nil {
		if !types.Identical(typ, x.Type()) {
			panic("closure type does not match order's assigned type")
		}
		addr.Alloc = x
		clo.Prealloc = nil
	}

	return walkexpr(cfn, init)
}

func walkpartialcall(n *ir.CallPartExpr, init *ir.Nodes) ir.Node {
	// Create closure in the form of a composite literal.
	// For x.M with receiver (x) type T, the generated code looks like:
	//
	//	clos = &struct{F uintptr; R T}{T.M·f, x}
	//
	// Like walkclosure above.

	if n.X.Type().IsInterface() {
		// Trigger panic for method on nil interface now.
		// Otherwise it happens in the wrapper and is confusing.
		n.X = cheapexpr(n.X, init)
		n.X = walkexpr(n.X, nil)

		tab := typecheck.Expr(ir.NewUnaryExpr(base.Pos, ir.OITAB, n.X))

		c := ir.NewUnaryExpr(base.Pos, ir.OCHECKNIL, tab)
		c.SetTypecheck(1)
		init.Append(c)
	}

	typ := typecheck.PartialCallType(n)

	clos := ir.NewCompLitExpr(base.Pos, ir.OCOMPLIT, ir.TypeNode(typ).(ir.Ntype), nil)
	clos.SetEsc(n.Esc())
	clos.List = []ir.Node{ir.NewUnaryExpr(base.Pos, ir.OCFUNC, n.Func.Nname), n.X}

	addr := typecheck.NodAddr(clos)
	addr.SetEsc(n.Esc())

	// Force type conversion from *struct to the func type.
	cfn := typecheck.ConvNop(addr, n.Type())

	// non-escaping temp to use, if any.
	if x := n.Prealloc; x != nil {
		if !types.Identical(typ, x.Type()) {
			panic("partial call type does not match order's assigned type")
		}
		addr.Alloc = x
		n.Prealloc = nil
	}

	return walkexpr(cfn, init)
}
