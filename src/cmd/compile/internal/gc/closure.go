// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"fmt"
)

func (p *noder) funcLit(expr *syntax.FuncLit) ir.Node {
	xtype := p.typeExpr(expr.Type)
	ntype := p.typeExpr(expr.Type)

	fn := ir.NewFunc(p.pos(expr))
	fn.SetIsHiddenClosure(Curfn != nil)
	fn.Nname = newFuncNameAt(p.pos(expr), ir.BlankNode.Sym(), fn) // filled in by typecheckclosure
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

// typecheckclosure typechecks an OCLOSURE node. It also creates the named
// function associated with the closure.
// TODO: This creation of the named function should probably really be done in a
// separate pass from type-checking.
func typecheckclosure(clo ir.Node, top int) {
	fn := clo.Func()
	// Set current associated iota value, so iota can be used inside
	// function in ConstSpec, see issue #22344
	if x := getIotaValue(); x >= 0 {
		fn.SetIota(x)
	}

	fn.ClosureType = typecheck(fn.ClosureType, ctxType)
	clo.SetType(fn.ClosureType.Type())
	fn.SetClosureCalled(top&ctxCallee != 0)

	// Do not typecheck fn twice, otherwise, we will end up pushing
	// fn to xtop multiple times, causing initLSym called twice.
	// See #30709
	if fn.Typecheck() == 1 {
		return
	}

	for _, ln := range fn.ClosureVars {
		n := ln.Defn
		if !n.Name().Captured() {
			n.Name().SetCaptured(true)
			if n.Name().Decldepth == 0 {
				base.Fatalf("typecheckclosure: var %S does not have decldepth assigned", n)
			}

			// Ignore assignments to the variable in straightline code
			// preceding the first capturing by a closure.
			if n.Name().Decldepth == decldepth {
				n.Name().SetAssigned(false)
			}
		}
	}

	fn.Nname.SetSym(closurename(Curfn))
	setNodeNameFunc(fn.Nname)
	typecheckFunc(fn)

	// Type check the body now, but only if we're inside a function.
	// At top level (in a variable initialization: curfn==nil) we're not
	// ready to type check code yet; we'll check it later, because the
	// underlying closure function we create is added to xtop.
	if Curfn != nil && clo.Type() != nil {
		oldfn := Curfn
		Curfn = fn
		olddd := decldepth
		decldepth = 1
		typecheckslice(fn.Body().Slice(), ctxStmt)
		decldepth = olddd
		Curfn = oldfn
	}

	xtop = append(xtop, fn)
}

// globClosgen is like Func.Closgen, but for the global scope.
var globClosgen int32

// closurename generates a new unique name for a closure within
// outerfunc.
func closurename(outerfunc *ir.Func) *types.Sym {
	outer := "glob."
	prefix := "func"
	gen := &globClosgen

	if outerfunc != nil {
		if outerfunc.OClosure != nil {
			prefix = ""
		}

		outer = ir.FuncName(outerfunc)

		// There may be multiple functions named "_". In those
		// cases, we can't use their individual Closgens as it
		// would lead to name clashes.
		if !ir.IsBlank(outerfunc.Nname) {
			gen = &outerfunc.Closgen
		}
	}

	*gen++
	return lookup(fmt.Sprintf("%s.%s%d", outer, prefix, *gen))
}

// capturevarscomplete is set to true when the capturevars phase is done.
var capturevarscomplete bool

// capturevars is called in a separate phase after all typechecking is done.
// It decides whether each variable captured by a closure should be captured
// by value or by reference.
// We use value capturing for values <= 128 bytes that are never reassigned
// after capturing (effectively constant).
func capturevars(fn *ir.Func) {
	lno := base.Pos
	base.Pos = fn.Pos()
	cvars := fn.ClosureVars
	out := cvars[:0]
	for _, v := range cvars {
		if v.Type() == nil {
			// If v.Type is nil, it means v looked like it
			// was going to be used in the closure, but
			// isn't. This happens in struct literals like
			// s{f: x} where we can't distinguish whether
			// f is a field identifier or expression until
			// resolving s.
			continue
		}
		out = append(out, v)

		// type check the & of closed variables outside the closure,
		// so that the outer frame also grabs them and knows they escape.
		dowidth(v.Type())

		var outer ir.Node
		outer = v.Outer
		outermost := v.Defn

		// out parameters will be assigned to implicitly upon return.
		if outermost.Class() != ir.PPARAMOUT && !outermost.Name().Addrtaken() && !outermost.Name().Assigned() && v.Type().Width <= 128 {
			v.SetByval(true)
		} else {
			outermost.Name().SetAddrtaken(true)
			outer = ir.Nod(ir.OADDR, outer, nil)
		}

		if base.Flag.LowerM > 1 {
			var name *types.Sym
			if v.Curfn != nil && v.Curfn.Nname != nil {
				name = v.Curfn.Sym()
			}
			how := "ref"
			if v.Byval() {
				how = "value"
			}
			base.WarnfAt(v.Pos(), "%v capturing by %s: %v (addr=%v assign=%v width=%d)", name, how, v.Sym(), outermost.Name().Addrtaken(), outermost.Name().Assigned(), int32(v.Type().Width))
		}

		outer = typecheck(outer, ctxExpr)
		fn.ClosureEnter.Append(outer)
	}

	fn.ClosureVars = out
	base.Pos = lno
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
				addr := NewName(lookup("&" + v.Sym().Name))
				addr.SetType(types.NewPtr(v.Type()))
				v.Heapaddr = addr
				v = addr
			}

			v.SetClass(ir.PPARAM)
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

		dowidth(f.Type())
		fn.SetType(f.Type()) // update type of ODCLFUNC
	} else {
		// The closure is not called, so it is going to stay as closure.
		var body []ir.Node
		offset := int64(Widthptr)
		for _, v := range fn.ClosureVars {
			// cv refers to the field inside of closure OSTRUCTLIT.
			typ := v.Type()
			if !v.Byval() {
				typ = types.NewPtr(typ)
			}
			offset = Rnd(offset, int64(typ.Align))
			cr := ir.NewClosureRead(typ, offset)
			offset += typ.Width

			if v.Byval() && v.Type().Width <= int64(2*Widthptr) {
				// If it is a small variable captured by value, downgrade it to PAUTO.
				v.SetClass(ir.PAUTO)
				fn.Dcl = append(fn.Dcl, v)
				body = append(body, ir.Nod(ir.OAS, v, cr))
			} else {
				// Declare variable holding addresses taken from closure
				// and initialize in entry prologue.
				addr := NewName(lookup("&" + v.Sym().Name))
				addr.SetType(types.NewPtr(v.Type()))
				addr.SetClass(ir.PAUTO)
				addr.SetUsed(true)
				addr.Curfn = fn
				fn.Dcl = append(fn.Dcl, addr)
				v.Heapaddr = addr
				var src ir.Node = cr
				if v.Byval() {
					src = ir.Nod(ir.OADDR, cr, nil)
				}
				body = append(body, ir.Nod(ir.OAS, addr, src))
			}
		}

		if len(body) > 0 {
			typecheckslice(body, ctxStmt)
			fn.Enter.Set(body)
			fn.SetNeedctxt(true)
		}
	}

	base.Pos = lno
}

// hasemptycvars reports whether closure clo has an
// empty list of captured vars.
func hasemptycvars(clo ir.Node) bool {
	return len(clo.Func().ClosureVars) == 0
}

// closuredebugruntimecheck applies boilerplate checks for debug flags
// and compiling runtime
func closuredebugruntimecheck(clo ir.Node) {
	if base.Debug.Closure > 0 {
		if clo.Esc() == EscHeap {
			base.WarnfAt(clo.Pos(), "heap closure, captured vars = %v", clo.Func().ClosureVars)
		} else {
			base.WarnfAt(clo.Pos(), "stack closure, captured vars = %v", clo.Func().ClosureVars)
		}
	}
	if base.Flag.CompilingRuntime && clo.Esc() == EscHeap {
		base.ErrorfAt(clo.Pos(), "heap-allocated closure, not allowed in runtime")
	}
}

// closureType returns the struct type used to hold all the information
// needed in the closure for clo (clo must be a OCLOSURE node).
// The address of a variable of the returned type can be cast to a func.
func closureType(clo ir.Node) *types.Type {
	// Create closure in the form of a composite literal.
	// supposing the closure captures an int i and a string s
	// and has one float64 argument and no results,
	// the generated code looks like:
	//
	//	clos = &struct{.F uintptr; i *int; s *string}{func.1, &i, &s}
	//
	// The use of the struct provides type information to the garbage
	// collector so that it can walk the closure. We could use (in this case)
	// [3]unsafe.Pointer instead, but that would leave the gc in the dark.
	// The information appears in the binary in the form of type descriptors;
	// the struct is unnamed so that closures in multiple packages with the
	// same struct type can share the descriptor.
	fields := []*ir.Field{
		namedfield(".F", types.Types[types.TUINTPTR]),
	}
	for _, v := range clo.Func().ClosureVars {
		typ := v.Type()
		if !v.Byval() {
			typ = types.NewPtr(typ)
		}
		fields = append(fields, symfield(v.Sym(), typ))
	}
	typ := tostruct(fields)
	typ.SetNoalg(true)
	return typ
}

func walkclosure(clo ir.Node, init *ir.Nodes) ir.Node {
	fn := clo.Func()

	// If no closure vars, don't bother wrapping.
	if hasemptycvars(clo) {
		if base.Debug.Closure > 0 {
			base.WarnfAt(clo.Pos(), "closure converted to global")
		}
		return fn.Nname
	}
	closuredebugruntimecheck(clo)

	typ := closureType(clo)

	clos := ir.Nod(ir.OCOMPLIT, nil, ir.TypeNode(typ))
	clos.SetEsc(clo.Esc())
	clos.PtrList().Set(append([]ir.Node{ir.Nod(ir.OCFUNC, fn.Nname, nil)}, fn.ClosureEnter.Slice()...))

	clos = ir.Nod(ir.OADDR, clos, nil)
	clos.SetEsc(clo.Esc())

	// Force type conversion from *struct to the func type.
	clos = convnop(clos, clo.Type())

	// non-escaping temp to use, if any.
	if x := prealloc[clo]; x != nil {
		if !types.Identical(typ, x.Type()) {
			panic("closure type does not match order's assigned type")
		}
		clos.Left().SetRight(x)
		delete(prealloc, clo)
	}

	return walkexpr(clos, init)
}

func typecheckpartialcall(dot ir.Node, sym *types.Sym) *ir.CallPartExpr {
	switch dot.Op() {
	case ir.ODOTINTER, ir.ODOTMETH:
		break

	default:
		base.Fatalf("invalid typecheckpartialcall")
	}

	// Create top-level function.
	fn := makepartialcall(dot, dot.Type(), sym)
	fn.SetWrapper(true)

	return ir.NewCallPartExpr(dot.Pos(), dot.Left(), NewName(sym), fn)
}

// makepartialcall returns a DCLFUNC node representing the wrapper function (*-fm) needed
// for partial calls.
func makepartialcall(dot ir.Node, t0 *types.Type, meth *types.Sym) *ir.Func {
	rcvrtype := dot.Left().Type()
	sym := methodSymSuffix(rcvrtype, meth, "-fm")

	if sym.Uniq() {
		return sym.Def.(*ir.Func)
	}
	sym.SetUniq(true)

	savecurfn := Curfn
	saveLineNo := base.Pos
	Curfn = nil

	// Set line number equal to the line number where the method is declared.
	var m *types.Field
	if lookdot0(meth, rcvrtype, &m, false) == 1 && m.Pos.IsKnown() {
		base.Pos = m.Pos
	}
	// Note: !m.Pos.IsKnown() happens for method expressions where
	// the method is implicitly declared. The Error method of the
	// built-in error type is one such method.  We leave the line
	// number at the use of the method expression in this
	// case. See issue 29389.

	tfn := ir.NewFuncType(base.Pos, nil,
		structargs(t0.Params(), true),
		structargs(t0.Results(), false))

	fn := dclfunc(sym, tfn)
	fn.SetDupok(true)
	fn.SetNeedctxt(true)

	tfn.Type().SetPkg(t0.Pkg())

	// Declare and initialize variable holding receiver.
	cr := ir.NewClosureRead(rcvrtype, Rnd(int64(Widthptr), int64(rcvrtype.Align)))
	ptr := NewName(lookup(".this"))
	declare(ptr, ir.PAUTO)
	ptr.SetUsed(true)
	var body []ir.Node
	if rcvrtype.IsPtr() || rcvrtype.IsInterface() {
		ptr.SetType(rcvrtype)
		body = append(body, ir.Nod(ir.OAS, ptr, cr))
	} else {
		ptr.SetType(types.NewPtr(rcvrtype))
		body = append(body, ir.Nod(ir.OAS, ptr, ir.Nod(ir.OADDR, cr, nil)))
	}

	call := ir.Nod(ir.OCALL, nodSym(ir.OXDOT, ptr, meth), nil)
	call.PtrList().Set(paramNnames(tfn.Type()))
	call.SetIsDDD(tfn.Type().IsVariadic())
	if t0.NumResults() != 0 {
		n := ir.Nod(ir.ORETURN, nil, nil)
		n.PtrList().Set1(call)
		call = n
	}
	body = append(body, call)

	fn.PtrBody().Set(body)
	funcbody()

	typecheckFunc(fn)
	// Need to typecheck the body of the just-generated wrapper.
	// typecheckslice() requires that Curfn is set when processing an ORETURN.
	Curfn = fn
	typecheckslice(fn.Body().Slice(), ctxStmt)
	sym.Def = fn
	xtop = append(xtop, fn)
	Curfn = savecurfn
	base.Pos = saveLineNo

	return fn
}

// partialCallType returns the struct type used to hold all the information
// needed in the closure for n (n must be a OCALLPART node).
// The address of a variable of the returned type can be cast to a func.
func partialCallType(n ir.Node) *types.Type {
	t := tostruct([]*ir.Field{
		namedfield("F", types.Types[types.TUINTPTR]),
		namedfield("R", n.Left().Type()),
	})
	t.SetNoalg(true)
	return t
}

func walkpartialcall(n *ir.CallPartExpr, init *ir.Nodes) ir.Node {
	// Create closure in the form of a composite literal.
	// For x.M with receiver (x) type T, the generated code looks like:
	//
	//	clos = &struct{F uintptr; R T}{T.M·f, x}
	//
	// Like walkclosure above.

	if n.Left().Type().IsInterface() {
		// Trigger panic for method on nil interface now.
		// Otherwise it happens in the wrapper and is confusing.
		n.SetLeft(cheapexpr(n.Left(), init))
		n.SetLeft(walkexpr(n.Left(), nil))

		tab := ir.Nod(ir.OITAB, n.Left(), nil)
		tab = typecheck(tab, ctxExpr)

		c := ir.Nod(ir.OCHECKNIL, tab, nil)
		c.SetTypecheck(1)
		init.Append(c)
	}

	typ := partialCallType(n)

	clos := ir.Nod(ir.OCOMPLIT, nil, ir.TypeNode(typ))
	clos.SetEsc(n.Esc())
	clos.PtrList().Set2(ir.Nod(ir.OCFUNC, n.Func().Nname, nil), n.Left())

	clos = ir.Nod(ir.OADDR, clos, nil)
	clos.SetEsc(n.Esc())

	// Force type conversion from *struct to the func type.
	clos = convnop(clos, n.Type())

	// non-escaping temp to use, if any.
	if x := prealloc[n]; x != nil {
		if !types.Identical(typ, x.Type()) {
			panic("partial call type does not match order's assigned type")
		}
		clos.Left().SetRight(x)
		delete(prealloc, n)
	}

	return walkexpr(clos, init)
}

// callpartMethod returns the *types.Field representing the method
// referenced by method value n.
func callpartMethod(n ir.Node) *types.Field {
	if n.Op() != ir.OCALLPART {
		base.Fatalf("expected OCALLPART, got %v", n)
	}

	// TODO(mdempsky): Optimize this. If necessary,
	// makepartialcall could save m for us somewhere.
	var m *types.Field
	if lookdot0(n.Right().Sym(), n.Left().Type(), &m, false) != 1 {
		base.Fatalf("failed to find field for OCALLPART")
	}

	return m
}
