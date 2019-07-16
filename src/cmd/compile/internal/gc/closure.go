// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types"
	"fmt"
)

func (p *noder) funcLit(expr *syntax.FuncLit) *Node {
	xtype := p.typeExpr(expr.Type)
	ntype := p.typeExpr(expr.Type)

	xfunc := p.nod(expr, ODCLFUNC, nil, nil)
	xfunc.Func.SetIsHiddenClosure(Curfn != nil)
	xfunc.Func.Nname = newfuncnamel(p.pos(expr), nblank.Sym) // filled in by typecheckclosure
	xfunc.Func.Nname.Name.Param.Ntype = xtype
	xfunc.Func.Nname.Name.Defn = xfunc

	clo := p.nod(expr, OCLOSURE, nil, nil)
	clo.Func.Ntype = ntype

	xfunc.Func.Closure = clo
	clo.Func.Closure = xfunc

	p.funcBody(xfunc, expr.Body)

	// closure-specific variables are hanging off the
	// ordinary ones in the symbol table; see oldname.
	// unhook them.
	// make the list of pointers for the closure call.
	for _, v := range xfunc.Func.Cvars.Slice() {
		// Unlink from v1; see comment in syntax.go type Param for these fields.
		v1 := v.Name.Defn
		v1.Name.Param.Innermost = v.Name.Param.Outer

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
		v.Name.Param.Outer = oldname(v.Sym)
	}

	return clo
}

func typecheckclosure(clo *Node, top int) {
	xfunc := clo.Func.Closure
	clo.Func.Ntype = typecheck(clo.Func.Ntype, Etype)
	clo.Type = clo.Func.Ntype.Type
	clo.Func.Top = top

	// Do not typecheck xfunc twice, otherwise, we will end up pushing
	// xfunc to xtop multiple times, causing initLSym called twice.
	// See #30709
	if xfunc.Typecheck() == 1 {
		return
	}

	for _, ln := range xfunc.Func.Cvars.Slice() {
		n := ln.Name.Defn
		if !n.Name.Captured() {
			n.Name.SetCaptured(true)
			if n.Name.Decldepth == 0 {
				Fatalf("typecheckclosure: var %S does not have decldepth assigned", n)
			}

			// Ignore assignments to the variable in straightline code
			// preceding the first capturing by a closure.
			if n.Name.Decldepth == decldepth {
				n.SetAssigned(false)
			}
		}
	}

	xfunc.Func.Nname.Sym = closurename(Curfn)
	disableExport(xfunc.Func.Nname.Sym)
	declare(xfunc.Func.Nname, PFUNC)
	xfunc = typecheck(xfunc, ctxStmt)

	// Type check the body now, but only if we're inside a function.
	// At top level (in a variable initialization: curfn==nil) we're not
	// ready to type check code yet; we'll check it later, because the
	// underlying closure function we create is added to xtop.
	if Curfn != nil && clo.Type != nil {
		oldfn := Curfn
		Curfn = xfunc
		olddd := decldepth
		decldepth = 1
		typecheckslice(xfunc.Nbody.Slice(), ctxStmt)
		decldepth = olddd
		Curfn = oldfn
	}

	xtop = append(xtop, xfunc)
}

// globClosgen is like Func.Closgen, but for the global scope.
var globClosgen int

// closurename generates a new unique name for a closure within
// outerfunc.
func closurename(outerfunc *Node) *types.Sym {
	outer := "glob."
	prefix := "func"
	gen := &globClosgen

	if outerfunc != nil {
		if outerfunc.Func.Closure != nil {
			prefix = ""
		}

		outer = outerfunc.funcname()

		// There may be multiple functions named "_". In those
		// cases, we can't use their individual Closgens as it
		// would lead to name clashes.
		if !outerfunc.Func.Nname.isBlank() {
			gen = &outerfunc.Func.Closgen
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
func capturevars(xfunc *Node) {
	lno := lineno
	lineno = xfunc.Pos

	clo := xfunc.Func.Closure
	cvars := xfunc.Func.Cvars.Slice()
	out := cvars[:0]
	for _, v := range cvars {
		if v.Type == nil {
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
		dowidth(v.Type)

		outer := v.Name.Param.Outer
		outermost := v.Name.Defn

		// out parameters will be assigned to implicitly upon return.
		if outermost.Class() != PPARAMOUT && !outermost.Addrtaken() && !outermost.Assigned() && v.Type.Width <= 128 {
			v.Name.SetByval(true)
		} else {
			outermost.SetAddrtaken(true)
			outer = nod(OADDR, outer, nil)
		}

		if Debug['m'] > 1 {
			var name *types.Sym
			if v.Name.Curfn != nil && v.Name.Curfn.Func.Nname != nil {
				name = v.Name.Curfn.Func.Nname.Sym
			}
			how := "ref"
			if v.Name.Byval() {
				how = "value"
			}
			Warnl(v.Pos, "%v capturing by %s: %v (addr=%v assign=%v width=%d)", name, how, v.Sym, outermost.Addrtaken(), outermost.Assigned(), int32(v.Type.Width))
		}

		outer = typecheck(outer, ctxExpr)
		clo.Func.Enter.Append(outer)
	}

	xfunc.Func.Cvars.Set(out)
	lineno = lno
}

// transformclosure is called in a separate phase after escape analysis.
// It transform closure bodies to properly reference captured variables.
func transformclosure(xfunc *Node) {
	lno := lineno
	lineno = xfunc.Pos
	clo := xfunc.Func.Closure

	if clo.Func.Top&ctxCallee != 0 {
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
		f := xfunc.Func.Nname

		// We are going to insert captured variables before input args.
		var params []*types.Field
		var decls []*Node
		for _, v := range xfunc.Func.Cvars.Slice() {
			if !v.Name.Byval() {
				// If v of type T is captured by reference,
				// we introduce function param &v *T
				// and v remains PAUTOHEAP with &v heapaddr
				// (accesses will implicitly deref &v).
				addr := newname(lookup("&" + v.Sym.Name))
				addr.Type = types.NewPtr(v.Type)
				v.Name.Param.Heapaddr = addr
				v = addr
			}

			v.SetClass(PPARAM)
			decls = append(decls, v)

			fld := types.NewField()
			fld.Nname = asTypesNode(v)
			fld.Type = v.Type
			fld.Sym = v.Sym
			params = append(params, fld)
		}

		if len(params) > 0 {
			// Prepend params and decls.
			f.Type.Params().SetFields(append(params, f.Type.Params().FieldSlice()...))
			xfunc.Func.Dcl = append(decls, xfunc.Func.Dcl...)
		}

		dowidth(f.Type)
		xfunc.Type = f.Type // update type of ODCLFUNC
	} else {
		// The closure is not called, so it is going to stay as closure.
		var body []*Node
		offset := int64(Widthptr)
		for _, v := range xfunc.Func.Cvars.Slice() {
			// cv refers to the field inside of closure OSTRUCTLIT.
			cv := nod(OCLOSUREVAR, nil, nil)

			cv.Type = v.Type
			if !v.Name.Byval() {
				cv.Type = types.NewPtr(v.Type)
			}
			offset = Rnd(offset, int64(cv.Type.Align))
			cv.Xoffset = offset
			offset += cv.Type.Width

			if v.Name.Byval() && v.Type.Width <= int64(2*Widthptr) {
				// If it is a small variable captured by value, downgrade it to PAUTO.
				v.SetClass(PAUTO)
				xfunc.Func.Dcl = append(xfunc.Func.Dcl, v)
				body = append(body, nod(OAS, v, cv))
			} else {
				// Declare variable holding addresses taken from closure
				// and initialize in entry prologue.
				addr := newname(lookup("&" + v.Sym.Name))
				addr.Type = types.NewPtr(v.Type)
				addr.SetClass(PAUTO)
				addr.Name.SetUsed(true)
				addr.Name.Curfn = xfunc
				xfunc.Func.Dcl = append(xfunc.Func.Dcl, addr)
				v.Name.Param.Heapaddr = addr
				if v.Name.Byval() {
					cv = nod(OADDR, cv, nil)
				}
				body = append(body, nod(OAS, addr, cv))
			}
		}

		if len(body) > 0 {
			typecheckslice(body, ctxStmt)
			xfunc.Func.Enter.Set(body)
			xfunc.Func.SetNeedctxt(true)
		}
	}

	lineno = lno
}

// hasemptycvars reports whether closure clo has an
// empty list of captured vars.
func hasemptycvars(clo *Node) bool {
	xfunc := clo.Func.Closure
	return xfunc.Func.Cvars.Len() == 0
}

// closuredebugruntimecheck applies boilerplate checks for debug flags
// and compiling runtime
func closuredebugruntimecheck(clo *Node) {
	if Debug_closure > 0 {
		xfunc := clo.Func.Closure
		if clo.Esc == EscHeap {
			Warnl(clo.Pos, "heap closure, captured vars = %v", xfunc.Func.Cvars)
		} else {
			Warnl(clo.Pos, "stack closure, captured vars = %v", xfunc.Func.Cvars)
		}
	}
	if compiling_runtime && clo.Esc == EscHeap {
		yyerrorl(clo.Pos, "heap-allocated closure, not allowed in runtime.")
	}
}

// closureType returns the struct type used to hold all the information
// needed in the closure for clo (clo must be a OCLOSURE node).
// The address of a variable of the returned type can be cast to a func.
func closureType(clo *Node) *types.Type {
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
	fields := []*Node{
		namedfield(".F", types.Types[TUINTPTR]),
	}
	for _, v := range clo.Func.Closure.Func.Cvars.Slice() {
		typ := v.Type
		if !v.Name.Byval() {
			typ = types.NewPtr(typ)
		}
		fields = append(fields, symfield(v.Sym, typ))
	}
	typ := tostruct(fields)
	typ.SetNoalg(true)
	return typ
}

func walkclosure(clo *Node, init *Nodes) *Node {
	xfunc := clo.Func.Closure

	// If no closure vars, don't bother wrapping.
	if hasemptycvars(clo) {
		if Debug_closure > 0 {
			Warnl(clo.Pos, "closure converted to global")
		}
		return xfunc.Func.Nname
	}
	closuredebugruntimecheck(clo)

	typ := closureType(clo)

	clos := nod(OCOMPLIT, nil, nod(ODEREF, typenod(typ), nil))
	clos.Esc = clo.Esc
	clos.Right.SetImplicit(true)
	clos.List.Set(append([]*Node{nod(OCFUNC, xfunc.Func.Nname, nil)}, clo.Func.Enter.Slice()...))

	// Force type conversion from *struct to the func type.
	clos = convnop(clos, clo.Type)

	// typecheck will insert a PTRLIT node under CONVNOP,
	// tag it with escape analysis result.
	clos.Left.Esc = clo.Esc

	// non-escaping temp to use, if any.
	if x := prealloc[clo]; x != nil {
		if !types.Identical(typ, x.Type) {
			panic("closure type does not match order's assigned type")
		}
		clos.Left.Right = x
		delete(prealloc, clo)
	}

	return walkexpr(clos, init)
}

func typecheckpartialcall(fn *Node, sym *types.Sym) {
	switch fn.Op {
	case ODOTINTER, ODOTMETH:
		break

	default:
		Fatalf("invalid typecheckpartialcall")
	}

	// Create top-level function.
	xfunc := makepartialcall(fn, fn.Type, sym)
	fn.Func = xfunc.Func
	fn.Right = newname(sym)
	fn.Op = OCALLPART
	fn.Type = xfunc.Type
}

func makepartialcall(fn *Node, t0 *types.Type, meth *types.Sym) *Node {
	rcvrtype := fn.Left.Type
	sym := methodSymSuffix(rcvrtype, meth, "-fm")

	if sym.Uniq() {
		return asNode(sym.Def)
	}
	sym.SetUniq(true)

	savecurfn := Curfn
	saveLineNo := lineno
	Curfn = nil

	// Set line number equal to the line number where the method is declared.
	var m *types.Field
	if lookdot0(meth, rcvrtype, &m, false) == 1 && m.Pos.IsKnown() {
		lineno = m.Pos
	}
	// Note: !m.Pos.IsKnown() happens for method expressions where
	// the method is implicitly declared. The Error method of the
	// built-in error type is one such method.  We leave the line
	// number at the use of the method expression in this
	// case. See issue 29389.

	tfn := nod(OTFUNC, nil, nil)
	tfn.List.Set(structargs(t0.Params(), true))
	tfn.Rlist.Set(structargs(t0.Results(), false))

	disableExport(sym)
	xfunc := dclfunc(sym, tfn)
	xfunc.Func.SetDupok(true)
	xfunc.Func.SetNeedctxt(true)

	tfn.Type.SetPkg(t0.Pkg())

	// Declare and initialize variable holding receiver.

	cv := nod(OCLOSUREVAR, nil, nil)
	cv.Type = rcvrtype
	cv.Xoffset = Rnd(int64(Widthptr), int64(cv.Type.Align))

	ptr := newname(lookup(".this"))
	declare(ptr, PAUTO)
	ptr.Name.SetUsed(true)
	var body []*Node
	if rcvrtype.IsPtr() || rcvrtype.IsInterface() {
		ptr.Type = rcvrtype
		body = append(body, nod(OAS, ptr, cv))
	} else {
		ptr.Type = types.NewPtr(rcvrtype)
		body = append(body, nod(OAS, ptr, nod(OADDR, cv, nil)))
	}

	call := nod(OCALL, nodSym(OXDOT, ptr, meth), nil)
	call.List.Set(paramNnames(tfn.Type))
	call.SetIsDDD(tfn.Type.IsVariadic())
	if t0.NumResults() != 0 {
		n := nod(ORETURN, nil, nil)
		n.List.Set1(call)
		call = n
	}
	body = append(body, call)

	xfunc.Nbody.Set(body)
	funcbody()

	xfunc = typecheck(xfunc, ctxStmt)
	sym.Def = asTypesNode(xfunc)
	xtop = append(xtop, xfunc)
	Curfn = savecurfn
	lineno = saveLineNo

	return xfunc
}

// partialCallType returns the struct type used to hold all the information
// needed in the closure for n (n must be a OCALLPART node).
// The address of a variable of the returned type can be cast to a func.
func partialCallType(n *Node) *types.Type {
	t := tostruct([]*Node{
		namedfield("F", types.Types[TUINTPTR]),
		namedfield("R", n.Left.Type),
	})
	t.SetNoalg(true)
	return t
}

func walkpartialcall(n *Node, init *Nodes) *Node {
	// Create closure in the form of a composite literal.
	// For x.M with receiver (x) type T, the generated code looks like:
	//
	//	clos = &struct{F uintptr; R T}{M.T·f, x}
	//
	// Like walkclosure above.

	if n.Left.Type.IsInterface() {
		// Trigger panic for method on nil interface now.
		// Otherwise it happens in the wrapper and is confusing.
		n.Left = cheapexpr(n.Left, init)
		n.Left = walkexpr(n.Left, nil)

		tab := nod(OITAB, n.Left, nil)
		tab = typecheck(tab, ctxExpr)

		c := nod(OCHECKNIL, tab, nil)
		c.SetTypecheck(1)
		init.Append(c)
	}

	typ := partialCallType(n)

	clos := nod(OCOMPLIT, nil, nod(ODEREF, typenod(typ), nil))
	clos.Esc = n.Esc
	clos.Right.SetImplicit(true)
	clos.List.Set2(nod(OCFUNC, n.Func.Nname, nil), n.Left)

	// Force type conversion from *struct to the func type.
	clos = convnop(clos, n.Type)

	// The typecheck inside convnop will insert a PTRLIT node under CONVNOP.
	// Tag it with escape analysis result.
	clos.Left.Esc = n.Esc

	// non-escaping temp to use, if any.
	if x := prealloc[n]; x != nil {
		if !types.Identical(typ, x.Type) {
			panic("partial call type does not match order's assigned type")
		}
		clos.Left.Right = x
		delete(prealloc, n)
	}

	return walkexpr(clos, init)
}
