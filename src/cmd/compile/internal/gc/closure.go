// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/obj"
	"fmt"
)

// function literals aka closures
func closurehdr(ntype *Node) {
	var name *Node
	var a *Node

	n := Nod(OCLOSURE, nil, nil)
	n.Func.Ntype = ntype
	n.Func.Depth = Funcdepth
	n.Func.Outerfunc = Curfn

	funchdr(n)

	// steal ntype's argument names and
	// leave a fresh copy in their place.
	// references to these variables need to
	// refer to the variables in the external
	// function declared below; see walkclosure.
	n.List = ntype.List

	n.Rlist = ntype.Rlist
	ntype.List = nil
	ntype.Rlist = nil
	for l := n.List; l != nil; l = l.Next {
		name = l.N.Left
		if name != nil {
			name = newname(name.Sym)
		}
		a = Nod(ODCLFIELD, name, l.N.Right)
		a.Isddd = l.N.Isddd
		if name != nil {
			name.Isddd = a.Isddd
		}
		ntype.List = list(ntype.List, a)
	}

	for l := n.Rlist; l != nil; l = l.Next {
		name = l.N.Left
		if name != nil {
			name = newname(name.Sym)
		}
		ntype.Rlist = list(ntype.Rlist, Nod(ODCLFIELD, name, l.N.Right))
	}
}

func closurebody(body *NodeList) *Node {
	if body == nil {
		body = list1(Nod(OEMPTY, nil, nil))
	}

	func_ := Curfn
	func_.Nbody = body
	func_.Func.Endlineno = lineno
	funcbody(func_)

	// closure-specific variables are hanging off the
	// ordinary ones in the symbol table; see oldname.
	// unhook them.
	// make the list of pointers for the closure call.
	var v *Node
	for l := func_.Func.Cvars; l != nil; l = l.Next {
		v = l.N
		v.Name.Param.Closure.Name.Param.Closure = v.Name.Param.Outer
		v.Name.Param.Outerexpr = oldname(v.Sym)
	}

	return func_
}

func typecheckclosure(func_ *Node, top int) {
	var n *Node

	for l := func_.Func.Cvars; l != nil; l = l.Next {
		n = l.N.Name.Param.Closure
		if !n.Name.Captured {
			n.Name.Captured = true
			if n.Name.Decldepth == 0 {
				Fatalf("typecheckclosure: var %v does not have decldepth assigned", Nconv(n, obj.FmtShort))
			}

			// Ignore assignments to the variable in straightline code
			// preceding the first capturing by a closure.
			if n.Name.Decldepth == decldepth {
				n.Assigned = false
			}
		}
	}

	for l := func_.Func.Dcl; l != nil; l = l.Next {
		if l.N.Op == ONAME && (l.N.Class == PPARAM || l.N.Class == PPARAMOUT) {
			l.N.Name.Decldepth = 1
		}
	}

	oldfn := Curfn
	typecheck(&func_.Func.Ntype, Etype)
	func_.Type = func_.Func.Ntype.Type
	func_.Func.Top = top

	// Type check the body now, but only if we're inside a function.
	// At top level (in a variable initialization: curfn==nil) we're not
	// ready to type check code yet; we'll check it later, because the
	// underlying closure function we create is added to xtop.
	if Curfn != nil && func_.Type != nil {
		Curfn = func_
		olddd := decldepth
		decldepth = 1
		typechecklist(func_.Nbody, Etop)
		decldepth = olddd
		Curfn = oldfn
	}

	// Create top-level function
	xtop = list(xtop, makeclosure(func_))
}

// closurename returns name for OCLOSURE n.
// It is not as simple as it ought to be, because we typecheck nested closures
// starting from the innermost one. So when we check the inner closure,
// we don't yet have name for the outer closure. This function uses recursion
// to generate names all the way up if necessary.

var closurename_closgen int

func closurename(n *Node) *Sym {
	if n.Sym != nil {
		return n.Sym
	}
	gen := 0
	outer := ""
	prefix := ""
	if n.Func.Outerfunc == nil {
		// Global closure.
		outer = "glob"

		prefix = "func"
		closurename_closgen++
		gen = closurename_closgen
	} else if n.Func.Outerfunc.Op == ODCLFUNC {
		// The outermost closure inside of a named function.
		outer = n.Func.Outerfunc.Func.Nname.Sym.Name

		prefix = "func"

		// Yes, functions can be named _.
		// Can't use function closgen in such case,
		// because it would lead to name clashes.
		if !isblank(n.Func.Outerfunc.Func.Nname) {
			n.Func.Outerfunc.Func.Closgen++
			gen = n.Func.Outerfunc.Func.Closgen
		} else {
			closurename_closgen++
			gen = closurename_closgen
		}
	} else if n.Func.Outerfunc.Op == OCLOSURE {
		// Nested closure, recurse.
		outer = closurename(n.Func.Outerfunc).Name

		prefix = ""
		n.Func.Outerfunc.Func.Closgen++
		gen = n.Func.Outerfunc.Func.Closgen
	} else {
		Fatalf("closurename called for %v", Nconv(n, obj.FmtShort))
	}
	n.Sym = Lookupf("%s.%s%d", outer, prefix, gen)
	return n.Sym
}

func makeclosure(func_ *Node) *Node {
	// wrap body in external function
	// that begins by reading closure parameters.
	xtype := Nod(OTFUNC, nil, nil)

	xtype.List = func_.List
	xtype.Rlist = func_.Rlist

	// create the function
	xfunc := Nod(ODCLFUNC, nil, nil)

	xfunc.Func.Nname = newfuncname(closurename(func_))
	xfunc.Func.Nname.Sym.Flags |= SymExported // disable export
	xfunc.Func.Nname.Name.Param.Ntype = xtype
	xfunc.Func.Nname.Name.Defn = xfunc
	declare(xfunc.Func.Nname, PFUNC)
	xfunc.Func.Nname.Name.Funcdepth = func_.Func.Depth
	xfunc.Func.Depth = func_.Func.Depth
	xfunc.Func.Endlineno = func_.Func.Endlineno
	makefuncsym(xfunc.Func.Nname.Sym)

	xfunc.Nbody = func_.Nbody
	xfunc.Func.Dcl = concat(func_.Func.Dcl, xfunc.Func.Dcl)
	if xfunc.Nbody == nil {
		Fatalf("empty body - won't generate any code")
	}
	typecheck(&xfunc, Etop)

	xfunc.Func.Closure = func_
	func_.Func.Closure = xfunc

	func_.Nbody = nil
	func_.List = nil
	func_.Rlist = nil

	return xfunc
}

// capturevars is called in a separate phase after all typechecking is done.
// It decides whether each variable captured by a closure should be captured
// by value or by reference.
// We use value capturing for values <= 128 bytes that are never reassigned
// after capturing (effectively constant).
func capturevars(xfunc *Node) {
	var v *Node
	var outer *Node

	lno := int(lineno)
	lineno = xfunc.Lineno

	func_ := xfunc.Func.Closure
	func_.Func.Enter = nil
	for l := func_.Func.Cvars; l != nil; l = l.Next {
		v = l.N
		if v.Type == nil {
			// if v->type is nil, it means v looked like it was
			// going to be used in the closure but wasn't.
			// this happens because when parsing a, b, c := f()
			// the a, b, c gets parsed as references to older
			// a, b, c before the parser figures out this is a
			// declaration.
			v.Op = OXXX

			continue
		}

		// type check the & of closed variables outside the closure,
		// so that the outer frame also grabs them and knows they escape.
		dowidth(v.Type)

		outer = v.Name.Param.Outerexpr
		v.Name.Param.Outerexpr = nil

		// out parameters will be assigned to implicitly upon return.
		if outer.Class != PPARAMOUT && !v.Name.Param.Closure.Addrtaken && !v.Name.Param.Closure.Assigned && v.Type.Width <= 128 {
			v.Name.Byval = true
		} else {
			v.Name.Param.Closure.Addrtaken = true
			outer = Nod(OADDR, outer, nil)
		}

		if Debug['m'] > 1 {
			var name *Sym
			if v.Name.Curfn != nil && v.Name.Curfn.Func.Nname != nil {
				name = v.Name.Curfn.Func.Nname.Sym
			}
			how := "ref"
			if v.Name.Byval {
				how = "value"
			}
			Warnl(int(v.Lineno), "%v capturing by %s: %v (addr=%v assign=%v width=%d)", name, how, v.Sym, v.Name.Param.Closure.Addrtaken, v.Name.Param.Closure.Assigned, int32(v.Type.Width))
		}

		typecheck(&outer, Erv)
		func_.Func.Enter = list(func_.Func.Enter, outer)
	}

	lineno = int32(lno)
}

// transformclosure is called in a separate phase after escape analysis.
// It transform closure bodies to properly reference captured variables.
func transformclosure(xfunc *Node) {
	lno := int(lineno)
	lineno = xfunc.Lineno
	func_ := xfunc.Func.Closure

	if func_.Func.Top&Ecall != 0 {
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
		//	func(a int, byval int, &byref *int) {
		//		println(byval)
		//		(*&byref)++
		//	}(byval, &byref, 42)

		// f is ONAME of the actual function.
		f := xfunc.Func.Nname

		// Get pointer to input arguments.
		// We are going to insert captured variables before input args.
		param := &getinargx(f.Type).Type
		original_args := *param // old input args
		original_dcl := xfunc.Func.Dcl
		xfunc.Func.Dcl = nil

		var v *Node
		var addr *Node
		var fld *Type
		for l := func_.Func.Cvars; l != nil; l = l.Next {
			v = l.N
			if v.Op == OXXX {
				continue
			}
			fld = typ(TFIELD)
			fld.Funarg = true
			if v.Name.Byval {
				// If v is captured by value, we merely downgrade it to PPARAM.
				v.Class = PPARAM

				v.Ullman = 1
				fld.Nname = v
			} else {
				// If v of type T is captured by reference,
				// we introduce function param &v *T
				// and v remains PPARAMREF with &v heapaddr
				// (accesses will implicitly deref &v).
				addr = newname(Lookupf("&%s", v.Sym.Name))
				addr.Type = Ptrto(v.Type)
				addr.Class = PPARAM
				v.Name.Heapaddr = addr
				fld.Nname = addr
			}

			fld.Type = fld.Nname.Type
			fld.Sym = fld.Nname.Sym

			// Declare the new param and add it the first part of the input arguments.
			xfunc.Func.Dcl = list(xfunc.Func.Dcl, fld.Nname)

			*param = fld
			param = &fld.Down
		}
		*param = original_args
		xfunc.Func.Dcl = concat(xfunc.Func.Dcl, original_dcl)

		// Recalculate param offsets.
		if f.Type.Width > 0 {
			Fatalf("transformclosure: width is already calculated")
		}
		dowidth(f.Type)
		xfunc.Type = f.Type // update type of ODCLFUNC
	} else {
		// The closure is not called, so it is going to stay as closure.
		nvar := 0

		var body *NodeList
		offset := int64(Widthptr)
		var addr *Node
		var v *Node
		var cv *Node
		for l := func_.Func.Cvars; l != nil; l = l.Next {
			v = l.N
			if v.Op == OXXX {
				continue
			}
			nvar++

			// cv refers to the field inside of closure OSTRUCTLIT.
			cv = Nod(OCLOSUREVAR, nil, nil)

			cv.Type = v.Type
			if !v.Name.Byval {
				cv.Type = Ptrto(v.Type)
			}
			offset = Rnd(offset, int64(cv.Type.Align))
			cv.Xoffset = offset
			offset += cv.Type.Width

			if v.Name.Byval && v.Type.Width <= int64(2*Widthptr) {
				// If it is a small variable captured by value, downgrade it to PAUTO.
				v.Class = PAUTO
				v.Ullman = 1
				xfunc.Func.Dcl = list(xfunc.Func.Dcl, v)
				body = list(body, Nod(OAS, v, cv))
			} else {
				// Declare variable holding addresses taken from closure
				// and initialize in entry prologue.
				addr = newname(Lookupf("&%s", v.Sym.Name))
				addr.Name.Param.Ntype = Nod(OIND, typenod(v.Type), nil)
				addr.Class = PAUTO
				addr.Used = true
				addr.Name.Curfn = xfunc
				xfunc.Func.Dcl = list(xfunc.Func.Dcl, addr)
				v.Name.Heapaddr = addr
				if v.Name.Byval {
					cv = Nod(OADDR, cv, nil)
				}
				body = list(body, Nod(OAS, addr, cv))
			}
		}

		typechecklist(body, Etop)
		walkstmtlist(body)
		xfunc.Func.Enter = body
		xfunc.Func.Needctxt = nvar > 0
	}

	lineno = int32(lno)
}

func walkclosure(func_ *Node, init **NodeList) *Node {
	// If no closure vars, don't bother wrapping.
	if func_.Func.Cvars == nil {
		return func_.Func.Closure.Func.Nname
	}

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

	typ := Nod(OTSTRUCT, nil, nil)

	typ.List = list1(Nod(ODCLFIELD, newname(Lookup(".F")), typenod(Types[TUINTPTR])))
	var typ1 *Node
	var v *Node
	for l := func_.Func.Cvars; l != nil; l = l.Next {
		v = l.N
		if v.Op == OXXX {
			continue
		}
		typ1 = typenod(v.Type)
		if !v.Name.Byval {
			typ1 = Nod(OIND, typ1, nil)
		}
		typ.List = list(typ.List, Nod(ODCLFIELD, newname(v.Sym), typ1))
	}

	clos := Nod(OCOMPLIT, nil, Nod(OIND, typ, nil))
	clos.Esc = func_.Esc
	clos.Right.Implicit = true
	clos.List = concat(list1(Nod(OCFUNC, func_.Func.Closure.Func.Nname, nil)), func_.Func.Enter)

	// Force type conversion from *struct to the func type.
	clos = Nod(OCONVNOP, clos, nil)

	clos.Type = func_.Type

	typecheck(&clos, Erv)

	// typecheck will insert a PTRLIT node under CONVNOP,
	// tag it with escape analysis result.
	clos.Left.Esc = func_.Esc

	// non-escaping temp to use, if any.
	// orderexpr did not compute the type; fill it in now.
	if x := prealloc[func_]; x != nil {
		x.Type = clos.Left.Left.Type
		x.Orig.Type = x.Type
		clos.Left.Right = x
		delete(prealloc, func_)
	}

	walkexpr(&clos, init)

	return clos
}

func typecheckpartialcall(fn *Node, sym *Node) {
	switch fn.Op {
	case ODOTINTER, ODOTMETH:
		break

	default:
		Fatalf("invalid typecheckpartialcall")
	}

	// Create top-level function.
	xfunc := makepartialcall(fn, fn.Type, sym)
	fn.Func = xfunc.Func
	fn.Right = sym
	fn.Op = OCALLPART
	fn.Type = xfunc.Type
}

var makepartialcall_gopkg *Pkg

func makepartialcall(fn *Node, t0 *Type, meth *Node) *Node {
	var p string

	rcvrtype := fn.Left.Type
	if exportname(meth.Sym.Name) {
		p = fmt.Sprintf("(%v).%s-fm", Tconv(rcvrtype, obj.FmtLeft|obj.FmtShort), meth.Sym.Name)
	} else {
		p = fmt.Sprintf("(%v).(%v)-fm", Tconv(rcvrtype, obj.FmtLeft|obj.FmtShort), Sconv(meth.Sym, obj.FmtLeft))
	}
	basetype := rcvrtype
	if Isptr[rcvrtype.Etype] {
		basetype = basetype.Type
	}
	if basetype.Etype != TINTER && basetype.Sym == nil {
		Fatalf("missing base type for %v", rcvrtype)
	}

	var spkg *Pkg
	if basetype.Sym != nil {
		spkg = basetype.Sym.Pkg
	}
	if spkg == nil {
		if makepartialcall_gopkg == nil {
			makepartialcall_gopkg = mkpkg("go")
		}
		spkg = makepartialcall_gopkg
	}

	sym := Pkglookup(p, spkg)

	if sym.Flags&SymUniq != 0 {
		return sym.Def
	}
	sym.Flags |= SymUniq

	savecurfn := Curfn
	Curfn = nil

	xtype := Nod(OTFUNC, nil, nil)
	i := 0
	var l *NodeList
	var callargs *NodeList
	ddd := false
	xfunc := Nod(ODCLFUNC, nil, nil)
	Curfn = xfunc
	var fld *Node
	var n *Node
	for t := getinargx(t0).Type; t != nil; t = t.Down {
		n = newname(Lookupf("a%d", i))
		i++
		n.Class = PPARAM
		xfunc.Func.Dcl = list(xfunc.Func.Dcl, n)
		callargs = list(callargs, n)
		fld = Nod(ODCLFIELD, n, typenod(t.Type))
		if t.Isddd {
			fld.Isddd = true
			ddd = true
		}

		l = list(l, fld)
	}

	xtype.List = l
	i = 0
	l = nil
	var retargs *NodeList
	for t := getoutargx(t0).Type; t != nil; t = t.Down {
		n = newname(Lookupf("r%d", i))
		i++
		n.Class = PPARAMOUT
		xfunc.Func.Dcl = list(xfunc.Func.Dcl, n)
		retargs = list(retargs, n)
		l = list(l, Nod(ODCLFIELD, n, typenod(t.Type)))
	}

	xtype.Rlist = l

	xfunc.Func.Dupok = true
	xfunc.Func.Nname = newfuncname(sym)
	xfunc.Func.Nname.Sym.Flags |= SymExported // disable export
	xfunc.Func.Nname.Name.Param.Ntype = xtype
	xfunc.Func.Nname.Name.Defn = xfunc
	declare(xfunc.Func.Nname, PFUNC)

	// Declare and initialize variable holding receiver.

	xfunc.Func.Needctxt = true
	cv := Nod(OCLOSUREVAR, nil, nil)
	cv.Xoffset = int64(Widthptr)
	cv.Type = rcvrtype
	if int(cv.Type.Align) > Widthptr {
		cv.Xoffset = int64(cv.Type.Align)
	}
	ptr := Nod(ONAME, nil, nil)
	ptr.Sym = Lookup("rcvr")
	ptr.Class = PAUTO
	ptr.Addable = true
	ptr.Ullman = 1
	ptr.Used = true
	ptr.Name.Curfn = xfunc
	xfunc.Func.Dcl = list(xfunc.Func.Dcl, ptr)
	var body *NodeList
	if Isptr[rcvrtype.Etype] || Isinter(rcvrtype) {
		ptr.Name.Param.Ntype = typenod(rcvrtype)
		body = list(body, Nod(OAS, ptr, cv))
	} else {
		ptr.Name.Param.Ntype = typenod(Ptrto(rcvrtype))
		body = list(body, Nod(OAS, ptr, Nod(OADDR, cv, nil)))
	}

	call := Nod(OCALL, Nod(OXDOT, ptr, meth), nil)
	call.List = callargs
	call.Isddd = ddd
	if t0.Outtuple == 0 {
		body = list(body, call)
	} else {
		n := Nod(OAS2, nil, nil)
		n.List = retargs
		n.Rlist = list1(call)
		body = list(body, n)
		n = Nod(ORETURN, nil, nil)
		body = list(body, n)
	}

	xfunc.Nbody = body

	typecheck(&xfunc, Etop)
	sym.Def = xfunc
	xtop = list(xtop, xfunc)
	Curfn = savecurfn

	return xfunc
}

func walkpartialcall(n *Node, init **NodeList) *Node {
	// Create closure in the form of a composite literal.
	// For x.M with receiver (x) type T, the generated code looks like:
	//
	//	clos = &struct{F uintptr; R T}{M.T·f, x}
	//
	// Like walkclosure above.

	if Isinter(n.Left.Type) {
		// Trigger panic for method on nil interface now.
		// Otherwise it happens in the wrapper and is confusing.
		n.Left = cheapexpr(n.Left, init)

		checknil(n.Left, init)
	}

	typ := Nod(OTSTRUCT, nil, nil)
	typ.List = list1(Nod(ODCLFIELD, newname(Lookup("F")), typenod(Types[TUINTPTR])))
	typ.List = list(typ.List, Nod(ODCLFIELD, newname(Lookup("R")), typenod(n.Left.Type)))

	clos := Nod(OCOMPLIT, nil, Nod(OIND, typ, nil))
	clos.Esc = n.Esc
	clos.Right.Implicit = true
	clos.List = list1(Nod(OCFUNC, n.Func.Nname, nil))
	clos.List = list(clos.List, n.Left)

	// Force type conversion from *struct to the func type.
	clos = Nod(OCONVNOP, clos, nil)

	clos.Type = n.Type

	typecheck(&clos, Erv)

	// typecheck will insert a PTRLIT node under CONVNOP,
	// tag it with escape analysis result.
	clos.Left.Esc = n.Esc

	// non-escaping temp to use, if any.
	// orderexpr did not compute the type; fill it in now.
	if x := prealloc[n]; x != nil {
		x.Type = clos.Left.Left.Type
		x.Orig.Type = x.Type
		clos.Left.Right = x
		delete(prealloc, n)
	}

	walkexpr(&clos, init)

	return clos
}
