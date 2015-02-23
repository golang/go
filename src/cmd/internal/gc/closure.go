// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/obj"
	"fmt"
)

/*
 * function literals aka closures
 */
func closurehdr(ntype *Node) {
	var name *Node
	var a *Node

	n := Nod(OCLOSURE, nil, nil)
	n.Ntype = ntype
	n.Funcdepth = Funcdepth
	n.Outerfunc = Curfn

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
	func_.Endlineno = lineno
	funcbody(func_)

	// closure-specific variables are hanging off the
	// ordinary ones in the symbol table; see oldname.
	// unhook them.
	// make the list of pointers for the closure call.
	var v *Node
	for l := func_.Cvars; l != nil; l = l.Next {
		v = l.N
		v.Closure.Closure = v.Outer
		v.Outerexpr = oldname(v.Sym)
	}

	return func_
}

func typecheckclosure(func_ *Node, top int) {
	var n *Node

	for l := func_.Cvars; l != nil; l = l.Next {
		n = l.N.Closure
		if n.Captured == 0 {
			n.Captured = 1
			if n.Decldepth == 0 {
				Fatal("typecheckclosure: var %v does not have decldepth assigned", Nconv(n, obj.FmtShort))
			}

			// Ignore assignments to the variable in straightline code
			// preceding the first capturing by a closure.
			if n.Decldepth == decldepth {
				n.Assigned = 0
			}
		}
	}

	for l := func_.Dcl; l != nil; l = l.Next {
		if l.N.Op == ONAME && (l.N.Class == PPARAM || l.N.Class == PPARAMOUT) {
			l.N.Decldepth = 1
		}
	}

	oldfn := Curfn
	typecheck(&func_.Ntype, Etype)
	func_.Type = func_.Ntype.Type
	func_.Top = top

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
	if n.Outerfunc == nil {
		// Global closure.
		outer = "glob"

		prefix = "func"
		closurename_closgen++
		gen = closurename_closgen
	} else if n.Outerfunc.Op == ODCLFUNC {
		// The outermost closure inside of a named function.
		outer = n.Outerfunc.Nname.Sym.Name

		prefix = "func"

		// Yes, functions can be named _.
		// Can't use function closgen in such case,
		// because it would lead to name clashes.
		if !isblank(n.Outerfunc.Nname) {
			n.Outerfunc.Closgen++
			gen = n.Outerfunc.Closgen
		} else {
			closurename_closgen++
			gen = closurename_closgen
		}
	} else if n.Outerfunc.Op == OCLOSURE {
		// Nested closure, recurse.
		outer = closurename(n.Outerfunc).Name

		prefix = ""
		n.Outerfunc.Closgen++
		gen = n.Outerfunc.Closgen
	} else {
		Fatal("closurename called for %v", Nconv(n, obj.FmtShort))
	}
	namebuf = fmt.Sprintf("%s.%s%d", outer, prefix, gen)
	n.Sym = Lookup(namebuf)
	return n.Sym
}

func makeclosure(func_ *Node) *Node {
	/*
	 * wrap body in external function
	 * that begins by reading closure parameters.
	 */
	xtype := Nod(OTFUNC, nil, nil)

	xtype.List = func_.List
	xtype.Rlist = func_.Rlist

	// create the function
	xfunc := Nod(ODCLFUNC, nil, nil)

	xfunc.Nname = newname(closurename(func_))
	xfunc.Nname.Sym.Flags |= SymExported // disable export
	xfunc.Nname.Ntype = xtype
	xfunc.Nname.Defn = xfunc
	declare(xfunc.Nname, PFUNC)
	xfunc.Nname.Funcdepth = func_.Funcdepth
	xfunc.Funcdepth = func_.Funcdepth
	xfunc.Endlineno = func_.Endlineno

	xfunc.Nbody = func_.Nbody
	xfunc.Dcl = concat(func_.Dcl, xfunc.Dcl)
	if xfunc.Nbody == nil {
		Fatal("empty body - won't generate any code")
	}
	typecheck(&xfunc, Etop)

	xfunc.Closure = func_
	func_.Closure = xfunc

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

	func_ := xfunc.Closure
	func_.Enter = nil
	for l := func_.Cvars; l != nil; l = l.Next {
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

		outer = v.Outerexpr
		v.Outerexpr = nil

		// out parameters will be assigned to implicitly upon return.
		if outer.Class != PPARAMOUT && v.Closure.Addrtaken == 0 && v.Closure.Assigned == 0 && v.Type.Width <= 128 {
			v.Byval = 1
		} else {
			v.Closure.Addrtaken = 1
			outer = Nod(OADDR, outer, nil)
		}

		if Debug['m'] > 1 {
			name := (*Sym)(nil)
			if v.Curfn != nil && v.Curfn.Nname != nil {
				name = v.Curfn.Nname.Sym
			}
			how := "ref"
			if v.Byval != 0 {
				how = "value"
			}
			Warnl(int(v.Lineno), "%v capturing by %s: %v (addr=%d assign=%d width=%d)", Sconv(name, 0), how, Sconv(v.Sym, 0), v.Closure.Addrtaken, v.Closure.Assigned, int32(v.Type.Width))
		}

		typecheck(&outer, Erv)
		func_.Enter = list(func_.Enter, outer)
	}

	lineno = int32(lno)
}

// transformclosure is called in a separate phase after escape analysis.
// It transform closure bodies to properly reference captured variables.
func transformclosure(xfunc *Node) {
	lno := int(lineno)
	lineno = xfunc.Lineno
	func_ := xfunc.Closure

	if func_.Top&Ecall != 0 {
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
		//	}(42, byval, &byref)

		// f is ONAME of the actual function.
		f := xfunc.Nname

		// Get pointer to input arguments and rewind to the end.
		// We are going to append captured variables to input args.
		param := &getinargx(f.Type).Type

		for ; *param != nil; param = &(*param).Down {
		}
		var v *Node
		var addr *Node
		var fld *Type
		for l := func_.Cvars; l != nil; l = l.Next {
			v = l.N
			if v.Op == OXXX {
				continue
			}
			fld = typ(TFIELD)
			fld.Funarg = 1
			if v.Byval != 0 {
				// If v is captured by value, we merely downgrade it to PPARAM.
				v.Class = PPARAM

				v.Ullman = 1
				fld.Nname = v
			} else {
				// If v of type T is captured by reference,
				// we introduce function param &v *T
				// and v remains PPARAMREF with &v heapaddr
				// (accesses will implicitly deref &v).
				namebuf = fmt.Sprintf("&%s", v.Sym.Name)

				addr = newname(Lookup(namebuf))
				addr.Type = Ptrto(v.Type)
				addr.Class = PPARAM
				v.Heapaddr = addr
				fld.Nname = addr
			}

			fld.Type = fld.Nname.Type
			fld.Sym = fld.Nname.Sym

			// Declare the new param and append it to input arguments.
			xfunc.Dcl = list(xfunc.Dcl, fld.Nname)

			*param = fld
			param = &fld.Down
		}

		// Recalculate param offsets.
		if f.Type.Width > 0 {
			Fatal("transformclosure: width is already calculated")
		}
		dowidth(f.Type)
		xfunc.Type = f.Type // update type of ODCLFUNC
	} else {
		// The closure is not called, so it is going to stay as closure.
		nvar := 0

		body := (*NodeList)(nil)
		offset := int64(Widthptr)
		var addr *Node
		var v *Node
		var cv *Node
		for l := func_.Cvars; l != nil; l = l.Next {
			v = l.N
			if v.Op == OXXX {
				continue
			}
			nvar++

			// cv refers to the field inside of closure OSTRUCTLIT.
			cv = Nod(OCLOSUREVAR, nil, nil)

			cv.Type = v.Type
			if v.Byval == 0 {
				cv.Type = Ptrto(v.Type)
			}
			offset = Rnd(offset, int64(cv.Type.Align))
			cv.Xoffset = offset
			offset += cv.Type.Width

			if v.Byval != 0 && v.Type.Width <= int64(2*Widthptr) && Thearch.Thechar == '6' {
				//  If it is a small variable captured by value, downgrade it to PAUTO.
				// This optimization is currently enabled only for amd64, see:
				// https://github.com/golang/go/issues/9865
				v.Class = PAUTO

				v.Ullman = 1
				xfunc.Dcl = list(xfunc.Dcl, v)
				body = list(body, Nod(OAS, v, cv))
			} else {
				// Declare variable holding addresses taken from closure
				// and initialize in entry prologue.
				namebuf = fmt.Sprintf("&%s", v.Sym.Name)

				addr = newname(Lookup(namebuf))
				addr.Ntype = Nod(OIND, typenod(v.Type), nil)
				addr.Class = PAUTO
				addr.Used = 1
				addr.Curfn = xfunc
				xfunc.Dcl = list(xfunc.Dcl, addr)
				v.Heapaddr = addr
				if v.Byval != 0 {
					cv = Nod(OADDR, cv, nil)
				}
				body = list(body, Nod(OAS, addr, cv))
			}
		}

		typechecklist(body, Etop)
		walkstmtlist(body)
		xfunc.Enter = body
		xfunc.Needctxt = nvar > 0
	}

	lineno = int32(lno)
}

func walkclosure(func_ *Node, init **NodeList) *Node {
	// If no closure vars, don't bother wrapping.
	if func_.Cvars == nil {
		return func_.Closure.Nname
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
	for l := func_.Cvars; l != nil; l = l.Next {
		v = l.N
		if v.Op == OXXX {
			continue
		}
		typ1 = typenod(v.Type)
		if v.Byval == 0 {
			typ1 = Nod(OIND, typ1, nil)
		}
		typ.List = list(typ.List, Nod(ODCLFIELD, newname(v.Sym), typ1))
	}

	clos := Nod(OCOMPLIT, nil, Nod(OIND, typ, nil))
	clos.Esc = func_.Esc
	clos.Right.Implicit = 1
	clos.List = concat(list1(Nod(OCFUNC, func_.Closure.Nname, nil)), func_.Enter)

	// Force type conversion from *struct to the func type.
	clos = Nod(OCONVNOP, clos, nil)

	clos.Type = func_.Type

	typecheck(&clos, Erv)

	// typecheck will insert a PTRLIT node under CONVNOP,
	// tag it with escape analysis result.
	clos.Left.Esc = func_.Esc

	// non-escaping temp to use, if any.
	// orderexpr did not compute the type; fill it in now.
	if func_.Alloc != nil {
		func_.Alloc.Type = clos.Left.Left.Type
		func_.Alloc.Orig.Type = func_.Alloc.Type
		clos.Left.Right = func_.Alloc
		func_.Alloc = nil
	}

	walkexpr(&clos, init)

	return clos
}

func typecheckpartialcall(fn *Node, sym *Node) {
	switch fn.Op {
	case ODOTINTER,
		ODOTMETH:
		break

	default:
		Fatal("invalid typecheckpartialcall")
	}

	// Create top-level function.
	fn.Nname = makepartialcall(fn, fn.Type, sym)

	fn.Right = sym
	fn.Op = OCALLPART
	fn.Type = fn.Nname.Type
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
	if Isptr[rcvrtype.Etype] != 0 {
		basetype = basetype.Type
	}
	if basetype.Etype != TINTER && basetype.Sym == nil {
		Fatal("missing base type for %v", Tconv(rcvrtype, 0))
	}

	spkg := (*Pkg)(nil)
	if basetype.Sym != nil {
		spkg = basetype.Sym.Pkg
	}
	if spkg == nil {
		if makepartialcall_gopkg == nil {
			makepartialcall_gopkg = mkpkg(newstrlit("go"))
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
	l := (*NodeList)(nil)
	callargs := (*NodeList)(nil)
	ddd := 0
	xfunc := Nod(ODCLFUNC, nil, nil)
	Curfn = xfunc
	var fld *Node
	var n *Node
	for t := getinargx(t0).Type; t != nil; t = t.Down {
		namebuf = fmt.Sprintf("a%d", i)
		i++
		n = newname(Lookup(namebuf))
		n.Class = PPARAM
		xfunc.Dcl = list(xfunc.Dcl, n)
		callargs = list(callargs, n)
		fld = Nod(ODCLFIELD, n, typenod(t.Type))
		if t.Isddd != 0 {
			fld.Isddd = 1
			ddd = 1
		}

		l = list(l, fld)
	}

	xtype.List = l
	i = 0
	l = nil
	retargs := (*NodeList)(nil)
	for t := getoutargx(t0).Type; t != nil; t = t.Down {
		namebuf = fmt.Sprintf("r%d", i)
		i++
		n = newname(Lookup(namebuf))
		n.Class = PPARAMOUT
		xfunc.Dcl = list(xfunc.Dcl, n)
		retargs = list(retargs, n)
		l = list(l, Nod(ODCLFIELD, n, typenod(t.Type)))
	}

	xtype.Rlist = l

	xfunc.Dupok = 1
	xfunc.Nname = newname(sym)
	xfunc.Nname.Sym.Flags |= SymExported // disable export
	xfunc.Nname.Ntype = xtype
	xfunc.Nname.Defn = xfunc
	declare(xfunc.Nname, PFUNC)

	// Declare and initialize variable holding receiver.
	body := (*NodeList)(nil)

	xfunc.Needctxt = true
	cv := Nod(OCLOSUREVAR, nil, nil)
	cv.Xoffset = int64(Widthptr)
	cv.Type = rcvrtype
	if int(cv.Type.Align) > Widthptr {
		cv.Xoffset = int64(cv.Type.Align)
	}
	ptr := Nod(ONAME, nil, nil)
	ptr.Sym = Lookup("rcvr")
	ptr.Class = PAUTO
	ptr.Addable = 1
	ptr.Ullman = 1
	ptr.Used = 1
	ptr.Curfn = xfunc
	xfunc.Dcl = list(xfunc.Dcl, ptr)
	if Isptr[rcvrtype.Etype] != 0 || Isinter(rcvrtype) {
		ptr.Ntype = typenod(rcvrtype)
		body = list(body, Nod(OAS, ptr, cv))
	} else {
		ptr.Ntype = typenod(Ptrto(rcvrtype))
		body = list(body, Nod(OAS, ptr, Nod(OADDR, cv, nil)))
	}

	call := Nod(OCALL, Nod(OXDOT, ptr, meth), nil)
	call.List = callargs
	call.Isddd = uint8(ddd)
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
	clos.Right.Implicit = 1
	clos.List = list1(Nod(OCFUNC, n.Nname.Nname, nil))
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
	if n.Alloc != nil {
		n.Alloc.Type = clos.Left.Left.Type
		n.Alloc.Orig.Type = n.Alloc.Type
		clos.Left.Right = n.Alloc
		n.Alloc = nil
	}

	walkexpr(&clos, init)

	return clos
}
