// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/obj"
	"fmt"
	"strings"
)

func dflag() bool {
	if Debug['d'] == 0 {
		return false
	}
	if Debug['y'] != 0 {
		return true
	}
	if incannedimport != 0 {
		return false
	}
	return true
}

// declaration stack & operations
func dcopy(a *Sym, b *Sym) {
	a.Pkg = b.Pkg
	a.Name = b.Name
	a.Def = b.Def
	a.Block = b.Block
	a.Lastlineno = b.Lastlineno
}

func push() *Sym {
	d := new(Sym)
	d.Lastlineno = lineno
	d.Link = dclstack
	dclstack = d
	return d
}

func pushdcl(s *Sym) *Sym {
	d := push()
	dcopy(d, s)
	if dflag() {
		fmt.Printf("\t%v push %v %p\n", Ctxt.Line(int(lineno)), s, s.Def)
	}
	return d
}

func popdcl() {
	var d *Sym
	var s *Sym
	var lno int

	//	if(dflag())
	//		print("revert\n");

	for d = dclstack; d != nil; d = d.Link {
		if d.Name == "" {
			break
		}
		s = Pkglookup(d.Name, d.Pkg)
		lno = int(s.Lastlineno)
		dcopy(s, d)
		d.Lastlineno = int32(lno)
		if dflag() {
			fmt.Printf("\t%v pop %v %p\n", Ctxt.Line(int(lineno)), s, s.Def)
		}
	}

	if d == nil {
		Fatalf("popdcl: no mark")
	}
	dclstack = d.Link
	block = d.Block
}

func markdcl() {
	d := push()
	d.Name = "" // used as a mark in fifo
	d.Block = block

	blockgen++
	block = blockgen
}

//	if(dflag())
//		print("markdcl\n");
func dumpdcl(st string) {
	var s *Sym

	i := 0
	for d := dclstack; d != nil; d = d.Link {
		i++
		fmt.Printf("    %.2d %p", i, d)
		if d.Name == "" {
			fmt.Printf("\n")
			continue
		}

		fmt.Printf(" '%s'", d.Name)
		s = Pkglookup(d.Name, d.Pkg)
		fmt.Printf(" %v\n", s)
	}
}

func testdclstack() {
	for d := dclstack; d != nil; d = d.Link {
		if d.Name == "" {
			if nerrors != 0 {
				errorexit()
			}
			Yyerror("mark left on the stack")
			continue
		}
	}
}

func redeclare(s *Sym, where string) {
	if s.Lastlineno == 0 {
		var tmp string
		if s.Origpkg != nil {
			tmp = s.Origpkg.Path
		} else {
			tmp = s.Pkg.Path
		}
		pkgstr := tmp
		Yyerror("%v redeclared %s\n"+"\tprevious declaration during import %q", s, where, pkgstr)
	} else {
		line1 := parserline()
		line2 := int(s.Lastlineno)

		// When an import and a declaration collide in separate files,
		// present the import as the "redeclared", because the declaration
		// is visible where the import is, but not vice versa.
		// See issue 4510.
		if s.Def == nil {
			line2 = line1
			line1 = int(s.Lastlineno)
		}

		yyerrorl(int(line1), "%v redeclared %s\n"+"\tprevious declaration at %v", s, where, Ctxt.Line(line2))
	}
}

var vargen int

// declare individual names - var, typ, const

var declare_typegen int

func declare(n *Node, ctxt Class) {
	if ctxt == PDISCARD {
		return
	}

	if isblank(n) {
		return
	}

	if n.Name == nil {
		// named OLITERAL needs Name; most OLITERALs don't.
		n.Name = new(Name)
	}
	n.Lineno = int32(parserline())
	s := n.Sym

	// kludgy: typecheckok means we're past parsing.  Eg genwrapper may declare out of package names later.
	if importpkg == nil && !typecheckok && s.Pkg != localpkg {
		Yyerror("cannot declare name %v", s)
	}

	if ctxt == PEXTERN && s.Name == "init" {
		Yyerror("cannot declare init - must be func")
	}

	gen := 0
	if ctxt == PEXTERN {
		externdcl = append(externdcl, n)
		if dflag() {
			fmt.Printf("\t%v global decl %v %p\n", Ctxt.Line(int(lineno)), s, n)
		}
	} else {
		if Curfn == nil && ctxt == PAUTO {
			Fatalf("automatic outside function")
		}
		if Curfn != nil {
			Curfn.Func.Dcl = list(Curfn.Func.Dcl, n)
		}
		if n.Op == OTYPE {
			declare_typegen++
			gen = declare_typegen
		} else if n.Op == ONAME && ctxt == PAUTO && !strings.Contains(s.Name, "·") {
			vargen++
			gen = vargen
		}
		pushdcl(s)
		n.Name.Curfn = Curfn
	}

	if ctxt == PAUTO {
		n.Xoffset = 0
	}

	if s.Block == block {
		// functype will print errors about duplicate function arguments.
		// Don't repeat the error here.
		if ctxt != PPARAM && ctxt != PPARAMOUT {
			redeclare(s, "in this block")
		}
	}

	s.Block = block
	s.Lastlineno = int32(parserline())
	s.Def = n
	n.Name.Vargen = int32(gen)
	n.Name.Funcdepth = Funcdepth
	n.Class = ctxt

	autoexport(n, ctxt)
}

func addvar(n *Node, t *Type, ctxt Class) {
	if n == nil || n.Sym == nil || (n.Op != ONAME && n.Op != ONONAME) || t == nil {
		Fatalf("addvar: n=%v t=%v nil", n, t)
	}

	n.Op = ONAME
	declare(n, ctxt)
	n.Type = t
}

// declare variables from grammar
// new_name_list (type | [type] = expr_list)
func variter(vl *NodeList, t *Node, el *NodeList) *NodeList {
	var init *NodeList
	doexpr := el != nil

	if count(el) == 1 && count(vl) > 1 {
		e := el.N
		as2 := Nod(OAS2, nil, nil)
		as2.List = vl
		as2.Rlist = list1(e)
		var v *Node
		for ; vl != nil; vl = vl.Next {
			v = vl.N
			v.Op = ONAME
			declare(v, dclcontext)
			v.Name.Param.Ntype = t
			v.Name.Defn = as2
			if Funcdepth > 0 {
				init = list(init, Nod(ODCL, v, nil))
			}
		}

		return list(init, as2)
	}

	var v *Node
	var e *Node
	for ; vl != nil; vl = vl.Next {
		if doexpr {
			if el == nil {
				Yyerror("missing expression in var declaration")
				break
			}

			e = el.N
			el = el.Next
		} else {
			e = nil
		}

		v = vl.N
		v.Op = ONAME
		declare(v, dclcontext)
		v.Name.Param.Ntype = t

		if e != nil || Funcdepth > 0 || isblank(v) {
			if Funcdepth > 0 {
				init = list(init, Nod(ODCL, v, nil))
			}
			e = Nod(OAS, v, e)
			init = list(init, e)
			if e.Right != nil {
				v.Name.Defn = e
			}
		}
	}

	if el != nil {
		Yyerror("extra expression in var declaration")
	}
	return init
}

// declare constants from grammar
// new_name_list [[type] = expr_list]
func constiter(vl *NodeList, t *Node, cl *NodeList) *NodeList {
	lno := int32(0) // default is to leave line number alone in listtreecopy
	if cl == nil {
		if t != nil {
			Yyerror("const declaration cannot have type without expression")
		}
		cl = lastconst
		t = lasttype
		lno = vl.N.Lineno
	} else {
		lastconst = cl
		lasttype = t
	}
	cl = listtreecopy(cl, lno)

	var v *Node
	var c *Node
	var vv *NodeList
	for ; vl != nil; vl = vl.Next {
		if cl == nil {
			Yyerror("missing value in const declaration")
			break
		}

		c = cl.N
		cl = cl.Next

		v = vl.N
		v.Op = OLITERAL
		declare(v, dclcontext)

		v.Name.Param.Ntype = t
		v.Name.Defn = c

		vv = list(vv, Nod(ODCLCONST, v, nil))
	}

	if cl != nil {
		Yyerror("extra expression in const declaration")
	}
	iota_ += 1
	return vv
}

// this generates a new name node,
// typically for labels or other one-off names.
func newname(s *Sym) *Node {
	if s == nil {
		Fatalf("newname nil")
	}

	n := Nod(ONAME, nil, nil)
	n.Sym = s
	n.Type = nil
	n.Addable = true
	n.Ullman = 1
	n.Xoffset = 0
	return n
}

// newfuncname generates a new name node for a function or method.
// TODO(rsc): Use an ODCLFUNC node instead. See comment in CL 7360.
func newfuncname(s *Sym) *Node {
	n := newname(s)
	n.Func = new(Func)
	n.Func.FCurfn = Curfn
	return n
}

// this generates a new name node for a name
// being declared.
func dclname(s *Sym) *Node {
	n := newname(s)
	n.Op = ONONAME // caller will correct it
	return n
}

func typenod(t *Type) *Node {
	// if we copied another type with *t = *u
	// then t->nod might be out of date, so
	// check t->nod->type too
	if t.Nod == nil || t.Nod.Type != t {
		t.Nod = Nod(OTYPE, nil, nil)
		t.Nod.Type = t
		t.Nod.Sym = t.Sym
	}

	return t.Nod
}

// this will return an old name
// that has already been pushed on the
// declaration list. a diagnostic is
// generated if no name has been defined.
func oldname(s *Sym) *Node {
	n := s.Def
	if n == nil {
		// maybe a top-level name will come along
		// to give this a definition later.
		// walkdef will check s->def again once
		// all the input source has been processed.
		n = newname(s)
		n.Op = ONONAME
		n.Name.Iota = iota_ // save current iota value in const declarations
	}

	if Curfn != nil && n.Op == ONAME && n.Name.Funcdepth > 0 && n.Name.Funcdepth != Funcdepth {
		// inner func is referring to var in outer func.
		//
		// TODO(rsc): If there is an outer variable x and we
		// are parsing x := 5 inside the closure, until we get to
		// the := it looks like a reference to the outer x so we'll
		// make x a closure variable unnecessarily.
		if n.Name.Param.Closure == nil || n.Name.Param.Closure.Name.Funcdepth != Funcdepth {
			// create new closure var.
			c := Nod(ONAME, nil, nil)

			c.Sym = s
			c.Class = PPARAMREF
			c.Isddd = n.Isddd
			c.Name.Defn = n
			c.Addable = false
			c.Ullman = 2
			c.Name.Funcdepth = Funcdepth
			c.Name.Param.Outer = n.Name.Param.Closure
			n.Name.Param.Closure = c
			c.Name.Param.Closure = n
			c.Xoffset = 0
			Curfn.Func.Cvars = list(Curfn.Func.Cvars, c)
		}

		// return ref to closure var, not original
		return n.Name.Param.Closure
	}

	return n
}

// := declarations
func colasname(n *Node) bool {
	switch n.Op {
	case ONAME,
		ONONAME,
		OPACK,
		OTYPE,
		OLITERAL:
		return n.Sym != nil
	}

	return false
}

func colasdefn(left *NodeList, defn *Node) {
	for l := left; l != nil; l = l.Next {
		if l.N.Sym != nil {
			l.N.Sym.Flags |= SymUniq
		}
	}

	nnew := 0
	nerr := 0
	var n *Node
	for l := left; l != nil; l = l.Next {
		n = l.N
		if isblank(n) {
			continue
		}
		if !colasname(n) {
			yyerrorl(int(defn.Lineno), "non-name %v on left side of :=", n)
			nerr++
			continue
		}

		if n.Sym.Flags&SymUniq == 0 {
			yyerrorl(int(defn.Lineno), "%v repeated on left side of :=", n.Sym)
			n.Diag++
			nerr++
			continue
		}

		n.Sym.Flags &^= SymUniq
		if n.Sym.Block == block {
			continue
		}

		nnew++
		n = newname(n.Sym)
		declare(n, dclcontext)
		n.Name.Defn = defn
		defn.Ninit = list(defn.Ninit, Nod(ODCL, n, nil))
		l.N = n
	}

	if nnew == 0 && nerr == 0 {
		yyerrorl(int(defn.Lineno), "no new variables on left side of :=")
	}
}

func colas(left *NodeList, right *NodeList, lno int32) *Node {
	as := Nod(OAS2, nil, nil)
	as.List = left
	as.Rlist = right
	as.Colas = true
	as.Lineno = lno
	colasdefn(left, as)

	// make the tree prettier; not necessary
	if count(left) == 1 && count(right) == 1 {
		as.Left = as.List.N
		as.Right = as.Rlist.N
		as.List = nil
		as.Rlist = nil
		as.Op = OAS
	}

	return as
}

// declare the arguments in an
// interface field declaration.
func ifacedcl(n *Node) {
	if n.Op != ODCLFIELD || n.Right == nil {
		Fatalf("ifacedcl")
	}

	if isblank(n.Left) {
		Yyerror("methods must have a unique non-blank name")
	}

	n.Func = new(Func)
	n.Func.FCurfn = Curfn
	dclcontext = PPARAM
	markdcl()
	Funcdepth++
	n.Func.Outer = Curfn
	Curfn = n
	funcargs(n.Right)

	// funcbody is normally called after the parser has
	// seen the body of a function but since an interface
	// field declaration does not have a body, we must
	// call it now to pop the current declaration context.
	dclcontext = PAUTO

	funcbody(n)
}

// declare the function proper
// and declare the arguments.
// called in extern-declaration context
// returns in auto-declaration context.
func funchdr(n *Node) {
	// change the declaration context from extern to auto
	if Funcdepth == 0 && dclcontext != PEXTERN {
		Fatalf("funchdr: dclcontext")
	}

	if importpkg == nil && n.Func.Nname != nil {
		makefuncsym(n.Func.Nname.Sym)
	}

	dclcontext = PAUTO
	markdcl()
	Funcdepth++

	n.Func.Outer = Curfn
	Curfn = n

	if n.Func.Nname != nil {
		funcargs(n.Func.Nname.Name.Param.Ntype)
	} else if n.Func.Ntype != nil {
		funcargs(n.Func.Ntype)
	} else {
		funcargs2(n.Type)
	}
}

func funcargs(nt *Node) {
	if nt.Op != OTFUNC {
		Fatalf("funcargs %v", Oconv(int(nt.Op), 0))
	}

	// re-start the variable generation number
	// we want to use small numbers for the return variables,
	// so let them have the chunk starting at 1.
	vargen = count(nt.Rlist)

	// declare the receiver and in arguments.
	// no n->defn because type checking of func header
	// will not fill in the types until later
	if nt.Left != nil {
		n := nt.Left
		if n.Op != ODCLFIELD {
			Fatalf("funcargs receiver %v", Oconv(int(n.Op), 0))
		}
		if n.Left != nil {
			n.Left.Op = ONAME
			n.Left.Name.Param.Ntype = n.Right
			declare(n.Left, PPARAM)
			if dclcontext == PAUTO {
				vargen++
				n.Left.Name.Vargen = int32(vargen)
			}
		}
	}

	var n *Node
	for l := nt.List; l != nil; l = l.Next {
		n = l.N
		if n.Op != ODCLFIELD {
			Fatalf("funcargs in %v", Oconv(int(n.Op), 0))
		}
		if n.Left != nil {
			n.Left.Op = ONAME
			n.Left.Name.Param.Ntype = n.Right
			declare(n.Left, PPARAM)
			if dclcontext == PAUTO {
				vargen++
				n.Left.Name.Vargen = int32(vargen)
			}
		}
	}

	// declare the out arguments.
	gen := count(nt.List)
	var i int = 0
	var nn *Node
	for l := nt.Rlist; l != nil; l = l.Next {
		n = l.N

		if n.Op != ODCLFIELD {
			Fatalf("funcargs out %v", Oconv(int(n.Op), 0))
		}

		if n.Left == nil {
			// Name so that escape analysis can track it. ~r stands for 'result'.
			n.Left = newname(Lookupf("~r%d", gen))
			gen++
		}

		// TODO: n->left->missing = 1;
		n.Left.Op = ONAME

		if isblank(n.Left) {
			// Give it a name so we can assign to it during return. ~b stands for 'blank'.
			// The name must be different from ~r above because if you have
			//	func f() (_ int)
			//	func g() int
			// f is allowed to use a plain 'return' with no arguments, while g is not.
			// So the two cases must be distinguished.
			// We do not record a pointer to the original node (n->orig).
			// Having multiple names causes too much confusion in later passes.
			nn = Nod(OXXX, nil, nil)

			*nn = *n.Left
			nn.Orig = nn
			nn.Sym = Lookupf("~b%d", gen)
			gen++
			n.Left = nn
		}

		n.Left.Name.Param.Ntype = n.Right
		declare(n.Left, PPARAMOUT)
		if dclcontext == PAUTO {
			i++
			n.Left.Name.Vargen = int32(i)
		}
	}
}

// Same as funcargs, except run over an already constructed TFUNC.
// This happens during import, where the hidden_fndcl rule has
// used functype directly to parse the function's type.
func funcargs2(t *Type) {
	if t.Etype != TFUNC {
		Fatalf("funcargs2 %v", t)
	}

	if t.Thistuple != 0 {
		var n *Node
		for ft := getthisx(t).Type; ft != nil; ft = ft.Down {
			if ft.Nname == nil || ft.Nname.Sym == nil {
				continue
			}
			n = ft.Nname // no need for newname(ft->nname->sym)
			n.Type = ft.Type
			declare(n, PPARAM)
		}
	}

	if t.Intuple != 0 {
		var n *Node
		for ft := getinargx(t).Type; ft != nil; ft = ft.Down {
			if ft.Nname == nil || ft.Nname.Sym == nil {
				continue
			}
			n = ft.Nname
			n.Type = ft.Type
			declare(n, PPARAM)
		}
	}

	if t.Outtuple != 0 {
		var n *Node
		for ft := getoutargx(t).Type; ft != nil; ft = ft.Down {
			if ft.Nname == nil || ft.Nname.Sym == nil {
				continue
			}
			n = ft.Nname
			n.Type = ft.Type
			declare(n, PPARAMOUT)
		}
	}
}

// finish the body.
// called in auto-declaration context.
// returns in extern-declaration context.
func funcbody(n *Node) {
	// change the declaration context from auto to extern
	if dclcontext != PAUTO {
		Fatalf("funcbody: dclcontext")
	}
	popdcl()
	Funcdepth--
	Curfn = n.Func.Outer
	n.Func.Outer = nil
	if Funcdepth == 0 {
		dclcontext = PEXTERN
	}
}

// new type being defined with name s.
func typedcl0(s *Sym) *Node {
	n := newname(s)
	n.Op = OTYPE
	declare(n, dclcontext)
	return n
}

// node n, which was returned by typedcl0
// is being declared to have uncompiled type t.
// return the ODCLTYPE node to use.
func typedcl1(n *Node, t *Node, local bool) *Node {
	n.Name.Param.Ntype = t
	n.Local = local
	return Nod(ODCLTYPE, n, nil)
}

// structs, functions, and methods.
// they don't belong here, but where do they belong?
func checkembeddedtype(t *Type) {
	if t == nil {
		return
	}

	if t.Sym == nil && Isptr[t.Etype] {
		t = t.Type
		if t.Etype == TINTER {
			Yyerror("embedded type cannot be a pointer to interface")
		}
	}

	if Isptr[t.Etype] {
		Yyerror("embedded type cannot be a pointer")
	} else if t.Etype == TFORW && t.Embedlineno == 0 {
		t.Embedlineno = lineno
	}
}

func structfield(n *Node) *Type {
	lno := int(lineno)
	lineno = n.Lineno

	if n.Op != ODCLFIELD {
		Fatalf("structfield: oops %v\n", n)
	}

	f := typ(TFIELD)
	f.Isddd = n.Isddd

	if n.Right != nil {
		typecheck(&n.Right, Etype)
		n.Type = n.Right.Type
		if n.Left != nil {
			n.Left.Type = n.Type
		}
		if n.Embedded != 0 {
			checkembeddedtype(n.Type)
		}
	}

	n.Right = nil

	f.Type = n.Type
	if f.Type == nil {
		f.Broke = true
	}

	switch n.Val().Ctype() {
	case CTSTR:
		f.Note = new(string)
		*f.Note = n.Val().U.(string)

	default:
		Yyerror("field annotation must be string")
		fallthrough

	case CTxxx:
		f.Note = nil
	}

	if n.Left != nil && n.Left.Op == ONAME {
		f.Nname = n.Left
		f.Embedded = n.Embedded
		f.Sym = f.Nname.Sym
	}

	lineno = int32(lno)
	return f
}

var uniqgen uint32

func checkdupfields(t *Type, what string) {
	lno := int(lineno)

	for ; t != nil; t = t.Down {
		if t.Sym != nil && t.Nname != nil && !isblank(t.Nname) {
			if t.Sym.Uniqgen == uniqgen {
				lineno = t.Nname.Lineno
				Yyerror("duplicate %s %s", what, t.Sym.Name)
			} else {
				t.Sym.Uniqgen = uniqgen
			}
		}
	}

	lineno = int32(lno)
}

// convert a parsed id/type list into
// a type for struct/interface/arglist
func tostruct(l *NodeList) *Type {
	t := typ(TSTRUCT)
	tostruct0(t, l)
	return t
}

func tostruct0(t *Type, l *NodeList) {
	if t == nil || t.Etype != TSTRUCT {
		Fatalf("struct expected")
	}

	for tp := &t.Type; l != nil; l = l.Next {
		f := structfield(l.N)

		*tp = f
		tp = &f.Down
	}

	for f := t.Type; f != nil && !t.Broke; f = f.Down {
		if f.Broke {
			t.Broke = true
		}
	}

	uniqgen++
	checkdupfields(t.Type, "field")

	if !t.Broke {
		checkwidth(t)
	}
}

func tofunargs(l *NodeList) *Type {
	var f *Type

	t := typ(TSTRUCT)
	t.Funarg = true

	for tp := &t.Type; l != nil; l = l.Next {
		f = structfield(l.N)
		f.Funarg = true

		// esc.go needs to find f given a PPARAM to add the tag.
		if l.N.Left != nil && l.N.Left.Class == PPARAM {
			l.N.Left.Name.Param.Field = f
		}

		*tp = f
		tp = &f.Down
	}

	for f := t.Type; f != nil && !t.Broke; f = f.Down {
		if f.Broke {
			t.Broke = true
		}
	}

	return t
}

func interfacefield(n *Node) *Type {
	lno := int(lineno)
	lineno = n.Lineno

	if n.Op != ODCLFIELD {
		Fatalf("interfacefield: oops %v\n", n)
	}

	if n.Val().Ctype() != CTxxx {
		Yyerror("interface method cannot have annotation")
	}

	f := typ(TFIELD)
	f.Isddd = n.Isddd

	if n.Right != nil {
		if n.Left != nil {
			// queue resolution of method type for later.
			// right now all we need is the name list.
			// avoids cycles for recursive interface types.
			n.Type = typ(TINTERMETH)

			n.Type.Nname = n.Right
			n.Left.Type = n.Type
			queuemethod(n)

			if n.Left.Op == ONAME {
				f.Nname = n.Left
				f.Embedded = n.Embedded
				f.Sym = f.Nname.Sym
			}
		} else {
			typecheck(&n.Right, Etype)
			n.Type = n.Right.Type

			if n.Embedded != 0 {
				checkembeddedtype(n.Type)
			}

			if n.Type != nil {
				switch n.Type.Etype {
				case TINTER:
					break

				case TFORW:
					Yyerror("interface type loop involving %v", n.Type)
					f.Broke = true

				default:
					Yyerror("interface contains embedded non-interface %v", n.Type)
					f.Broke = true
				}
			}
		}
	}

	n.Right = nil

	f.Type = n.Type
	if f.Type == nil {
		f.Broke = true
	}

	lineno = int32(lno)
	return f
}

func tointerface(l *NodeList) *Type {
	t := typ(TINTER)
	tointerface0(t, l)
	return t
}

func tointerface0(t *Type, l *NodeList) *Type {
	if t == nil || t.Etype != TINTER {
		Fatalf("interface expected")
	}

	tp := &t.Type
	for ; l != nil; l = l.Next {
		f := interfacefield(l.N)

		if l.N.Left == nil && f.Type.Etype == TINTER {
			// embedded interface, inline methods
			for t1 := f.Type.Type; t1 != nil; t1 = t1.Down {
				f = typ(TFIELD)
				f.Type = t1.Type
				f.Broke = t1.Broke
				f.Sym = t1.Sym
				if f.Sym != nil {
					f.Nname = newname(f.Sym)
				}
				*tp = f
				tp = &f.Down
			}
		} else {
			*tp = f
			tp = &f.Down
		}
	}

	for f := t.Type; f != nil && !t.Broke; f = f.Down {
		if f.Broke {
			t.Broke = true
		}
	}

	uniqgen++
	checkdupfields(t.Type, "method")
	t = sortinter(t)
	checkwidth(t)

	return t
}

func embedded(s *Sym, pkg *Pkg) *Node {
	const (
		CenterDot = 0xB7
	)
	// Names sometimes have disambiguation junk
	// appended after a center dot.  Discard it when
	// making the name for the embedded struct field.
	name := s.Name

	if i := strings.Index(s.Name, string(CenterDot)); i >= 0 {
		name = s.Name[:i]
	}

	var n *Node
	if exportname(name) {
		n = newname(Lookup(name))
	} else if s.Pkg == builtinpkg {
		// The name of embedded builtins belongs to pkg.
		n = newname(Pkglookup(name, pkg))
	} else {
		n = newname(Pkglookup(name, s.Pkg))
	}
	n = Nod(ODCLFIELD, n, oldname(s))
	n.Embedded = 1
	return n
}

// check that the list of declarations is either all anonymous or all named
func findtype(l *NodeList) *Node {
	for ; l != nil; l = l.Next {
		if l.N.Op == OKEY {
			return l.N.Right
		}
	}
	return nil
}

func checkarglist(all *NodeList, input int) *NodeList {
	named := 0
	for l := all; l != nil; l = l.Next {
		if l.N.Op == OKEY {
			named = 1
			break
		}
	}

	if named != 0 {
		var n *Node
		var l *NodeList
		for l = all; l != nil; l = l.Next {
			n = l.N
			if n.Op != OKEY && n.Sym == nil {
				Yyerror("mixed named and unnamed function parameters")
				break
			}
		}

		if l == nil && n != nil && n.Op != OKEY {
			Yyerror("final function parameter must have type")
		}
	}

	var nextt *Node
	var t *Node
	var n *Node
	for l := all; l != nil; l = l.Next {
		// can cache result from findtype to avoid
		// quadratic behavior here, but unlikely to matter.
		n = l.N

		if named != 0 {
			if n.Op == OKEY {
				t = n.Right
				n = n.Left
				nextt = nil
			} else {
				if nextt == nil {
					nextt = findtype(l)
				}
				t = nextt
			}
		} else {
			t = n
			n = nil
		}

		// during import l->n->op is OKEY, but l->n->left->sym == S
		// means it was a '?', not that it was
		// a lone type This doesn't matter for the exported
		// declarations, which are parsed by rules that don't
		// use checkargs, but can happen for func literals in
		// the inline bodies.
		// TODO(rsc) this can go when typefmt case TFIELD in exportmode fmt.go prints _ instead of ?
		if importpkg != nil && n.Sym == nil {
			n = nil
		}

		if n != nil && n.Sym == nil {
			t = n
			n = nil
		}

		if n != nil {
			n = newname(n.Sym)
		}
		n = Nod(ODCLFIELD, n, t)
		if n.Right != nil && n.Right.Op == ODDD {
			if input == 0 {
				Yyerror("cannot use ... in output argument list")
			} else if l.Next != nil {
				Yyerror("can only use ... as final argument in list")
			}
			n.Right.Op = OTARRAY
			n.Right.Right = n.Right.Left
			n.Right.Left = nil
			n.Isddd = true
			if n.Left != nil {
				n.Left.Isddd = true
			}
		}

		l.N = n
	}

	return all
}

func fakethis() *Node {
	n := Nod(ODCLFIELD, nil, typenod(Ptrto(typ(TSTRUCT))))
	return n
}

// Is this field a method on an interface?
// Those methods have an anonymous *struct{} as the receiver.
// (See fakethis above.)
func isifacemethod(f *Type) bool {
	rcvr := getthisx(f).Type
	if rcvr.Sym != nil {
		return false
	}
	t := rcvr.Type
	if !Isptr[t.Etype] {
		return false
	}
	t = t.Type
	if t.Sym != nil || t.Etype != TSTRUCT || t.Type != nil {
		return false
	}
	return true
}

// turn a parsed function declaration into a type
func functype(this *Node, in *NodeList, out *NodeList) *Type {
	t := typ(TFUNC)
	functype0(t, this, in, out)
	return t
}

func functype0(t *Type, this *Node, in *NodeList, out *NodeList) {
	if t == nil || t.Etype != TFUNC {
		Fatalf("function type expected")
	}

	var rcvr *NodeList
	if this != nil {
		rcvr = list1(this)
	}
	t.Type = tofunargs(rcvr)
	t.Type.Down = tofunargs(out)
	t.Type.Down.Down = tofunargs(in)

	uniqgen++
	checkdupfields(t.Type.Type, "argument")
	checkdupfields(t.Type.Down.Type, "argument")
	checkdupfields(t.Type.Down.Down.Type, "argument")

	if t.Type.Broke || t.Type.Down.Broke || t.Type.Down.Down.Broke {
		t.Broke = true
	}

	if this != nil {
		t.Thistuple = 1
	}
	t.Outtuple = count(out)
	t.Intuple = count(in)
	t.Outnamed = false
	if t.Outtuple > 0 && out.N.Left != nil && out.N.Left.Orig != nil {
		s := out.N.Left.Orig.Sym
		if s != nil && (s.Name[0] != '~' || s.Name[1] != 'r') { // ~r%d is the name invented for an unnamed result
			t.Outnamed = true
		}
	}
}

var methodsym_toppkg *Pkg

func methodsym(nsym *Sym, t0 *Type, iface int) *Sym {
	var s *Sym
	var p string
	var suffix string
	var spkg *Pkg

	t := t0
	if t == nil {
		goto bad
	}
	s = t.Sym
	if s == nil && Isptr[t.Etype] {
		t = t.Type
		if t == nil {
			goto bad
		}
		s = t.Sym
	}

	spkg = nil
	if s != nil {
		spkg = s.Pkg
	}

	// if t0 == *t and t0 has a sym,
	// we want to see *t, not t0, in the method name.
	if t != t0 && t0.Sym != nil {
		t0 = Ptrto(t)
	}

	suffix = ""
	if iface != 0 {
		dowidth(t0)
		if t0.Width < Types[Tptr].Width {
			suffix = "·i"
		}
	}

	if (spkg == nil || nsym.Pkg != spkg) && !exportname(nsym.Name) {
		if t0.Sym == nil && Isptr[t0.Etype] {
			p = fmt.Sprintf("(%v).%s.%s%s", Tconv(t0, obj.FmtLeft|obj.FmtShort), nsym.Pkg.Prefix, nsym.Name, suffix)
		} else {
			p = fmt.Sprintf("%v.%s.%s%s", Tconv(t0, obj.FmtLeft|obj.FmtShort), nsym.Pkg.Prefix, nsym.Name, suffix)
		}
	} else {
		if t0.Sym == nil && Isptr[t0.Etype] {
			p = fmt.Sprintf("(%v).%s%s", Tconv(t0, obj.FmtLeft|obj.FmtShort), nsym.Name, suffix)
		} else {
			p = fmt.Sprintf("%v.%s%s", Tconv(t0, obj.FmtLeft|obj.FmtShort), nsym.Name, suffix)
		}
	}

	if spkg == nil {
		if methodsym_toppkg == nil {
			methodsym_toppkg = mkpkg("go")
		}
		spkg = methodsym_toppkg
	}

	s = Pkglookup(p, spkg)

	return s

bad:
	Yyerror("illegal receiver type: %v", t0)
	return nil
}

func methodname(n *Node, t *Type) *Node {
	s := methodsym(n.Sym, t, 0)
	if s == nil {
		return n
	}
	return newname(s)
}

func methodname1(n *Node, t *Node) *Node {
	star := ""
	if t.Op == OIND {
		star = "*"
		t = t.Left
	}

	if t.Sym == nil || isblank(n) {
		return newfuncname(n.Sym)
	}

	var p string
	if star != "" {
		p = fmt.Sprintf("(%s%v).%v", star, t.Sym, n.Sym)
	} else {
		p = fmt.Sprintf("%v.%v", t.Sym, n.Sym)
	}

	if exportname(t.Sym.Name) {
		n = newfuncname(Lookup(p))
	} else {
		n = newfuncname(Pkglookup(p, t.Sym.Pkg))
	}

	return n
}

// add a method, declared as a function,
// n is fieldname, pa is base type, t is function type
func addmethod(sf *Sym, t *Type, local bool, nointerface bool) {
	// get field sym
	if sf == nil {
		Fatalf("no method symbol")
	}

	// get parent type sym
	pa := getthisx(t).Type // ptr to this structure
	if pa == nil {
		Yyerror("missing receiver")
		return
	}

	pa = pa.Type
	f := methtype(pa, 1)
	if f == nil {
		t = pa
		if t == nil { // rely on typecheck having complained before
			return
		}
		if t != nil {
			if Isptr[t.Etype] {
				if t.Sym != nil {
					Yyerror("invalid receiver type %v (%v is a pointer type)", pa, t)
					return
				}

				t = t.Type
			}

			if t.Broke { // rely on typecheck having complained before
				return
			}
			if t.Sym == nil {
				Yyerror("invalid receiver type %v (%v is an unnamed type)", pa, t)
				return
			}

			if Isptr[t.Etype] {
				Yyerror("invalid receiver type %v (%v is a pointer type)", pa, t)
				return
			}

			if t.Etype == TINTER {
				Yyerror("invalid receiver type %v (%v is an interface type)", pa, t)
				return
			}
		}

		// Should have picked off all the reasons above,
		// but just in case, fall back to generic error.
		Yyerror("invalid receiver type %v (%v / %v)", pa, Tconv(pa, obj.FmtLong), Tconv(t, obj.FmtLong))

		return
	}

	pa = f
	if local && !pa.Local {
		Yyerror("cannot define new methods on non-local type %v", pa)
		return
	}

	if isblanksym(sf) {
		return
	}

	if pa.Etype == TSTRUCT {
		for f := pa.Type; f != nil; f = f.Down {
			if f.Sym == sf {
				Yyerror("type %v has both field and method named %v", pa, sf)
				return
			}
		}
	}

	n := Nod(ODCLFIELD, newname(sf), nil)
	n.Type = t

	var d *Type // last found
	for f := pa.Method; f != nil; f = f.Down {
		d = f
		if f.Etype != TFIELD {
			Fatalf("addmethod: not TFIELD: %v", Tconv(f, obj.FmtLong))
		}
		if sf.Name != f.Sym.Name {
			continue
		}
		if !Eqtype(t, f.Type) {
			Yyerror("method redeclared: %v.%v\n\t%v\n\t%v", pa, sf, f.Type, t)
		}
		return
	}

	f = structfield(n)
	f.Nointerface = nointerface

	// during import unexported method names should be in the type's package
	if importpkg != nil && f.Sym != nil && !exportname(f.Sym.Name) && f.Sym.Pkg != structpkg {
		Fatalf("imported method name %v in wrong package %s\n", Sconv(f.Sym, obj.FmtSign), structpkg.Name)
	}

	if d == nil {
		pa.Method = f
	} else {
		d.Down = f
	}
	return
}

func funccompile(n *Node) {
	Stksize = BADWIDTH
	Maxarg = 0

	if n.Type == nil {
		if nerrors == 0 {
			Fatalf("funccompile missing type")
		}
		return
	}

	// assign parameter offsets
	checkwidth(n.Type)

	if Curfn != nil {
		Fatalf("funccompile %v inside %v", n.Func.Nname.Sym, Curfn.Func.Nname.Sym)
	}

	Stksize = 0
	dclcontext = PAUTO
	Funcdepth = n.Func.Depth + 1
	compile(n)
	Curfn = nil
	Funcdepth = 0
	dclcontext = PEXTERN
}

func funcsym(s *Sym) *Sym {
	if s.Fsym != nil {
		return s.Fsym
	}

	s1 := Pkglookup(s.Name+"·f", s.Pkg)
	s.Fsym = s1
	return s1
}

func makefuncsym(s *Sym) {
	if isblanksym(s) {
		return
	}
	if compiling_runtime != 0 && s.Name == "getg" {
		// runtime.getg() is not a real function and so does
		// not get a funcsym.
		return
	}
	s1 := funcsym(s)
	s1.Def = newfuncname(s1)
	s1.Def.Func.Shortname = newname(s)
	funcsyms = append(funcsyms, s1.Def)
}

type nowritebarrierrecChecker struct {
	curfn  *Node
	stable bool

	// best maps from the ODCLFUNC of each visited function that
	// recursively invokes a write barrier to the called function
	// on the shortest path to a write barrier.
	best map[*Node]nowritebarrierrecCall
}

type nowritebarrierrecCall struct {
	target *Node
	depth  int
	lineno int32
}

func checknowritebarrierrec() {
	c := nowritebarrierrecChecker{
		best: make(map[*Node]nowritebarrierrecCall),
	}
	visitBottomUp(xtop, func(list []*Node, recursive bool) {
		// Functions with write barriers have depth 0.
		for _, n := range list {
			if n.Func.WBLineno != 0 {
				c.best[n] = nowritebarrierrecCall{target: nil, depth: 0, lineno: n.Func.WBLineno}
			}
		}

		// Propagate write barrier depth up from callees. In
		// the recursive case, we have to update this at most
		// len(list) times and can stop when we an iteration
		// that doesn't change anything.
		for _ = range list {
			c.stable = false
			for _, n := range list {
				if n.Func.WBLineno == 0 {
					c.curfn = n
					c.visitcodelist(n.Nbody)
				}
			}
			if c.stable {
				break
			}
		}

		// Check nowritebarrierrec functions.
		for _, n := range list {
			if !n.Func.Nowritebarrierrec {
				continue
			}
			call, hasWB := c.best[n]
			if !hasWB {
				continue
			}

			// Build the error message in reverse.
			err := ""
			for call.target != nil {
				err = fmt.Sprintf("\n\t%v: called by %v%s", Ctxt.Line(int(call.lineno)), n.Func.Nname, err)
				n = call.target
				call = c.best[n]
			}
			err = fmt.Sprintf("write barrier prohibited by caller; %v%s", n.Func.Nname, err)
			yyerrorl(int(n.Func.WBLineno), err)
		}
	})
}

func (c *nowritebarrierrecChecker) visitcodelist(l *NodeList) {
	for ; l != nil; l = l.Next {
		c.visitcode(l.N)
	}
}

func (c *nowritebarrierrecChecker) visitcode(n *Node) {
	if n == nil {
		return
	}

	if n.Op == OCALLFUNC || n.Op == OCALLMETH {
		c.visitcall(n)
	}

	c.visitcodelist(n.Ninit)
	c.visitcode(n.Left)
	c.visitcode(n.Right)
	c.visitcodelist(n.List)
	c.visitcodelist(n.Nbody)
	c.visitcodelist(n.Rlist)
}

func (c *nowritebarrierrecChecker) visitcall(n *Node) {
	fn := n.Left
	if n.Op == OCALLMETH {
		fn = n.Left.Right.Sym.Def
	}
	if fn == nil || fn.Op != ONAME || fn.Class != PFUNC || fn.Name.Defn == nil {
		return
	}
	if (compiling_runtime != 0 || fn.Sym.Pkg == Runtimepkg) && fn.Sym.Name == "allocm" {
		return
	}
	defn := fn.Name.Defn

	fnbest, ok := c.best[defn]
	if !ok {
		return
	}
	best, ok := c.best[c.curfn]
	if ok && fnbest.depth+1 >= best.depth {
		return
	}
	c.best[c.curfn] = nowritebarrierrecCall{target: defn, depth: fnbest.depth + 1, lineno: n.Lineno}
	c.stable = false
}
