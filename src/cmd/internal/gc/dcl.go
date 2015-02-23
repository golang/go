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

/*
 * declaration stack & operations
 */
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
		fmt.Printf("\t%v push %v %p\n", Ctxt.Line(int(lineno)), Sconv(s, 0), s.Def)
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
			fmt.Printf("\t%v pop %v %p\n", Ctxt.Line(int(lineno)), Sconv(s, 0), s.Def)
		}
	}

	if d == nil {
		Fatal("popdcl: no mark")
	}
	dclstack = d.Link
	block = d.Block
}

func poptodcl() {
	// pop the old marker and push a new one
	// (cannot reuse the existing one)
	// because we use the markers to identify blocks
	// for the goto restriction checks.
	popdcl()

	markdcl()
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
		fmt.Printf(" %v\n", Sconv(s, 0))
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
		var tmp *Strlit
		if s.Origpkg != nil {
			tmp = s.Origpkg.Path
		} else {
			tmp = s.Pkg.Path
		}
		pkgstr := tmp
		Yyerror("%v redeclared %s\n"+"\tprevious declaration during import \"%v\"", Sconv(s, 0), where, Zconv(pkgstr, 0))
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

		yyerrorl(int(line1), "%v redeclared %s\n"+"\tprevious declaration at %v", Sconv(s, 0), where, Ctxt.Line(line2))
	}
}

var vargen int

/*
 * declare individual names - var, typ, const
 */

var declare_typegen int

func declare(n *Node, ctxt int) {
	if ctxt == PDISCARD {
		return
	}

	if isblank(n) {
		return
	}

	n.Lineno = int32(parserline())
	s := n.Sym

	// kludgy: typecheckok means we're past parsing.  Eg genwrapper may declare out of package names later.
	if importpkg == nil && typecheckok == 0 && s.Pkg != localpkg {
		Yyerror("cannot declare name %v", Sconv(s, 0))
	}

	if ctxt == PEXTERN && s.Name == "init" {
		Yyerror("cannot declare init - must be func", s)
	}

	gen := 0
	if ctxt == PEXTERN {
		externdcl = list(externdcl, n)
		if dflag() {
			fmt.Printf("\t%v global decl %v %p\n", Ctxt.Line(int(lineno)), Sconv(s, 0), n)
		}
	} else {
		if Curfn == nil && ctxt == PAUTO {
			Fatal("automatic outside function")
		}
		if Curfn != nil {
			Curfn.Dcl = list(Curfn.Dcl, n)
		}
		if n.Op == OTYPE {
			declare_typegen++
			gen = declare_typegen
		} else if n.Op == ONAME && ctxt == PAUTO && !strings.Contains(s.Name, "·") {
			vargen++
			gen = vargen
		}
		pushdcl(s)
		n.Curfn = Curfn
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
	n.Vargen = int32(gen)
	n.Funcdepth = Funcdepth
	n.Class = uint8(ctxt)

	autoexport(n, ctxt)
}

func addvar(n *Node, t *Type, ctxt int) {
	if n == nil || n.Sym == nil || (n.Op != ONAME && n.Op != ONONAME) || t == nil {
		Fatal("addvar: n=%v t=%v nil", Nconv(n, 0), Tconv(t, 0))
	}

	n.Op = ONAME
	declare(n, ctxt)
	n.Type = t
}

/*
 * declare variables from grammar
 * new_name_list (type | [type] = expr_list)
 */
func variter(vl *NodeList, t *Node, el *NodeList) *NodeList {
	init := (*NodeList)(nil)
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
			v.Ntype = t
			v.Defn = as2
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
		v.Ntype = t

		if e != nil || Funcdepth > 0 || isblank(v) {
			if Funcdepth > 0 {
				init = list(init, Nod(ODCL, v, nil))
			}
			e = Nod(OAS, v, e)
			init = list(init, e)
			if e.Right != nil {
				v.Defn = e
			}
		}
	}

	if el != nil {
		Yyerror("extra expression in var declaration")
	}
	return init
}

/*
 * declare constants from grammar
 * new_name_list [[type] = expr_list]
 */
func constiter(vl *NodeList, t *Node, cl *NodeList) *NodeList {
	vv := (*NodeList)(nil)
	if cl == nil {
		if t != nil {
			Yyerror("const declaration cannot have type without expression")
		}
		cl = lastconst
		t = lasttype
	} else {
		lastconst = cl
		lasttype = t
	}

	cl = listtreecopy(cl)

	var v *Node
	var c *Node
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

		v.Ntype = t
		v.Defn = c

		vv = list(vv, Nod(ODCLCONST, v, nil))
	}

	if cl != nil {
		Yyerror("extra expression in const declaration")
	}
	iota_ += 1
	return vv
}

/*
 * this generates a new name node,
 * typically for labels or other one-off names.
 */
func newname(s *Sym) *Node {
	if s == nil {
		Fatal("newname nil")
	}

	n := Nod(ONAME, nil, nil)
	n.Sym = s
	n.Type = nil
	n.Addable = 1
	n.Ullman = 1
	n.Xoffset = 0
	return n
}

/*
 * this generates a new name node for a name
 * being declared.
 */
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

/*
 * this will return an old name
 * that has already been pushed on the
 * declaration list. a diagnostic is
 * generated if no name has been defined.
 */
func oldname(s *Sym) *Node {
	n := s.Def
	if n == nil {
		// maybe a top-level name will come along
		// to give this a definition later.
		// walkdef will check s->def again once
		// all the input source has been processed.
		n = newname(s)

		n.Op = ONONAME
		n.Iota = iota_ // save current iota value in const declarations
	}

	if Curfn != nil && n.Funcdepth > 0 && n.Funcdepth != Funcdepth && n.Op == ONAME {
		// inner func is referring to var in outer func.
		//
		// TODO(rsc): If there is an outer variable x and we
		// are parsing x := 5 inside the closure, until we get to
		// the := it looks like a reference to the outer x so we'll
		// make x a closure variable unnecessarily.
		if n.Closure == nil || n.Closure.Funcdepth != Funcdepth {
			// create new closure var.
			c := Nod(ONAME, nil, nil)

			c.Sym = s
			c.Class = PPARAMREF
			c.Isddd = n.Isddd
			c.Defn = n
			c.Addable = 0
			c.Ullman = 2
			c.Funcdepth = Funcdepth
			c.Outer = n.Closure
			n.Closure = c
			c.Closure = n
			c.Xoffset = 0
			Curfn.Cvars = list(Curfn.Cvars, c)
		}

		// return ref to closure var, not original
		return n.Closure
	}

	return n
}

/*
 * := declarations
 */
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
			yyerrorl(int(defn.Lineno), "non-name %v on left side of :=", Nconv(n, 0))
			nerr++
			continue
		}

		if n.Sym.Flags&SymUniq == 0 {
			yyerrorl(int(defn.Lineno), "%v repeated on left side of :=", Sconv(n.Sym, 0))
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
		n.Defn = defn
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
	as.Colas = 1
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

/*
 * declare the arguments in an
 * interface field declaration.
 */
func ifacedcl(n *Node) {
	if n.Op != ODCLFIELD || n.Right == nil {
		Fatal("ifacedcl")
	}

	if isblank(n.Left) {
		Yyerror("methods must have a unique non-blank name")
	}

	dclcontext = PPARAM
	markdcl()
	Funcdepth++
	n.Outer = Curfn
	Curfn = n
	funcargs(n.Right)

	// funcbody is normally called after the parser has
	// seen the body of a function but since an interface
	// field declaration does not have a body, we must
	// call it now to pop the current declaration context.
	dclcontext = PAUTO

	funcbody(n)
}

/*
 * declare the function proper
 * and declare the arguments.
 * called in extern-declaration context
 * returns in auto-declaration context.
 */
func funchdr(n *Node) {
	// change the declaration context from extern to auto
	if Funcdepth == 0 && dclcontext != PEXTERN {
		Fatal("funchdr: dclcontext")
	}

	dclcontext = PAUTO
	markdcl()
	Funcdepth++

	n.Outer = Curfn
	Curfn = n

	if n.Nname != nil {
		funcargs(n.Nname.Ntype)
	} else if n.Ntype != nil {
		funcargs(n.Ntype)
	} else {
		funcargs2(n.Type)
	}
}

func funcargs(nt *Node) {
	if nt.Op != OTFUNC {
		Fatal("funcargs %v", Oconv(int(nt.Op), 0))
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
			Fatal("funcargs receiver %v", Oconv(int(n.Op), 0))
		}
		if n.Left != nil {
			n.Left.Op = ONAME
			n.Left.Ntype = n.Right
			declare(n.Left, PPARAM)
			if dclcontext == PAUTO {
				vargen++
				n.Left.Vargen = int32(vargen)
			}
		}
	}

	var n *Node
	for l := nt.List; l != nil; l = l.Next {
		n = l.N
		if n.Op != ODCLFIELD {
			Fatal("funcargs in %v", Oconv(int(n.Op), 0))
		}
		if n.Left != nil {
			n.Left.Op = ONAME
			n.Left.Ntype = n.Right
			declare(n.Left, PPARAM)
			if dclcontext == PAUTO {
				vargen++
				n.Left.Vargen = int32(vargen)
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
			Fatal("funcargs out %v", Oconv(int(n.Op), 0))
		}

		if n.Left == nil {
			// Name so that escape analysis can track it. ~r stands for 'result'.
			namebuf = fmt.Sprintf("~r%d", gen)
			gen++

			n.Left = newname(Lookup(namebuf))
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
			namebuf = fmt.Sprintf("~b%d", gen)
			gen++
			nn.Sym = Lookup(namebuf)
			n.Left = nn
		}

		n.Left.Ntype = n.Right
		declare(n.Left, PPARAMOUT)
		if dclcontext == PAUTO {
			i++
			n.Left.Vargen = int32(i)
		}
	}
}

/*
 * Same as funcargs, except run over an already constructed TFUNC.
 * This happens during import, where the hidden_fndcl rule has
 * used functype directly to parse the function's type.
 */
func funcargs2(t *Type) {
	if t.Etype != TFUNC {
		Fatal("funcargs2 %v", Tconv(t, 0))
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

/*
 * finish the body.
 * called in auto-declaration context.
 * returns in extern-declaration context.
 */
func funcbody(n *Node) {
	// change the declaration context from auto to extern
	if dclcontext != PAUTO {
		Fatal("funcbody: dclcontext")
	}
	popdcl()
	Funcdepth--
	Curfn = n.Outer
	n.Outer = nil
	if Funcdepth == 0 {
		dclcontext = PEXTERN
	}
}

/*
 * new type being defined with name s.
 */
func typedcl0(s *Sym) *Node {
	n := newname(s)
	n.Op = OTYPE
	declare(n, dclcontext)
	return n
}

/*
 * node n, which was returned by typedcl0
 * is being declared to have uncompiled type t.
 * return the ODCLTYPE node to use.
 */
func typedcl1(n *Node, t *Node, local int) *Node {
	n.Ntype = t
	n.Local = uint8(local)
	return Nod(ODCLTYPE, n, nil)
}

/*
 * structs, functions, and methods.
 * they don't belong here, but where do they belong?
 */
func checkembeddedtype(t *Type) {
	if t == nil {
		return
	}

	if t.Sym == nil && Isptr[t.Etype] != 0 {
		t = t.Type
		if t.Etype == TINTER {
			Yyerror("embedded type cannot be a pointer to interface")
		}
	}

	if Isptr[t.Etype] != 0 {
		Yyerror("embedded type cannot be a pointer")
	} else if t.Etype == TFORW && t.Embedlineno == 0 {
		t.Embedlineno = lineno
	}
}

func structfield(n *Node) *Type {
	lno := int(lineno)
	lineno = n.Lineno

	if n.Op != ODCLFIELD {
		Fatal("structfield: oops %v\n", Nconv(n, 0))
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
		f.Broke = 1
	}

	switch n.Val.Ctype {
	case CTSTR:
		f.Note = n.Val.U.Sval

	default:
		Yyerror("field annotation must be string")
		fallthrough

		// fallthrough
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

/*
 * convert a parsed id/type list into
 * a type for struct/interface/arglist
 */
func tostruct(l *NodeList) *Type {
	var f *Type
	t := typ(TSTRUCT)

	for tp := &t.Type; l != nil; l = l.Next {
		f = structfield(l.N)

		*tp = f
		tp = &f.Down
	}

	for f := t.Type; f != nil && t.Broke == 0; f = f.Down {
		if f.Broke != 0 {
			t.Broke = 1
		}
	}

	uniqgen++
	checkdupfields(t.Type, "field")

	if t.Broke == 0 {
		checkwidth(t)
	}

	return t
}

func tofunargs(l *NodeList) *Type {
	var f *Type

	t := typ(TSTRUCT)
	t.Funarg = 1

	for tp := &t.Type; l != nil; l = l.Next {
		f = structfield(l.N)
		f.Funarg = 1

		// esc.c needs to find f given a PPARAM to add the tag.
		if l.N.Left != nil && l.N.Left.Class == PPARAM {
			l.N.Left.Paramfld = f
		}

		*tp = f
		tp = &f.Down
	}

	for f := t.Type; f != nil && t.Broke == 0; f = f.Down {
		if f.Broke != 0 {
			t.Broke = 1
		}
	}

	return t
}

func interfacefield(n *Node) *Type {
	lno := int(lineno)
	lineno = n.Lineno

	if n.Op != ODCLFIELD {
		Fatal("interfacefield: oops %v\n", Nconv(n, 0))
	}

	if n.Val.Ctype != CTxxx {
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
					Yyerror("interface type loop involving %v", Tconv(n.Type, 0))
					f.Broke = 1

				default:
					Yyerror("interface contains embedded non-interface %v", Tconv(n.Type, 0))
					f.Broke = 1
				}
			}
		}
	}

	n.Right = nil

	f.Type = n.Type
	if f.Type == nil {
		f.Broke = 1
	}

	lineno = int32(lno)
	return f
}

func tointerface(l *NodeList) *Type {
	var f *Type
	var t1 *Type

	t := typ(TINTER)

	tp := &t.Type
	for ; l != nil; l = l.Next {
		f = interfacefield(l.N)

		if l.N.Left == nil && f.Type.Etype == TINTER {
			// embedded interface, inline methods
			for t1 = f.Type.Type; t1 != nil; t1 = t1.Down {
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

	for f := t.Type; f != nil && t.Broke == 0; f = f.Down {
		if f.Broke != 0 {
			t.Broke = 1
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

/*
 * check that the list of declarations is either all anonymous or all named
 */
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
		n := (*Node)(nil)
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

	nextt := (*Node)(nil)
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
		// TODO(rsc) this can go when typefmt case TFIELD in exportmode fmt.c prints _ instead of ?
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
			n.Isddd = 1
			if n.Left != nil {
				n.Left.Isddd = 1
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

/*
 * Is this field a method on an interface?
 * Those methods have an anonymous
 * *struct{} as the receiver.
 * (See fakethis above.)
 */
func isifacemethod(f *Type) bool {
	rcvr := getthisx(f).Type
	if rcvr.Sym != nil {
		return false
	}
	t := rcvr.Type
	if Isptr[t.Etype] == 0 {
		return false
	}
	t = t.Type
	if t.Sym != nil || t.Etype != TSTRUCT || t.Type != nil {
		return false
	}
	return true
}

/*
 * turn a parsed function declaration
 * into a type
 */
func functype(this *Node, in *NodeList, out *NodeList) *Type {
	t := typ(TFUNC)

	rcvr := (*NodeList)(nil)
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

	if t.Type.Broke != 0 || t.Type.Down.Broke != 0 || t.Type.Down.Down.Broke != 0 {
		t.Broke = 1
	}

	if this != nil {
		t.Thistuple = 1
	}
	t.Outtuple = count(out)
	t.Intuple = count(in)
	t.Outnamed = 0
	if t.Outtuple > 0 && out.N.Left != nil && out.N.Left.Orig != nil {
		s := out.N.Left.Orig.Sym
		if s != nil && (s.Name[0] != '~' || s.Name[1] != 'r') { // ~r%d is the name invented for an unnamed result
			t.Outnamed = 1
		}
	}

	return t
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
	if s == nil && Isptr[t.Etype] != 0 {
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
		if t0.Sym == nil && Isptr[t0.Etype] != 0 {
			p = fmt.Sprintf("(%v).%s.%s%s", Tconv(t0, obj.FmtLeft|obj.FmtShort), nsym.Pkg.Prefix, nsym.Name, suffix)
		} else {
			p = fmt.Sprintf("%v.%s.%s%s", Tconv(t0, obj.FmtLeft|obj.FmtShort), nsym.Pkg.Prefix, nsym.Name, suffix)
		}
	} else {
		if t0.Sym == nil && Isptr[t0.Etype] != 0 {
			p = fmt.Sprintf("(%v).%s%s", Tconv(t0, obj.FmtLeft|obj.FmtShort), nsym.Name, suffix)
		} else {
			p = fmt.Sprintf("%v.%s%s", Tconv(t0, obj.FmtLeft|obj.FmtShort), nsym.Name, suffix)
		}
	}

	if spkg == nil {
		if methodsym_toppkg == nil {
			methodsym_toppkg = mkpkg(newstrlit("go"))
		}
		spkg = methodsym_toppkg
	}

	s = Pkglookup(p, spkg)

	return s

bad:
	Yyerror("illegal receiver type: %v", Tconv(t0, 0))
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
		return newname(n.Sym)
	}

	var p string
	if star != "" {
		p = fmt.Sprintf("(%s%v).%v", star, Sconv(t.Sym, 0), Sconv(n.Sym, 0))
	} else {
		p = fmt.Sprintf("%v.%v", Sconv(t.Sym, 0), Sconv(n.Sym, 0))
	}

	if exportname(t.Sym.Name) {
		n = newname(Lookup(p))
	} else {
		n = newname(Pkglookup(p, t.Sym.Pkg))
	}

	return n
}

/*
 * add a method, declared as a function,
 * n is fieldname, pa is base type, t is function type
 */
func addmethod(sf *Sym, t *Type, local bool, nointerface bool) {
	// get field sym
	if sf == nil {
		Fatal("no method symbol")
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
			if Isptr[t.Etype] != 0 {
				if t.Sym != nil {
					Yyerror("invalid receiver type %v (%v is a pointer type)", Tconv(pa, 0), Tconv(t, 0))
					return
				}

				t = t.Type
			}

			if t.Broke != 0 { // rely on typecheck having complained before
				return
			}
			if t.Sym == nil {
				Yyerror("invalid receiver type %v (%v is an unnamed type)", Tconv(pa, 0), Tconv(t, 0))
				return
			}

			if Isptr[t.Etype] != 0 {
				Yyerror("invalid receiver type %v (%v is a pointer type)", Tconv(pa, 0), Tconv(t, 0))
				return
			}

			if t.Etype == TINTER {
				Yyerror("invalid receiver type %v (%v is an interface type)", Tconv(pa, 0), Tconv(t, 0))
				return
			}
		}

		// Should have picked off all the reasons above,
		// but just in case, fall back to generic error.
		Yyerror("invalid receiver type %v (%v / %v)", Tconv(pa, 0), Tconv(pa, obj.FmtLong), Tconv(t, obj.FmtLong))

		return
	}

	pa = f
	if pa.Etype == TSTRUCT {
		for f := pa.Type; f != nil; f = f.Down {
			if f.Sym == sf {
				Yyerror("type %v has both field and method named %v", Tconv(pa, 0), Sconv(sf, 0))
				return
			}
		}
	}

	if local && pa.Local == 0 {
		// defining method on non-local type.
		Yyerror("cannot define new methods on non-local type %v", Tconv(pa, 0))

		return
	}

	n := Nod(ODCLFIELD, newname(sf), nil)
	n.Type = t

	d := (*Type)(nil) // last found
	for f := pa.Method; f != nil; f = f.Down {
		d = f
		if f.Etype != TFIELD {
			Fatal("addmethod: not TFIELD: %v", Tconv(f, obj.FmtLong))
		}
		if sf.Name != f.Sym.Name {
			continue
		}
		if !Eqtype(t, f.Type) {
			Yyerror("method redeclared: %v.%v\n\t%v\n\t%v", Tconv(pa, 0), Sconv(sf, 0), Tconv(f.Type, 0), Tconv(t, 0))
		}
		return
	}

	f = structfield(n)
	f.Nointerface = nointerface

	// during import unexported method names should be in the type's package
	if importpkg != nil && f.Sym != nil && !exportname(f.Sym.Name) && f.Sym.Pkg != structpkg {
		Fatal("imported method name %v in wrong package %s\n", Sconv(f.Sym, obj.FmtSign), structpkg.Name)
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
			Fatal("funccompile missing type")
		}
		return
	}

	// assign parameter offsets
	checkwidth(n.Type)

	if Curfn != nil {
		Fatal("funccompile %v inside %v", Sconv(n.Nname.Sym, 0), Sconv(Curfn.Nname.Sym, 0))
	}

	Stksize = 0
	dclcontext = PAUTO
	Funcdepth = n.Funcdepth + 1
	compile(n)
	Curfn = nil
	Funcdepth = 0
	dclcontext = PEXTERN
}

func funcsym(s *Sym) *Sym {
	p := fmt.Sprintf("%s·f", s.Name)
	s1 := Pkglookup(p, s.Pkg)

	if s1.Def == nil {
		s1.Def = newname(s1)
		s1.Def.Shortname = newname(s)
		funcsyms = list(funcsyms, s1.Def)
	}

	return s1
}
