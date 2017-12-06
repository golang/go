// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bytes"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
	"fmt"
	"strings"
)

// Declaration stack & operations

var externdcl []*Node

func testdclstack() {
	if !types.IsDclstackValid() {
		if nerrors != 0 {
			errorexit()
		}
		Fatalf("mark left on the dclstack")
	}
}

// redeclare emits a diagnostic about symbol s being redeclared somewhere.
func redeclare(s *types.Sym, where string) {
	if !s.Lastlineno.IsKnown() {
		var tmp string
		if s.Origpkg != nil {
			tmp = s.Origpkg.Path
		} else {
			tmp = s.Pkg.Path
		}
		pkgstr := tmp
		yyerror("%v redeclared %s\n"+
			"\tprevious declaration during import %q", s, where, pkgstr)
	} else {
		line1 := lineno
		line2 := s.Lastlineno

		// When an import and a declaration collide in separate files,
		// present the import as the "redeclared", because the declaration
		// is visible where the import is, but not vice versa.
		// See issue 4510.
		if s.Def == nil {
			line2 = line1
			line1 = s.Lastlineno
		}

		yyerrorl(line1, "%v redeclared %s\n"+
			"\tprevious declaration at %v", s, where, linestr(line2))
	}
}

var vargen int

// declare individual names - var, typ, const

var declare_typegen int

// declare records that Node n declares symbol n.Sym in the specified
// declaration context.
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
	n.Pos = lineno
	s := n.Sym

	// kludgy: typecheckok means we're past parsing. Eg genwrapper may declare out of package names later.
	if !inimport && !typecheckok && s.Pkg != localpkg {
		yyerror("cannot declare name %v", s)
	}

	gen := 0
	if ctxt == PEXTERN {
		if s.Name == "init" {
			yyerror("cannot declare init - must be func")
		}
		if s.Name == "main" && localpkg.Name == "main" {
			yyerror("cannot declare main - must be func")
		}
		externdcl = append(externdcl, n)
	} else {
		if Curfn == nil && ctxt == PAUTO {
			Fatalf("automatic outside function")
		}
		if Curfn != nil {
			Curfn.Func.Dcl = append(Curfn.Func.Dcl, n)
		}
		if n.Op == OTYPE {
			declare_typegen++
			gen = declare_typegen
		} else if n.Op == ONAME && ctxt == PAUTO && !strings.Contains(s.Name, "·") {
			vargen++
			gen = vargen
		}
		types.Pushdcl(s)
		n.Name.Curfn = Curfn
	}

	if ctxt == PAUTO {
		n.Xoffset = 0
	}

	if s.Block == types.Block {
		// functype will print errors about duplicate function arguments.
		// Don't repeat the error here.
		if ctxt != PPARAM && ctxt != PPARAMOUT {
			redeclare(s, "in this block")
		}
	}

	s.Block = types.Block
	s.Lastlineno = lineno
	s.Def = asTypesNode(n)
	n.Name.Vargen = int32(gen)
	n.Name.Funcdepth = funcdepth
	n.SetClass(ctxt)

	autoexport(n, ctxt)
}

func addvar(n *Node, t *types.Type, ctxt Class) {
	if n == nil || n.Sym == nil || (n.Op != ONAME && n.Op != ONONAME) || t == nil {
		Fatalf("addvar: n=%v t=%v nil", n, t)
	}

	n.Op = ONAME
	declare(n, ctxt)
	n.Type = t
}

// declare variables from grammar
// new_name_list (type | [type] = expr_list)
func variter(vl []*Node, t *Node, el []*Node) []*Node {
	var init []*Node
	doexpr := len(el) > 0

	if len(el) == 1 && len(vl) > 1 {
		e := el[0]
		as2 := nod(OAS2, nil, nil)
		as2.List.Set(vl)
		as2.Rlist.Set1(e)
		for _, v := range vl {
			v.Op = ONAME
			declare(v, dclcontext)
			v.Name.Param.Ntype = t
			v.Name.Defn = as2
			if funcdepth > 0 {
				init = append(init, nod(ODCL, v, nil))
			}
		}

		return append(init, as2)
	}

	for _, v := range vl {
		var e *Node
		if doexpr {
			if len(el) == 0 {
				yyerror("missing expression in var declaration")
				break
			}
			e = el[0]
			el = el[1:]
		}

		v.Op = ONAME
		declare(v, dclcontext)
		v.Name.Param.Ntype = t

		if e != nil || funcdepth > 0 || isblank(v) {
			if funcdepth > 0 {
				init = append(init, nod(ODCL, v, nil))
			}
			e = nod(OAS, v, e)
			init = append(init, e)
			if e.Right != nil {
				v.Name.Defn = e
			}
		}
	}

	if len(el) != 0 {
		yyerror("extra expression in var declaration")
	}
	return init
}

// newnoname returns a new ONONAME Node associated with symbol s.
func newnoname(s *types.Sym) *Node {
	if s == nil {
		Fatalf("newnoname nil")
	}
	n := nod(ONONAME, nil, nil)
	n.Sym = s
	n.SetAddable(true)
	n.Xoffset = 0
	return n
}

// newfuncname generates a new name node for a function or method.
// TODO(rsc): Use an ODCLFUNC node instead. See comment in CL 7360.
func newfuncname(s *types.Sym) *Node {
	return newfuncnamel(lineno, s)
}

// newfuncnamel generates a new name node for a function or method.
// TODO(rsc): Use an ODCLFUNC node instead. See comment in CL 7360.
func newfuncnamel(pos src.XPos, s *types.Sym) *Node {
	n := newnamel(pos, s)
	n.Func = new(Func)
	n.Func.SetIsHiddenClosure(Curfn != nil)
	return n
}

// this generates a new name node for a name
// being declared.
func dclname(s *types.Sym) *Node {
	n := newname(s)
	n.Op = ONONAME // caller will correct it
	return n
}

func typenod(t *types.Type) *Node {
	return typenodl(src.NoXPos, t)
}

func typenodl(pos src.XPos, t *types.Type) *Node {
	// if we copied another type with *t = *u
	// then t->nod might be out of date, so
	// check t->nod->type too
	if asNode(t.Nod) == nil || asNode(t.Nod).Type != t {
		t.Nod = asTypesNode(nodl(pos, OTYPE, nil, nil))
		asNode(t.Nod).Type = t
		asNode(t.Nod).Sym = t.Sym
	}

	return asNode(t.Nod)
}

func anonfield(typ *types.Type) *Node {
	return nod(ODCLFIELD, nil, typenod(typ))
}

func namedfield(s string, typ *types.Type) *Node {
	return symfield(lookup(s), typ)
}

func symfield(s *types.Sym, typ *types.Type) *Node {
	return nod(ODCLFIELD, newname(s), typenod(typ))
}

// oldname returns the Node that declares symbol s in the current scope.
// If no such Node currently exists, an ONONAME Node is returned instead.
func oldname(s *types.Sym) *Node {
	n := asNode(s.Def)
	if n == nil {
		// Maybe a top-level declaration will come along later to
		// define s. resolve will check s.Def again once all input
		// source has been processed.
		return newnoname(s)
	}

	if Curfn != nil && n.Op == ONAME && n.Name.Funcdepth > 0 && n.Name.Funcdepth != funcdepth {
		// Inner func is referring to var in outer func.
		//
		// TODO(rsc): If there is an outer variable x and we
		// are parsing x := 5 inside the closure, until we get to
		// the := it looks like a reference to the outer x so we'll
		// make x a closure variable unnecessarily.
		c := n.Name.Param.Innermost
		if c == nil || c.Name.Funcdepth != funcdepth {
			// Do not have a closure var for the active closure yet; make one.
			c = newname(s)
			c.SetClass(PAUTOHEAP)
			c.SetIsClosureVar(true)
			c.SetIsddd(n.Isddd())
			c.Name.Defn = n
			c.SetAddable(false)
			c.Name.Funcdepth = funcdepth

			// Link into list of active closure variables.
			// Popped from list in func closurebody.
			c.Name.Param.Outer = n.Name.Param.Innermost
			n.Name.Param.Innermost = c

			Curfn.Func.Cvars.Append(c)
		}

		// return ref to closure var, not original
		return c
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

func colasdefn(left []*Node, defn *Node) {
	for _, n := range left {
		if n.Sym != nil {
			n.Sym.SetUniq(true)
		}
	}

	var nnew, nerr int
	for i, n := range left {
		if isblank(n) {
			continue
		}
		if !colasname(n) {
			yyerrorl(defn.Pos, "non-name %v on left side of :=", n)
			nerr++
			continue
		}

		if !n.Sym.Uniq() {
			yyerrorl(defn.Pos, "%v repeated on left side of :=", n.Sym)
			n.SetDiag(true)
			nerr++
			continue
		}

		n.Sym.SetUniq(false)
		if n.Sym.Block == types.Block {
			continue
		}

		nnew++
		n = newname(n.Sym)
		declare(n, dclcontext)
		n.Name.Defn = defn
		defn.Ninit.Append(nod(ODCL, n, nil))
		left[i] = n
	}

	if nnew == 0 && nerr == 0 {
		yyerrorl(defn.Pos, "no new variables on left side of :=")
	}
}

// declare the arguments in an
// interface field declaration.
func ifacedcl(n *Node) {
	if n.Op != ODCLFIELD || n.Right == nil {
		Fatalf("ifacedcl")
	}

	if isblank(n.Left) {
		yyerror("methods must have a unique non-blank name")
	}
}

// declare the function proper
// and declare the arguments.
// called in extern-declaration context
// returns in auto-declaration context.
func funchdr(n *Node) {
	// change the declaration context from extern to auto
	if funcdepth == 0 && dclcontext != PEXTERN {
		Fatalf("funchdr: dclcontext = %d", dclcontext)
	}

	dclcontext = PAUTO
	funcstart(n)

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
		Fatalf("funcargs %v", nt.Op)
	}

	// re-start the variable generation number
	// we want to use small numbers for the return variables,
	// so let them have the chunk starting at 1.
	vargen = nt.Rlist.Len()

	// declare the receiver and in arguments.
	// no n->defn because type checking of func header
	// will not fill in the types until later
	if nt.Left != nil {
		n := nt.Left
		if n.Op != ODCLFIELD {
			Fatalf("funcargs receiver %v", n.Op)
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

	for _, n := range nt.List.Slice() {
		if n.Op != ODCLFIELD {
			Fatalf("funcargs in %v", n.Op)
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
	gen := nt.List.Len()
	var i int = 0
	for _, n := range nt.Rlist.Slice() {
		if n.Op != ODCLFIELD {
			Fatalf("funcargs out %v", n.Op)
		}

		if n.Left == nil {
			// Name so that escape analysis can track it. ~r stands for 'result'.
			n.Left = newname(lookupN("~r", gen))
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
			nn := *n.Left
			nn.Orig = &nn
			nn.Sym = lookupN("~b", gen)
			gen++
			n.Left = &nn
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
func funcargs2(t *types.Type) {
	if t.Etype != TFUNC {
		Fatalf("funcargs2 %v", t)
	}

	for _, ft := range t.Recvs().Fields().Slice() {
		if asNode(ft.Nname) == nil || asNode(ft.Nname).Sym == nil {
			continue
		}
		n := asNode(ft.Nname) // no need for newname(ft->nname->sym)
		n.Type = ft.Type
		declare(n, PPARAM)
	}

	for _, ft := range t.Params().Fields().Slice() {
		if asNode(ft.Nname) == nil || asNode(ft.Nname).Sym == nil {
			continue
		}
		n := asNode(ft.Nname)
		n.Type = ft.Type
		declare(n, PPARAM)
	}

	for _, ft := range t.Results().Fields().Slice() {
		if asNode(ft.Nname) == nil || asNode(ft.Nname).Sym == nil {
			continue
		}
		n := asNode(ft.Nname)
		n.Type = ft.Type
		declare(n, PPARAMOUT)
	}
}

var funcstack []*Node // stack of previous values of Curfn
var funcdepth int32   // len(funcstack) during parsing, but then forced to be the same later during compilation

// start the function.
// called before funcargs; undone at end of funcbody.
func funcstart(n *Node) {
	types.Markdcl()
	funcstack = append(funcstack, Curfn)
	funcdepth++
	Curfn = n
}

// finish the body.
// called in auto-declaration context.
// returns in extern-declaration context.
func funcbody() {
	// change the declaration context from auto to extern
	if dclcontext != PAUTO {
		Fatalf("funcbody: unexpected dclcontext %d", dclcontext)
	}
	types.Popdcl()
	funcstack, Curfn = funcstack[:len(funcstack)-1], funcstack[len(funcstack)-1]
	funcdepth--
	if funcdepth == 0 {
		dclcontext = PEXTERN
	}
}

// structs, functions, and methods.
// they don't belong here, but where do they belong?
func checkembeddedtype(t *types.Type) {
	if t == nil {
		return
	}

	if t.Sym == nil && t.IsPtr() {
		t = t.Elem()
		if t.IsInterface() {
			yyerror("embedded type cannot be a pointer to interface")
		}
	}

	if t.IsPtr() || t.IsUnsafePtr() {
		yyerror("embedded type cannot be a pointer")
	} else if t.Etype == TFORW && !t.ForwardType().Embedlineno.IsKnown() {
		t.ForwardType().Embedlineno = lineno
	}
}

func structfield(n *Node) *types.Field {
	lno := lineno
	lineno = n.Pos

	if n.Op != ODCLFIELD {
		Fatalf("structfield: oops %v\n", n)
	}

	f := types.NewField()
	f.SetIsddd(n.Isddd())

	if n.Right != nil {
		n.Right = typecheck(n.Right, Etype)
		n.Type = n.Right.Type
		if n.Left != nil {
			n.Left.Type = n.Type
		}
		if n.Embedded() {
			checkembeddedtype(n.Type)
		}
	}

	n.Right = nil

	f.Type = n.Type
	if f.Type == nil {
		f.SetBroke(true)
	}

	switch u := n.Val().U.(type) {
	case string:
		f.Note = u
	default:
		yyerror("field tag must be a string")
	case nil:
		// no-op
	}

	if n.Left != nil && n.Left.Op == ONAME {
		f.Nname = asTypesNode(n.Left)
		if n.Embedded() {
			f.Embedded = 1
		} else {
			f.Embedded = 0
		}
		f.Sym = asNode(f.Nname).Sym
	}

	lineno = lno
	return f
}

// checkdupfields emits errors for duplicately named fields or methods in
// a list of struct or interface types.
func checkdupfields(what string, ts ...*types.Type) {
	seen := make(map[*types.Sym]bool)
	for _, t := range ts {
		for _, f := range t.Fields().Slice() {
			if f.Sym == nil || f.Sym.IsBlank() || asNode(f.Nname) == nil {
				continue
			}
			if seen[f.Sym] {
				yyerrorl(asNode(f.Nname).Pos, "duplicate %s %s", what, f.Sym.Name)
				continue
			}
			seen[f.Sym] = true
		}
	}
}

// convert a parsed id/type list into
// a type for struct/interface/arglist
func tostruct(l []*Node) *types.Type {
	t := types.New(TSTRUCT)
	tostruct0(t, l)
	return t
}

func tostruct0(t *types.Type, l []*Node) {
	if t == nil || !t.IsStruct() {
		Fatalf("struct expected")
	}

	fields := make([]*types.Field, len(l))
	for i, n := range l {
		f := structfield(n)
		if f.Broke() {
			t.SetBroke(true)
		}
		fields[i] = f
	}
	t.SetFields(fields)

	checkdupfields("field", t)

	if !t.Broke() {
		checkwidth(t)
	}
}

func tofunargs(l []*Node, funarg types.Funarg) *types.Type {
	t := types.New(TSTRUCT)
	t.StructType().Funarg = funarg

	fields := make([]*types.Field, len(l))
	for i, n := range l {
		f := structfield(n)
		f.Funarg = funarg

		// esc.go needs to find f given a PPARAM to add the tag.
		if n.Left != nil && n.Left.Class() == PPARAM {
			n.Left.Name.Param.Field = f
		}
		if f.Broke() {
			t.SetBroke(true)
		}
		fields[i] = f
	}
	t.SetFields(fields)
	return t
}

func tofunargsfield(fields []*types.Field, funarg types.Funarg) *types.Type {
	t := types.New(TSTRUCT)
	t.StructType().Funarg = funarg

	for _, f := range fields {
		f.Funarg = funarg

		// esc.go needs to find f given a PPARAM to add the tag.
		if asNode(f.Nname) != nil && asNode(f.Nname).Class() == PPARAM {
			asNode(f.Nname).Name.Param.Field = f
		}
	}
	t.SetFields(fields)
	return t
}

func interfacefield(n *Node) *types.Field {
	lno := lineno
	lineno = n.Pos

	if n.Op != ODCLFIELD {
		Fatalf("interfacefield: oops %v\n", n)
	}

	if n.Val().Ctype() != CTxxx {
		yyerror("interface method cannot have annotation")
	}

	// MethodSpec = MethodName Signature | InterfaceTypeName .
	//
	// If Left != nil, then Left is MethodName and Right is Signature.
	// Otherwise, Right is InterfaceTypeName.

	if n.Right != nil {
		n.Right = typecheck(n.Right, Etype)
		n.Type = n.Right.Type
		n.Right = nil
	}

	f := types.NewField()
	if n.Left != nil {
		f.Nname = asTypesNode(n.Left)
		f.Sym = asNode(f.Nname).Sym
	} else {
		// Placeholder ONAME just to hold Pos.
		// TODO(mdempsky): Add Pos directly to Field instead.
		f.Nname = asTypesNode(newname(nblank.Sym))
	}

	f.Type = n.Type
	if f.Type == nil {
		f.SetBroke(true)
	}

	lineno = lno
	return f
}

func tointerface(l []*Node) *types.Type {
	if len(l) == 0 {
		return types.Types[TINTER]
	}
	t := types.New(TINTER)
	tointerface0(t, l)
	return t
}

func tointerface0(t *types.Type, l []*Node) {
	if t == nil || !t.IsInterface() {
		Fatalf("interface expected")
	}

	var fields []*types.Field
	for _, n := range l {
		f := interfacefield(n)
		if f.Broke() {
			t.SetBroke(true)
		}
		fields = append(fields, f)
	}
	t.SetInterface(fields)
}

func fakeRecv() *Node {
	return anonfield(types.FakeRecvType())
}

func fakeRecvField() *types.Field {
	f := types.NewField()
	f.Type = types.FakeRecvType()
	return f
}

// isifacemethod reports whether (field) m is
// an interface method. Such methods have the
// special receiver type types.FakeRecvType().
func isifacemethod(f *types.Type) bool {
	return f.Recv().Type == types.FakeRecvType()
}

// turn a parsed function declaration into a type
func functype(this *Node, in, out []*Node) *types.Type {
	t := types.New(TFUNC)
	functype0(t, this, in, out)
	return t
}

func functype0(t *types.Type, this *Node, in, out []*Node) {
	if t == nil || t.Etype != TFUNC {
		Fatalf("function type expected")
	}

	var rcvr []*Node
	if this != nil {
		rcvr = []*Node{this}
	}
	t.FuncType().Receiver = tofunargs(rcvr, types.FunargRcvr)
	t.FuncType().Results = tofunargs(out, types.FunargResults)
	t.FuncType().Params = tofunargs(in, types.FunargParams)

	checkdupfields("argument", t.Recvs(), t.Results(), t.Params())

	if t.Recvs().Broke() || t.Results().Broke() || t.Params().Broke() {
		t.SetBroke(true)
	}

	t.FuncType().Outnamed = false
	if len(out) > 0 && out[0].Left != nil && out[0].Left.Orig != nil {
		s := out[0].Left.Orig.Sym
		if s != nil && (s.Name[0] != '~' || s.Name[1] != 'r') { // ~r%d is the name invented for an unnamed result
			t.FuncType().Outnamed = true
		}
	}
}

func functypefield(this *types.Field, in, out []*types.Field) *types.Type {
	t := types.New(TFUNC)
	functypefield0(t, this, in, out)
	return t
}

func functypefield0(t *types.Type, this *types.Field, in, out []*types.Field) {
	var rcvr []*types.Field
	if this != nil {
		rcvr = []*types.Field{this}
	}
	t.FuncType().Receiver = tofunargsfield(rcvr, types.FunargRcvr)
	t.FuncType().Results = tofunargsfield(out, types.FunargRcvr)
	t.FuncType().Params = tofunargsfield(in, types.FunargRcvr)

	t.FuncType().Outnamed = false
	if len(out) > 0 && asNode(out[0].Nname) != nil && asNode(out[0].Nname).Orig != nil {
		s := asNode(out[0].Nname).Orig.Sym
		if s != nil && (s.Name[0] != '~' || s.Name[1] != 'r') { // ~r%d is the name invented for an unnamed result
			t.FuncType().Outnamed = true
		}
	}
}

var methodsym_toppkg *types.Pkg

func methodsym(nsym *types.Sym, t0 *types.Type, iface bool) *types.Sym {
	if t0 == nil {
		Fatalf("methodsym: nil receiver type")
	}

	t := t0
	s := t.Sym
	if s == nil && t.IsPtr() {
		t = t.Elem()
		if t == nil {
			Fatalf("methodsym: ptrto nil")
		}
		s = t.Sym
	}

	// if t0 == *t and t0 has a sym,
	// we want to see *t, not t0, in the method name.
	if t != t0 && t0.Sym != nil {
		t0 = types.NewPtr(t)
	}

	suffix := ""
	if iface {
		dowidth(t0)
		if t0.Width < int64(Widthptr) {
			suffix = "·i"
		}
	}

	var spkg *types.Pkg
	if s != nil {
		spkg = s.Pkg
	}
	pkgprefix := ""
	if (spkg == nil || nsym.Pkg != spkg) && !exportname(nsym.Name) && nsym.Pkg.Prefix != `""` {
		pkgprefix = "." + nsym.Pkg.Prefix
	}
	var p string
	if t0.Sym == nil && t0.IsPtr() {
		p = fmt.Sprintf("(%-S)%s.%s%s", t0, pkgprefix, nsym.Name, suffix)
	} else {
		p = fmt.Sprintf("%-S%s.%s%s", t0, pkgprefix, nsym.Name, suffix)
	}

	if spkg == nil {
		if methodsym_toppkg == nil {
			methodsym_toppkg = types.NewPkg("go", "")
		}
		spkg = methodsym_toppkg
	}

	return spkg.Lookup(p)
}

// methodname is a misnomer because this now returns a Sym, rather
// than an ONAME.
// TODO(mdempsky): Reconcile with methodsym.
func methodname(s *types.Sym, recv *types.Type) *types.Sym {
	star := false
	if recv.IsPtr() {
		star = true
		recv = recv.Elem()
	}

	tsym := recv.Sym
	if tsym == nil || s.IsBlank() {
		return s
	}

	var p string
	if star {
		p = fmt.Sprintf("(*%v).%v", tsym.Name, s)
	} else {
		p = fmt.Sprintf("%v.%v", tsym, s)
	}

	s = tsym.Pkg.Lookup(p)

	return s
}

// Add a method, declared as a function.
// - msym is the method symbol
// - t is function type (with receiver)
// Returns a pointer to the existing or added Field.
func addmethod(msym *types.Sym, t *types.Type, local, nointerface bool) *types.Field {
	if msym == nil {
		Fatalf("no method symbol")
	}

	// get parent type sym
	rf := t.Recv() // ptr to this structure
	if rf == nil {
		yyerror("missing receiver")
		return nil
	}

	mt := methtype(rf.Type)
	if mt == nil || mt.Sym == nil {
		pa := rf.Type
		t := pa
		if t != nil && t.IsPtr() {
			if t.Sym != nil {
				yyerror("invalid receiver type %v (%v is a pointer type)", pa, t)
				return nil
			}
			t = t.Elem()
		}

		switch {
		case t == nil || t.Broke():
			// rely on typecheck having complained before
		case t.Sym == nil:
			yyerror("invalid receiver type %v (%v is an unnamed type)", pa, t)
		case t.IsPtr():
			yyerror("invalid receiver type %v (%v is a pointer type)", pa, t)
		case t.IsInterface():
			yyerror("invalid receiver type %v (%v is an interface type)", pa, t)
		default:
			// Should have picked off all the reasons above,
			// but just in case, fall back to generic error.
			yyerror("invalid receiver type %v (%L / %L)", pa, pa, t)
		}
		return nil
	}

	if local && mt.Sym.Pkg != localpkg {
		yyerror("cannot define new methods on non-local type %v", mt)
		return nil
	}

	if msym.IsBlank() {
		return nil
	}

	if mt.IsStruct() {
		for _, f := range mt.Fields().Slice() {
			if f.Sym == msym {
				yyerror("type %v has both field and method named %v", mt, msym)
				return nil
			}
		}
	}

	for _, f := range mt.Methods().Slice() {
		if msym.Name != f.Sym.Name {
			continue
		}
		// eqtype only checks that incoming and result parameters match,
		// so explicitly check that the receiver parameters match too.
		if !eqtype(t, f.Type) || !eqtype(t.Recv().Type, f.Type.Recv().Type) {
			yyerror("method redeclared: %v.%v\n\t%v\n\t%v", mt, msym, f.Type, t)
		}
		return f
	}

	f := types.NewField()
	f.Sym = msym
	f.Nname = asTypesNode(newname(msym))
	f.Type = t
	f.SetNointerface(nointerface)

	mt.Methods().Append(f)
	return f
}

func funccompile(n *Node) {
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

	dclcontext = PAUTO
	funcdepth = n.Func.Depth + 1
	compile(n)
	Curfn = nil
	funcdepth = 0
	dclcontext = PEXTERN
}

func funcsymname(s *types.Sym) string {
	return s.Name + "·f"
}

// funcsym returns s·f.
func funcsym(s *types.Sym) *types.Sym {
	// funcsymsmu here serves to protect not just mutations of funcsyms (below),
	// but also the package lookup of the func sym name,
	// since this function gets called concurrently from the backend.
	// There are no other concurrent package lookups in the backend,
	// except for the types package, which is protected separately.
	// Reusing funcsymsmu to also cover this package lookup
	// avoids a general, broader, expensive package lookup mutex.
	// Note makefuncsym also does package look-up of func sym names,
	// but that it is only called serially, from the front end.
	funcsymsmu.Lock()
	sf, existed := s.Pkg.LookupOK(funcsymname(s))
	// Don't export s·f when compiling for dynamic linking.
	// When dynamically linking, the necessary function
	// symbols will be created explicitly with makefuncsym.
	// See the makefuncsym comment for details.
	if !Ctxt.Flag_dynlink && !existed {
		funcsyms = append(funcsyms, s)
	}
	funcsymsmu.Unlock()
	return sf
}

// makefuncsym ensures that s·f is exported.
// It is only used with -dynlink.
// When not compiling for dynamic linking,
// the funcsyms are created as needed by
// the packages that use them.
// Normally we emit the s·f stubs as DUPOK syms,
// but DUPOK doesn't work across shared library boundaries.
// So instead, when dynamic linking, we only create
// the s·f stubs in s's package.
func makefuncsym(s *types.Sym) {
	if !Ctxt.Flag_dynlink {
		Fatalf("makefuncsym dynlink")
	}
	if s.IsBlank() {
		return
	}
	if compiling_runtime && (s.Name == "getg" || s.Name == "getclosureptr" || s.Name == "getcallerpc" || s.Name == "getcallersp") {
		// runtime.getg(), getclosureptr(), getcallerpc(), and
		// getcallersp() are not real functions and so do not
		// get funcsyms.
		return
	}
	if _, existed := s.Pkg.LookupOK(funcsymname(s)); !existed {
		funcsyms = append(funcsyms, s)
	}
}

func dclfunc(sym *types.Sym, tfn *Node) *Node {
	if tfn.Op != OTFUNC {
		Fatalf("expected OTFUNC node, got %v", tfn)
	}

	fn := nod(ODCLFUNC, nil, nil)
	fn.Func.Nname = newname(sym)
	fn.Func.Nname.Name.Defn = fn
	fn.Func.Nname.Name.Param.Ntype = tfn
	declare(fn.Func.Nname, PFUNC)
	funchdr(fn)
	fn.Func.Nname.Name.Param.Ntype = typecheck(fn.Func.Nname.Name.Param.Ntype, Etype)
	return fn
}

type nowritebarrierrecChecker struct {
	// extraCalls contains extra function calls that may not be
	// visible during later analysis. It maps from the ODCLFUNC of
	// the caller to a list of callees.
	extraCalls map[*Node][]nowritebarrierrecCall

	// curfn is the current function during AST walks.
	curfn *Node
}

type nowritebarrierrecCall struct {
	target *Node    // ODCLFUNC of caller or callee
	lineno src.XPos // line of call
}

type nowritebarrierrecCallSym struct {
	target *obj.LSym // LSym of callee
	lineno src.XPos  // line of call
}

// newNowritebarrierrecChecker creates a nowritebarrierrecChecker. It
// must be called before transformclosure and walk.
func newNowritebarrierrecChecker() *nowritebarrierrecChecker {
	c := &nowritebarrierrecChecker{
		extraCalls: make(map[*Node][]nowritebarrierrecCall),
	}

	// Find all systemstack calls and record their targets. In
	// general, flow analysis can't see into systemstack, but it's
	// important to handle it for this check, so we model it
	// directly. This has to happen before transformclosure since
	// it's a lot harder to work out the argument after.
	for _, n := range xtop {
		if n.Op != ODCLFUNC {
			continue
		}
		c.curfn = n
		inspect(n, c.findExtraCalls)
	}
	c.curfn = nil
	return c
}

func (c *nowritebarrierrecChecker) findExtraCalls(n *Node) bool {
	if n.Op != OCALLFUNC {
		return true
	}
	fn := n.Left
	if fn == nil || fn.Op != ONAME || fn.Class() != PFUNC || fn.Name.Defn == nil {
		return true
	}
	if !isRuntimePkg(fn.Sym.Pkg) || fn.Sym.Name != "systemstack" {
		return true
	}

	var callee *Node
	arg := n.List.First()
	switch arg.Op {
	case ONAME:
		callee = arg.Name.Defn
	case OCLOSURE:
		callee = arg.Func.Closure
	default:
		Fatalf("expected ONAME or OCLOSURE node, got %+v", arg)
	}
	if callee.Op != ODCLFUNC {
		Fatalf("expected ODCLFUNC node, got %+v", callee)
	}
	c.extraCalls[c.curfn] = append(c.extraCalls[c.curfn], nowritebarrierrecCall{callee, n.Pos})
	return true
}

// recordCall records a call from ODCLFUNC node "from", to function
// symbol "to" at position pos.
//
// This should be done as late as possible during compilation to
// capture precise call graphs. The target of the call is an LSym
// because that's all we know after we start SSA.
//
// This can be called concurrently for different from Nodes.
func (c *nowritebarrierrecChecker) recordCall(from *Node, to *obj.LSym, pos src.XPos) {
	if from.Op != ODCLFUNC {
		Fatalf("expected ODCLFUNC, got %v", from)
	}
	// We record this information on the *Func so this is
	// concurrent-safe.
	fn := from.Func
	if fn.nwbrCalls == nil {
		fn.nwbrCalls = new([]nowritebarrierrecCallSym)
	}
	*fn.nwbrCalls = append(*fn.nwbrCalls, nowritebarrierrecCallSym{to, pos})
}

func (c *nowritebarrierrecChecker) check() {
	// We walk the call graph as late as possible so we can
	// capture all calls created by lowering, but this means we
	// only get to see the obj.LSyms of calls. symToFunc lets us
	// get back to the ODCLFUNCs.
	symToFunc := make(map[*obj.LSym]*Node)
	// funcs records the back-edges of the BFS call graph walk. It
	// maps from the ODCLFUNC of each function that must not have
	// write barriers to the call that inhibits them. Functions
	// that are directly marked go:nowritebarrierrec are in this
	// map with a zero-valued nowritebarrierrecCall. This also
	// acts as the set of marks for the BFS of the call graph.
	funcs := make(map[*Node]nowritebarrierrecCall)
	// q is the queue of ODCLFUNC Nodes to visit in BFS order.
	var q nodeQueue

	for _, n := range xtop {
		if n.Op != ODCLFUNC {
			continue
		}

		symToFunc[n.Func.lsym] = n

		// Make nowritebarrierrec functions BFS roots.
		if n.Func.Pragma&Nowritebarrierrec != 0 {
			funcs[n] = nowritebarrierrecCall{}
			q.pushRight(n)
		}
		// Check go:nowritebarrier functions.
		if n.Func.Pragma&Nowritebarrier != 0 && n.Func.WBPos.IsKnown() {
			yyerrorl(n.Func.WBPos, "write barrier prohibited")
		}
	}

	// Perform a BFS of the call graph from all
	// go:nowritebarrierrec functions.
	enqueue := func(src, target *Node, pos src.XPos) {
		if target.Func.Pragma&Yeswritebarrierrec != 0 {
			// Don't flow into this function.
			return
		}
		if _, ok := funcs[target]; ok {
			// Already found a path to target.
			return
		}

		// Record the path.
		funcs[target] = nowritebarrierrecCall{target: src, lineno: pos}
		q.pushRight(target)
	}
	for !q.empty() {
		fn := q.popLeft()

		// Check fn.
		if fn.Func.WBPos.IsKnown() {
			var err bytes.Buffer
			call := funcs[fn]
			for call.target != nil {
				fmt.Fprintf(&err, "\n\t%v: called by %v", linestr(call.lineno), call.target.Func.Nname)
				call = funcs[call.target]
			}
			yyerrorl(fn.Func.WBPos, "write barrier prohibited by caller; %v%s", fn.Func.Nname, err.String())
			continue
		}

		// Enqueue fn's calls.
		for _, callee := range c.extraCalls[fn] {
			enqueue(fn, callee.target, callee.lineno)
		}
		if fn.Func.nwbrCalls == nil {
			continue
		}
		for _, callee := range *fn.Func.nwbrCalls {
			target := symToFunc[callee.target]
			if target != nil {
				enqueue(fn, target, callee.lineno)
			}
		}
	}
}
