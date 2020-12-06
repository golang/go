// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bytes"
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
	"fmt"
	"strings"
)

// Declaration stack & operations

var externdcl []ir.Node

func testdclstack() {
	if !types.IsDclstackValid() {
		base.Fatalf("mark left on the dclstack")
	}
}

// redeclare emits a diagnostic about symbol s being redeclared at pos.
func redeclare(pos src.XPos, s *types.Sym, where string) {
	if !s.Lastlineno.IsKnown() {
		pkg := s.Origpkg
		if pkg == nil {
			pkg = s.Pkg
		}
		base.ErrorfAt(pos, "%v redeclared %s\n"+
			"\tprevious declaration during import %q", s, where, pkg.Path)
	} else {
		prevPos := s.Lastlineno

		// When an import and a declaration collide in separate files,
		// present the import as the "redeclared", because the declaration
		// is visible where the import is, but not vice versa.
		// See issue 4510.
		if s.Def == nil {
			pos, prevPos = prevPos, pos
		}

		base.ErrorfAt(pos, "%v redeclared %s\n"+
			"\tprevious declaration at %v", s, where, base.FmtPos(prevPos))
	}
}

var vargen int

// declare individual names - var, typ, const

var declare_typegen int

// declare records that Node n declares symbol n.Sym in the specified
// declaration context.
func declare(n *ir.Name, ctxt ir.Class) {
	if ir.IsBlank(n) {
		return
	}

	s := n.Sym()

	// kludgy: typecheckok means we're past parsing. Eg genwrapper may declare out of package names later.
	if !inimport && !typecheckok && s.Pkg != types.LocalPkg {
		base.ErrorfAt(n.Pos(), "cannot declare name %v", s)
	}

	gen := 0
	if ctxt == ir.PEXTERN {
		if s.Name == "init" {
			base.ErrorfAt(n.Pos(), "cannot declare init - must be func")
		}
		if s.Name == "main" && s.Pkg.Name == "main" {
			base.ErrorfAt(n.Pos(), "cannot declare main - must be func")
		}
		externdcl = append(externdcl, n)
	} else {
		if Curfn == nil && ctxt == ir.PAUTO {
			base.Pos = n.Pos()
			base.Fatalf("automatic outside function")
		}
		if Curfn != nil && ctxt != ir.PFUNC && n.Op() == ir.ONAME {
			Curfn.Dcl = append(Curfn.Dcl, n)
		}
		if n.Op() == ir.OTYPE {
			declare_typegen++
			gen = declare_typegen
		} else if n.Op() == ir.ONAME && ctxt == ir.PAUTO && !strings.Contains(s.Name, "·") {
			vargen++
			gen = vargen
		}
		types.Pushdcl(s)
		n.Curfn = Curfn
	}

	if ctxt == ir.PAUTO {
		n.SetOffset(0)
	}

	if s.Block == types.Block {
		// functype will print errors about duplicate function arguments.
		// Don't repeat the error here.
		if ctxt != ir.PPARAM && ctxt != ir.PPARAMOUT {
			redeclare(n.Pos(), s, "in this block")
		}
	}

	s.Block = types.Block
	s.Lastlineno = base.Pos
	s.Def = n
	n.Vargen = int32(gen)
	n.SetClass(ctxt)
	if ctxt == ir.PFUNC {
		n.Sym().SetFunc(true)
	}

	autoexport(n, ctxt)
}

// declare variables from grammar
// new_name_list (type | [type] = expr_list)
func variter(vl []ir.Node, t ir.Ntype, el []ir.Node) []ir.Node {
	var init []ir.Node
	doexpr := len(el) > 0

	if len(el) == 1 && len(vl) > 1 {
		e := el[0]
		as2 := ir.Nod(ir.OAS2, nil, nil)
		as2.PtrList().Set(vl)
		as2.PtrRlist().Set1(e)
		for _, v := range vl {
			v := v.(*ir.Name)
			v.SetOp(ir.ONAME)
			declare(v, dclcontext)
			v.Ntype = t
			v.Defn = as2
			if Curfn != nil {
				init = append(init, ir.Nod(ir.ODCL, v, nil))
			}
		}

		return append(init, as2)
	}

	nel := len(el)
	for _, v := range vl {
		v := v.(*ir.Name)
		var e ir.Node
		if doexpr {
			if len(el) == 0 {
				base.Errorf("assignment mismatch: %d variables but %d values", len(vl), nel)
				break
			}
			e = el[0]
			el = el[1:]
		}

		v.SetOp(ir.ONAME)
		declare(v, dclcontext)
		v.Ntype = t

		if e != nil || Curfn != nil || ir.IsBlank(v) {
			if Curfn != nil {
				init = append(init, ir.Nod(ir.ODCL, v, nil))
			}
			e = ir.Nod(ir.OAS, v, e)
			init = append(init, e)
			if e.Right() != nil {
				v.Defn = e
			}
		}
	}

	if len(el) != 0 {
		base.Errorf("assignment mismatch: %d variables but %d values", len(vl), nel)
	}
	return init
}

// newFuncNameAt generates a new name node for a function or method.
func newFuncNameAt(pos src.XPos, s *types.Sym, fn *ir.Func) *ir.Name {
	if fn.Nname != nil {
		base.Fatalf("newFuncName - already have name")
	}
	n := ir.NewNameAt(pos, s)
	n.SetFunc(fn)
	fn.Nname = n
	return n
}

func anonfield(typ *types.Type) *ir.Field {
	return symfield(nil, typ)
}

func namedfield(s string, typ *types.Type) *ir.Field {
	return symfield(lookup(s), typ)
}

func symfield(s *types.Sym, typ *types.Type) *ir.Field {
	return ir.NewField(base.Pos, s, nil, typ)
}

// oldname returns the Node that declares symbol s in the current scope.
// If no such Node currently exists, an ONONAME Node is returned instead.
// Automatically creates a new closure variable if the referenced symbol was
// declared in a different (containing) function.
func oldname(s *types.Sym) ir.Node {
	n := ir.AsNode(s.Def)
	if n == nil {
		// Maybe a top-level declaration will come along later to
		// define s. resolve will check s.Def again once all input
		// source has been processed.
		return ir.NewDeclNameAt(base.Pos, s)
	}

	if Curfn != nil && n.Op() == ir.ONAME && n.Name().Curfn != nil && n.Name().Curfn != Curfn {
		// Inner func is referring to var in outer func.
		//
		// TODO(rsc): If there is an outer variable x and we
		// are parsing x := 5 inside the closure, until we get to
		// the := it looks like a reference to the outer x so we'll
		// make x a closure variable unnecessarily.
		c := n.Name().Innermost
		if c == nil || c.Curfn != Curfn {
			// Do not have a closure var for the active closure yet; make one.
			c = NewName(s)
			c.SetClass(ir.PAUTOHEAP)
			c.SetIsClosureVar(true)
			c.SetIsDDD(n.IsDDD())
			c.Defn = n

			// Link into list of active closure variables.
			// Popped from list in func funcLit.
			c.Outer = n.Name().Innermost
			n.Name().Innermost = c

			Curfn.ClosureVars = append(Curfn.ClosureVars, c)
		}

		// return ref to closure var, not original
		return c
	}

	return n
}

// importName is like oldname,
// but it reports an error if sym is from another package and not exported.
func importName(sym *types.Sym) ir.Node {
	n := oldname(sym)
	if !types.IsExported(sym.Name) && sym.Pkg != types.LocalPkg {
		n.SetDiag(true)
		base.Errorf("cannot refer to unexported name %s.%s", sym.Pkg.Name, sym.Name)
	}
	return n
}

// := declarations
func colasname(n ir.Node) bool {
	switch n.Op() {
	case ir.ONAME,
		ir.ONONAME,
		ir.OPACK,
		ir.OTYPE,
		ir.OLITERAL:
		return n.Sym() != nil
	}

	return false
}

func colasdefn(left []ir.Node, defn ir.Node) {
	for _, n := range left {
		if n.Sym() != nil {
			n.Sym().SetUniq(true)
		}
	}

	var nnew, nerr int
	for i, n := range left {
		if ir.IsBlank(n) {
			continue
		}
		if !colasname(n) {
			base.ErrorfAt(defn.Pos(), "non-name %v on left side of :=", n)
			nerr++
			continue
		}

		if !n.Sym().Uniq() {
			base.ErrorfAt(defn.Pos(), "%v repeated on left side of :=", n.Sym())
			n.SetDiag(true)
			nerr++
			continue
		}

		n.Sym().SetUniq(false)
		if n.Sym().Block == types.Block {
			continue
		}

		nnew++
		n := NewName(n.Sym())
		declare(n, dclcontext)
		n.Defn = defn
		defn.PtrInit().Append(ir.Nod(ir.ODCL, n, nil))
		left[i] = n
	}

	if nnew == 0 && nerr == 0 {
		base.ErrorfAt(defn.Pos(), "no new variables on left side of :=")
	}
}

// declare the function proper
// and declare the arguments.
// called in extern-declaration context
// returns in auto-declaration context.
func funchdr(fn *ir.Func) {
	// change the declaration context from extern to auto
	funcStack = append(funcStack, funcStackEnt{Curfn, dclcontext})
	Curfn = fn
	dclcontext = ir.PAUTO

	types.Markdcl()

	if fn.Nname.Ntype != nil {
		funcargs(fn.Nname.Ntype.(*ir.FuncType))
	} else {
		funcargs2(fn.Type())
	}
}

func funcargs(nt *ir.FuncType) {
	if nt.Op() != ir.OTFUNC {
		base.Fatalf("funcargs %v", nt.Op())
	}

	// re-start the variable generation number
	// we want to use small numbers for the return variables,
	// so let them have the chunk starting at 1.
	//
	// TODO(mdempsky): This is ugly, and only necessary because
	// esc.go uses Vargen to figure out result parameters' index
	// within the result tuple.
	vargen = len(nt.Results)

	// declare the receiver and in arguments.
	if nt.Recv != nil {
		funcarg(nt.Recv, ir.PPARAM)
	}
	for _, n := range nt.Params {
		funcarg(n, ir.PPARAM)
	}

	oldvargen := vargen
	vargen = 0

	// declare the out arguments.
	gen := len(nt.Params)
	for _, n := range nt.Results {
		if n.Sym == nil {
			// Name so that escape analysis can track it. ~r stands for 'result'.
			n.Sym = lookupN("~r", gen)
			gen++
		}
		if n.Sym.IsBlank() {
			// Give it a name so we can assign to it during return. ~b stands for 'blank'.
			// The name must be different from ~r above because if you have
			//	func f() (_ int)
			//	func g() int
			// f is allowed to use a plain 'return' with no arguments, while g is not.
			// So the two cases must be distinguished.
			n.Sym = lookupN("~b", gen)
			gen++
		}

		funcarg(n, ir.PPARAMOUT)
	}

	vargen = oldvargen
}

func funcarg(n *ir.Field, ctxt ir.Class) {
	if n.Sym == nil {
		return
	}

	name := ir.NewNameAt(n.Pos, n.Sym)
	n.Decl = name
	name.Ntype = n.Ntype
	name.SetIsDDD(n.IsDDD)
	declare(name, ctxt)

	vargen++
	n.Decl.Vargen = int32(vargen)
}

// Same as funcargs, except run over an already constructed TFUNC.
// This happens during import, where the hidden_fndcl rule has
// used functype directly to parse the function's type.
func funcargs2(t *types.Type) {
	if t.Kind() != types.TFUNC {
		base.Fatalf("funcargs2 %v", t)
	}

	for _, f := range t.Recvs().Fields().Slice() {
		funcarg2(f, ir.PPARAM)
	}
	for _, f := range t.Params().Fields().Slice() {
		funcarg2(f, ir.PPARAM)
	}
	for _, f := range t.Results().Fields().Slice() {
		funcarg2(f, ir.PPARAMOUT)
	}
}

func funcarg2(f *types.Field, ctxt ir.Class) {
	if f.Sym == nil {
		return
	}
	n := ir.NewNameAt(f.Pos, f.Sym)
	f.Nname = n
	n.SetType(f.Type)
	n.SetIsDDD(f.IsDDD())
	declare(n, ctxt)
}

var funcStack []funcStackEnt // stack of previous values of Curfn/dclcontext

type funcStackEnt struct {
	curfn      *ir.Func
	dclcontext ir.Class
}

// finish the body.
// called in auto-declaration context.
// returns in extern-declaration context.
func funcbody() {
	// change the declaration context from auto to previous context
	types.Popdcl()
	var e funcStackEnt
	funcStack, e = funcStack[:len(funcStack)-1], funcStack[len(funcStack)-1]
	Curfn, dclcontext = e.curfn, e.dclcontext
}

// structs, functions, and methods.
// they don't belong here, but where do they belong?
func checkembeddedtype(t *types.Type) {
	if t == nil {
		return
	}

	if t.Sym() == nil && t.IsPtr() {
		t = t.Elem()
		if t.IsInterface() {
			base.Errorf("embedded type cannot be a pointer to interface")
		}
	}

	if t.IsPtr() || t.IsUnsafePtr() {
		base.Errorf("embedded type cannot be a pointer")
	} else if t.Kind() == types.TFORW && !t.ForwardType().Embedlineno.IsKnown() {
		t.ForwardType().Embedlineno = base.Pos
	}
}

// checkdupfields emits errors for duplicately named fields or methods in
// a list of struct or interface types.
func checkdupfields(what string, fss ...[]*types.Field) {
	seen := make(map[*types.Sym]bool)
	for _, fs := range fss {
		for _, f := range fs {
			if f.Sym == nil || f.Sym.IsBlank() {
				continue
			}
			if seen[f.Sym] {
				base.ErrorfAt(f.Pos, "duplicate %s %s", what, f.Sym.Name)
				continue
			}
			seen[f.Sym] = true
		}
	}
}

// convert a parsed id/type list into
// a type for struct/interface/arglist
func tostruct(l []*ir.Field) *types.Type {
	lno := base.Pos

	fields := make([]*types.Field, len(l))
	for i, n := range l {
		base.Pos = n.Pos

		if n.Ntype != nil {
			n.Type = typecheckNtype(n.Ntype).Type()
			n.Ntype = nil
		}
		f := types.NewField(n.Pos, n.Sym, n.Type)
		if n.Embedded {
			checkembeddedtype(n.Type)
			f.Embedded = 1
		}
		f.Note = n.Note
		fields[i] = f
	}
	checkdupfields("field", fields)

	base.Pos = lno
	return types.NewStruct(types.LocalPkg, fields)
}

func tointerface(nmethods []*ir.Field) *types.Type {
	if len(nmethods) == 0 {
		return types.Types[types.TINTER]
	}

	lno := base.Pos

	methods := make([]*types.Field, len(nmethods))
	for i, n := range nmethods {
		base.Pos = n.Pos
		if n.Ntype != nil {
			n.Type = typecheckNtype(n.Ntype).Type()
			n.Ntype = nil
		}
		methods[i] = types.NewField(n.Pos, n.Sym, n.Type)
	}

	base.Pos = lno
	return types.NewInterface(types.LocalPkg, methods)
}

func fakeRecv() *ir.Field {
	return anonfield(types.FakeRecvType())
}

func fakeRecvField() *types.Field {
	return types.NewField(src.NoXPos, nil, types.FakeRecvType())
}

// isifacemethod reports whether (field) m is
// an interface method. Such methods have the
// special receiver type types.FakeRecvType().
func isifacemethod(f *types.Type) bool {
	return f.Recv().Type == types.FakeRecvType()
}

// turn a parsed function declaration into a type
func functype(nrecv *ir.Field, nparams, nresults []*ir.Field) *types.Type {
	funarg := func(n *ir.Field) *types.Field {
		lno := base.Pos
		base.Pos = n.Pos

		if n.Ntype != nil {
			n.Type = typecheckNtype(n.Ntype).Type()
			n.Ntype = nil
		}

		f := types.NewField(n.Pos, n.Sym, n.Type)
		f.SetIsDDD(n.IsDDD)
		if n.Decl != nil {
			n.Decl.SetType(f.Type)
			f.Nname = n.Decl
		}

		base.Pos = lno
		return f
	}
	funargs := func(nn []*ir.Field) []*types.Field {
		res := make([]*types.Field, len(nn))
		for i, n := range nn {
			res[i] = funarg(n)
		}
		return res
	}

	var recv *types.Field
	if nrecv != nil {
		recv = funarg(nrecv)
	}

	t := types.NewSignature(types.LocalPkg, recv, funargs(nparams), funargs(nresults))
	checkdupfields("argument", t.Recvs().FieldSlice(), t.Params().FieldSlice(), t.Results().FieldSlice())
	return t
}

func hasNamedResults(fn *ir.Func) bool {
	typ := fn.Type()
	return typ.NumResults() > 0 && types.OrigSym(typ.Results().Field(0).Sym) != nil
}

// methodSym returns the method symbol representing a method name
// associated with a specific receiver type.
//
// Method symbols can be used to distinguish the same method appearing
// in different method sets. For example, T.M and (*T).M have distinct
// method symbols.
//
// The returned symbol will be marked as a function.
func methodSym(recv *types.Type, msym *types.Sym) *types.Sym {
	sym := methodSymSuffix(recv, msym, "")
	sym.SetFunc(true)
	return sym
}

// methodSymSuffix is like methodsym, but allows attaching a
// distinguisher suffix. To avoid collisions, the suffix must not
// start with a letter, number, or period.
func methodSymSuffix(recv *types.Type, msym *types.Sym, suffix string) *types.Sym {
	if msym.IsBlank() {
		base.Fatalf("blank method name")
	}

	rsym := recv.Sym()
	if recv.IsPtr() {
		if rsym != nil {
			base.Fatalf("declared pointer receiver type: %v", recv)
		}
		rsym = recv.Elem().Sym()
	}

	// Find the package the receiver type appeared in. For
	// anonymous receiver types (i.e., anonymous structs with
	// embedded fields), use the "go" pseudo-package instead.
	rpkg := gopkg
	if rsym != nil {
		rpkg = rsym.Pkg
	}

	var b bytes.Buffer
	if recv.IsPtr() {
		// The parentheses aren't really necessary, but
		// they're pretty traditional at this point.
		fmt.Fprintf(&b, "(%-S)", recv)
	} else {
		fmt.Fprintf(&b, "%-S", recv)
	}

	// A particular receiver type may have multiple non-exported
	// methods with the same name. To disambiguate them, include a
	// package qualifier for names that came from a different
	// package than the receiver type.
	if !types.IsExported(msym.Name) && msym.Pkg != rpkg {
		b.WriteString(".")
		b.WriteString(msym.Pkg.Prefix)
	}

	b.WriteString(".")
	b.WriteString(msym.Name)
	b.WriteString(suffix)

	return rpkg.LookupBytes(b.Bytes())
}

// Add a method, declared as a function.
// - msym is the method symbol
// - t is function type (with receiver)
// Returns a pointer to the existing or added Field; or nil if there's an error.
func addmethod(n *ir.Func, msym *types.Sym, t *types.Type, local, nointerface bool) *types.Field {
	if msym == nil {
		base.Fatalf("no method symbol")
	}

	// get parent type sym
	rf := t.Recv() // ptr to this structure
	if rf == nil {
		base.Errorf("missing receiver")
		return nil
	}

	mt := methtype(rf.Type)
	if mt == nil || mt.Sym() == nil {
		pa := rf.Type
		t := pa
		if t != nil && t.IsPtr() {
			if t.Sym() != nil {
				base.Errorf("invalid receiver type %v (%v is a pointer type)", pa, t)
				return nil
			}
			t = t.Elem()
		}

		switch {
		case t == nil || t.Broke():
			// rely on typecheck having complained before
		case t.Sym() == nil:
			base.Errorf("invalid receiver type %v (%v is not a defined type)", pa, t)
		case t.IsPtr():
			base.Errorf("invalid receiver type %v (%v is a pointer type)", pa, t)
		case t.IsInterface():
			base.Errorf("invalid receiver type %v (%v is an interface type)", pa, t)
		default:
			// Should have picked off all the reasons above,
			// but just in case, fall back to generic error.
			base.Errorf("invalid receiver type %v (%L / %L)", pa, pa, t)
		}
		return nil
	}

	if local && mt.Sym().Pkg != types.LocalPkg {
		base.Errorf("cannot define new methods on non-local type %v", mt)
		return nil
	}

	if msym.IsBlank() {
		return nil
	}

	if mt.IsStruct() {
		for _, f := range mt.Fields().Slice() {
			if f.Sym == msym {
				base.Errorf("type %v has both field and method named %v", mt, msym)
				f.SetBroke(true)
				return nil
			}
		}
	}

	for _, f := range mt.Methods().Slice() {
		if msym.Name != f.Sym.Name {
			continue
		}
		// types.Identical only checks that incoming and result parameters match,
		// so explicitly check that the receiver parameters match too.
		if !types.Identical(t, f.Type) || !types.Identical(t.Recv().Type, f.Type.Recv().Type) {
			base.Errorf("method redeclared: %v.%v\n\t%v\n\t%v", mt, msym, f.Type, t)
		}
		return f
	}

	f := types.NewField(base.Pos, msym, t)
	f.Nname = n.Nname
	f.SetNointerface(nointerface)

	mt.Methods().Append(f)
	return f
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
	if !base.Ctxt.Flag_dynlink && !existed {
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
	if !base.Ctxt.Flag_dynlink {
		base.Fatalf("makefuncsym dynlink")
	}
	if s.IsBlank() {
		return
	}
	if base.Flag.CompilingRuntime && (s.Name == "getg" || s.Name == "getclosureptr" || s.Name == "getcallerpc" || s.Name == "getcallersp") {
		// runtime.getg(), getclosureptr(), getcallerpc(), and
		// getcallersp() are not real functions and so do not
		// get funcsyms.
		return
	}
	if _, existed := s.Pkg.LookupOK(funcsymname(s)); !existed {
		funcsyms = append(funcsyms, s)
	}
}

// setNodeNameFunc marks a node as a function.
func setNodeNameFunc(n ir.Node) {
	if n.Op() != ir.ONAME || n.Class() != ir.Pxxx {
		base.Fatalf("expected ONAME/Pxxx node, got %v", n)
	}

	n.SetClass(ir.PFUNC)
	n.Sym().SetFunc(true)
}

func dclfunc(sym *types.Sym, tfn ir.Ntype) *ir.Func {
	if tfn.Op() != ir.OTFUNC {
		base.Fatalf("expected OTFUNC node, got %v", tfn)
	}

	fn := ir.NewFunc(base.Pos)
	fn.Nname = newFuncNameAt(base.Pos, sym, fn)
	fn.Nname.Defn = fn
	fn.Nname.Ntype = tfn
	setNodeNameFunc(fn.Nname)
	funchdr(fn)
	fn.Nname.Ntype = typecheckNtype(fn.Nname.Ntype)
	return fn
}

type nowritebarrierrecChecker struct {
	// extraCalls contains extra function calls that may not be
	// visible during later analysis. It maps from the ODCLFUNC of
	// the caller to a list of callees.
	extraCalls map[*ir.Func][]nowritebarrierrecCall

	// curfn is the current function during AST walks.
	curfn *ir.Func
}

type nowritebarrierrecCall struct {
	target *ir.Func // caller or callee
	lineno src.XPos // line of call
}

// newNowritebarrierrecChecker creates a nowritebarrierrecChecker. It
// must be called before transformclosure and walk.
func newNowritebarrierrecChecker() *nowritebarrierrecChecker {
	c := &nowritebarrierrecChecker{
		extraCalls: make(map[*ir.Func][]nowritebarrierrecCall),
	}

	// Find all systemstack calls and record their targets. In
	// general, flow analysis can't see into systemstack, but it's
	// important to handle it for this check, so we model it
	// directly. This has to happen before transformclosure since
	// it's a lot harder to work out the argument after.
	for _, n := range xtop {
		if n.Op() != ir.ODCLFUNC {
			continue
		}
		c.curfn = n.(*ir.Func)
		ir.Inspect(n, c.findExtraCalls)
	}
	c.curfn = nil
	return c
}

func (c *nowritebarrierrecChecker) findExtraCalls(n ir.Node) bool {
	if n.Op() != ir.OCALLFUNC {
		return true
	}
	fn := n.Left()
	if fn == nil || fn.Op() != ir.ONAME || fn.Class() != ir.PFUNC || fn.Name().Defn == nil {
		return true
	}
	if !isRuntimePkg(fn.Sym().Pkg) || fn.Sym().Name != "systemstack" {
		return true
	}

	var callee *ir.Func
	arg := n.List().First()
	switch arg.Op() {
	case ir.ONAME:
		callee = arg.Name().Defn.(*ir.Func)
	case ir.OCLOSURE:
		callee = arg.Func()
	default:
		base.Fatalf("expected ONAME or OCLOSURE node, got %+v", arg)
	}
	if callee.Op() != ir.ODCLFUNC {
		base.Fatalf("expected ODCLFUNC node, got %+v", callee)
	}
	c.extraCalls[c.curfn] = append(c.extraCalls[c.curfn], nowritebarrierrecCall{callee, n.Pos()})
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
func (c *nowritebarrierrecChecker) recordCall(fn *ir.Func, to *obj.LSym, pos src.XPos) {
	// We record this information on the *Func so this is concurrent-safe.
	if fn.NWBRCalls == nil {
		fn.NWBRCalls = new([]ir.SymAndPos)
	}
	*fn.NWBRCalls = append(*fn.NWBRCalls, ir.SymAndPos{Sym: to, Pos: pos})
}

func (c *nowritebarrierrecChecker) check() {
	// We walk the call graph as late as possible so we can
	// capture all calls created by lowering, but this means we
	// only get to see the obj.LSyms of calls. symToFunc lets us
	// get back to the ODCLFUNCs.
	symToFunc := make(map[*obj.LSym]*ir.Func)
	// funcs records the back-edges of the BFS call graph walk. It
	// maps from the ODCLFUNC of each function that must not have
	// write barriers to the call that inhibits them. Functions
	// that are directly marked go:nowritebarrierrec are in this
	// map with a zero-valued nowritebarrierrecCall. This also
	// acts as the set of marks for the BFS of the call graph.
	funcs := make(map[*ir.Func]nowritebarrierrecCall)
	// q is the queue of ODCLFUNC Nodes to visit in BFS order.
	var q ir.NameQueue

	for _, n := range xtop {
		if n.Op() != ir.ODCLFUNC {
			continue
		}
		fn := n.(*ir.Func)

		symToFunc[fn.LSym] = fn

		// Make nowritebarrierrec functions BFS roots.
		if fn.Pragma&ir.Nowritebarrierrec != 0 {
			funcs[fn] = nowritebarrierrecCall{}
			q.PushRight(fn.Nname)
		}
		// Check go:nowritebarrier functions.
		if fn.Pragma&ir.Nowritebarrier != 0 && fn.WBPos.IsKnown() {
			base.ErrorfAt(fn.WBPos, "write barrier prohibited")
		}
	}

	// Perform a BFS of the call graph from all
	// go:nowritebarrierrec functions.
	enqueue := func(src, target *ir.Func, pos src.XPos) {
		if target.Pragma&ir.Yeswritebarrierrec != 0 {
			// Don't flow into this function.
			return
		}
		if _, ok := funcs[target]; ok {
			// Already found a path to target.
			return
		}

		// Record the path.
		funcs[target] = nowritebarrierrecCall{target: src, lineno: pos}
		q.PushRight(target.Nname)
	}
	for !q.Empty() {
		fn := q.PopLeft().Func()

		// Check fn.
		if fn.WBPos.IsKnown() {
			var err bytes.Buffer
			call := funcs[fn]
			for call.target != nil {
				fmt.Fprintf(&err, "\n\t%v: called by %v", base.FmtPos(call.lineno), call.target.Nname)
				call = funcs[call.target]
			}
			base.ErrorfAt(fn.WBPos, "write barrier prohibited by caller; %v%s", fn.Nname, err.String())
			continue
		}

		// Enqueue fn's calls.
		for _, callee := range c.extraCalls[fn] {
			enqueue(fn, callee.target, callee.lineno)
		}
		if fn.NWBRCalls == nil {
			continue
		}
		for _, callee := range *fn.NWBRCalls {
			target := symToFunc[callee.Sym]
			if target != nil {
				enqueue(fn, target, callee.Pos)
			}
		}
	}
}
