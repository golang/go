// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typecheck

import (
	"fmt"
	"strconv"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

var DeclContext ir.Class // PEXTERN/PAUTO

func AssignDefn(left []ir.Node, defn ir.Node) {
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
		if !assignableName(n) {
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
		Declare(n, DeclContext)
		n.Defn = defn
		defn.PtrInit().Append(ir.NewDecl(base.Pos, ir.ODCL, n))
		left[i] = n
	}

	if nnew == 0 && nerr == 0 {
		base.ErrorfAt(defn.Pos(), "no new variables on left side of :=")
	}
}

// := declarations
func assignableName(n ir.Node) bool {
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

func DeclFunc(sym *types.Sym, tfn ir.Ntype) *ir.Func {
	if tfn.Op() != ir.OTFUNC {
		base.Fatalf("expected OTFUNC node, got %v", tfn)
	}

	fn := ir.NewFunc(base.Pos)
	fn.Nname = ir.NewFuncNameAt(base.Pos, sym, fn)
	fn.Nname.Defn = fn
	fn.Nname.Ntype = tfn
	ir.MarkFunc(fn.Nname)
	StartFuncBody(fn)
	fn.Nname.Ntype = typecheckNtype(fn.Nname.Ntype)
	return fn
}

// declare variables from grammar
// new_name_list (type | [type] = expr_list)
func DeclVars(vl []*ir.Name, t ir.Ntype, el []ir.Node) []ir.Node {
	var init []ir.Node
	doexpr := len(el) > 0

	if len(el) == 1 && len(vl) > 1 {
		e := el[0]
		as2 := ir.NewAssignListStmt(base.Pos, ir.OAS2, nil, nil)
		as2.Rhs = []ir.Node{e}
		for _, v := range vl {
			as2.Lhs.Append(v)
			Declare(v, DeclContext)
			v.Ntype = t
			v.Defn = as2
			if ir.CurFunc != nil {
				init = append(init, ir.NewDecl(base.Pos, ir.ODCL, v))
			}
		}

		return append(init, as2)
	}

	for i, v := range vl {
		var e ir.Node
		if doexpr {
			if i >= len(el) {
				base.Errorf("assignment mismatch: %d variables but %d values", len(vl), len(el))
				break
			}
			e = el[i]
		}

		Declare(v, DeclContext)
		v.Ntype = t

		if e != nil || ir.CurFunc != nil || ir.IsBlank(v) {
			if ir.CurFunc != nil {
				init = append(init, ir.NewDecl(base.Pos, ir.ODCL, v))
			}
			as := ir.NewAssignStmt(base.Pos, v, e)
			init = append(init, as)
			if e != nil {
				v.Defn = as
			}
		}
	}

	if len(el) > len(vl) {
		base.Errorf("assignment mismatch: %d variables but %d values", len(vl), len(el))
	}
	return init
}

// Declare records that Node n declares symbol n.Sym in the specified
// declaration context.
func Declare(n *ir.Name, ctxt ir.Class) {
	if ir.IsBlank(n) {
		return
	}

	s := n.Sym()

	// kludgy: typecheckok means we're past parsing. Eg genwrapper may declare out of package names later.
	if !inimport && !TypecheckAllowed && s.Pkg != types.LocalPkg {
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
		Target.Externs = append(Target.Externs, n)
	} else {
		if ir.CurFunc == nil && ctxt == ir.PAUTO {
			base.Pos = n.Pos()
			base.Fatalf("automatic outside function")
		}
		if ir.CurFunc != nil && ctxt != ir.PFUNC && n.Op() == ir.ONAME {
			ir.CurFunc.Dcl = append(ir.CurFunc.Dcl, n)
		}
		if n.Op() == ir.OTYPE {
			declare_typegen++
			gen = declare_typegen
		} else if n.Op() == ir.ONAME && ctxt == ir.PAUTO && !strings.Contains(s.Name, "·") {
			vargen++
			gen = vargen
		}
		types.Pushdcl(s)
		n.Curfn = ir.CurFunc
	}

	if ctxt == ir.PAUTO {
		n.SetFrameOffset(0)
	}

	if s.Block == types.Block {
		// functype will print errors about duplicate function arguments.
		// Don't repeat the error here.
		if ctxt != ir.PPARAM && ctxt != ir.PPARAMOUT {
			Redeclared(n.Pos(), s, "in this block")
		}
	}

	s.Block = types.Block
	s.Lastlineno = base.Pos
	s.Def = n
	n.Vargen = int32(gen)
	n.Class_ = ctxt
	if ctxt == ir.PFUNC {
		n.Sym().SetFunc(true)
	}

	autoexport(n, ctxt)
}

// Export marks n for export (or reexport).
func Export(n *ir.Name) {
	if n.Sym().OnExportList() {
		return
	}
	n.Sym().SetOnExportList(true)

	if base.Flag.E != 0 {
		fmt.Printf("export symbol %v\n", n.Sym())
	}

	Target.Exports = append(Target.Exports, n)
}

// Redeclared emits a diagnostic about symbol s being redeclared at pos.
func Redeclared(pos src.XPos, s *types.Sym, where string) {
	if !s.Lastlineno.IsKnown() {
		pkgName := DotImportRefs[s.Def.(*ir.Ident)]
		base.ErrorfAt(pos, "%v redeclared %s\n"+
			"\t%v: previous declaration during import %q", s, where, base.FmtPos(pkgName.Pos()), pkgName.Pkg.Path)
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
			"\t%v: previous declaration", s, where, base.FmtPos(prevPos))
	}
}

// declare the function proper
// and declare the arguments.
// called in extern-declaration context
// returns in auto-declaration context.
func StartFuncBody(fn *ir.Func) {
	// change the declaration context from extern to auto
	funcStack = append(funcStack, funcStackEnt{ir.CurFunc, DeclContext})
	ir.CurFunc = fn
	DeclContext = ir.PAUTO

	types.Markdcl()

	if fn.Nname.Ntype != nil {
		funcargs(fn.Nname.Ntype.(*ir.FuncType))
	} else {
		funcargs2(fn.Type())
	}
}

// finish the body.
// called in auto-declaration context.
// returns in extern-declaration context.
func FinishFuncBody() {
	// change the declaration context from auto to previous context
	types.Popdcl()
	var e funcStackEnt
	funcStack, e = funcStack[:len(funcStack)-1], funcStack[len(funcStack)-1]
	ir.CurFunc, DeclContext = e.curfn, e.dclcontext
}

func CheckFuncStack() {
	if len(funcStack) != 0 {
		base.Fatalf("funcStack is non-empty: %v", len(funcStack))
	}
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

	mt := types.ReceiverBaseType(rf.Type)
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

func autoexport(n *ir.Name, ctxt ir.Class) {
	if n.Sym().Pkg != types.LocalPkg {
		return
	}
	if (ctxt != ir.PEXTERN && ctxt != ir.PFUNC) || DeclContext != ir.PEXTERN {
		return
	}
	if n.Type() != nil && n.Type().IsKind(types.TFUNC) && ir.IsMethod(n) {
		return
	}

	if types.IsExported(n.Sym().Name) || initname(n.Sym().Name) {
		Export(n)
	}
	if base.Flag.AsmHdr != "" && !n.Sym().Asm() {
		n.Sym().SetAsm(true)
		Target.Asms = append(Target.Asms, n)
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

// declare individual names - var, typ, const

var declare_typegen int

func fakeRecvField() *types.Field {
	return types.NewField(src.NoXPos, nil, types.FakeRecvType())
}

var funcStack []funcStackEnt // stack of previous values of Curfn/dclcontext

type funcStackEnt struct {
	curfn      *ir.Func
	dclcontext ir.Class
}

func funcarg(n *ir.Field, ctxt ir.Class) {
	if n.Sym == nil {
		return
	}

	name := ir.NewNameAt(n.Pos, n.Sym)
	n.Decl = name
	name.Ntype = n.Ntype
	Declare(name, ctxt)

	vargen++
	n.Decl.Vargen = int32(vargen)
}

func funcarg2(f *types.Field, ctxt ir.Class) {
	if f.Sym == nil {
		return
	}
	n := ir.NewNameAt(f.Pos, f.Sym)
	f.Nname = n
	n.SetType(f.Type)
	Declare(n, ctxt)
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
			n.Sym = LookupNum("~r", gen)
			gen++
		}
		if n.Sym.IsBlank() {
			// Give it a name so we can assign to it during return. ~b stands for 'blank'.
			// The name must be different from ~r above because if you have
			//	func f() (_ int)
			//	func g() int
			// f is allowed to use a plain 'return' with no arguments, while g is not.
			// So the two cases must be distinguished.
			n.Sym = LookupNum("~b", gen)
			gen++
		}

		funcarg(n, ir.PPARAMOUT)
	}

	vargen = oldvargen
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

func initname(s string) bool {
	return s == "init"
}

var vargen int

func Temp(t *types.Type) *ir.Name {
	return TempAt(base.Pos, ir.CurFunc, t)
}

// make a new Node off the books
func TempAt(pos src.XPos, curfn *ir.Func, t *types.Type) *ir.Name {
	if curfn == nil {
		base.Fatalf("no curfn for tempAt")
	}
	if curfn.Op() == ir.OCLOSURE {
		ir.Dump("tempAt", curfn)
		base.Fatalf("adding tempAt to wrong closure function")
	}
	if t == nil {
		base.Fatalf("tempAt called with nil type")
	}
	if t.Kind() == types.TFUNC && t.Recv() != nil {
		base.Fatalf("misuse of method type: %v", t)
	}

	s := &types.Sym{
		Name: autotmpname(len(curfn.Dcl)),
		Pkg:  types.LocalPkg,
	}
	n := ir.NewNameAt(pos, s)
	s.Def = n
	n.SetType(t)
	n.Class_ = ir.PAUTO
	n.SetEsc(ir.EscNever)
	n.Curfn = curfn
	n.SetUsed(true)
	n.SetAutoTemp(true)
	curfn.Dcl = append(curfn.Dcl, n)

	types.CalcSize(t)

	return n
}

// autotmpname returns the name for an autotmp variable numbered n.
func autotmpname(n int) string {
	// Give each tmp a different name so that they can be registerized.
	// Add a preceding . to avoid clashing with legal names.
	const prefix = ".autotmp_"
	// Start with a buffer big enough to hold a large n.
	b := []byte(prefix + "      ")[:len(prefix)]
	b = strconv.AppendInt(b, int64(n), 10)
	return types.InternString(b)
}

// f is method type, with receiver.
// return function type, receiver as first argument (or not).
func NewMethodType(sig *types.Type, recv *types.Type) *types.Type {
	nrecvs := 0
	if recv != nil {
		nrecvs++
	}

	params := make([]*types.Field, nrecvs+sig.Params().Fields().Len())
	if recv != nil {
		params[0] = types.NewField(base.Pos, nil, recv)
	}
	for i, param := range sig.Params().Fields().Slice() {
		d := types.NewField(base.Pos, nil, param.Type)
		d.SetIsDDD(param.IsDDD())
		params[nrecvs+i] = d
	}

	results := make([]*types.Field, sig.Results().Fields().Len())
	for i, t := range sig.Results().Fields().Slice() {
		results[i] = types.NewField(base.Pos, nil, t.Type)
	}

	return types.NewSignature(types.LocalPkg, nil, params, results)
}
