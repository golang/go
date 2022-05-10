// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typecheck

import (
	"fmt"
	"sync"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

var DeclContext ir.Class = ir.PEXTERN // PEXTERN/PAUTO

func DeclFunc(sym *types.Sym, recv *ir.Field, params, results []*ir.Field) *ir.Func {
	fn := ir.NewFunc(base.Pos)
	fn.Nname = ir.NewNameAt(base.Pos, sym)
	fn.Nname.Func = fn
	fn.Nname.Defn = fn
	ir.MarkFunc(fn.Nname)
	StartFuncBody(fn)

	var recv1 *types.Field
	if recv != nil {
		recv1 = declareParam(ir.PPARAM, -1, recv)
	}

	typ := types.NewSignature(types.LocalPkg, recv1, nil, declareParams(ir.PPARAM, params), declareParams(ir.PPARAMOUT, results))
	checkdupfields("argument", typ.Recvs().FieldSlice(), typ.Params().FieldSlice(), typ.Results().FieldSlice())
	fn.Nname.SetType(typ)
	fn.Nname.SetTypecheck(1)

	return fn
}

// Declare records that Node n declares symbol n.Sym in the specified
// declaration context.
func Declare(n *ir.Name, ctxt ir.Class) {
	if ir.IsBlank(n) {
		return
	}

	s := n.Sym()

	// kludgy: TypecheckAllowed means we're past parsing. Eg reflectdata.methodWrapper may declare out of package names later.
	if !inimport && !TypecheckAllowed && s.Pkg != types.LocalPkg {
		base.ErrorfAt(n.Pos(), "cannot declare name %v", s)
	}

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
		types.Pushdcl(s)
		n.Curfn = ir.CurFunc
	}

	if ctxt == ir.PAUTO {
		n.SetFrameOffset(0)
	}

	s.Def = n
	n.Class = ctxt
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

	if types.IsExported(n.Sym().Name) || n.Sym().Name == "init" {
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

var funcStack []funcStackEnt // stack of previous values of ir.CurFunc/DeclContext

type funcStackEnt struct {
	curfn      *ir.Func
	dclcontext ir.Class
}

func declareParams(ctxt ir.Class, l []*ir.Field) []*types.Field {
	fields := make([]*types.Field, len(l))
	for i, n := range l {
		fields[i] = declareParam(ctxt, i, n)
	}
	return fields
}

func declareParam(ctxt ir.Class, i int, param *ir.Field) *types.Field {
	f := types.NewField(param.Pos, param.Sym, param.Type)
	f.SetIsDDD(param.IsDDD)

	sym := param.Sym
	if ctxt == ir.PPARAMOUT {
		if sym == nil {
			// Name so that escape analysis can track it. ~r stands for 'result'.
			sym = LookupNum("~r", i)
		} else if sym.IsBlank() {
			// Give it a name so we can assign to it during return. ~b stands for 'blank'.
			// The name must be different from ~r above because if you have
			//	func f() (_ int)
			//	func g() int
			// f is allowed to use a plain 'return' with no arguments, while g is not.
			// So the two cases must be distinguished.
			sym = LookupNum("~b", i)
		}
	}

	if sym != nil {
		name := ir.NewNameAt(param.Pos, sym)
		name.SetType(f.Type)
		name.SetTypecheck(1)
		Declare(name, ctxt)

		f.Nname = name
	}

	return f
}

func Temp(t *types.Type) *ir.Name {
	return TempAt(base.Pos, ir.CurFunc, t)
}

// make a new Node off the books
func TempAt(pos src.XPos, curfn *ir.Func, t *types.Type) *ir.Name {
	if curfn == nil {
		base.Fatalf("no curfn for TempAt")
	}
	if curfn.Op() == ir.OCLOSURE {
		ir.Dump("TempAt", curfn)
		base.Fatalf("adding TempAt to wrong closure function")
	}
	if t == nil {
		base.Fatalf("TempAt called with nil type")
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
	n.SetTypecheck(1)
	n.Class = ir.PAUTO
	n.SetEsc(ir.EscNever)
	n.Curfn = curfn
	n.SetUsed(true)
	n.SetAutoTemp(true)
	curfn.Dcl = append(curfn.Dcl, n)

	types.CalcSize(t)

	return n
}

var (
	autotmpnamesmu sync.Mutex
	autotmpnames   []string
)

// autotmpname returns the name for an autotmp variable numbered n.
func autotmpname(n int) string {
	autotmpnamesmu.Lock()
	defer autotmpnamesmu.Unlock()

	// Grow autotmpnames, if needed.
	if n >= len(autotmpnames) {
		autotmpnames = append(autotmpnames, make([]string, n+1-len(autotmpnames))...)
		autotmpnames = autotmpnames[:cap(autotmpnames)]
	}

	s := autotmpnames[n]
	if s == "" {
		// Give each tmp a different name so that they can be registerized.
		// Add a preceding . to avoid clashing with legal names.
		prefix := ".autotmp_%d"

		s = fmt.Sprintf(prefix, n)
		autotmpnames[n] = s
	}
	return s
}

// f is method type, with receiver.
// return function type, receiver as first argument (or not).
func NewMethodType(sig *types.Type, recv *types.Type) *types.Type {
	if sig.HasTParam() {
		base.Fatalf("NewMethodType with type parameters in signature %+v", sig)
	}
	if recv != nil && recv.HasTParam() {
		base.Fatalf("NewMethodType with type parameters in receiver %+v", recv)
	}
	nrecvs := 0
	if recv != nil {
		nrecvs++
	}

	// TODO(mdempsky): Move this function to types.
	// TODO(mdempsky): Preserve positions, names, and package from sig+recv.

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

	return types.NewSignature(types.LocalPkg, nil, nil, params, results)
}
