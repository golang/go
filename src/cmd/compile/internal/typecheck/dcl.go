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

var funcStack []*ir.Func // stack of previous values of ir.CurFunc

// DeclFunc creates and returns ONAMEs for the parameters and results
// of the given function. It also sets ir.CurFunc, and adds fn to
// Target.Funcs.
//
// After the caller is done constructing fn, it must call
// FinishFuncBody.
func DeclFunc(fn *ir.Func) (params, results []*ir.Name) {
	typ := fn.Type()

	// Currently, DeclFunc is only used to create normal functions, not
	// methods. If a use case for creating methods shows up, we can
	// extend it to support those too.
	if typ.Recv() != nil {
		base.FatalfAt(fn.Pos(), "unexpected receiver parameter")
	}

	params = declareParams(fn, ir.PPARAM, typ.Params())
	results = declareParams(fn, ir.PPARAMOUT, typ.Results())

	funcStack = append(funcStack, ir.CurFunc)
	ir.CurFunc = fn

	fn.Nname.Defn = fn
	Target.Funcs = append(Target.Funcs, fn)

	return
}

// FinishFuncBody restores ir.CurFunc to its state before the last
// call to DeclFunc.
func FinishFuncBody() {
	funcStack, ir.CurFunc = funcStack[:len(funcStack)-1], funcStack[len(funcStack)-1]
}

func CheckFuncStack() {
	if len(funcStack) != 0 {
		base.Fatalf("funcStack is non-empty: %v", len(funcStack))
	}
}

func declareParams(fn *ir.Func, ctxt ir.Class, params []*types.Field) []*ir.Name {
	names := make([]*ir.Name, len(params))
	for i, param := range params {
		names[i] = declareParam(fn, ctxt, i, param)
	}
	return names
}

func declareParam(fn *ir.Func, ctxt ir.Class, i int, param *types.Field) *ir.Name {
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

	if sym == nil {
		return nil
	}

	name := fn.NewLocal(param.Pos, sym, ctxt, param.Type)
	param.Nname = name
	return name
}

// make a new Node off the books.
func TempAt(pos src.XPos, curfn *ir.Func, t *types.Type) *ir.Name {
	if curfn == nil {
		base.Fatalf("no curfn for TempAt")
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
	n := curfn.NewLocal(pos, s, ir.PAUTO, t)
	s.Def = n // TODO(mdempsky): Should be unnecessary.
	n.SetEsc(ir.EscNever)
	n.SetUsed(true)
	n.SetAutoTemp(true)

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
	nrecvs := 0
	if recv != nil {
		nrecvs++
	}

	// TODO(mdempsky): Move this function to types.
	// TODO(mdempsky): Preserve positions, names, and package from sig+recv.

	params := make([]*types.Field, nrecvs+sig.NumParams())
	if recv != nil {
		params[0] = types.NewField(base.Pos, nil, recv)
	}
	for i, param := range sig.Params() {
		d := types.NewField(base.Pos, nil, param.Type)
		d.SetIsDDD(param.IsDDD())
		params[nrecvs+i] = d
	}

	results := make([]*types.Field, sig.NumResults())
	for i, t := range sig.Results() {
		results[i] = types.NewField(base.Pos, nil, t.Type)
	}

	return types.NewSignature(nil, params, results)
}
