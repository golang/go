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

// DeclFunc declares the parameters for fn and adds it to
// Target.Funcs.
//
// Before returning, it sets CurFunc to fn. When the caller is done
// constructing fn, it must call FinishFuncBody to restore CurFunc.
func DeclFunc(fn *ir.Func) {
	fn.DeclareParams(true)
	fn.Nname.Defn = fn
	Target.Funcs = append(Target.Funcs, fn)

	funcStack = append(funcStack, ir.CurFunc)
	ir.CurFunc = fn
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

// make a new Node off the books.
func TempAt(pos src.XPos, curfn *ir.Func, typ *types.Type) *ir.Name {
	if curfn == nil {
		base.FatalfAt(pos, "no curfn for TempAt")
	}
	if typ == nil {
		base.FatalfAt(pos, "TempAt called with nil type")
	}
	if typ.Kind() == types.TFUNC && typ.Recv() != nil {
		base.FatalfAt(pos, "misuse of method type: %v", typ)
	}
	types.CalcSize(typ)

	sym := &types.Sym{
		Name: autotmpname(len(curfn.Dcl)),
		Pkg:  types.LocalPkg,
	}
	name := curfn.NewLocal(pos, sym, typ)
	name.SetEsc(ir.EscNever)
	name.SetUsed(true)
	name.SetAutoTemp(true)

	return name
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
