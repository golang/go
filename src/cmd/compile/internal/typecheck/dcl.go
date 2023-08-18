// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typecheck

import (
	"fmt"
	"internal/types/errors"
	"sync"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

var funcStack []*ir.Func // stack of previous values of ir.CurFunc

func DeclFunc(sym *types.Sym, recv *ir.Field, params, results []*ir.Field) *ir.Func {
	fn := ir.NewFunc(base.Pos, base.Pos, sym, nil)

	funcStack = append(funcStack, ir.CurFunc)
	ir.CurFunc = fn

	var recv1 *types.Field
	if recv != nil {
		recv1 = declareParam(fn, ir.PPARAM, -1, recv)
	}

	typ := types.NewSignature(recv1, declareParams(fn, ir.PPARAM, params), declareParams(fn, ir.PPARAMOUT, results))
	checkdupfields("argument", typ.Recvs().FieldSlice(), typ.Params().FieldSlice(), typ.Results().FieldSlice())

	fn.Nname.SetType(typ)
	fn.Nname.SetTypecheck(1)

	fn.Nname.Defn = fn
	Target.Funcs = append(Target.Funcs, fn)

	return fn
}

// finish the body.
// called in auto-declaration context.
// returns in extern-declaration context.
func FinishFuncBody() {
	funcStack, ir.CurFunc = funcStack[:len(funcStack)-1], funcStack[len(funcStack)-1]
}

func CheckFuncStack() {
	if len(funcStack) != 0 {
		base.Fatalf("funcStack is non-empty: %v", len(funcStack))
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
				base.ErrorfAt(f.Pos, errors.DuplicateFieldAndMethod, "duplicate %s %s", what, f.Sym.Name)
				continue
			}
			seen[f.Sym] = true
		}
	}
}

func declareParams(fn *ir.Func, ctxt ir.Class, l []*ir.Field) []*types.Field {
	fields := make([]*types.Field, len(l))
	for i, n := range l {
		fields[i] = declareParam(fn, ctxt, i, n)
	}
	return fields
}

func declareParam(fn *ir.Func, ctxt ir.Class, i int, param *ir.Field) *types.Field {
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
		f.Nname = fn.NewLocal(param.Pos, sym, ctxt, f.Type)
	}

	return f
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

	return types.NewSignature(nil, params, results)
}
