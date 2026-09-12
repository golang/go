// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package devirtualize

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/inline"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
)

// This file implements return-value devirtualization: splitting a function into
// a devirtualized variant plus a boxing thunk.
//
// A function f whose recorded result set is a single concrete type T still
// returns it boxed in an interface, so every caller pays for the boxing and,
// without [StaticCall]'s rewrite, an indirect call for each method used on the
// result and the entailing heap escapes. Splitting moves f's body into a new
// function f.dv whose result type is T itself, and rebuilds f as a thunk that
// calls f.dv and boxes the result:
//
//	func f(args) I    { return I(f.dv(args)) }
//	func f.dv(args) T { ...original body, returns unboxed... }
//
// f keeps its symbol, signature, and inline body, so func values, itabs,
// reflection, and importers see no difference. Static calls to
// f are then rewritten by [StaticResults] to call f.dv directly and
// re-box at the call site, where the boxing is visible to the caller's own
// analysis and often melts away, or at least enables some stack promotion.

// SplitResultFunc splits fn if its recorded result sets prove that an
// interface-typed result always holds one concrete type.
//
// It reports the devirtualized variant, or nil when fn is left alone.
// The caller is responsible for pass bookkeeping of the new function;
// its body is fn's old body, moved, and so may hold state private to
// the running pass (such as the interleaved pass's ParenExprs).
func (s *State) SplitResultFunc(fn *ir.Func) *ir.Func {
	devirt := s.splitResultTypes(fn)
	if devirt == nil {
		return nil
	}

	if fn.IsClosure() {
		// A capturing closure's variant would need the capture
		// context forwarded; instead, callers see through closure
		// calls via the recorded result sets alone.
		return nil
	}

	if fn.Pragma != 0 || fn.Wrapper() || fn.WasmImport != nil || fn.WasmExport != nil {
		// Pragmas change calling or scheduling behavior in ways a
		// split cannot be assumed to preserve: nosplit stack limits,
		// cgo argument pinning, and so on. A plain //go:noinline is
		// also respected, as splitting rewrites the function much
		// like inlining would.
		return nil
	}

	if fn.Sym().Linkname != "" || base.Flag.CompilingRuntime {
		// Currently we skip splitting in -+ to avoid potential surprises.
		// The runtime generally does not make use of interfaces, because
		// implicit allocation is forbidden.
		return nil
	}

	if fn.Inl != nil && fn.Inl.Cost < inline.MinSplitCost {
		// Cheap and inlinable; leave it to the inliner.
		return nil
	}

	// The thunk forwards the receiver and every parameter by name,
	// and blank or anonymous ones cannot be read.
	//
	// TODO(mcy): synthesize zero values for those instead.
	for _, p := range fn.Type().RecvParams() {
		n, ok := p.Nname.(*ir.Name)
		if !ok || n.Sym() == nil || n.Sym().IsBlank() {
			return nil
		}
	}

	reads, closures := scanBody(fn)

	// A named result that the body reads cannot change type. The
	// analysis already treats results assigned outside of return
	// statements as unknown, so references here are rare shapes like
	// a result read after an assignment inside a return expression,
	// or a result captured by value in a function literal; drop just
	// those slots.
	for _, n := range reads {
		for i, f := range fn.Type().Results() {
			if f.Nname == n {
				devirt[i] = nil
			}
		}
	}

	can := false
	for _, t := range devirt {
		can = can || t != nil
	}
	if !can {
		return nil
	}

	fdv := s.buildSplit(fn, devirt, closures)

	fn.DevirtVariant = fdv
	fdv.DevirtOriginal = fn
	fdv.ResultTypeSets = fn.ResultTypeSets

	// The variant has no export-data body of its own. When the
	// original is inlinable, a call to the variant inlines as the
	// original's body with the devirtualized results unboxed; see
	// noder's unifiedInlineCall.
	fdv.Inl = fn.Inl

	if base.Flag.LowerM != 0 {
		base.WarnfAt(fn.Pos(), "splitting %v into %v", fn, fdv)
	}
	return fdv
}

// splitResultTypes reports, per result slot of fn, the concrete type
// to devirtualize that slot to, or nil for slots to leave alone.
//
// A nil slice means no slot qualifies.
func (s *State) splitResultTypes(fn *ir.Func) []*types.Type {
	rts := fn.ResultTypeSets
	if rts == nil {
		return nil
	}

	var devirt []*types.Type
	for i, set := range rts {
		if len(set) != 1 || set[0] == nilType || set[0] == unknownType {
			continue
		}
		if devirt == nil {
			devirt = make([]*types.Type, len(rts))
		}
		devirt[i] = set[0]
	}
	return devirt
}

// scanBody finds the named result parameters that fn's body reads,
// and the function literals it contains, at any nesting depth.
//
// The walk does not descend into the literals' bodies by itself, so
// a result they capture is counted as read through their
// ClosureVars. The Defn link is the one to check: at any depth it
// names the canonical variable, where Outer names the capture one
// level up.
func scanBody(fn *ir.Func) (reads []*ir.Name, closures []*ir.Func) {
	var scan func(body ir.Nodes)
	scan = func(body ir.Nodes) {
		ir.VisitList(body, func(n ir.Node) {
			switch n := n.(type) {
			case *ir.Name:
				if n.Class == ir.PPARAMOUT {
					reads = append(reads, n)
				}
			case *ir.ClosureExpr:
				closures = append(closures, n.Func)
				for _, cv := range n.Func.ClosureVars {
					if defn, ok := cv.Defn.(*ir.Name); ok && defn.Class == ir.PPARAMOUT {
						reads = append(reads, defn)
					}
				}
				scan(n.Func.Body)
			}
		})
	}
	scan(fn.Body)
	return reads, closures
}

// buildSplit moves fn's body into a new devirtualized variant and
// rebuilds fn as a thunk around it.
func (s *State) buildSplit(fn *ir.Func, devirt []*types.Type, closures []*ir.Func) *ir.Func {
	pos := fn.Pos()

	// The variant's signature is fn's with the devirtualized result
	// types substituted. The fields are fresh so that DeclareParams
	// can bind new parameter names owned by the variant.
	clone := func(f *types.Field, typ *types.Type) *types.Field {
		res := types.NewField(f.Pos, f.Sym, typ)
		res.SetIsDDD(f.IsDDD())
		return res
	}

	// A method's receiver is promoted to a leading parameter of the
	// variant, the way shaped methods promote theirs: only static
	// calls reach the variant, so nothing needs it to be a method,
	// and the method itself stays behind as the thunk that itabs
	// point to.
	params := make([]*types.Field, len(fn.Type().RecvParams()))
	for i, f := range fn.Type().RecvParams() {
		params[i] = clone(f, f.Type)
	}

	results := make([]*types.Field, len(fn.Type().Results()))
	for i, f := range fn.Type().Results() {
		typ := f.Type
		if devirt[i] != nil {
			typ = devirt[i]
		}
		results[i] = clone(f, typ)
	}

	sym := fn.Sym().Pkg.Lookup(fn.Sym().Name + ".dv")
	fdv := ir.NewFunc(pos, fn.Nname.Pos(), sym, types.NewSignature(nil, params, results))
	fdv.SetDupok(fn.Dupok())
	typecheck.DeclFunc(fdv)

	// Move the body, the locals, and the body-scoped debug info.
	fdv.Body, fn.Body = fn.Body, nil
	fdv.Parents, fn.Parents = fn.Parents, nil
	fdv.Marks, fn.Marks = fn.Marks, nil
	fdv.Endlineno = fn.Endlineno
	fdv.Label = fn.Label

	keep := fn.Dcl[:0]
	for _, n := range fn.Dcl {
		if n.Class == ir.PAUTO {
			n.Curfn = fdv
			fdv.Dcl = append(fdv.Dcl, n)
		} else {
			keep = append(keep, n)
		}
	}
	fn.Dcl = keep

	// The moved body still names fn's parameters; retarget it to the
	// variant's. Result parameters are rebound too: a retained named
	// result (an interface slot that did not qualify) may be read or
	// assigned by the body. Devirtualized slots get a mapping as
	// well, but it is never exercised, since slots the body reads
	// were dropped from devirt above.
	subst := make(map[*ir.Name]*ir.Name)
	for i, f := range fn.Type().RecvParams() {
		subst[f.Nname.(*ir.Name)] = fdv.Type().Params()[i].Nname.(*ir.Name)
	}
	for i, f := range fn.Type().Results() {
		if old, ok := f.Nname.(*ir.Name); ok {
			subst[old] = fdv.Type().Results()[i].Nname.(*ir.Name)
		}
	}

	var edit func(ir.Node) ir.Node
	edit = func(n ir.Node) ir.Node {
		if n, ok := n.(*ir.Name); ok {
			if repl, ok := subst[n]; ok {
				return repl
			}
			return n
		}
		// WithHidden: the reader stows dictionary and runtime-type
		// operands in fields EditChildren skips, like IndexExpr.RType,
		// and those reference the parameters too.
		ir.EditChildrenWithHidden(n, edit)
		return n
	}
	for i, n := range fdv.Body {
		fdv.Body[i] = edit(n)
	}

	// Update captures to point at the new function. Both links need
	// rebinding: Outer names the capture one level up, and Defn the
	// canonical variable, which Name.Canonical follows; a nested
	// literal's Defn reaches fn's parameters directly even when its
	// Outer does not.
	for _, cf := range closures {
		if cf.ClosureParent == fn {
			cf.ClosureParent = fdv
		}
		for _, cv := range cf.ClosureVars {
			if repl, ok := subst[cv.Outer]; ok {
				cv.Outer = repl
			}
			if defn, ok := cv.Defn.(*ir.Name); ok {
				if repl, ok := subst[defn]; ok {
					cv.Defn = repl
				}
			}
		}
	}

	// Unbox the devirtualized slots at every return: strip the
	// boxing conversion where one is directly present, and extract
	// the value word otherwise. The extraction is unchecked: the
	// analysis proved the dynamic type, and a checked assertion
	// would survive to run time whenever the boxing site is not
	// visible in this function, such as a value returned by a
	// recorded callee that did not itself split.
	ir.VisitList(fdv.Body, func(n ir.Node) {
		ret, ok := n.(*ir.ReturnStmt)
		if !ok {
			return
		}
		if len(ret.Results) != len(devirt) {
			base.FatalfAt(ret.Pos(), "split function %v has a bare return", fdv)
		}
		for i, t := range devirt {
			if t == nil {
				continue
			}
			e := ret.Results[i]
			for {
				if p, ok := e.(*ir.ParenExpr); ok {
					e = p.X
					continue
				}
				break
			}
			if conv, ok := e.(*ir.ConvExpr); ok && conv.Op() == ir.OCONVIFACE && types.Identical(conv.X.Type(), t) {
				// The conversion node may hold the statements that
				// define its operand: rewriting "return g()" for a
				// multi-valued g leaves the call and its temporaries
				// in the implicit conversion's init list. Hoist them
				// onto the return statement rather than dropping
				// them with the conversion.
				ret.PtrInit().Append(ir.TakeInit(conv)...)
				ret.Results[i] = conv.X
				continue
			}
			ret.Results[i] = Unbox(ret.Results[i], t)
		}
	})

	typecheck.FinishFuncBody()

	// Rebuild fn as the boxing thunk. Typechecking inserts appropriate
	// boxing conversions.
	ir.WithFunc(fn, func() {
		recvParams := fn.Type().RecvParams()
		args := make([]ir.Node, len(recvParams))
		for i, f := range recvParams {
			args[i] = f.Nname.(*ir.Name)
		}
		call := typecheck.Call(pos, fdv.Nname, args, fn.Type().IsVariadic())

		ret := ir.NewReturnStmt(pos, nil)
		ret.Results = []ir.Node{call}
		fn.Body = []ir.Node{ret}
		typecheck.Stmts(fn.Body)
	})

	return fdv
}

// Unbox extracts the value of concrete type t from an interface
// value known to hold it, with no runtime check.
func Unbox(v ir.Node, t *types.Type) ir.Node {
	idata := ir.NewUnaryExpr(v.Pos(), ir.OIDATA, v)
	idata.SetTypecheck(1)
	if types.IsDirectIface(t) {
		idata.SetType(t)
		return idata
	}

	// The value is boxed behind a pointer, and the proven dynamic
	// type also proves the interface is not nil, so the load cannot
	// fault.
	idata.SetType(types.NewPtr(t))
	deref := ir.NewStarExpr(v.Pos(), idata)
	deref.SetType(t)
	deref.SetTypecheck(1)
	return deref
}

// StaticResults rewrites a static call to a split function into a
// call of its devirtualized variant, re-boxing the results at the
// call site.
//
// The rewrite is returned as an OINLCALL, as if the thunk left
// behind by [State.SplitResultFunc] had been inlined there: the
// boxing conversions sit in ReturnVars, where the caller's
// concrete-type analysis sees them, so method calls on the results
// devirtualize in turn.
func StaticResults(s *State, curfn *ir.Func, call *ir.CallExpr) *ir.InlinedCallExpr {
	if call.GoDefer || call.Op() != ir.OCALLFUNC {
		// A go or defer statement needs a plain call.
		return nil
	}
	callee := staticCallee(call)
	if callee == nil {
		return nil
	}
	fdv := callee.DevirtVariant
	if fdv == nil {
		return nil
	}

	pos := call.Pos()
	origResults := callee.Type().Results()
	origType := call.Type()

	// Retarget the call and update its type for the new result
	// stack offsets, as [StaticCall] does.
	call.Fun = fdv.Nname
	types.CheckSize(fdv.Type())
	switch ft := fdv.Type(); ft.NumResults() {
	case 1:
		call.SetType(ft.Result(0).Type)
	default:
		call.SetType(ft.ResultsTuple())
	}

	results := fdv.Type().Results()
	tmps := make([]ir.Node, len(results))
	retvars := make([]ir.Node, len(results))
	for i, f := range results {
		tmp := typecheck.TempAt(pos, curfn, f.Type)
		tmps[i] = tmp
		if orig := origResults[i].Type; orig.IsInterface() && !f.Type.IsInterface() {
			retvars[i] = typecheck.AssignConv(tmp, orig, "devirtualized result")
		} else {
			retvars[i] = tmp
		}
	}

	var as ir.Node
	if len(tmps) == 1 {
		n := ir.NewAssignStmt(pos, tmps[0], call)
		n.SetTypecheck(1)
		as = n
	} else {
		n := ir.NewAssignListStmt(pos, ir.OAS2FUNC, tmps, []ir.Node{call})
		n.SetTypecheck(1)
		as = n
	}

	inl := ir.NewInlinedCallExpr(pos, []ir.Node{as}, retvars)
	inl.SetType(origType)
	inl.SetTypecheck(1)

	if base.Flag.LowerM != 0 {
		base.WarnfAt(pos, "devirtualizing call to %v", fdv)
	}
	return inl
}
