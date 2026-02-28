// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package devirtualize

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/inline"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/pgoir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// CallStat summarizes a single call site.
//
// This is used only for debug logging.
type CallStat struct {
	Pkg string // base.Ctxt.Pkgpath
	Pos string // file:line:col of call.

	Caller string // Linker symbol name of calling function.

	// Direct or indirect call.
	Direct bool

	// For indirect calls, interface call or other indirect function call.
	Interface bool

	// Total edge weight from this call site.
	Weight int64

	// Hottest callee from this call site, regardless of type
	// compatibility.
	Hottest       string
	HottestWeight int64

	// Devirtualized callee if != "".
	//
	// Note that this may be different than Hottest because we apply
	// type-check restrictions, which helps distinguish multiple calls on
	// the same line.
	Devirtualized       string
	DevirtualizedWeight int64
}

// ProfileGuided performs call devirtualization of indirect calls based on
// profile information.
//
// Specifically, it performs conditional devirtualization of interface calls or
// function value calls for the hottest callee.
//
// That is, for interface calls it performs a transformation like:
//
//	type Iface interface {
//		Foo()
//	}
//
//	type Concrete struct{}
//
//	func (Concrete) Foo() {}
//
//	func foo(i Iface) {
//		i.Foo()
//	}
//
// to:
//
//	func foo(i Iface) {
//		if c, ok := i.(Concrete); ok {
//			c.Foo()
//		} else {
//			i.Foo()
//		}
//	}
//
// For function value calls it performs a transformation like:
//
//	func Concrete() {}
//
//	func foo(fn func()) {
//		fn()
//	}
//
// to:
//
//	func foo(fn func()) {
//		if internal/abi.FuncPCABIInternal(fn) == internal/abi.FuncPCABIInternal(Concrete) {
//			Concrete()
//		} else {
//			fn()
//		}
//	}
//
// The primary benefit of this transformation is enabling inlining of the
// direct call.
func ProfileGuided(fn *ir.Func, p *pgoir.Profile) {
	ir.CurFunc = fn

	name := ir.LinkFuncName(fn)

	var jsonW *json.Encoder
	if base.Debug.PGODebug >= 3 {
		jsonW = json.NewEncoder(os.Stdout)
	}

	var edit func(n ir.Node) ir.Node
	edit = func(n ir.Node) ir.Node {
		if n == nil {
			return n
		}

		ir.EditChildren(n, edit)

		call, ok := n.(*ir.CallExpr)
		if !ok {
			return n
		}

		var stat *CallStat
		if base.Debug.PGODebug >= 3 {
			// Statistics about every single call. Handy for external data analysis.
			//
			// TODO(prattmic): Log via logopt?
			stat = constructCallStat(p, fn, name, call)
			if stat != nil {
				defer func() {
					jsonW.Encode(&stat)
				}()
			}
		}

		op := call.Op()
		if op != ir.OCALLFUNC && op != ir.OCALLINTER {
			return n
		}

		if base.Debug.PGODebug >= 2 {
			fmt.Printf("%v: PGO devirtualize considering call %v\n", ir.Line(call), call)
		}

		if call.GoDefer {
			if base.Debug.PGODebug >= 2 {
				fmt.Printf("%v: can't PGO devirtualize go/defer call %v\n", ir.Line(call), call)
			}
			return n
		}

		var newNode ir.Node
		var callee *ir.Func
		var weight int64
		switch op {
		case ir.OCALLFUNC:
			newNode, callee, weight = maybeDevirtualizeFunctionCall(p, fn, call)
		case ir.OCALLINTER:
			newNode, callee, weight = maybeDevirtualizeInterfaceCall(p, fn, call)
		default:
			panic("unreachable")
		}

		if newNode == nil {
			return n
		}

		if stat != nil {
			stat.Devirtualized = ir.LinkFuncName(callee)
			stat.DevirtualizedWeight = weight
		}

		return newNode
	}

	ir.EditChildren(fn, edit)
}

// Devirtualize interface call if possible and eligible. Returns the new
// ir.Node if call was devirtualized, and if so also the callee and weight of
// the devirtualized edge.
func maybeDevirtualizeInterfaceCall(p *pgoir.Profile, fn *ir.Func, call *ir.CallExpr) (ir.Node, *ir.Func, int64) {
	if base.Debug.PGODevirtualize < 1 {
		return nil, nil, 0
	}

	// Bail if we do not have a hot callee.
	callee, weight := findHotConcreteInterfaceCallee(p, fn, call)
	if callee == nil {
		return nil, nil, 0
	}
	// Bail if we do not have a Type node for the hot callee.
	ctyp := methodRecvType(callee)
	if ctyp == nil {
		return nil, nil, 0
	}
	// Bail if we know for sure it won't inline.
	if !shouldPGODevirt(callee) {
		return nil, nil, 0
	}
	// Bail if de-selected by PGO Hash.
	if !base.PGOHash.MatchPosWithInfo(call.Pos(), "devirt", nil) {
		return nil, nil, 0
	}

	return rewriteInterfaceCall(call, fn, callee, ctyp), callee, weight
}

// Devirtualize an indirect function call if possible and eligible. Returns the new
// ir.Node if call was devirtualized, and if so also the callee and weight of
// the devirtualized edge.
func maybeDevirtualizeFunctionCall(p *pgoir.Profile, fn *ir.Func, call *ir.CallExpr) (ir.Node, *ir.Func, int64) {
	if base.Debug.PGODevirtualize < 2 {
		return nil, nil, 0
	}

	// Bail if this is a direct call; no devirtualization necessary.
	callee := pgoir.DirectCallee(call.Fun)
	if callee != nil {
		return nil, nil, 0
	}

	// Bail if we do not have a hot callee.
	callee, weight := findHotConcreteFunctionCallee(p, fn, call)
	if callee == nil {
		return nil, nil, 0
	}

	// TODO(go.dev/issue/61577): Closures need the closure context passed
	// via the context register. That requires extra plumbing that we
	// haven't done yet.
	if callee.OClosure != nil {
		if base.Debug.PGODebug >= 3 {
			fmt.Printf("callee %s is a closure, skipping\n", ir.FuncName(callee))
		}
		return nil, nil, 0
	}
	// runtime.memhash_varlen does not look like a closure, but it uses
	// internal/runtime/sys.GetClosurePtr to access data encoded by
	// callers, which are generated by
	// cmd/compile/internal/reflectdata.genhash.
	if callee.Sym().Pkg.Path == "runtime" && callee.Sym().Name == "memhash_varlen" {
		if base.Debug.PGODebug >= 3 {
			fmt.Printf("callee %s is a closure (runtime.memhash_varlen), skipping\n", ir.FuncName(callee))
		}
		return nil, nil, 0
	}
	// TODO(prattmic): We don't properly handle methods as callees in two
	// different dimensions:
	//
	// 1. Method expressions. e.g.,
	//
	//      var fn func(*os.File, []byte) (int, error) = (*os.File).Read
	//
	// In this case, typ will report *os.File as the receiver while
	// ctyp reports it as the first argument. types.Identical ignores
	// receiver parameters, so it treats these as different, even though
	// they are still call compatible.
	//
	// 2. Method values. e.g.,
	//
	//      var f *os.File
	//      var fn func([]byte) (int, error) = f.Read
	//
	// types.Identical will treat these as compatible (since receiver
	// parameters are ignored). However, in this case, we do not call
	// (*os.File).Read directly. Instead, f is stored in closure context
	// and we call the wrapper (*os.File).Read-fm. However, runtime/pprof
	// hides wrappers from profiles, making it appear that there is a call
	// directly to the method. We could recognize this pattern return the
	// wrapper rather than the method.
	//
	// N.B. perf profiles will report wrapper symbols directly, so
	// ideally we should support direct wrapper references as well.
	if callee.Type().Recv() != nil {
		if base.Debug.PGODebug >= 3 {
			fmt.Printf("callee %s is a method, skipping\n", ir.FuncName(callee))
		}
		return nil, nil, 0
	}

	// Bail if we know for sure it won't inline.
	if !shouldPGODevirt(callee) {
		return nil, nil, 0
	}
	// Bail if de-selected by PGO Hash.
	if !base.PGOHash.MatchPosWithInfo(call.Pos(), "devirt", nil) {
		return nil, nil, 0
	}

	return rewriteFunctionCall(call, fn, callee), callee, weight
}

// shouldPGODevirt checks if we should perform PGO devirtualization to the
// target function.
//
// PGO devirtualization is most valuable when the callee is inlined, so if it
// won't inline we can skip devirtualizing.
func shouldPGODevirt(fn *ir.Func) bool {
	var reason string
	if base.Flag.LowerM > 1 || logopt.Enabled() {
		defer func() {
			if reason != "" {
				if base.Flag.LowerM > 1 {
					fmt.Printf("%v: should not PGO devirtualize %v: %s\n", ir.Line(fn), ir.FuncName(fn), reason)
				}
				if logopt.Enabled() {
					logopt.LogOpt(fn.Pos(), ": should not PGO devirtualize function", "pgoir-devirtualize", ir.FuncName(fn), reason)
				}
			}
		}()
	}

	reason = inline.InlineImpossible(fn)
	if reason != "" {
		return false
	}

	// TODO(prattmic): checking only InlineImpossible is very conservative,
	// primarily excluding only functions with pragmas. We probably want to
	// move in either direction. Either:
	//
	// 1. Don't even bother to check InlineImpossible, as it affects so few
	// functions.
	//
	// 2. Or consider the function body (notably cost) to better determine
	// if the function will actually inline.

	return true
}

// constructCallStat builds an initial CallStat describing this call, for
// logging. If the call is devirtualized, the devirtualization fields should be
// updated.
func constructCallStat(p *pgoir.Profile, fn *ir.Func, name string, call *ir.CallExpr) *CallStat {
	switch call.Op() {
	case ir.OCALLFUNC, ir.OCALLINTER, ir.OCALLMETH:
	default:
		// We don't care about logging builtin functions.
		return nil
	}

	stat := CallStat{
		Pkg:    base.Ctxt.Pkgpath,
		Pos:    ir.Line(call),
		Caller: name,
	}

	offset := pgoir.NodeLineOffset(call, fn)

	hotter := func(e *pgoir.IREdge) bool {
		if stat.Hottest == "" {
			return true
		}
		if e.Weight != stat.HottestWeight {
			return e.Weight > stat.HottestWeight
		}
		// If weight is the same, arbitrarily sort lexicographally, as
		// findHotConcreteCallee does.
		return e.Dst.Name() < stat.Hottest
	}

	callerNode := p.WeightedCG.IRNodes[name]
	if callerNode == nil {
		return nil
	}

	// Sum of all edges from this callsite, regardless of callee.
	// For direct calls, this should be the same as the single edge
	// weight (except for multiple calls on one line, which we
	// can't distinguish).
	for _, edge := range callerNode.OutEdges {
		if edge.CallSiteOffset != offset {
			continue
		}
		stat.Weight += edge.Weight
		if hotter(edge) {
			stat.HottestWeight = edge.Weight
			stat.Hottest = edge.Dst.Name()
		}
	}

	switch call.Op() {
	case ir.OCALLFUNC:
		stat.Interface = false

		callee := pgoir.DirectCallee(call.Fun)
		if callee != nil {
			stat.Direct = true
			if stat.Hottest == "" {
				stat.Hottest = ir.LinkFuncName(callee)
			}
		} else {
			stat.Direct = false
		}
	case ir.OCALLINTER:
		stat.Direct = false
		stat.Interface = true
	case ir.OCALLMETH:
		base.FatalfAt(call.Pos(), "OCALLMETH missed by typecheck")
	}

	return &stat
}

// copyInputs copies the inputs to a call: the receiver (for interface calls)
// or function value (for function value calls) and the arguments. These
// expressions are evaluated once and assigned to temporaries.
//
// The assignment statement is added to init and the copied receiver/fn
// expression and copied arguments expressions are returned.
func copyInputs(curfn *ir.Func, pos src.XPos, recvOrFn ir.Node, args []ir.Node, init *ir.Nodes) (ir.Node, []ir.Node) {
	// Evaluate receiver/fn and argument expressions. The receiver/fn is
	// used twice but we don't want to cause side effects twice. The
	// arguments are used in two different calls and we can't trivially
	// copy them.
	//
	// recvOrFn must be first in the assignment list as its side effects
	// must be ordered before argument side effects.
	var lhs, rhs []ir.Node
	newRecvOrFn := typecheck.TempAt(pos, curfn, recvOrFn.Type())
	lhs = append(lhs, newRecvOrFn)
	rhs = append(rhs, recvOrFn)

	for _, arg := range args {
		argvar := typecheck.TempAt(pos, curfn, arg.Type())

		lhs = append(lhs, argvar)
		rhs = append(rhs, arg)
	}

	asList := ir.NewAssignListStmt(pos, ir.OAS2, lhs, rhs)
	init.Append(typecheck.Stmt(asList))

	return newRecvOrFn, lhs[1:]
}

// retTemps returns a slice of temporaries to be used for storing result values from call.
func retTemps(curfn *ir.Func, pos src.XPos, call *ir.CallExpr) []ir.Node {
	sig := call.Fun.Type()
	var retvars []ir.Node
	for _, ret := range sig.Results() {
		retvars = append(retvars, typecheck.TempAt(pos, curfn, ret.Type))
	}
	return retvars
}

// condCall returns an ir.InlinedCallExpr that performs a call to thenCall if
// cond is true and elseCall if cond is false. The return variables of the
// InlinedCallExpr evaluate to the return values from the call.
func condCall(curfn *ir.Func, pos src.XPos, cond ir.Node, thenCall, elseCall *ir.CallExpr, init ir.Nodes) *ir.InlinedCallExpr {
	// Doesn't matter whether we use thenCall or elseCall, they must have
	// the same return types.
	retvars := retTemps(curfn, pos, thenCall)

	var thenBlock, elseBlock ir.Nodes
	if len(retvars) == 0 {
		thenBlock.Append(thenCall)
		elseBlock.Append(elseCall)
	} else {
		// Copy slice so edits in one location don't affect another.
		thenRet := append([]ir.Node(nil), retvars...)
		thenAsList := ir.NewAssignListStmt(pos, ir.OAS2, thenRet, []ir.Node{thenCall})
		thenBlock.Append(typecheck.Stmt(thenAsList))

		elseRet := append([]ir.Node(nil), retvars...)
		elseAsList := ir.NewAssignListStmt(pos, ir.OAS2, elseRet, []ir.Node{elseCall})
		elseBlock.Append(typecheck.Stmt(elseAsList))
	}

	nif := ir.NewIfStmt(pos, cond, thenBlock, elseBlock)
	nif.SetInit(init)
	nif.Likely = true

	body := []ir.Node{typecheck.Stmt(nif)}

	// This isn't really an inlined call of course, but InlinedCallExpr
	// makes handling reassignment of return values easier.
	res := ir.NewInlinedCallExpr(pos, body, retvars)
	res.SetType(thenCall.Type())
	res.SetTypecheck(1)
	return res
}

// rewriteInterfaceCall devirtualizes the given interface call using a direct
// method call to concretetyp.
func rewriteInterfaceCall(call *ir.CallExpr, curfn, callee *ir.Func, concretetyp *types.Type) ir.Node {
	if base.Flag.LowerM != 0 {
		fmt.Printf("%v: PGO devirtualizing interface call %v to %v\n", ir.Line(call), call.Fun, callee)
	}

	// We generate an OINCALL of:
	//
	// var recv Iface
	//
	// var arg1 A1
	// var argN AN
	//
	// var ret1 R1
	// var retN RN
	//
	// recv, arg1, argN = recv expr, arg1 expr, argN expr
	//
	// t, ok := recv.(Concrete)
	// if ok {
	//   ret1, retN = t.Method(arg1, ... argN)
	// } else {
	//   ret1, retN = recv.Method(arg1, ... argN)
	// }
	//
	// OINCALL retvars: ret1, ... retN
	//
	// This isn't really an inlined call of course, but InlinedCallExpr
	// makes handling reassignment of return values easier.
	//
	// TODO(prattmic): This increases the size of the AST in the caller,
	// making it less like to inline. We may want to compensate for this
	// somehow.

	sel := call.Fun.(*ir.SelectorExpr)
	method := sel.Sel
	pos := call.Pos()
	init := ir.TakeInit(call)

	recv, args := copyInputs(curfn, pos, sel.X, call.Args.Take(), &init)

	// Copy slice so edits in one location don't affect another.
	argvars := append([]ir.Node(nil), args...)
	call.Args = argvars

	tmpnode := typecheck.TempAt(base.Pos, curfn, concretetyp)
	tmpok := typecheck.TempAt(base.Pos, curfn, types.Types[types.TBOOL])

	assert := ir.NewTypeAssertExpr(pos, recv, concretetyp)

	assertAsList := ir.NewAssignListStmt(pos, ir.OAS2, []ir.Node{tmpnode, tmpok}, []ir.Node{typecheck.Expr(assert)})
	init.Append(typecheck.Stmt(assertAsList))

	concreteCallee := typecheck.XDotMethod(pos, tmpnode, method, true)
	// Copy slice so edits in one location don't affect another.
	argvars = append([]ir.Node(nil), argvars...)
	concreteCall := typecheck.Call(pos, concreteCallee, argvars, call.IsDDD).(*ir.CallExpr)

	res := condCall(curfn, pos, tmpok, concreteCall, call, init)

	if base.Debug.PGODebug >= 3 {
		fmt.Printf("PGO devirtualizing interface call to %+v. After: %+v\n", concretetyp, res)
	}

	return res
}

// rewriteFunctionCall devirtualizes the given OCALLFUNC using a direct
// function call to callee.
func rewriteFunctionCall(call *ir.CallExpr, curfn, callee *ir.Func) ir.Node {
	if base.Flag.LowerM != 0 {
		fmt.Printf("%v: PGO devirtualizing function call %v to %v\n", ir.Line(call), call.Fun, callee)
	}

	// We generate an OINCALL of:
	//
	// var fn FuncType
	//
	// var arg1 A1
	// var argN AN
	//
	// var ret1 R1
	// var retN RN
	//
	// fn, arg1, argN = fn expr, arg1 expr, argN expr
	//
	// fnPC := internal/abi.FuncPCABIInternal(fn)
	// concretePC := internal/abi.FuncPCABIInternal(concrete)
	//
	// if fnPC == concretePC {
	//   ret1, retN = concrete(arg1, ... argN) // Same closure context passed (TODO)
	// } else {
	//   ret1, retN = fn(arg1, ... argN)
	// }
	//
	// OINCALL retvars: ret1, ... retN
	//
	// This isn't really an inlined call of course, but InlinedCallExpr
	// makes handling reassignment of return values easier.

	pos := call.Pos()
	init := ir.TakeInit(call)

	fn, args := copyInputs(curfn, pos, call.Fun, call.Args.Take(), &init)

	// Copy slice so edits in one location don't affect another.
	argvars := append([]ir.Node(nil), args...)
	call.Args = argvars

	// FuncPCABIInternal takes an interface{}, emulate that. This is needed
	// for to ensure we get the MAKEFACE we need for SSA.
	fnIface := typecheck.Expr(ir.NewConvExpr(pos, ir.OCONV, types.Types[types.TINTER], fn))
	calleeIface := typecheck.Expr(ir.NewConvExpr(pos, ir.OCONV, types.Types[types.TINTER], callee.Nname))

	fnPC := ir.FuncPC(pos, fnIface, obj.ABIInternal)
	concretePC := ir.FuncPC(pos, calleeIface, obj.ABIInternal)

	pcEq := typecheck.Expr(ir.NewBinaryExpr(base.Pos, ir.OEQ, fnPC, concretePC))

	// TODO(go.dev/issue/61577): Handle callees that a closures and need a
	// copy of the closure context from call. For now, we skip callees that
	// are closures in maybeDevirtualizeFunctionCall.
	if callee.OClosure != nil {
		base.Fatalf("Callee is a closure: %+v", callee)
	}

	// Copy slice so edits in one location don't affect another.
	argvars = append([]ir.Node(nil), argvars...)
	concreteCall := typecheck.Call(pos, callee.Nname, argvars, call.IsDDD).(*ir.CallExpr)

	res := condCall(curfn, pos, pcEq, concreteCall, call, init)

	if base.Debug.PGODebug >= 3 {
		fmt.Printf("PGO devirtualizing function call to %+v. After: %+v\n", ir.FuncName(callee), res)
	}

	return res
}

// methodRecvType returns the type containing method fn. Returns nil if fn
// is not a method.
func methodRecvType(fn *ir.Func) *types.Type {
	recv := fn.Nname.Type().Recv()
	if recv == nil {
		return nil
	}
	return recv.Type
}

// interfaceCallRecvTypeAndMethod returns the type and the method of the interface
// used in an interface call.
func interfaceCallRecvTypeAndMethod(call *ir.CallExpr) (*types.Type, *types.Sym) {
	if call.Op() != ir.OCALLINTER {
		base.Fatalf("Call isn't OCALLINTER: %+v", call)
	}

	sel, ok := call.Fun.(*ir.SelectorExpr)
	if !ok {
		base.Fatalf("OCALLINTER doesn't contain SelectorExpr: %+v", call)
	}

	return sel.X.Type(), sel.Sel
}

// findHotConcreteCallee returns the *ir.Func of the hottest callee of a call,
// if available, and its edge weight. extraFn can perform additional
// applicability checks on each candidate edge. If extraFn returns false,
// candidate will not be considered a valid callee candidate.
func findHotConcreteCallee(p *pgoir.Profile, caller *ir.Func, call *ir.CallExpr, extraFn func(callerName string, callOffset int, candidate *pgoir.IREdge) bool) (*ir.Func, int64) {
	callerName := ir.LinkFuncName(caller)
	callerNode := p.WeightedCG.IRNodes[callerName]
	callOffset := pgoir.NodeLineOffset(call, caller)

	if callerNode == nil {
		return nil, 0
	}

	var hottest *pgoir.IREdge

	// Returns true if e is hotter than hottest.
	//
	// Naively this is just e.Weight > hottest.Weight, but because OutEdges
	// has arbitrary iteration order, we need to apply additional sort
	// criteria when e.Weight == hottest.Weight to ensure we have stable
	// selection.
	hotter := func(e *pgoir.IREdge) bool {
		if hottest == nil {
			return true
		}
		if e.Weight != hottest.Weight {
			return e.Weight > hottest.Weight
		}

		// Now e.Weight == hottest.Weight, we must select on other
		// criteria.

		// If only one edge has IR, prefer that one.
		if (hottest.Dst.AST == nil) != (e.Dst.AST == nil) {
			if e.Dst.AST != nil {
				return true
			}
			return false
		}

		// Arbitrary, but the callee names will always differ. Select
		// the lexicographically first callee.
		return e.Dst.Name() < hottest.Dst.Name()
	}

	for _, e := range callerNode.OutEdges {
		if e.CallSiteOffset != callOffset {
			continue
		}

		if !hotter(e) {
			// TODO(prattmic): consider total caller weight? i.e.,
			// if the hottest callee is only 10% of the weight,
			// maybe don't devirtualize? Similarly, if this is call
			// is globally very cold, there is not much value in
			// devirtualizing.
			if base.Debug.PGODebug >= 2 {
				fmt.Printf("%v: edge %s:%d -> %s (weight %d): too cold (hottest %d)\n", ir.Line(call), callerName, callOffset, e.Dst.Name(), e.Weight, hottest.Weight)
			}
			continue
		}

		if e.Dst.AST == nil {
			// Destination isn't visible from this package
			// compilation.
			//
			// We must assume it implements the interface.
			//
			// We still record this as the hottest callee so far
			// because we only want to return the #1 hottest
			// callee. If we skip this then we'd return the #2
			// hottest callee.
			if base.Debug.PGODebug >= 2 {
				fmt.Printf("%v: edge %s:%d -> %s (weight %d) (missing IR): hottest so far\n", ir.Line(call), callerName, callOffset, e.Dst.Name(), e.Weight)
			}
			hottest = e
			continue
		}

		if extraFn != nil && !extraFn(callerName, callOffset, e) {
			continue
		}

		if base.Debug.PGODebug >= 2 {
			fmt.Printf("%v: edge %s:%d -> %s (weight %d): hottest so far\n", ir.Line(call), callerName, callOffset, e.Dst.Name(), e.Weight)
		}
		hottest = e
	}

	if hottest == nil || hottest.Weight == 0 {
		if base.Debug.PGODebug >= 2 {
			fmt.Printf("%v: call %s:%d: no hot callee\n", ir.Line(call), callerName, callOffset)
		}
		return nil, 0
	}

	if base.Debug.PGODebug >= 2 {
		fmt.Printf("%v: call %s:%d: hottest callee %s (weight %d)\n", ir.Line(call), callerName, callOffset, hottest.Dst.Name(), hottest.Weight)
	}
	return hottest.Dst.AST, hottest.Weight
}

// findHotConcreteInterfaceCallee returns the *ir.Func of the hottest callee of an
// interface call, if available, and its edge weight.
func findHotConcreteInterfaceCallee(p *pgoir.Profile, caller *ir.Func, call *ir.CallExpr) (*ir.Func, int64) {
	inter, method := interfaceCallRecvTypeAndMethod(call)

	return findHotConcreteCallee(p, caller, call, func(callerName string, callOffset int, e *pgoir.IREdge) bool {
		ctyp := methodRecvType(e.Dst.AST)
		if ctyp == nil {
			// Not a method.
			// TODO(prattmic): Support non-interface indirect calls.
			if base.Debug.PGODebug >= 2 {
				fmt.Printf("%v: edge %s:%d -> %s (weight %d): callee not a method\n", ir.Line(call), callerName, callOffset, e.Dst.Name(), e.Weight)
			}
			return false
		}

		// If ctyp doesn't implement inter it is most likely from a
		// different call on the same line
		if !typecheck.Implements(ctyp, inter) {
			// TODO(prattmic): this is overly strict. Consider if
			// ctyp is a partial implementation of an interface
			// that gets embedded in types that complete the
			// interface. It would still be OK to devirtualize a
			// call to this method.
			//
			// What we'd need to do is check that the function
			// pointer in the itab matches the method we want,
			// rather than doing a full type assertion.
			if base.Debug.PGODebug >= 2 {
				why := typecheck.ImplementsExplain(ctyp, inter)
				fmt.Printf("%v: edge %s:%d -> %s (weight %d): %v doesn't implement %v (%s)\n", ir.Line(call), callerName, callOffset, e.Dst.Name(), e.Weight, ctyp, inter, why)
			}
			return false
		}

		// If the method name is different it is most likely from a
		// different call on the same line
		if !strings.HasSuffix(e.Dst.Name(), "."+method.Name) {
			if base.Debug.PGODebug >= 2 {
				fmt.Printf("%v: edge %s:%d -> %s (weight %d): callee is a different method\n", ir.Line(call), callerName, callOffset, e.Dst.Name(), e.Weight)
			}
			return false
		}

		return true
	})
}

// findHotConcreteFunctionCallee returns the *ir.Func of the hottest callee of an
// indirect function call, if available, and its edge weight.
func findHotConcreteFunctionCallee(p *pgoir.Profile, caller *ir.Func, call *ir.CallExpr) (*ir.Func, int64) {
	typ := call.Fun.Type().Underlying()

	return findHotConcreteCallee(p, caller, call, func(callerName string, callOffset int, e *pgoir.IREdge) bool {
		ctyp := e.Dst.AST.Type().Underlying()

		// If ctyp doesn't match typ it is most likely from a different
		// call on the same line.
		//
		// Note that we are comparing underlying types, as different
		// defined types are OK. e.g., a call to a value of type
		// net/http.HandlerFunc can be devirtualized to a function with
		// the same underlying type.
		if !types.Identical(typ, ctyp) {
			if base.Debug.PGODebug >= 2 {
				fmt.Printf("%v: edge %s:%d -> %s (weight %d): %v doesn't match %v\n", ir.Line(call), callerName, callOffset, e.Dst.Name(), e.Weight, ctyp, typ)
			}
			return false
		}

		return true
	})
}
