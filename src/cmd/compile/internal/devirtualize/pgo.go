// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package devirtualize

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/inline"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/pgo"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
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
// Specifically, it performs conditional devirtualization of interface calls
// for the hottest callee. That is, it performs a transformation like:
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
// The primary benefit of this transformation is enabling inlining of the
// direct call.
func ProfileGuided(fn *ir.Func, p *pgo.Profile) {
	ir.CurFunc = fn

	name := ir.LinkFuncName(fn)

	// Can't devirtualize go/defer calls. See comment in Static.
	goDeferCall := make(map[*ir.CallExpr]bool)

	var jsonW *json.Encoder
	if base.Debug.PGODebug >= 3 {
		jsonW = json.NewEncoder(os.Stdout)
	}

	var edit func(n ir.Node) ir.Node
	edit = func(n ir.Node) ir.Node {
		if n == nil {
			return n
		}

		if gds, ok := n.(*ir.GoDeferStmt); ok {
			if call, ok := gds.Call.(*ir.CallExpr); ok {
				goDeferCall[call] = true
			}
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

		if call.Op() != ir.OCALLINTER {
			return n
		}

		if base.Debug.PGODebug >= 2 {
			fmt.Printf("%v: PGO devirtualize considering call %v\n", ir.Line(call), call)
		}

		if goDeferCall[call] {
			if base.Debug.PGODebug >= 2 {
				fmt.Printf("%v: can't PGO devirtualize go/defer call %v\n", ir.Line(call), call)
			}
			return n
		}

		// Bail if we do not have a hot callee.
		callee, weight := findHotConcreteCallee(p, fn, call)
		if callee == nil {
			return n
		}
		// Bail if we do not have a Type node for the hot callee.
		ctyp := methodRecvType(callee)
		if ctyp == nil {
			return n
		}
		// Bail if we know for sure it won't inline.
		if !shouldPGODevirt(callee) {
			return n
		}

		if !base.PGOHash.MatchPosWithInfo(n.Pos(), "devirt", nil) {
			// De-selected by PGO Hash.
			return n
		}

		if stat != nil {
			stat.Devirtualized = ir.LinkFuncName(callee)
			stat.DevirtualizedWeight = weight
		}

		return rewriteCondCall(call, fn, callee, ctyp)
	}

	ir.EditChildren(fn, edit)
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
					logopt.LogOpt(fn.Pos(), ": should not PGO devirtualize function", "pgo-devirtualize", ir.FuncName(fn), reason)
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
func constructCallStat(p *pgo.Profile, fn *ir.Func, name string, call *ir.CallExpr) *CallStat {
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

	offset := pgo.NodeLineOffset(call, fn)

	hotter := func(e *pgo.IREdge) bool {
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

	// Sum of all edges from this callsite, regardless of callee.
	// For direct calls, this should be the same as the single edge
	// weight (except for multiple calls on one line, which we
	// can't distinguish).
	callerNode := p.WeightedCG.IRNodes[name]
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

		callee := pgo.DirectCallee(call.Fun)
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

// rewriteCondCall devirtualizes the given call using a direct method call to
// concretetyp.
func rewriteCondCall(call *ir.CallExpr, curfn, callee *ir.Func, concretetyp *types.Type) ir.Node {
	if base.Flag.LowerM != 0 {
		fmt.Printf("%v: PGO devirtualizing %v to %v\n", ir.Line(call), call.Fun, callee)
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

	var retvars []ir.Node

	sig := call.Fun.Type()

	for _, ret := range sig.Results() {
		retvars = append(retvars, typecheck.TempAt(base.Pos, curfn, ret.Type))
	}

	sel := call.Fun.(*ir.SelectorExpr)
	method := sel.Sel
	pos := call.Pos()
	init := ir.TakeInit(call)

	// Evaluate receiver and argument expressions. The receiver is used
	// twice but we don't want to cause side effects twice. The arguments
	// are used in two different calls and we can't trivially copy them.
	//
	// recv must be first in the assignment list as its side effects must
	// be ordered before argument side effects.
	var lhs, rhs []ir.Node
	recv := typecheck.TempAt(base.Pos, curfn, sel.X.Type())
	lhs = append(lhs, recv)
	rhs = append(rhs, sel.X)

	// Move arguments to assignments prior to the if statement. We cannot
	// simply copy the args' IR, as some IR constructs cannot be copied,
	// such as labels (possible in InlinedCall nodes).
	args := call.Args.Take()
	for _, arg := range args {
		argvar := typecheck.TempAt(base.Pos, curfn, arg.Type())

		lhs = append(lhs, argvar)
		rhs = append(rhs, arg)
	}

	asList := ir.NewAssignListStmt(pos, ir.OAS2, lhs, rhs)
	init.Append(typecheck.Stmt(asList))

	// Copy slice so edits in one location don't affect another.
	argvars := append([]ir.Node(nil), lhs[1:]...)
	call.Args = argvars

	tmpnode := typecheck.TempAt(base.Pos, curfn, concretetyp)
	tmpok := typecheck.TempAt(base.Pos, curfn, types.Types[types.TBOOL])

	assert := ir.NewTypeAssertExpr(pos, recv, concretetyp)

	assertAsList := ir.NewAssignListStmt(pos, ir.OAS2, []ir.Node{tmpnode, tmpok}, []ir.Node{typecheck.Expr(assert)})
	init.Append(typecheck.Stmt(assertAsList))

	concreteCallee := typecheck.XDotMethod(pos, tmpnode, method, true)
	// Copy slice so edits in one location don't affect another.
	argvars = append([]ir.Node(nil), argvars...)
	concreteCall := typecheck.Call(pos, concreteCallee, argvars, call.IsDDD)

	var thenBlock, elseBlock ir.Nodes
	if len(retvars) == 0 {
		thenBlock.Append(concreteCall)
		elseBlock.Append(call)
	} else {
		// Copy slice so edits in one location don't affect another.
		thenRet := append([]ir.Node(nil), retvars...)
		thenAsList := ir.NewAssignListStmt(pos, ir.OAS2, thenRet, []ir.Node{concreteCall})
		thenBlock.Append(typecheck.Stmt(thenAsList))

		elseRet := append([]ir.Node(nil), retvars...)
		elseAsList := ir.NewAssignListStmt(pos, ir.OAS2, elseRet, []ir.Node{call})
		elseBlock.Append(typecheck.Stmt(elseAsList))
	}

	cond := ir.NewIfStmt(pos, nil, nil, nil)
	cond.SetInit(init)
	cond.Cond = tmpok
	cond.Body = thenBlock
	cond.Else = elseBlock
	cond.Likely = true

	body := []ir.Node{typecheck.Stmt(cond)}

	res := ir.NewInlinedCallExpr(pos, body, retvars)
	res.SetType(call.Type())
	res.SetTypecheck(1)

	if base.Debug.PGODebug >= 3 {
		fmt.Printf("PGO devirtualizing call to %+v. After: %+v\n", concretetyp, res)
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

// findHotConcreteCallee returns the *ir.Func of the hottest callee of an
// indirect call, if available, and its edge weight.
func findHotConcreteCallee(p *pgo.Profile, caller *ir.Func, call *ir.CallExpr) (*ir.Func, int64) {
	callerName := ir.LinkFuncName(caller)
	callerNode := p.WeightedCG.IRNodes[callerName]
	callOffset := pgo.NodeLineOffset(call, caller)

	inter, method := interfaceCallRecvTypeAndMethod(call)

	var hottest *pgo.IREdge

	// Returns true if e is hotter than hottest.
	//
	// Naively this is just e.Weight > hottest.Weight, but because OutEdges
	// has arbitrary iteration order, we need to apply additional sort
	// criteria when e.Weight == hottest.Weight to ensure we have stable
	// selection.
	hotter := func(e *pgo.IREdge) bool {
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

		ctyp := methodRecvType(e.Dst.AST)
		if ctyp == nil {
			// Not a method.
			// TODO(prattmic): Support non-interface indirect calls.
			if base.Debug.PGODebug >= 2 {
				fmt.Printf("%v: edge %s:%d -> %s (weight %d): callee not a method\n", ir.Line(call), callerName, callOffset, e.Dst.Name(), e.Weight)
			}
			continue
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
			continue
		}

		// If the method name is different it is most likely from a
		// different call on the same line
		if !strings.HasSuffix(e.Dst.Name(), "."+method.Name) {
			if base.Debug.PGODebug >= 2 {
				fmt.Printf("%v: edge %s:%d -> %s (weight %d): callee is a different method\n", ir.Line(call), callerName, callOffset, e.Dst.Name(), e.Weight)
			}
			continue
		}

		if base.Debug.PGODebug >= 2 {
			fmt.Printf("%v: edge %s:%d -> %s (weight %d): hottest so far\n", ir.Line(call), callerName, callOffset, e.Dst.Name(), e.Weight)
		}
		hottest = e
	}

	if hottest == nil {
		if base.Debug.PGODebug >= 2 {
			fmt.Printf("%v: call %s:%d: no hot callee\n", ir.Line(call), callerName, callOffset)
		}
		return nil, 0
	}

	if base.Debug.PGODebug >= 2 {
		fmt.Printf("%v call %s:%d: hottest callee %s (weight %d)\n", ir.Line(call), callerName, callOffset, hottest.Dst.Name(), hottest.Weight)
	}
	return hottest.Dst.AST, hottest.Weight
}
