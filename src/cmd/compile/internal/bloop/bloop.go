// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bloop

// This file contains support routines for keeping
// statements alive
// in such loops (example):
//
//	for b.Loop() {
//		var a, b int
//		a = 5
//		b = 6
//		f(a, b)
//	}
//
// The results of a, b and f(a, b) will be kept alive.
//
// Formally, the lhs (if they are [ir.Name]-s) of
// [ir.AssignStmt], [ir.AssignListStmt],
// [ir.AssignOpStmt], and the results of [ir.CallExpr]
// or its args if it doesn't return a value will be kept
// alive.
//
// The keep alive logic is implemented with as wrapping a
// runtime.KeepAlive around the Name.
//
// TODO: currently this is implemented with KeepAlive
// because it will prevent DSE and DCE which is probably
// what we want right now. And KeepAlive takes an ssa
// value instead of a symbol, which is easier to manage.
// But since KeepAlive's context was mainly in the runtime
// and GC, should we implement a new intrinsic that lowers
// to OpVarLive? Peeling out the symbols is a bit tricky
// and also VarLive seems to assume that there exists a
// VarDef on the same symbol that dominates it.

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

// getNameFromNode tries to iteratively peel down the node to
// get the name.
func getNameFromNode(n ir.Node) *ir.Name {
	// Tries to iteratively peel down the node to get the names.
	for n != nil {
		switch n.Op() {
		case ir.ONAME:
			// Found the name, stop the loop.
			return n.(*ir.Name)
		case ir.OSLICE, ir.OSLICE3:
			n = n.(*ir.SliceExpr).X
		case ir.ODOT:
			n = n.(*ir.SelectorExpr).X
		case ir.OCONV, ir.OCONVIFACE, ir.OCONVNOP:
			n = n.(*ir.ConvExpr).X
		case ir.OADDR:
			n = n.(*ir.AddrExpr).X
		case ir.ODOTPTR:
			n = n.(*ir.SelectorExpr).X
		case ir.OINDEX, ir.OINDEXMAP:
			n = n.(*ir.IndexExpr).X
		default:
			n = nil
		}
	}
	return nil
}

// getAddressableNameFromNode is like getNameFromNode but returns nil if the node is not addressable.
func getAddressableNameFromNode(n ir.Node) *ir.Name {
	if name := getNameFromNode(n); name != nil && ir.IsAddressable(name) {
		return name
	}
	return nil
}

// getKeepAliveNodes analyzes an IR node and returns a list of nodes that must be kept alive.
func getKeepAliveNodes(pos src.XPos, n ir.Node) ir.Nodes {
	name := getAddressableNameFromNode(n)
	if name != nil {
		debugName(name, pos)
		return ir.Nodes{name}
	} else if deref := n.(*ir.StarExpr); deref != nil {
		if base.Flag.LowerM > 1 {
			base.WarnfAt(pos, "dereference will be kept alive")
		}
		return ir.Nodes{deref}
	} else if base.Flag.LowerM > 1 {
		base.WarnfAt(pos, "expr is unknown to bloop pass")
	}
	return nil
}

// keepAliveAt returns a statement that is either curNode, or a
// block containing curNode followed by a call to runtime.KeepAlive for each
// node in ns. These calls ensure that nodes in ns will be live until
// after curNode's execution.
func keepAliveAt(ns ir.Nodes, curNode ir.Node) ir.Node {
	if len(ns) == 0 {
		return curNode
	}

	pos := curNode.Pos()
	calls := ir.Nodes{curNode}
	for _, n := range ns {
		if n == nil || n.Sym() == nil || n.Sym().IsBlank() {
			continue
		}
		if !ir.IsAddressable(n) {
			base.FatalfAt(n.Pos(), "keepAliveAt: node %v is not addressable", n)
		}
		arg := ir.NewConvExpr(pos, ir.OCONV, types.Types[types.TUNSAFEPTR], typecheck.NodAddr(n))
		callExpr := typecheck.Call(pos, typecheck.LookupRuntime("KeepAlive"), ir.Nodes{arg}, false).(*ir.CallExpr)
		callExpr.IsCompilerVarLive = true
		callExpr.NoInline = true
		calls = append(calls, callExpr)
	}

	return ir.NewBlockStmt(pos, calls)
}

func debugName(name *ir.Name, pos src.XPos) {
	if base.Flag.LowerM > 1 {
		if name.Linksym() != nil {
			base.WarnfAt(pos, "%s will be kept alive", name.Linksym().Name)
		} else {
			base.WarnfAt(pos, "expr will be kept alive")
		}
	}
}

// preserveCallResults assigns the results of a call statement to temporary variables to ensure they remain alive.
func preserveCallResults(curFn *ir.Func, call *ir.CallExpr) ir.Node {
	var ns ir.Nodes
	lhs := make(ir.Nodes, call.Fun.Type().NumResults())
	for i, res := range call.Fun.Type().Results() {
		tmp := typecheck.TempAt(call.Pos(), curFn, res.Type)
		lhs[i] = tmp
		ns = append(ns, tmp)
	}

	if base.Flag.LowerM > 1 {
		plural := ""
		if call.Fun.Type().NumResults() > 1 {
			plural = "s"
		}
		base.WarnfAt(call.Pos(), "function result%s will be kept alive", plural)
	}

	assign := typecheck.AssignExpr(ir.NewAssignListStmt(call.Pos(), ir.OAS2, lhs, ir.Nodes{call})).(*ir.AssignListStmt)
	assign.Def = true
	for _, tmp := range lhs {
		// Place temp declarations in the loop body to help escape analysis.
		assign.PtrInit().Append(typecheck.Stmt(ir.NewDecl(assign.Pos(), ir.ODCL, tmp.(*ir.Name))))
	}
	return keepAliveAt(ns, assign)
}

// preserveCallArgs ensures the arguments of a call statement are kept alive by transforming them into temporaries if necessary.
func preserveCallArgs(curFn *ir.Func, call *ir.CallExpr) ir.Node {
	var argTmps ir.Nodes
	var names ir.Nodes
	preserveTmp := func(pos src.XPos, n ir.Node) ir.Node {
		tmp := typecheck.TempAt(pos, curFn, n.Type())
		assign := ir.NewAssignStmt(pos, tmp, n)
		assign.Def = true
		// Place temp declarations in the loop body to help escape analysis.
		assign.PtrInit().Append(typecheck.Stmt(ir.NewDecl(assign.Pos(), ir.ODCL, tmp)))
		argTmps = append(argTmps, typecheck.AssignExpr(assign))
		names = append(names, tmp)
		if base.Flag.LowerM > 1 {
			base.WarnfAt(call.Pos(), "function arg will be kept alive")
		}
		return tmp
	}
	for i, a := range call.Args {
		if name := getAddressableNameFromNode(a); name != nil {
			// If they are name, keep them alive directly.
			debugName(name, call.Pos())
			names = append(names, name)
		} else if a.Op() == ir.OSLICELIT {
			// variadic args are encoded as slice literal.
			s := a.(*ir.CompLitExpr)
			var ns ir.Nodes
			for i, elem := range s.List {
				if name := getAddressableNameFromNode(elem); name != nil {
					debugName(name, call.Pos())
					ns = append(ns, name)
				} else {
					// We need a temporary to save this arg.
					s.List[i] = preserveTmp(elem.Pos(), elem)
				}
			}
			names = append(names, ns...)
		} else {
			// expressions, we need to assign them to temps and change the original arg to reference them.
			call.Args[i] = preserveTmp(call.Pos(), a)
		}
	}
	if len(argTmps) > 0 {
		argTmps = append(argTmps, call)
		return keepAliveAt(names, ir.NewBlockStmt(call.Pos(), argTmps))
	}
	return keepAliveAt(names, call)
}

// preserveStmt transforms stmt so that any names defined/assigned within it
// are used after stmt's execution, preventing their dead code elimination
// and dead store elimination. The return value is the transformed statement.
func preserveStmt(curFn *ir.Func, stmt ir.Node) ir.Node {
	switch n := stmt.(type) {
	case *ir.AssignStmt:
		return keepAliveAt(getKeepAliveNodes(n.Pos(), n.X), n)
	case *ir.AssignListStmt:
		var ns ir.Nodes
		for _, lhs := range n.Lhs {
			ns = append(ns, getKeepAliveNodes(n.Pos(), lhs)...)
		}
		return keepAliveAt(ns, n)
	case *ir.AssignOpStmt:
		return keepAliveAt(getKeepAliveNodes(n.Pos(), n.X), n)
	case *ir.CallExpr:
		// The function's results are not assigned, preserve them.
		if n.Fun != nil && n.Fun.Type() != nil && n.Fun.Type().NumResults() != 0 {
			return preserveCallResults(curFn, n)
		}
		// This function doesn't return anything, keep its args alive.
		return preserveCallArgs(curFn, n)
	}
	return stmt
}

func preserveStmts(curFn *ir.Func, list ir.Nodes) {
	for i := range list {
		list[i] = preserveStmt(curFn, list[i])
	}
}

// isTestingBLoop returns true if it matches the node as a
// testing.(*B).Loop. See issue #61515.
func isTestingBLoop(t ir.Node) bool {
	if t.Op() != ir.OFOR {
		return false
	}
	nFor, ok := t.(*ir.ForStmt)
	if !ok || nFor.Cond == nil || nFor.Cond.Op() != ir.OCALLFUNC {
		return false
	}
	n, ok := nFor.Cond.(*ir.CallExpr)
	if !ok || n.Fun == nil || n.Fun.Op() != ir.OMETHEXPR {
		return false
	}
	name := ir.MethodExprName(n.Fun)
	if name == nil {
		return false
	}
	if fSym := name.Sym(); fSym != nil && name.Class == ir.PFUNC && fSym.Pkg != nil &&
		fSym.Name == "(*B).Loop" && fSym.Pkg.Path == "testing" {
		// Attempting to match a function call to testing.(*B).Loop
		return true
	}
	return false
}

type editor struct {
	inBloop bool
	curFn   *ir.Func
}

func (e editor) edit(n ir.Node) ir.Node {
	e.inBloop = isTestingBLoop(n) || e.inBloop
	// It's in bloop, mark the stmts with bodies.
	ir.EditChildren(n, e.edit)
	if e.inBloop {
		switch n := n.(type) {
		case *ir.ForStmt:
			preserveStmts(e.curFn, n.Body)
		case *ir.IfStmt:
			preserveStmts(e.curFn, n.Body)
			preserveStmts(e.curFn, n.Else)
		case *ir.BlockStmt:
			preserveStmts(e.curFn, n.List)
		case *ir.CaseClause:
			preserveStmts(e.curFn, n.List)
			preserveStmts(e.curFn, n.Body)
		case *ir.CommClause:
			preserveStmts(e.curFn, n.Body)
		case *ir.RangeStmt:
			preserveStmts(e.curFn, n.Body)
		}
	}
	return n
}

// Walk performs a walk on all functions in the package
// if it imports testing and wrap the results of all qualified
// statements in a runtime.KeepAlive intrinsic call. See package
// doc for more details.
//
//	for b.Loop() {...}
//
// loop's body.
func Walk(pkg *ir.Package) {
	hasTesting := false
	for _, i := range pkg.Imports {
		if i.Path == "testing" {
			hasTesting = true
			break
		}
	}
	if !hasTesting {
		return
	}
	for _, fn := range pkg.Funcs {
		e := editor{false, fn}
		ir.EditChildren(fn, e.edit)
		if ir.MatchAstDump(fn, "bloop") {
			ir.AstDump(fn, "bloop, "+ir.FuncName(fn))
		}
	}

}
