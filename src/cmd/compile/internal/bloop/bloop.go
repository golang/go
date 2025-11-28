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
	"cmd/compile/internal/reflectdata"
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

// keepAliveAt returns a statement that is either curNode, or a
// block containing curNode followed by a call to runtime.keepAlive for each
// node in ns. These calls ensure that nodes in ns will be live until
// after curNode's execution.
func keepAliveAt(ns []ir.Node, curNode ir.Node) ir.Node {
	if len(ns) == 0 {
		return curNode
	}

	pos := curNode.Pos()
	calls := []ir.Node{curNode}
	for _, n := range ns {
		if n == nil {
			continue
		}
		if n.Sym() == nil {
			continue
		}
		if n.Sym().IsBlank() {
			continue
		}
		arg := ir.NewConvExpr(pos, ir.OCONV, types.Types[types.TINTER], n)
		if !n.Type().IsInterface() {
			srcRType0 := reflectdata.TypePtrAt(pos, n.Type())
			arg.TypeWord = srcRType0
			arg.SrcRType = srcRType0
		}
		callExpr := typecheck.Call(pos,
			typecheck.LookupRuntime("KeepAlive"),
			[]ir.Node{arg}, false).(*ir.CallExpr)
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

// preserveStmt transforms stmt so that any names defined/assigned within it
// are used after stmt's execution, preventing their dead code elimination
// and dead store elimination. The return value is the transformed statement.
func preserveStmt(curFn *ir.Func, stmt ir.Node) (ret ir.Node) {
	ret = stmt
	switch n := stmt.(type) {
	case *ir.AssignStmt:
		// Peel down struct and slice indexing to get the names
		name := getNameFromNode(n.X)
		if name != nil {
			debugName(name, n.Pos())
			ret = keepAliveAt([]ir.Node{name}, n)
		} else if deref := n.X.(*ir.StarExpr); deref != nil {
			ret = keepAliveAt([]ir.Node{deref}, n)
			if base.Flag.LowerM > 1 {
				base.WarnfAt(n.Pos(), "dereference will be kept alive")
			}
		} else if base.Flag.LowerM > 1 {
			base.WarnfAt(n.Pos(), "expr is unknown to bloop pass")
		}
	case *ir.AssignListStmt:
		ns := []ir.Node{}
		for _, lhs := range n.Lhs {
			name := getNameFromNode(lhs)
			if name != nil {
				debugName(name, n.Pos())
				ns = append(ns, name)
			} else if deref := lhs.(*ir.StarExpr); deref != nil {
				ns = append(ns, deref)
				if base.Flag.LowerM > 1 {
					base.WarnfAt(n.Pos(), "dereference will be kept alive")
				}
			} else if base.Flag.LowerM > 1 {
				base.WarnfAt(n.Pos(), "expr is unknown to bloop pass")
			}
		}
		ret = keepAliveAt(ns, n)
	case *ir.AssignOpStmt:
		name := getNameFromNode(n.X)
		if name != nil {
			debugName(name, n.Pos())
			ret = keepAliveAt([]ir.Node{name}, n)
		} else if deref := n.X.(*ir.StarExpr); deref != nil {
			ret = keepAliveAt([]ir.Node{deref}, n)
			if base.Flag.LowerM > 1 {
				base.WarnfAt(n.Pos(), "dereference will be kept alive")
			}
		} else if base.Flag.LowerM > 1 {
			base.WarnfAt(n.Pos(), "expr is unknown to bloop pass")
		}
	case *ir.CallExpr:
		curNode := stmt
		if n.Fun != nil && n.Fun.Type() != nil && n.Fun.Type().NumResults() != 0 {
			ns := []ir.Node{}
			// This function's results are not assigned, assign them to
			// auto tmps and then keepAliveAt these autos.
			// Note: markStmt assumes the context that it's called - this CallExpr is
			// not within another OAS2, which is guaranteed by the case above.
			results := n.Fun.Type().Results()
			lhs := make([]ir.Node, len(results))
			for i, res := range results {
				tmp := typecheck.TempAt(n.Pos(), curFn, res.Type)
				lhs[i] = tmp
				ns = append(ns, tmp)
			}

			// Create an assignment statement.
			assign := typecheck.AssignExpr(
				ir.NewAssignListStmt(n.Pos(), ir.OAS2, lhs,
					[]ir.Node{n})).(*ir.AssignListStmt)
			assign.Def = true
			curNode = assign
			plural := ""
			if len(results) > 1 {
				plural = "s"
			}
			if base.Flag.LowerM > 1 {
				base.WarnfAt(n.Pos(), "function result%s will be kept alive", plural)
			}
			ret = keepAliveAt(ns, curNode)
		} else {
			// This function probably doesn't return anything, keep its args alive.
			argTmps := []ir.Node{}
			names := []ir.Node{}
			for i, a := range n.Args {
				if name := getNameFromNode(a); name != nil {
					// If they are name, keep them alive directly.
					debugName(name, n.Pos())
					names = append(names, name)
				} else if a.Op() == ir.OSLICELIT {
					// variadic args are encoded as slice literal.
					s := a.(*ir.CompLitExpr)
					ns := []ir.Node{}
					for i, elem := range s.List {
						if name := getNameFromNode(elem); name != nil {
							debugName(name, n.Pos())
							ns = append(ns, name)
						} else {
							// We need a temporary to save this arg.
							tmp := typecheck.TempAt(elem.Pos(), curFn, elem.Type())
							argTmps = append(argTmps, typecheck.AssignExpr(ir.NewAssignStmt(elem.Pos(), tmp, elem)))
							names = append(names, tmp)
							s.List[i] = tmp
							if base.Flag.LowerM > 1 {
								base.WarnfAt(n.Pos(), "function arg will be kept alive")
							}
						}
					}
					names = append(names, ns...)
				} else {
					// expressions, we need to assign them to temps and change the original arg to reference
					// them.
					tmp := typecheck.TempAt(n.Pos(), curFn, a.Type())
					argTmps = append(argTmps, typecheck.AssignExpr(ir.NewAssignStmt(n.Pos(), tmp, a)))
					names = append(names, tmp)
					n.Args[i] = tmp
					if base.Flag.LowerM > 1 {
						base.WarnfAt(n.Pos(), "function arg will be kept alive")
					}
				}
			}
			if len(argTmps) > 0 {
				argTmps = append(argTmps, n)
				curNode = ir.NewBlockStmt(n.Pos(), argTmps)
			}
			ret = keepAliveAt(names, curNode)
		}
	}
	return
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

// BloopWalk performs a walk on all functions in the package
// if it imports testing and wrap the results of all qualified
// statements in a runtime.KeepAlive intrinsic call. See package
// doc for more details.
//
//	for b.Loop() {...}
//
// loop's body.
func BloopWalk(pkg *ir.Package) {
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
	}
}
