// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package loopvar applies the proper variable capture, according
// to experiment, flags, language version, etc.
package loopvar

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"fmt"
)

type VarAndLoop struct {
	Name    *ir.Name
	Loop    ir.Node  // the *ir.RangeStmt or *ir.ForStmt. Used for identity and position
	LastPos src.XPos // the last position observed within Loop
}

// ForCapture transforms for and range loops that declare variables that might be
// captured by a closure or escaped to the heap, using a syntactic check that
// conservatively overestimates the loops where capture occurs, but still avoids
// transforming the (large) majority of loops. It returns the list of names
// subject to this change, that may (once transformed) be heap allocated in the
// process. (This allows checking after escape analysis to call out any such
// variables, in case it causes allocation/performance problems).
//
// The decision to transform loops is normally encoded in the For/Range loop node
// field DistinctVars but is also dependent on base.LoopVarHash, and some values
// of base.Debug.LoopVar (which is set per-package).  Decisions encoded in DistinctVars
// are preserved across inlining, so if package a calls b.F and loops in b.F are
// transformed, then they are always transformed, whether b.F is inlined or not.
//
// Per-package, the debug flag settings that affect this transformer:
//
// base.LoopVarHash != nil => use hash setting to govern transformation.
// note that LoopVarHash != nil sets base.Debug.LoopVar to 1 (unless it is >= 11, for testing/debugging).
//
// base.Debug.LoopVar == 11 => transform ALL loops ignoring syntactic/potential escape. Do not log, can be in addition to GOEXPERIMENT.
//
// The effect of GOEXPERIMENT=loopvar is to change the default value (0) of base.Debug.LoopVar to 1 for all packages.
func ForCapture(fn *ir.Func) []VarAndLoop {
	// if a loop variable is transformed it is appended to this slice for later logging
	var transformed []VarAndLoop

	describe := func(n *ir.Name) string {
		pos := n.Pos()
		inner := base.Ctxt.InnermostPos(pos)
		outer := base.Ctxt.OutermostPos(pos)
		if inner == outer {
			return fmt.Sprintf("loop variable %v now per-iteration", n)
		}
		return fmt.Sprintf("loop variable %v now per-iteration (loop inlined into %s:%d)", n, outer.Filename(), outer.Line())
	}

	forCapture := func() {
		seq := 1

		dclFixups := make(map[*ir.Name]ir.Stmt)

		// possibly leaked includes names of declared loop variables that may be leaked;
		// the mapped value is true if the name is *syntactically* leaked, and those loops
		// will be transformed.
		possiblyLeaked := make(map[*ir.Name]bool)

		// these enable an optimization of "escape" under return statements
		loopDepth := 0
		returnInLoopDepth := 0

		// noteMayLeak is called for candidate variables in for range/3-clause, and
		// adds them (mapped to false) to possiblyLeaked.
		noteMayLeak := func(x ir.Node) {
			if n, ok := x.(*ir.Name); ok {
				if n.Type().Kind() == types.TBLANK {
					return
				}
				// default is false (leak candidate, not yet known to leak), but flag can make all variables "leak"
				possiblyLeaked[n] = base.Debug.LoopVar >= 11
			}
		}

		// For reporting, keep track of the last position within any loop.
		// Loops nest, also need to be sensitive to inlining.
		var lastPos src.XPos

		updateLastPos := func(p src.XPos) {
			pl, ll := p.Line(), lastPos.Line()
			if p.SameFile(lastPos) &&
				(pl > ll || pl == ll && p.Col() > lastPos.Col()) {
				lastPos = p
			}
		}

		// maybeReplaceVar unshares an iteration variable for a range loop,
		// if that variable was actually (syntactically) leaked,
		// subject to hash-variable debugging.
		maybeReplaceVar := func(k ir.Node, x *ir.RangeStmt) ir.Node {
			if n, ok := k.(*ir.Name); ok && possiblyLeaked[n] {
				desc := func { describe(n) }
				if base.LoopVarHash.MatchPos(n.Pos(), desc) {
					// Rename the loop key, prefix body with assignment from loop key
					transformed = append(transformed, VarAndLoop{n, x, lastPos})
					tk := typecheck.TempAt(base.Pos, fn, n.Type())
					tk.SetTypecheck(1)
					as := ir.NewAssignStmt(x.Pos(), n, tk)
					as.Def = true
					as.SetTypecheck(1)
					x.Body.Prepend(as)
					dclFixups[n] = as
					return tk
				}
			}
			return k
		}

		// scanChildrenThenTransform processes node x to:
		//  1. if x is a for/range w/ DistinctVars, note declared iteration variables possiblyLeaked (PL)
		//  2. search all of x's children for syntactically escaping references to v in PL,
		//     meaning either address-of-v or v-captured-by-a-closure
		//  3. for all v in PL that had a syntactically escaping reference, transform the declaration
		//     and (in case of 3-clause loop) the loop to the unshared loop semantics.
		//  This is all much simpler for range loops; 3-clause loops can have an arbitrary number
		//  of iteration variables and the transformation is more involved, range loops have at most 2.
		var scanChildrenThenTransform func(x ir.Node) bool
		scanChildrenThenTransform = func { n ->

			if loopDepth > 0 {
				updateLastPos(n.Pos())
			}

			switch x := n.(type) {
			case *ir.ClosureExpr:
				if returnInLoopDepth >= loopDepth {
					// This expression is a child of a return, which escapes all loops above
					// the return, but not those between this expression and the return.
					break
				}
				for _, cv := range x.Func.ClosureVars {
					v := cv.Canonical()
					if _, ok := possiblyLeaked[v]; ok {
						possiblyLeaked[v] = true
					}
				}

			case *ir.AddrExpr:
				if returnInLoopDepth >= loopDepth {
					// This expression is a child of a return, which escapes all loops above
					// the return, but not those between this expression and the return.
					break
				}
				// Explicitly note address-taken so that return-statements can be excluded
				y := ir.OuterValue(x.X)
				if y.Op() != ir.ONAME {
					break
				}
				z, ok := y.(*ir.Name)
				if !ok {
					break
				}
				switch z.Class {
				case ir.PAUTO, ir.PPARAM, ir.PPARAMOUT, ir.PAUTOHEAP:
					if _, ok := possiblyLeaked[z]; ok {
						possiblyLeaked[z] = true
					}
				}

			case *ir.ReturnStmt:
				savedRILD := returnInLoopDepth
				returnInLoopDepth = loopDepth
				defer func() { returnInLoopDepth = savedRILD }()

			case *ir.RangeStmt:
				if !(x.Def && x.DistinctVars) {
					// range loop must define its iteration variables AND have distinctVars.
					x.DistinctVars = false
					break
				}
				noteMayLeak(x.Key)
				noteMayLeak(x.Value)
				loopDepth++
				savedLastPos := lastPos
				lastPos = x.Pos() // this sets the file.
				ir.DoChildren(n, scanChildrenThenTransform)
				loopDepth--
				x.Key = maybeReplaceVar(x.Key, x)
				x.Value = maybeReplaceVar(x.Value, x)
				thisLastPos := lastPos
				lastPos = savedLastPos
				updateLastPos(thisLastPos) // this will propagate lastPos if in the same file.
				x.DistinctVars = false
				return false

			case *ir.ForStmt:
				if !x.DistinctVars {
					break
				}
				forAllDefInInit(x, noteMayLeak)
				loopDepth++
				savedLastPos := lastPos
				lastPos = x.Pos() // this sets the file.
				ir.DoChildren(n, scanChildrenThenTransform)
				loopDepth--
				var leaked []*ir.Name
				// Collect the leaking variables for the much-more-complex transformation.
				forAllDefInInit(x, func { z ->
					if n, ok := z.(*ir.Name); ok && possiblyLeaked[n] {
						desc := func { describe(n) }
						// Hash on n.Pos() for most precise failure location.
						if base.LoopVarHash.MatchPos(n.Pos(), desc) {
							leaked = append(leaked, n)
						}
					}
				})

				if len(leaked) > 0 {
					// need to transform the for loop just so.

					/* Contrived example, w/ numbered comments from the transformation:
									BEFORE:
										var escape []*int
										for z := 0; z < n; z++ {
											if reason() {
												escape = append(escape, &z)
												continue
											}
											z = z + z
											stuff
										}
									AFTER:
										for z', tmp_first := 0, true; ; { // (4)
											                              // (5) body' follows:
											z := z'                       // (1)
											if tmp_first {tmp_first = false} else {z++} // (6)
											if ! (z < n) { break }        // (7)
											                              // (3, 8) body_continue
											if reason() {
					                            escape = append(escape, &z)
												goto next                 // rewritten continue
											}
											z = z + z
											stuff
										next:                             // (9)
											z' = z                       // (2)
										}

										In the case that the loop contains no increment (z++),
										there is no need for step 6,
										and thus no need to test, update, or declare tmp_first (part of step 4).
										Similarly if the loop contains no exit test (z < n),
										then there is no need for step 7.
					*/

					// Expressed in terms of the input ForStmt
					//
					// 	type ForStmt struct {
					// 	init     Nodes
					// 	Label    *types.Sym
					// 	Cond     Node  // empty if OFORUNTIL
					// 	Post     Node
					// 	Body     Nodes
					// 	HasBreak bool
					// }

					// OFOR: init; loop: if !Cond {break}; Body; Post; goto loop

					// (1) prebody = {z := z' for z in leaked}
					// (2) postbody = {z' = z for z in leaked}
					// (3) body_continue = {body : s/continue/goto next}
					// (4) init' = (init : s/z/z' for z in leaked) + tmp_first := true
					// (5) body' = prebody +        // appears out of order below
					// (6)         if tmp_first {tmp_first = false} else {Post} +
					// (7)         if !cond {break} +
					// (8)         body_continue (3) +
					// (9)         next: postbody (2)
					// (10) cond' = {}
					// (11) post' = {}

					// minor optimizations:
					//   if Post is empty, tmp_first and step 6 can be skipped.
					//   if Cond is empty, that code can also be skipped.

					var preBody, postBody ir.Nodes

					// Given original iteration variable z, what is the corresponding z'
					// that carries the value from iteration to iteration?
					zPrimeForZ := make(map[*ir.Name]*ir.Name)

					// (1,2) initialize preBody and postBody
					for _, z := range leaked {
						transformed = append(transformed, VarAndLoop{z, x, lastPos})

						tz := typecheck.TempAt(base.Pos, fn, z.Type())
						tz.SetTypecheck(1)
						zPrimeForZ[z] = tz

						as := ir.NewAssignStmt(x.Pos(), z, tz)
						as.Def = true
						as.SetTypecheck(1)
						preBody.Append(as)
						dclFixups[z] = as

						as = ir.NewAssignStmt(x.Pos(), tz, z)
						as.SetTypecheck(1)
						postBody.Append(as)

					}

					// (3) rewrite continues in body -- rewrite is inplace, so works for top level visit, too.
					label := typecheck.Lookup(fmt.Sprintf(".3clNext_%d", seq))
					seq++
					labelStmt := ir.NewLabelStmt(x.Pos(), label)
					labelStmt.SetTypecheck(1)

					loopLabel := x.Label
					loopDepth := 0
					var editContinues func(x ir.Node) bool
					editContinues = func { x ->

						switch c := x.(type) {
						case *ir.BranchStmt:
							// If this is a continue targeting the loop currently being rewritten, transform it to an appropriate GOTO
							if c.Op() == ir.OCONTINUE && (loopDepth == 0 && c.Label == nil || loopLabel != nil && c.Label == loopLabel) {
								c.Label = label
								c.SetOp(ir.OGOTO)
							}
						case *ir.RangeStmt, *ir.ForStmt:
							loopDepth++
							ir.DoChildren(x, editContinues)
							loopDepth--
							return false
						}
						ir.DoChildren(x, editContinues)
						return false
					}
					for _, y := range x.Body {
						editContinues(y)
					}
					bodyContinue := x.Body

					// (4) rewrite init
					forAllDefInInitUpdate(x, func { z, pz ->
						// note tempFor[n] can be nil if hash searching.
						if n, ok := z.(*ir.Name); ok && possiblyLeaked[n] && zPrimeForZ[n] != nil {
							*pz = zPrimeForZ[n]
						}
					})

					postNotNil := x.Post != nil
					var tmpFirstDcl ir.Node
					if postNotNil {
						// body' = prebody +
						// (6)     if tmp_first {tmp_first = false} else {Post} +
						//         if !cond {break} + ...
						tmpFirst := typecheck.TempAt(base.Pos, fn, types.Types[types.TBOOL])
						tmpFirstDcl = typecheck.Stmt(ir.NewAssignStmt(x.Pos(), tmpFirst, ir.NewBool(base.Pos, true)))
						tmpFirstSetFalse := typecheck.Stmt(ir.NewAssignStmt(x.Pos(), tmpFirst, ir.NewBool(base.Pos, false)))
						ifTmpFirst := ir.NewIfStmt(x.Pos(), tmpFirst, ir.Nodes{tmpFirstSetFalse}, ir.Nodes{x.Post})
						ifTmpFirst.PtrInit().Append(typecheck.Stmt(ir.NewDecl(base.Pos, ir.ODCL, tmpFirst))) // declares tmpFirst
						preBody.Append(typecheck.Stmt(ifTmpFirst))
					}

					// body' = prebody +
					//         if tmp_first {tmp_first = false} else {Post} +
					// (7)     if !cond {break} + ...
					if x.Cond != nil {
						notCond := ir.NewUnaryExpr(x.Cond.Pos(), ir.ONOT, x.Cond)
						notCond.SetType(x.Cond.Type())
						notCond.SetTypecheck(1)
						newBreak := ir.NewBranchStmt(x.Pos(), ir.OBREAK, nil)
						newBreak.SetTypecheck(1)
						ifNotCond := ir.NewIfStmt(x.Pos(), notCond, ir.Nodes{newBreak}, nil)
						ifNotCond.SetTypecheck(1)
						preBody.Append(ifNotCond)
					}

					if postNotNil {
						x.PtrInit().Append(tmpFirstDcl)
					}

					// (8)
					preBody.Append(bodyContinue...)
					// (9)
					preBody.Append(labelStmt)
					preBody.Append(postBody...)

					// (5) body' = prebody + ...
					x.Body = preBody

					// (10) cond' = {}
					x.Cond = nil

					// (11) post' = {}
					x.Post = nil
				}
				thisLastPos := lastPos
				lastPos = savedLastPos
				updateLastPos(thisLastPos) // this will propagate lastPos if in the same file.
				x.DistinctVars = false

				return false
			}

			ir.DoChildren(n, scanChildrenThenTransform)

			return false
		}
		scanChildrenThenTransform(fn)
		if len(transformed) > 0 {
			// editNodes scans a slice C of ir.Node, looking for declarations that
			// appear in dclFixups.  Any declaration D whose "fixup" is an assignmnt
			// statement A is removed from the C and relocated to the Init
			// of A.  editNodes returns the modified slice of ir.Node.
			editNodes := func(c ir.Nodes) ir.Nodes {
				j := 0
				for _, n := range c {
					if d, ok := n.(*ir.Decl); ok {
						if s := dclFixups[d.X]; s != nil {
							switch a := s.(type) {
							case *ir.AssignStmt:
								a.PtrInit().Prepend(d)
								delete(dclFixups, d.X) // can't be sure of visit order, wouldn't want to visit twice.
							default:
								base.Fatalf("not implemented yet for node type %v", s.Op())
							}
							continue // do not copy this node, and do not increment j
						}
					}
					c[j] = n
					j++
				}
				for k := j; k < len(c); k++ {
					c[k] = nil
				}
				return c[:j]
			}
			// fixup all tagged declarations in all the statements lists in fn.
			rewriteNodes(fn, editNodes)
		}
	}
	ir.WithFunc(fn, forCapture)
	return transformed
}

// forAllDefInInitUpdate applies "do" to all the defining assignments in the Init clause of a ForStmt.
// This abstracts away some of the boilerplate from the already complex and verbose for-3-clause case.
func forAllDefInInitUpdate(x *ir.ForStmt, do func(z ir.Node, update *ir.Node)) {
	for _, s := range x.Init() {
		switch y := s.(type) {
		case *ir.AssignListStmt:
			if !y.Def {
				continue
			}
			for i, z := range y.Lhs {
				do(z, &y.Lhs[i])
			}
		case *ir.AssignStmt:
			if !y.Def {
				continue
			}
			do(y.X, &y.X)
		}
	}
}

// forAllDefInInit is forAllDefInInitUpdate without the update option.
func forAllDefInInit(x *ir.ForStmt, do func(z ir.Node)) {
	forAllDefInInitUpdate(x, func { z, _ -> do(z) })
}

// rewriteNodes applies editNodes to all statement lists in fn.
func rewriteNodes(fn *ir.Func, editNodes func(c ir.Nodes) ir.Nodes) {
	var forNodes func(x ir.Node) bool
	forNodes = func { n ->
		if stmt, ok := n.(ir.InitNode); ok {
			// process init list
			stmt.SetInit(editNodes(stmt.Init()))
		}
		switch x := n.(type) {
		case *ir.Func:
			x.Body = editNodes(x.Body)
		case *ir.InlinedCallExpr:
			x.Body = editNodes(x.Body)

		case *ir.CaseClause:
			x.Body = editNodes(x.Body)
		case *ir.CommClause:
			x.Body = editNodes(x.Body)

		case *ir.BlockStmt:
			x.List = editNodes(x.List)

		case *ir.ForStmt:
			x.Body = editNodes(x.Body)
		case *ir.RangeStmt:
			x.Body = editNodes(x.Body)
		case *ir.IfStmt:
			x.Body = editNodes(x.Body)
			x.Else = editNodes(x.Else)
		case *ir.SelectStmt:
			x.Compiled = editNodes(x.Compiled)
		case *ir.SwitchStmt:
			x.Compiled = editNodes(x.Compiled)
		}
		ir.DoChildren(n, forNodes)
		return false
	}
	forNodes(fn)
}

func LogTransformations(transformed []VarAndLoop) {
	print := 2 <= base.Debug.LoopVar && base.Debug.LoopVar != 11

	if print || logopt.Enabled() { // 11 is do them all, quietly, 12 includes debugging.
		fileToPosBase := make(map[string]*src.PosBase) // used to remove inline context for innermost reporting.

		// trueInlinedPos rebases inner w/o inline context so that it prints correctly in WarnfAt; otherwise it prints as outer.
		trueInlinedPos := func(inner src.Pos) src.XPos {
			afn := inner.AbsFilename()
			pb, ok := fileToPosBase[afn]
			if !ok {
				pb = src.NewFileBase(inner.Filename(), afn)
				fileToPosBase[afn] = pb
			}
			inner.SetBase(pb)
			return base.Ctxt.PosTable.XPos(inner)
		}

		type unit struct{}
		loopsSeen := make(map[ir.Node]unit)
		type loopPos struct {
			loop  ir.Node
			last  src.XPos
			curfn *ir.Func
		}
		var loops []loopPos
		for _, lv := range transformed {
			n := lv.Name
			if _, ok := loopsSeen[lv.Loop]; !ok {
				l := lv.Loop
				loopsSeen[l] = unit{}
				loops = append(loops, loopPos{l, lv.LastPos, n.Curfn})
			}
			pos := n.Pos()

			inner := base.Ctxt.InnermostPos(pos)
			outer := base.Ctxt.OutermostPos(pos)

			if logopt.Enabled() {
				// For automated checking of coverage of this transformation, include this in the JSON information.
				var nString interface{} = n
				if inner != outer {
					nString = fmt.Sprintf("%v (from inline)", n)
				}
				if n.Esc() == ir.EscHeap {
					logopt.LogOpt(pos, "iteration-variable-to-heap", "loopvar", ir.FuncName(n.Curfn), nString)
				} else {
					logopt.LogOpt(pos, "iteration-variable-to-stack", "loopvar", ir.FuncName(n.Curfn), nString)
				}
			}
			if print {
				if inner == outer {
					if n.Esc() == ir.EscHeap {
						base.WarnfAt(pos, "loop variable %v now per-iteration, heap-allocated", n)
					} else {
						base.WarnfAt(pos, "loop variable %v now per-iteration, stack-allocated", n)
					}
				} else {
					innerXPos := trueInlinedPos(inner)
					if n.Esc() == ir.EscHeap {
						base.WarnfAt(innerXPos, "loop variable %v now per-iteration, heap-allocated (loop inlined into %s:%d)", n, outer.Filename(), outer.Line())
					} else {
						base.WarnfAt(innerXPos, "loop variable %v now per-iteration, stack-allocated (loop inlined into %s:%d)", n, outer.Filename(), outer.Line())
					}
				}
			}
		}
		for _, l := range loops {
			pos := l.loop.Pos()
			last := l.last
			loopKind := "range"
			if _, ok := l.loop.(*ir.ForStmt); ok {
				loopKind = "for"
			}
			if logopt.Enabled() {
				// Intended to help with performance debugging, we record whole loop ranges
				logopt.LogOptRange(pos, last, "loop-modified-"+loopKind, "loopvar", ir.FuncName(l.curfn))
			}
			if print && 4 <= base.Debug.LoopVar {
				// TODO decide if we want to keep this, or not.  It was helpful for validating logopt, otherwise, eh.
				inner := base.Ctxt.InnermostPos(pos)
				outer := base.Ctxt.OutermostPos(pos)

				if inner == outer {
					base.WarnfAt(pos, "%s loop ending at %d:%d was modified", loopKind, last.Line(), last.Col())
				} else {
					pos = trueInlinedPos(inner)
					last = trueInlinedPos(base.Ctxt.InnermostPos(last))
					base.WarnfAt(pos, "%s loop ending at %d:%d was modified (loop inlined into %s:%d)", loopKind, last.Line(), last.Col(), outer.Filename(), outer.Line())
				}
			}
		}
	}
}
