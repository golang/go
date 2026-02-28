// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/internal/src"
	"fmt"
	"maps"
	"path/filepath"
)

type aliasAnalysis struct {
	// fn is the function being analyzed.
	fn *ir.Func

	// candidateSlices are declared slices that
	// start unaliased and might still be unaliased.
	candidateSlices map[*ir.Name]candidateSlice

	// noAliasAppends are appends that have been
	// proven to use an unaliased slice.
	noAliasAppends []*ir.CallExpr

	// loops is a stack of observed loops,
	// each with a list of candidate appends.
	loops [][]candidateAppend

	// State for optional validation checking (doubleCheck mode):
	processed   map[ir.Node]int // count of times each node was processed, for doubleCheck mode
	doubleCheck bool            // whether to do doubleCheck mode
}

// candidateSlice tracks information about a declared slice
// that might be unaliased.
type candidateSlice struct {
	loopDepth int // depth of loop when slice was declared
}

// candidateAppend tracks information about an OAPPEND that
// might be using an unaliased slice.
type candidateAppend struct {
	s    *ir.Name     // the slice argument in 's = append(s, ...)'
	call *ir.CallExpr // the append call
}

// aliasAnalysis looks for specific patterns of slice usage and proves
// that certain appends are operating on non-aliased slices.
//
// This allows us to emit calls to free the backing arrays for certain
// non-aliased slices at runtime when we know the memory is logically dead.
//
// The analysis is conservative, giving up on any operation we do not
// explicitly understand.
func (aa *aliasAnalysis) analyze(fn *ir.Func) {
	// Walk the function body to discover slice declarations, their uses,
	// and any append that we can prove is using an unaliased slice.
	//
	// An example is:
	//
	//   var s []T
	//   for _, v := range input {
	//      f()
	//      s = append(s, g(v))	   // s cannot be aliased here
	//      h()
	//   }
	//   return s
	//
	// Here, we can prove that the append to s is operating on an unaliased slice,
	// and that conclusion is unaffected by s later being returned and escaping.
	//
	// In contrast, in this example, the aliasing of s in the loop body means the
	// append can be operating on an aliased slice, so we do not record s as unaliased:
	//
	//   var s []T
	//   var alias []T
	//   for _, v := range input {
	//      s = append(s, v)	   // s is aliased on second pass through loop body
	//      alias = s
	//   }
	//
	// Arbitrary uses of s after an append do not affect the aliasing conclusion
	// for that append, but only if the append cannot be revisited at execution time
	// via a loop or goto.
	//
	// We track the loop depth when a slice was declared and verify all uses of a slice
	// are non-aliasing until we return to that depth. In other words, we make sure
	// we have processed any possible execution-time revisiting of the slice prior
	// to making our final determination.
	//
	// This approach helps for example with nested loops, such as:
	//
	//   var s []int
	//   for range 10 {
	//       for range 10 {
	//           s = append(s, 0)  // s is proven as non-aliased here
	//       }
	//   }
	//   alias = s                 // both loops are complete
	//
	// Or in contrast:
	//
	//   var s []int
	//   for range 10 {
	//       for range 10 {
	//           s = append(s, 0)  // s is treated as aliased here
	//       }
	//       alias = s             // aliased, and outermost loop cycles back
	//   }
	//
	// As we walk the function, we look for things like:
	//
	// 1. Slice declarations (currently supporting 'var s []T', 's := make([]T, ...)',
	//    and 's := []T{...}').
	// 2. Appends to a slice of the form 's = append(s, ...)'.
	// 3. Other uses of the slice, which we treat as potential aliasing outside
	//    of a few known safe cases.
	// 4. A start of a loop, which we track in a stack so that
	//    any uses of a slice within a loop body are treated as potential
	//    aliasing, including statements in the loop body after an append.
	//    Candidate appends are stored in the loop stack at the loop depth of their
	//    corresponding slice declaration (rather than the loop depth of the append),
	//    which essentially postpones a decision about the candidate append.
	// 5. An end of a loop, which pops the loop stack and allows us to
	//    conclusively treat candidate appends from the loop body based
	//    on the loop depth of the slice declaration.
	//
	// Note that as we pop a candidate append at the end of a loop, we know
	// its corresponding slice was unaliased throughout the loop being popped
	// if the slice is still in the candidate slice map (without having been
	// removed for potential aliasing), and we know we can make a final decision
	// about a candidate append if we have returned to the loop depth
	// where its slice was declared. In other words, there is no unanalyzed
	// control flow that could take us back at execution-time to the
	// candidate append in the now analyzed loop. This helps for example
	// with nested loops, such as in our examples just above.
	//
	// We give up on a particular candidate slice if we see any use of it
	// that we don't explicitly understand, and we give up on all of
	// our candidate slices if we see any goto or label, which could be
	// unstructured control flow. (TODO(thepudds): we remove the goto/label
	// restriction in a subsequent CL.)
	//
	// Note that the intended use is to indicate that a slice is safe to pass
	// to runtime.freegc, which currently requires that the passed pointer
	// point to the base of its heap object.
	//
	// Therefore, we currently do not allow any re-slicing of the slice, though we could
	// potentially allow s[0:x] or s[:x] or similar. (Slice expressions that alter
	// the capacity might be possible to allow with freegc changes, though they are
	// currently disallowed here like all slice expressions).
	//
	// TODO(thepudds): we could support the slice being used as non-escaping function call parameter
	// but to do that, we need to verify any creation of specials via user code triggers an escape,
	// or mail better runtime.freegc support for specials, or have a temporary compile-time solution
	// for specials. (Currently, this analysis side-steps specials because any use of a slice
	// that might cause a user-created special will cause it to be treated as aliased, and
	// separately, runtime.freegc handles profiling-related specials).

	// Initialize.
	aa.fn = fn
	aa.candidateSlices = make(map[*ir.Name]candidateSlice) // slices that might be unaliased

	// doubleCheck controls whether we do a sanity check of our processing logic
	// by counting each node visited in our main pass, and then comparing those counts
	// against a simple walk at the end. The main intent is to help catch missing
	// any nodes squirreled away in some spot we forgot to examine in our main pass.
	aa.doubleCheck = base.Debug.EscapeAliasCheck > 0
	aa.processed = make(map[ir.Node]int)

	if base.Debug.EscapeAlias >= 2 {
		aa.diag(fn.Pos(), fn, "====== starting func", "======")
	}

	ir.DoChildren(fn, aa.visit)

	for _, call := range aa.noAliasAppends {
		if base.Debug.EscapeAlias >= 1 {
			base.WarnfAt(call.Pos(), "alias analysis: append using non-aliased slice: %v in func %v",
				call, fn)
		}
		if base.Debug.FreeAppend > 0 {
			call.AppendNoAlias = true
		}
	}

	if aa.doubleCheck {
		doubleCheckProcessed(fn, aa.processed)
	}
}

func (aa *aliasAnalysis) visit(n ir.Node) bool {
	if n == nil {
		return false
	}

	if base.Debug.EscapeAlias >= 3 {
		fmt.Printf("%-25s alias analysis: visiting node: %12s  %-18T  %v\n",
			fmtPosShort(n.Pos())+":", n.Op().String(), n, n)
	}

	// As we visit nodes, we want to ensure we handle all children
	// without missing any (through ignorance or future changes).
	// We do this by counting nodes as we visit them or otherwise
	// declare a node to be fully processed.
	//
	// In particular, we want to ensure we don't miss the use
	// of a slice in some expression that might be an aliasing usage.
	//
	// When doubleCheck is enabled, we compare the counts
	// accumulated in our analysis against counts from a trivial walk,
	// failing if there is any mismatch.
	//
	// This call here counts that we have visited this node n
	// via our main visit method. (In contrast, some nodes won't
	// be visited by the main visit method, but instead will be
	// manually marked via countProcessed when we believe we have fully
	// dealt with the node).
	aa.countProcessed(n)

	switch n.Op() {
	case ir.ODCL:
		decl := n.(*ir.Decl)

		if decl.X != nil && decl.X.Type().IsSlice() && decl.X.Class == ir.PAUTO {
			s := decl.X
			if _, ok := aa.candidateSlices[s]; ok {
				base.FatalfAt(n.Pos(), "candidate slice already tracked as candidate: %v", s)
			}
			if base.Debug.EscapeAlias >= 2 {
				aa.diag(n.Pos(), s, "adding candidate slice", "(loop depth: %d)", len(aa.loops))
			}
			aa.candidateSlices[s] = candidateSlice{loopDepth: len(aa.loops)}
		}
		// No children aside from the declared ONAME.
		aa.countProcessed(decl.X)
		return false

	case ir.ONAME:

		// We are seeing a name we have not already handled in another case,
		// so remove any corresponding candidate slice.
		if n.Type().IsSlice() {
			name := n.(*ir.Name)
			_, ok := aa.candidateSlices[name]
			if ok {
				delete(aa.candidateSlices, name)
				if base.Debug.EscapeAlias >= 2 {
					aa.diag(n.Pos(), name, "removing candidate slice", "")
				}
			}
		}
		// No children.
		return false

	case ir.OAS2:
		n := n.(*ir.AssignListStmt)
		aa.analyzeAssign(n, n.Lhs, n.Rhs)
		return false

	case ir.OAS:
		assign := n.(*ir.AssignStmt)
		aa.analyzeAssign(n, []ir.Node{assign.X}, []ir.Node{assign.Y})
		return false

	case ir.OFOR, ir.ORANGE:
		aa.visitList(n.Init())

		if n.Op() == ir.ORANGE {
			// TODO(thepudds): previously we visited this range expression
			// in the switch just below, after pushing the loop. This current placement
			// is more correct, but generate a test or find an example in stdlib or similar
			// where it matters. (Our current tests do not complain.)
			aa.visit(n.(*ir.RangeStmt).X)
		}

		// Push a new loop.
		aa.loops = append(aa.loops, nil)

		// Process the loop.
		switch n.Op() {
		case ir.OFOR:
			forstmt := n.(*ir.ForStmt)
			aa.visit(forstmt.Cond)
			aa.visitList(forstmt.Body)
			aa.visit(forstmt.Post)
		case ir.ORANGE:
			rangestmt := n.(*ir.RangeStmt)
			aa.visit(rangestmt.Key)
			aa.visit(rangestmt.Value)
			aa.visitList(rangestmt.Body)
		default:
			base.Fatalf("loop not OFOR or ORANGE: %v", n)
		}

		// Pop the loop.
		var candidateAppends []candidateAppend
		candidateAppends, aa.loops = aa.loops[len(aa.loops)-1], aa.loops[:len(aa.loops)-1]
		for _, a := range candidateAppends {
			// We are done with the loop, so we can validate any candidate appends
			// that have not had their slice removed yet. We know a slice is unaliased
			// throughout the loop if the slice is still in the candidate slice map.
			if cs, ok := aa.candidateSlices[a.s]; ok {
				if cs.loopDepth == len(aa.loops) {
					// We've returned to the loop depth where the slice was declared and
					// hence made it all the way through any loops that started after
					// that declaration.
					if base.Debug.EscapeAlias >= 2 {
						aa.diag(n.Pos(), a.s, "proved non-aliased append",
							"(completed loop, decl at depth: %d)", cs.loopDepth)
					}
					aa.noAliasAppends = append(aa.noAliasAppends, a.call)
				} else if cs.loopDepth < len(aa.loops) {
					if base.Debug.EscapeAlias >= 2 {
						aa.diag(n.Pos(), a.s, "cannot prove non-aliased append",
							"(completed loop, decl at depth: %d)", cs.loopDepth)
					}
				} else {
					panic("impossible: candidate slice loopDepth > current loop depth")
				}
			}
		}
		return false

	case ir.OLEN, ir.OCAP:
		n := n.(*ir.UnaryExpr)
		if n.X.Op() == ir.ONAME {
			// This does not disqualify a candidate slice.
			aa.visitList(n.Init())
			aa.countProcessed(n.X)
		} else {
			ir.DoChildren(n, aa.visit)
		}
		return false

	case ir.OCLOSURE:
		// Give up on all our in-progress slices.
		closure := n.(*ir.ClosureExpr)
		if base.Debug.EscapeAlias >= 2 {
			aa.diag(n.Pos(), closure.Func, "clearing all in-progress slices due to OCLOSURE",
				"(was %d in-progress slices)", len(aa.candidateSlices))
		}
		clear(aa.candidateSlices)
		return ir.DoChildren(n, aa.visit)

	case ir.OLABEL, ir.OGOTO:
		// Give up on all our in-progress slices.
		if base.Debug.EscapeAlias >= 2 {
			aa.diag(n.Pos(), n, "clearing all in-progress slices due to label or goto",
				"(was %d in-progress slices)", len(aa.candidateSlices))
		}
		clear(aa.candidateSlices)
		return false

	default:
		return ir.DoChildren(n, aa.visit)
	}
}

func (aa *aliasAnalysis) visitList(nodes []ir.Node) {
	for _, n := range nodes {
		aa.visit(n)
	}
}

// analyzeAssign evaluates the assignment dsts... = srcs...
//
// assign is an *ir.AssignStmt or *ir.AssignListStmt.
func (aa *aliasAnalysis) analyzeAssign(assign ir.Node, dsts, srcs []ir.Node) {
	aa.visitList(assign.Init())
	for i := range dsts {
		dst := dsts[i]
		src := srcs[i]

		if dst.Op() != ir.ONAME || !dst.Type().IsSlice() {
			// Nothing for us to do aside from visiting the remaining children.
			aa.visit(dst)
			aa.visit(src)
			continue
		}

		// We have a slice being assigned to an ONAME.

		// Check for simple zero value assignments to an ONAME, which we ignore.
		if src == nil {
			aa.countProcessed(dst)
			continue
		}

		if base.Debug.EscapeAlias >= 4 {
			srcfn := ""
			if src.Op() == ir.ONAME {
				srcfn = fmt.Sprintf("%v.", src.Name().Curfn)
			}
			aa.diag(assign.Pos(), assign, "visiting slice assignment", "%v.%v = %s%v (%s %T = %s %T)",
				dst.Name().Curfn, dst, srcfn, src, dst.Op().String(), dst, src.Op().String(), src)
		}

		// Now check what we have on the RHS.
		switch src.Op() {
		// Cases:

		// Check for s := make([]T, ...) or s := []T{...}, along with the '=' version
		// of those which does not alias s as long as s is not used in the make.
		//
		// TODO(thepudds): we need to be sure that 's := []T{1,2,3}' does not end up backed by a
		// global static. Ad-hoc testing indicates that example and similar seem to be
		// stack allocated, but that was not exhaustive testing. We do have runtime.freegc
		// able to throw if it finds a global static, but should test more.
		//
		// TODO(thepudds): could also possibly allow 's := append([]T(nil), ...)'
		// and 's := append([]T{}, ...)'.
		case ir.OMAKESLICE, ir.OSLICELIT:
			name := dst.(*ir.Name)
			if name.Class == ir.PAUTO {
				if base.Debug.EscapeAlias > 1 {
					aa.diag(assign.Pos(), assign, "assignment from make or slice literal", "")
				}
				// If this is Def=true, the ODCL in the init will causes this to be tracked
				// as a candidate slice. We walk the init and RHS but avoid visiting the name
				// in the LHS, which would remove the slice from the candidate list after it
				// was just added.
				aa.visit(src)
				aa.countProcessed(name)
				continue
			}

		// Check for s = append(s, <...>).
		case ir.OAPPEND:
			s := dst.(*ir.Name)
			call := src.(*ir.CallExpr)
			if call.Args[0] == s {
				// Matches s = append(s, <...>).
				// First visit other arguments in case they use s.
				aa.visitList(call.Args[1:])
				// Mark the call as processed, and s twice.
				aa.countProcessed(s, call, s)

				// We have now examined all non-ONAME children of assign.

				// This is now the heart of the analysis.
				// Check to see if this slice is a live candidate.
				cs, ok := aa.candidateSlices[s]
				if ok {
					if cs.loopDepth == len(aa.loops) {
						// No new loop has started after the declaration of s,
						// so this is definitive.
						if base.Debug.EscapeAlias >= 2 {
							aa.diag(assign.Pos(), assign, "proved non-aliased append",
								"(loop depth: %d, equals decl depth)", len(aa.loops))
						}
						aa.noAliasAppends = append(aa.noAliasAppends, call)
					} else if cs.loopDepth < len(aa.loops) {
						// A new loop has started since the declaration of s,
						// so we can't validate this append yet, but
						// remember it in case we can validate it later when
						// all loops using s are done.
						aa.loops[cs.loopDepth] = append(aa.loops[cs.loopDepth],
							candidateAppend{s: s, call: call})
					} else {
						panic("impossible: candidate slice loopDepth > current loop depth")
					}
				}
				continue
			}
		} // End of switch on src.Op().

		// Reached bottom of the loop over assignments.
		// If we get here, we need to visit the dst and src normally.
		aa.visit(dst)
		aa.visit(src)
	}
}

func (aa *aliasAnalysis) countProcessed(nodes ...ir.Node) {
	if aa.doubleCheck {
		for _, n := range nodes {
			aa.processed[n]++
		}
	}
}

func (aa *aliasAnalysis) diag(pos src.XPos, n ir.Node, what string, format string, args ...any) {
	fmt.Printf("%-25s alias analysis: %-30s  %-20s  %s\n",
		fmtPosShort(pos)+":",
		what+":",
		fmt.Sprintf("%v", n),
		fmt.Sprintf(format, args...))
}

// doubleCheckProcessed does a sanity check for missed nodes in our visit.
func doubleCheckProcessed(fn *ir.Func, processed map[ir.Node]int) {
	// Do a trivial walk while counting the nodes
	// to compare against the counts in processed.

	observed := make(map[ir.Node]int)
	var walk func(n ir.Node) bool
	walk = func(n ir.Node) bool {
		observed[n]++
		return ir.DoChildren(n, walk)
	}
	ir.DoChildren(fn, walk)

	if !maps.Equal(processed, observed) {
		// The most likely mistake might be something was missed while building processed,
		// so print extra details in that direction.
		for n, observedCount := range observed {
			processedCount, ok := processed[n]
			if processedCount != observedCount || !ok {
				base.WarnfAt(n.Pos(),
					"alias analysis: mismatch for %T: %v: processed %d times, observed %d times",
					n, n, processedCount, observedCount)
			}
		}
		base.FatalfAt(fn.Pos(), "alias analysis: mismatch in visited nodes")
	}
}

func fmtPosShort(xpos src.XPos) string {
	// TODO(thepudds): I think I did this a simpler way a while ago? Or maybe add base.FmtPosShort
	// or similar? Or maybe just use base.FmtPos and give up on nicely aligned log messages?
	pos := base.Ctxt.PosTable.Pos(xpos)
	shortLine := filepath.Base(pos.AbsFilename()) + ":" + pos.LineNumber()
	return shortLine
}
