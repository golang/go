// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"fmt"
)

// an edgeMem records a backedge, together with the memory
// phi functions at the target of the backedge that must
// be updated when a rescheduling check replaces the backedge.
type edgeMem struct {
	e Edge
	m *Value // phi for memory at dest of e
}

// a rewriteTarget is a value-argindex pair indicating
// where a rewrite is applied.  Note that this is for values,
// not for block controls, because block controls are not targets
// for the rewrites performed in inserting rescheduling checks.
type rewriteTarget struct {
	v *Value
	i int
}

type rewrite struct {
	before, after *Value          // before is the expected value before rewrite, after is the new value installed.
	rewrites      []rewriteTarget // all the targets for this rewrite.
}

func (r *rewrite) String() string {
	s := "\n\tbefore=" + r.before.String() + ", after=" + r.after.String()
	for _, rw := range r.rewrites {
		s += ", (i=" + fmt.Sprint(rw.i) + ", v=" + rw.v.LongString() + ")"
	}
	s += "\n"
	return s
}

// insertLoopReschedChecks inserts rescheduling checks on loop backedges.
func insertLoopReschedChecks(f *Func) {
	// TODO: when split information is recorded in export data, insert checks only on backedges that can be reached on a split-call-free path.

	// Loop reschedule checks compare the stack pointer with
	// the per-g stack bound.  If the pointer appears invalid,
	// that means a reschedule check is needed.
	//
	// Steps:
	// 1. locate backedges.
	// 2. Record memory definitions at block end so that
	//    the SSA graph for mem can be properly modified.
	// 3. Ensure that phi functions that will-be-needed for mem
	//    are present in the graph, initially with trivial inputs.
	// 4. Record all to-be-modified uses of mem;
	//    apply modifications (split into two steps to simplify and
	//    avoided nagging order-dependencies).
	// 5. Rewrite backedges to include reschedule check,
	//    and modify destination phi function appropriately with new
	//    definitions for mem.

	if f.NoSplit { // nosplit functions don't reschedule.
		return
	}

	backedges := backedges(f)
	if len(backedges) == 0 { // no backedges means no rescheduling checks.
		return
	}

	lastMems := findLastMems(f)

	idom := f.Idom()
	po := f.postorder()
	// The ordering in the dominator tree matters; it's important that
	// the walk of the dominator tree also be a preorder (i.e., a node is
	// visited only after all its non-backedge predecessors have been visited).
	sdom := newSparseOrderedTree(f, idom, po)

	if f.pass.debug > 1 {
		fmt.Printf("before %s = %s\n", f.Name, sdom.treestructure(f.Entry))
	}

	tofixBackedges := []edgeMem{}

	for _, e := range backedges { // TODO: could filter here by calls in loops, if declared and inferred nosplit are recorded in export data.
		tofixBackedges = append(tofixBackedges, edgeMem{e, nil})
	}

	// It's possible that there is no memory state (no global/pointer loads/stores or calls)
	if lastMems[f.Entry.ID] == nil {
		lastMems[f.Entry.ID] = f.Entry.NewValue0(f.Entry.Pos, OpInitMem, types.TypeMem)
	}

	memDefsAtBlockEnds := make([]*Value, f.NumBlocks()) // For each block, the mem def seen at its bottom. Could be from earlier block.

	// Propagate last mem definitions forward through successor blocks.
	for i := len(po) - 1; i >= 0; i-- {
		b := po[i]
		mem := lastMems[b.ID]
		for j := 0; mem == nil; j++ { // if there's no def, then there's no phi, so the visible mem is identical in all predecessors.
			// loop because there might be backedges that haven't been visited yet.
			mem = memDefsAtBlockEnds[b.Preds[j].b.ID]
		}
		memDefsAtBlockEnds[b.ID] = mem
		if f.pass.debug > 2 {
			fmt.Printf("memDefsAtBlockEnds[%s] = %s\n", b, mem)
		}
	}

	// Maps from block to newly-inserted phi function in block.
	newmemphis := make(map[*Block]rewrite)

	// Insert phi functions as necessary for future changes to flow graph.
	for i, emc := range tofixBackedges {
		e := emc.e
		h := e.b

		// find the phi function for the memory input at "h", if there is one.
		var headerMemPhi *Value // look for header mem phi

		for _, v := range h.Values {
			if v.Op == OpPhi && v.Type.IsMemory() {
				headerMemPhi = v
			}
		}

		if headerMemPhi == nil {
			// if the header is nil, make a trivial phi from the dominator
			mem0 := memDefsAtBlockEnds[idom[h.ID].ID]
			headerMemPhi = newPhiFor(h, mem0)
			newmemphis[h] = rewrite{before: mem0, after: headerMemPhi}
			addDFphis(mem0, h, h, f, memDefsAtBlockEnds, newmemphis, sdom)

		}
		tofixBackedges[i].m = headerMemPhi

	}
	if f.pass.debug > 0 {
		for b, r := range newmemphis {
			fmt.Printf("before b=%s, rewrite=%s\n", b, r.String())
		}
	}

	// dfPhiTargets notes inputs to phis in dominance frontiers that should not
	// be rewritten as part of the dominated children of some outer rewrite.
	dfPhiTargets := make(map[rewriteTarget]bool)

	rewriteNewPhis(f.Entry, f.Entry, f, memDefsAtBlockEnds, newmemphis, dfPhiTargets, sdom)

	if f.pass.debug > 0 {
		for b, r := range newmemphis {
			fmt.Printf("after b=%s, rewrite=%s\n", b, r.String())
		}
	}

	// Apply collected rewrites.
	for _, r := range newmemphis {
		for _, rw := range r.rewrites {
			rw.v.SetArg(rw.i, r.after)
		}
	}

	// Rewrite backedges to include reschedule checks.
	for _, emc := range tofixBackedges {
		e := emc.e
		headerMemPhi := emc.m
		h := e.b
		i := e.i
		p := h.Preds[i]
		bb := p.b
		mem0 := headerMemPhi.Args[i]
		// bb e->p h,
		// Because we're going to insert a rare-call, make sure the
		// looping edge still looks likely.
		likely := BranchLikely
		if p.i != 0 {
			likely = BranchUnlikely
		}
		if bb.Kind != BlockPlain { // backedges can be unconditional. e.g., if x { something; continue }
			bb.Likely = likely
		}

		// rewrite edge to include reschedule check
		// existing edges:
		//
		// bb.Succs[p.i] == Edge{h, i}
		// h.Preds[i] == p == Edge{bb,p.i}
		//
		// new block(s):
		// test:
		//    if sp < g.limit { goto sched }
		//    goto join
		// sched:
		//    mem1 := call resched (mem0)
		//    goto join
		// join:
		//    mem2 := phi(mem0, mem1)
		//    goto h
		//
		// and correct arg i of headerMemPhi and headerCtrPhi
		//
		// EXCEPT: join block containing only phi functions is bad
		// for the register allocator.  Therefore, there is no
		// join, and branches targeting join must instead target
		// the header, and the other phi functions within header are
		// adjusted for the additional input.

		test := f.NewBlock(BlockIf)
		sched := f.NewBlock(BlockPlain)

		test.Pos = bb.Pos
		sched.Pos = bb.Pos

		// if sp < g.limit { goto sched }
		// goto header

		cfgtypes := &f.Config.Types
		pt := cfgtypes.Uintptr
		g := test.NewValue1(bb.Pos, OpGetG, pt, mem0)
		sp := test.NewValue0(bb.Pos, OpSP, pt)
		cmpOp := OpLess64U
		if pt.Size() == 4 {
			cmpOp = OpLess32U
		}
		limaddr := test.NewValue1I(bb.Pos, OpOffPtr, pt, 2*pt.Size(), g)
		lim := test.NewValue2(bb.Pos, OpLoad, pt, limaddr, mem0)
		cmp := test.NewValue2(bb.Pos, cmpOp, cfgtypes.Bool, sp, lim)
		test.SetControl(cmp)

		// if true, goto sched
		test.AddEdgeTo(sched)

		// if false, rewrite edge to header.
		// do NOT remove+add, because that will perturb all the other phi functions
		// as well as messing up other edges to the header.
		test.Succs = append(test.Succs, Edge{h, i})
		h.Preds[i] = Edge{test, 1}
		headerMemPhi.SetArg(i, mem0)

		test.Likely = BranchUnlikely

		// sched:
		//    mem1 := call resched (mem0)
		//    goto header
		resched := f.fe.Syslook("goschedguarded")
		mem1 := sched.NewValue1A(bb.Pos, OpStaticCall, types.TypeMem, StaticAuxCall(resched, nil, nil), mem0)
		sched.AddEdgeTo(h)
		headerMemPhi.AddArg(mem1)

		bb.Succs[p.i] = Edge{test, 0}
		test.Preds = append(test.Preds, Edge{bb, p.i})

		// Must correct all the other phi functions in the header for new incoming edge.
		// Except for mem phis, it will be the same value seen on the original
		// backedge at index i.
		for _, v := range h.Values {
			if v.Op == OpPhi && v != headerMemPhi {
				v.AddArg(v.Args[i])
			}
		}
	}

	f.invalidateCFG()

	if f.pass.debug > 1 {
		sdom = newSparseTree(f, f.Idom())
		fmt.Printf("after %s = %s\n", f.Name, sdom.treestructure(f.Entry))
	}
}

// newPhiFor inserts a new Phi function into b,
// with all inputs set to v.
func newPhiFor(b *Block, v *Value) *Value {
	phiV := b.NewValue0(b.Pos, OpPhi, v.Type)

	for range b.Preds {
		phiV.AddArg(v)
	}
	return phiV
}

// rewriteNewPhis updates newphis[h] to record all places where the new phi function inserted
// in block h will replace a previous definition.  Block b is the block currently being processed;
// if b has its own phi definition then it takes the place of h.
// defsForUses provides information about other definitions of the variable that are present
// (and if nil, indicates that the variable is no longer live)
// sdom must yield a preorder of the flow graph if recursively walked, root-to-children.
// The result of newSparseOrderedTree with order supplied by a dfs-postorder satisfies this
// requirement.
func rewriteNewPhis(h, b *Block, f *Func, defsForUses []*Value, newphis map[*Block]rewrite, dfPhiTargets map[rewriteTarget]bool, sdom SparseTree) {
	// If b is a block with a new phi, then a new rewrite applies below it in the dominator tree.
	if _, ok := newphis[b]; ok {
		h = b
	}
	change := newphis[h]
	x := change.before
	y := change.after

	// Apply rewrites to this block
	if x != nil { // don't waste time on the common case of no definition.
		p := &change.rewrites
		for _, v := range b.Values {
			if v == y { // don't rewrite self -- phi inputs are handled below.
				continue
			}
			for i, w := range v.Args {
				if w != x {
					continue
				}
				tgt := rewriteTarget{v, i}

				// It's possible dominated control flow will rewrite this instead.
				// Visiting in preorder (a property of how sdom was constructed)
				// ensures that these are seen in the proper order.
				if dfPhiTargets[tgt] {
					continue
				}
				*p = append(*p, tgt)
				if f.pass.debug > 1 {
					fmt.Printf("added block target for h=%v, b=%v, x=%v, y=%v, tgt.v=%s, tgt.i=%d\n",
						h, b, x, y, v, i)
				}
			}
		}

		// Rewrite appropriate inputs of phis reached in successors
		// in dominance frontier, self, and dominated.
		// If the variable def reaching uses in b is itself defined in b, then the new phi function
		// does not reach the successors of b.  (This assumes a bit about the structure of the
		// phi use-def graph, but it's true for memory.)
		if dfu := defsForUses[b.ID]; dfu != nil && dfu.Block != b {
			for _, e := range b.Succs {
				s := e.b

				for _, v := range s.Values {
					if v.Op == OpPhi && v.Args[e.i] == x {
						tgt := rewriteTarget{v, e.i}
						*p = append(*p, tgt)
						dfPhiTargets[tgt] = true
						if f.pass.debug > 1 {
							fmt.Printf("added phi target for h=%v, b=%v, s=%v, x=%v, y=%v, tgt.v=%s, tgt.i=%d\n",
								h, b, s, x, y, v.LongString(), e.i)
						}
						break
					}
				}
			}
		}
		newphis[h] = change
	}

	for c := sdom[b.ID].child; c != nil; c = sdom[c.ID].sibling {
		rewriteNewPhis(h, c, f, defsForUses, newphis, dfPhiTargets, sdom) // TODO: convert to explicit stack from recursion.
	}
}

// addDFphis creates new trivial phis that are necessary to correctly reflect (within SSA)
// a new definition for variable "x" inserted at h (usually but not necessarily a phi).
// These new phis can only occur at the dominance frontier of h; block s is in the dominance
// frontier of h if h does not strictly dominate s and if s is a successor of a block b where
// either b = h or h strictly dominates b.
// These newly created phis are themselves new definitions that may require addition of their
// own trivial phi functions in their own dominance frontier, and this is handled recursively.
func addDFphis(x *Value, h, b *Block, f *Func, defForUses []*Value, newphis map[*Block]rewrite, sdom SparseTree) {
	oldv := defForUses[b.ID]
	if oldv != x { // either a new definition replacing x, or nil if it is proven that there are no uses reachable from b
		return
	}
	idom := f.Idom()
outer:
	for _, e := range b.Succs {
		s := e.b
		// check phi functions in the dominance frontier
		if sdom.isAncestor(h, s) {
			continue // h dominates s, successor of b, therefore s is not in the frontier.
		}
		if _, ok := newphis[s]; ok {
			continue // successor s of b already has a new phi function, so there is no need to add another.
		}
		if x != nil {
			for _, v := range s.Values {
				if v.Op == OpPhi && v.Args[e.i] == x {
					continue outer // successor s of b has an old phi function, so there is no need to add another.
				}
			}
		}

		old := defForUses[idom[s.ID].ID] // new phi function is correct-but-redundant, combining value "old" on all inputs.
		headerPhi := newPhiFor(s, old)
		// the new phi will replace "old" in block s and all blocks dominated by s.
		newphis[s] = rewrite{before: old, after: headerPhi} // record new phi, to have inputs labeled "old" rewritten to "headerPhi"
		addDFphis(old, s, s, f, defForUses, newphis, sdom)  // the new definition may also create new phi functions.
	}
	for c := sdom[b.ID].child; c != nil; c = sdom[c.ID].sibling {
		addDFphis(x, h, c, f, defForUses, newphis, sdom) // TODO: convert to explicit stack from recursion.
	}
}

// findLastMems maps block ids to last memory-output op in a block, if any
func findLastMems(f *Func) []*Value {

	var stores []*Value
	lastMems := make([]*Value, f.NumBlocks())
	storeUse := f.newSparseSet(f.NumValues())
	defer f.retSparseSet(storeUse)
	for _, b := range f.Blocks {
		// Find all the stores in this block. Categorize their uses:
		//  storeUse contains stores which are used by a subsequent store.
		storeUse.clear()
		stores = stores[:0]
		var memPhi *Value
		for _, v := range b.Values {
			if v.Op == OpPhi {
				if v.Type.IsMemory() {
					memPhi = v
				}
				continue
			}
			if v.Type.IsMemory() {
				stores = append(stores, v)
				for _, a := range v.Args {
					if a.Block == b && a.Type.IsMemory() {
						storeUse.add(a.ID)
					}
				}
			}
		}
		if len(stores) == 0 {
			lastMems[b.ID] = memPhi
			continue
		}

		// find last store in the block
		var last *Value
		for _, v := range stores {
			if storeUse.contains(v.ID) {
				continue
			}
			if last != nil {
				b.Fatalf("two final stores - simultaneous live stores %s %s", last, v)
			}
			last = v
		}
		if last == nil {
			b.Fatalf("no last store found - cycle?")
		}
		lastMems[b.ID] = last
	}
	return lastMems
}

// mark values
type markKind uint8

const (
	notFound    markKind = iota // block has not been discovered yet
	notExplored                 // discovered and in queue, outedges not processed yet
	explored                    // discovered and in queue, outedges processed
	done                        // all done, in output ordering
)

type backedgesState struct {
	b *Block
	i int
}

// backedges returns a slice of successor edges that are back
// edges.  For reducible loops, edge.b is the header.
func backedges(f *Func) []Edge {
	edges := []Edge{}
	mark := make([]markKind, f.NumBlocks())
	stack := []backedgesState{}

	mark[f.Entry.ID] = notExplored
	stack = append(stack, backedgesState{f.Entry, 0})

	for len(stack) > 0 {
		l := len(stack)
		x := stack[l-1]
		if x.i < len(x.b.Succs) {
			e := x.b.Succs[x.i]
			stack[l-1].i++
			s := e.b
			if mark[s.ID] == notFound {
				mark[s.ID] = notExplored
				stack = append(stack, backedgesState{s, 0})
			} else if mark[s.ID] == notExplored {
				edges = append(edges, e)
			}
		} else {
			mark[x.b.ID] = done
			stack = stack[0 : l-1]
		}
	}
	return edges
}
