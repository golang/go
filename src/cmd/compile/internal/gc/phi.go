// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"container/heap"
	"fmt"
)

// This file contains the algorithm to place phi nodes in a function.
// For small functions, we use Braun, Buchwald, Hack, Leißa, Mallon, and Zwinkau.
// https://pp.info.uni-karlsruhe.de/uploads/publikationen/braun13cc.pdf
// For large functions, we use Sreedhar & Gao: A Linear Time Algorithm for Placing Φ-Nodes.
// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.8.1979&rep=rep1&type=pdf

const smallBlocks = 500

const debugPhi = false

// insertPhis finds all the places in the function where a phi is
// necessary and inserts them.
// Uses FwdRef ops to find all uses of variables, and s.defvars to find
// all definitions.
// Phi values are inserted, and all FwdRefs are changed to a Copy
// of the appropriate phi or definition.
// TODO: make this part of cmd/compile/internal/ssa somehow?
func (s *state) insertPhis() {
	if len(s.f.Blocks) <= smallBlocks {
		sps := simplePhiState{s: s, f: s.f, defvars: s.defvars}
		sps.insertPhis()
		return
	}
	ps := phiState{s: s, f: s.f, defvars: s.defvars}
	ps.insertPhis()
}

type phiState struct {
	s       *state                 // SSA state
	f       *ssa.Func              // function to work on
	defvars []map[*Node]*ssa.Value // defined variables at end of each block

	varnum map[*Node]int32 // variable numbering

	// properties of the dominator tree
	idom  []*ssa.Block // dominator parents
	tree  []domBlock   // dominator child+sibling
	level []int32      // level in dominator tree (0 = root or unreachable, 1 = children of root, ...)

	// scratch locations
	priq   blockHeap    // priority queue of blocks, higher level (toward leaves) = higher priority
	q      []*ssa.Block // inner loop queue
	queued *sparseSet   // has been put in q
	hasPhi *sparseSet   // has a phi
	hasDef *sparseSet   // has a write of the variable we're processing

	// miscellaneous
	placeholder *ssa.Value // dummy value to use as a "not set yet" placeholder.
}

func (s *phiState) insertPhis() {
	if debugPhi {
		fmt.Println(s.f.String())
	}

	// Find all the variables for which we need to match up reads & writes.
	// This step prunes any basic-block-only variables from consideration.
	// Generate a numbering for these variables.
	s.varnum = map[*Node]int32{}
	var vars []*Node
	var vartypes []*types.Type
	for _, b := range s.f.Blocks {
		for _, v := range b.Values {
			if v.Op != ssa.OpFwdRef {
				continue
			}
			var_ := v.Aux.(*Node)

			// Optimization: look back 1 block for the definition.
			if len(b.Preds) == 1 {
				c := b.Preds[0].Block()
				if w := s.defvars[c.ID][var_]; w != nil {
					v.Op = ssa.OpCopy
					v.Aux = nil
					v.AddArg(w)
					continue
				}
			}

			if _, ok := s.varnum[var_]; ok {
				continue
			}
			s.varnum[var_] = int32(len(vartypes))
			if debugPhi {
				fmt.Printf("var%d = %v\n", len(vartypes), var_)
			}
			vars = append(vars, var_)
			vartypes = append(vartypes, v.Type)
		}
	}

	if len(vartypes) == 0 {
		return
	}

	// Find all definitions of the variables we need to process.
	// defs[n] contains all the blocks in which variable number n is assigned.
	defs := make([][]*ssa.Block, len(vartypes))
	for _, b := range s.f.Blocks {
		for var_ := range s.defvars[b.ID] { // TODO: encode defvars some other way (explicit ops)? make defvars[n] a slice instead of a map.
			if n, ok := s.varnum[var_]; ok {
				defs[n] = append(defs[n], b)
			}
		}
	}

	// Make dominator tree.
	s.idom = s.f.Idom()
	s.tree = make([]domBlock, s.f.NumBlocks())
	for _, b := range s.f.Blocks {
		p := s.idom[b.ID]
		if p != nil {
			s.tree[b.ID].sibling = s.tree[p.ID].firstChild
			s.tree[p.ID].firstChild = b
		}
	}
	// Compute levels in dominator tree.
	// With parent pointers we can do a depth-first walk without
	// any auxiliary storage.
	s.level = make([]int32, s.f.NumBlocks())
	b := s.f.Entry
levels:
	for {
		if p := s.idom[b.ID]; p != nil {
			s.level[b.ID] = s.level[p.ID] + 1
			if debugPhi {
				fmt.Printf("level %s = %d\n", b, s.level[b.ID])
			}
		}
		if c := s.tree[b.ID].firstChild; c != nil {
			b = c
			continue
		}
		for {
			if c := s.tree[b.ID].sibling; c != nil {
				b = c
				continue levels
			}
			b = s.idom[b.ID]
			if b == nil {
				break levels
			}
		}
	}

	// Allocate scratch locations.
	s.priq.level = s.level
	s.q = make([]*ssa.Block, 0, s.f.NumBlocks())
	s.queued = newSparseSet(s.f.NumBlocks())
	s.hasPhi = newSparseSet(s.f.NumBlocks())
	s.hasDef = newSparseSet(s.f.NumBlocks())
	s.placeholder = s.s.entryNewValue0(ssa.OpUnknown, types.TypeInvalid)

	// Generate phi ops for each variable.
	for n := range vartypes {
		s.insertVarPhis(n, vars[n], defs[n], vartypes[n])
	}

	// Resolve FwdRefs to the correct write or phi.
	s.resolveFwdRefs()

	// Erase variable numbers stored in AuxInt fields of phi ops. They are no longer needed.
	for _, b := range s.f.Blocks {
		for _, v := range b.Values {
			if v.Op == ssa.OpPhi {
				v.AuxInt = 0
			}
		}
	}
}

func (s *phiState) insertVarPhis(n int, var_ *Node, defs []*ssa.Block, typ *types.Type) {
	priq := &s.priq
	q := s.q
	queued := s.queued
	queued.clear()
	hasPhi := s.hasPhi
	hasPhi.clear()
	hasDef := s.hasDef
	hasDef.clear()

	// Add defining blocks to priority queue.
	for _, b := range defs {
		priq.a = append(priq.a, b)
		hasDef.add(b.ID)
		if debugPhi {
			fmt.Printf("def of var%d in %s\n", n, b)
		}
	}
	heap.Init(priq)

	// Visit blocks defining variable n, from deepest to shallowest.
	for len(priq.a) > 0 {
		currentRoot := heap.Pop(priq).(*ssa.Block)
		if debugPhi {
			fmt.Printf("currentRoot %s\n", currentRoot)
		}
		// Walk subtree below definition.
		// Skip subtrees we've done in previous iterations.
		// Find edges exiting tree dominated by definition (the dominance frontier).
		// Insert phis at target blocks.
		if queued.contains(currentRoot.ID) {
			s.s.Fatalf("root already in queue")
		}
		q = append(q, currentRoot)
		queued.add(currentRoot.ID)
		for len(q) > 0 {
			b := q[len(q)-1]
			q = q[:len(q)-1]
			if debugPhi {
				fmt.Printf("  processing %s\n", b)
			}

			currentRootLevel := s.level[currentRoot.ID]
			for _, e := range b.Succs {
				c := e.Block()
				// TODO: if the variable is dead at c, skip it.
				if s.level[c.ID] > currentRootLevel {
					// a D-edge, or an edge whose target is in currentRoot's subtree.
					continue
				}
				if hasPhi.contains(c.ID) {
					continue
				}
				// Add a phi to block c for variable n.
				hasPhi.add(c.ID)
				v := c.NewValue0I(currentRoot.Pos, ssa.OpPhi, typ, int64(n)) // TODO: line number right?
				// Note: we store the variable number in the phi's AuxInt field. Used temporarily by phi building.
				s.s.addNamedValue(var_, v)
				for range c.Preds {
					v.AddArg(s.placeholder) // Actual args will be filled in by resolveFwdRefs.
				}
				if debugPhi {
					fmt.Printf("new phi for var%d in %s: %s\n", n, c, v)
				}
				if !hasDef.contains(c.ID) {
					// There's now a new definition of this variable in block c.
					// Add it to the priority queue to explore.
					heap.Push(priq, c)
					hasDef.add(c.ID)
				}
			}

			// Visit children if they have not been visited yet.
			for c := s.tree[b.ID].firstChild; c != nil; c = s.tree[c.ID].sibling {
				if !queued.contains(c.ID) {
					q = append(q, c)
					queued.add(c.ID)
				}
			}
		}
	}
}

// resolveFwdRefs links all FwdRef uses up to their nearest dominating definition.
func (s *phiState) resolveFwdRefs() {
	// Do a depth-first walk of the dominator tree, keeping track
	// of the most-recently-seen value for each variable.

	// Map from variable ID to SSA value at the current point of the walk.
	values := make([]*ssa.Value, len(s.varnum))
	for i := range values {
		values[i] = s.placeholder
	}

	// Stack of work to do.
	type stackEntry struct {
		b *ssa.Block // block to explore

		// variable/value pair to reinstate on exit
		n int32 // variable ID
		v *ssa.Value

		// Note: only one of b or n,v will be set.
	}
	var stk []stackEntry

	stk = append(stk, stackEntry{b: s.f.Entry})
	for len(stk) > 0 {
		work := stk[len(stk)-1]
		stk = stk[:len(stk)-1]

		b := work.b
		if b == nil {
			// On exit from a block, this case will undo any assignments done below.
			values[work.n] = work.v
			continue
		}

		// Process phis as new defs. They come before FwdRefs in this block.
		for _, v := range b.Values {
			if v.Op != ssa.OpPhi {
				continue
			}
			n := int32(v.AuxInt)
			// Remember the old assignment so we can undo it when we exit b.
			stk = append(stk, stackEntry{n: n, v: values[n]})
			// Record the new assignment.
			values[n] = v
		}

		// Replace a FwdRef op with the current incoming value for its variable.
		for _, v := range b.Values {
			if v.Op != ssa.OpFwdRef {
				continue
			}
			n := s.varnum[v.Aux.(*Node)]
			v.Op = ssa.OpCopy
			v.Aux = nil
			v.AddArg(values[n])
		}

		// Establish values for variables defined in b.
		for var_, v := range s.defvars[b.ID] {
			n, ok := s.varnum[var_]
			if !ok {
				// some variable not live across a basic block boundary.
				continue
			}
			// Remember the old assignment so we can undo it when we exit b.
			stk = append(stk, stackEntry{n: n, v: values[n]})
			// Record the new assignment.
			values[n] = v
		}

		// Replace phi args in successors with the current incoming value.
		for _, e := range b.Succs {
			c, i := e.Block(), e.Index()
			for j := len(c.Values) - 1; j >= 0; j-- {
				v := c.Values[j]
				if v.Op != ssa.OpPhi {
					break // All phis will be at the end of the block during phi building.
				}
				// Only set arguments that have been resolved.
				// For very wide CFGs, this significantly speeds up phi resolution.
				// See golang.org/issue/8225.
				if w := values[v.AuxInt]; w.Op != ssa.OpUnknown {
					v.SetArg(i, w)
				}
			}
		}

		// Walk children in dominator tree.
		for c := s.tree[b.ID].firstChild; c != nil; c = s.tree[c.ID].sibling {
			stk = append(stk, stackEntry{b: c})
		}
	}
}

// domBlock contains extra per-block information to record the dominator tree.
type domBlock struct {
	firstChild *ssa.Block // first child of block in dominator tree
	sibling    *ssa.Block // next child of parent in dominator tree
}

// A block heap is used as a priority queue to implement the PiggyBank
// from Sreedhar and Gao.  That paper uses an array which is better
// asymptotically but worse in the common case when the PiggyBank
// holds a sparse set of blocks.
type blockHeap struct {
	a     []*ssa.Block // block IDs in heap
	level []int32      // depth in dominator tree (static, used for determining priority)
}

func (h *blockHeap) Len() int      { return len(h.a) }
func (h *blockHeap) Swap(i, j int) { a := h.a; a[i], a[j] = a[j], a[i] }

func (h *blockHeap) Push(x interface{}) {
	v := x.(*ssa.Block)
	h.a = append(h.a, v)
}
func (h *blockHeap) Pop() interface{} {
	old := h.a
	n := len(old)
	x := old[n-1]
	h.a = old[:n-1]
	return x
}
func (h *blockHeap) Less(i, j int) bool {
	return h.level[h.a[i].ID] > h.level[h.a[j].ID]
}

// TODO: stop walking the iterated domininance frontier when
// the variable is dead. Maybe detect that by checking if the
// node we're on is reverse dominated by all the reads?
// Reverse dominated by the highest common successor of all the reads?

// copy of ../ssa/sparseset.go
// TODO: move this file to ../ssa, then use sparseSet there.
type sparseSet struct {
	dense  []ssa.ID
	sparse []int32
}

// newSparseSet returns a sparseSet that can represent
// integers between 0 and n-1
func newSparseSet(n int) *sparseSet {
	return &sparseSet{dense: nil, sparse: make([]int32, n)}
}

func (s *sparseSet) contains(x ssa.ID) bool {
	i := s.sparse[x]
	return i < int32(len(s.dense)) && s.dense[i] == x
}

func (s *sparseSet) add(x ssa.ID) {
	i := s.sparse[x]
	if i < int32(len(s.dense)) && s.dense[i] == x {
		return
	}
	s.dense = append(s.dense, x)
	s.sparse[x] = int32(len(s.dense)) - 1
}

func (s *sparseSet) clear() {
	s.dense = s.dense[:0]
}

// Variant to use for small functions.
type simplePhiState struct {
	s         *state                 // SSA state
	f         *ssa.Func              // function to work on
	fwdrefs   []*ssa.Value           // list of FwdRefs to be processed
	defvars   []map[*Node]*ssa.Value // defined variables at end of each block
	reachable []bool                 // which blocks are reachable
}

func (s *simplePhiState) insertPhis() {
	s.reachable = ssa.ReachableBlocks(s.f)

	// Find FwdRef ops.
	for _, b := range s.f.Blocks {
		for _, v := range b.Values {
			if v.Op != ssa.OpFwdRef {
				continue
			}
			s.fwdrefs = append(s.fwdrefs, v)
			var_ := v.Aux.(*Node)
			if _, ok := s.defvars[b.ID][var_]; !ok {
				s.defvars[b.ID][var_] = v // treat FwdDefs as definitions.
			}
		}
	}

	var args []*ssa.Value

loop:
	for len(s.fwdrefs) > 0 {
		v := s.fwdrefs[len(s.fwdrefs)-1]
		s.fwdrefs = s.fwdrefs[:len(s.fwdrefs)-1]
		b := v.Block
		var_ := v.Aux.(*Node)
		if b == s.f.Entry {
			// No variable should be live at entry.
			s.s.Fatalf("Value live at entry. It shouldn't be. func %s, node %v, value %v", s.f.Name, var_, v)
		}
		if !s.reachable[b.ID] {
			// This block is dead.
			// It doesn't matter what we use here as long as it is well-formed.
			v.Op = ssa.OpUnknown
			v.Aux = nil
			continue
		}
		// Find variable value on each predecessor.
		args = args[:0]
		for _, e := range b.Preds {
			args = append(args, s.lookupVarOutgoing(e.Block(), v.Type, var_, v.Pos))
		}

		// Decide if we need a phi or not. We need a phi if there
		// are two different args (which are both not v).
		var w *ssa.Value
		for _, a := range args {
			if a == v {
				continue // self-reference
			}
			if a == w {
				continue // already have this witness
			}
			if w != nil {
				// two witnesses, need a phi value
				v.Op = ssa.OpPhi
				v.AddArgs(args...)
				v.Aux = nil
				continue loop
			}
			w = a // save witness
		}
		if w == nil {
			s.s.Fatalf("no witness for reachable phi %s", v)
		}
		// One witness. Make v a copy of w.
		v.Op = ssa.OpCopy
		v.Aux = nil
		v.AddArg(w)
	}
}

// lookupVarOutgoing finds the variable's value at the end of block b.
func (s *simplePhiState) lookupVarOutgoing(b *ssa.Block, t *types.Type, var_ *Node, line src.XPos) *ssa.Value {
	for {
		if v := s.defvars[b.ID][var_]; v != nil {
			return v
		}
		// The variable is not defined by b and we haven't looked it up yet.
		// If b has exactly one predecessor, loop to look it up there.
		// Otherwise, give up and insert a new FwdRef and resolve it later.
		if len(b.Preds) != 1 {
			break
		}
		b = b.Preds[0].Block()
		if !s.reachable[b.ID] {
			// This is rare; it happens with oddly interleaved infinite loops in dead code.
			// See issue 19783.
			break
		}
	}
	// Generate a FwdRef for the variable and return that.
	v := b.NewValue0A(line, ssa.OpFwdRef, t, var_)
	s.defvars[b.ID][var_] = v
	s.s.addNamedValue(var_, v)
	s.fwdrefs = append(s.fwdrefs, v)
	return v
}
