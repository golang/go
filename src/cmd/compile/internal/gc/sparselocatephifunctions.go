// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/ssa"
	"fmt"
	"math"
)

// sparseDefState contains a Go map from ONAMEs (*Node) to sparse definition trees, and
// a search helper for the CFG's dominator tree in which those definitions are embedded.
// Once initialized, given a use of an ONAME within a block, the ssa definition for
// that ONAME can be discovered in time roughly proportional to the log of the number
// of SSA definitions of that ONAME (thus avoiding pathological quadratic behavior for
// very large programs).  The helper contains state (a dominator tree numbering) common
// to all the sparse definition trees, as well as some necessary data obtained from
// the ssa package.
//
// This algorithm has improved asymptotic complexity, but the constant factor is
// rather large and thus it is only preferred for very large inputs containing
// 1000s of blocks and variables.
type sparseDefState struct {
	helper         *ssa.SparseTreeHelper // contains one copy of information needed to do sparse mapping
	defmapForOname map[*Node]*onameDefs  // for each ONAME, its definition set (normal and phi)
}

// onameDefs contains a record of definitions (ordinary and implied phi function) for a single OName.
// stm is the set of definitions for the OName.
// firstdef and lastuse are postorder block numberings that
// conservatively bracket the entire lifetime of the OName.
type onameDefs struct {
	stm *ssa.SparseTreeMap
	// firstdef and lastuse define an interval in the postorder numbering
	// that is guaranteed to include the entire lifetime of an ONAME.
	// In the postorder numbering, math.MaxInt32 is before anything,
	// and 0 is after-or-equal all exit nodes and infinite loops.
	firstdef int32 // the first definition of this ONAME *in the postorder numbering*
	lastuse  int32 // the last use of this ONAME *in the postorder numbering*
}

// defsFor finds or creates-and-inserts-in-map the definition information
// (sparse tree and live range) for a given OName.
func (m *sparseDefState) defsFor(n *Node) *onameDefs {
	d := m.defmapForOname[n]
	if d != nil {
		return d
	}
	// Reminder: firstdef/lastuse are postorder indices, not block indices,
	// so these default values define an empty interval, not the entire one.
	d = &onameDefs{stm: m.helper.NewTree(), firstdef: 0, lastuse: math.MaxInt32}
	m.defmapForOname[n] = d
	return d
}

// Insert adds a definition at b (with specified before/within/after adjustment)
// to sparse tree onameDefs.  The lifetime is extended as necessary.
func (m *sparseDefState) Insert(tree *onameDefs, b *ssa.Block, adjust int32) {
	bponum := m.helper.Ponums[b.ID]
	if bponum > tree.firstdef {
		tree.firstdef = bponum
	}
	tree.stm.Insert(b, adjust, b, m.helper)
}

// Use updates tree to record a use within b, extending the lifetime as necessary.
func (m *sparseDefState) Use(tree *onameDefs, b *ssa.Block) {
	bponum := m.helper.Ponums[b.ID]
	if bponum < tree.lastuse {
		tree.lastuse = bponum
	}
}

// locatePotentialPhiFunctions finds all the places where phi functions
// will be inserted into a program and records those and ordinary definitions
// in a "map" (not a Go map) that given an OName and use site, returns the
// SSA definition for that OName that will reach the use site (that is,
// the use site's nearest def/phi site in the dominator tree.)
func (s *state) locatePotentialPhiFunctions(fn *Node) *sparseDefState {
	// s.config.SparsePhiCutoff() is compared with product of numblocks and numvalues,
	// if product is smaller than cutoff, use old non-sparse method.
	// cutoff == 0 implies all sparse
	// cutoff == uint(-1) implies all non-sparse
	if uint64(s.f.NumValues())*uint64(s.f.NumBlocks()) < s.config.SparsePhiCutoff() {
		return nil
	}

	helper := ssa.NewSparseTreeHelper(s.f)
	po := helper.Po // index by block.ID to obtain postorder # of block.
	trees := make(map[*Node]*onameDefs)
	dm := &sparseDefState{defmapForOname: trees, helper: helper}

	// Process params, taking note of their special lifetimes
	b := s.f.Entry
	for _, n := range fn.Func.Dcl {
		switch n.Class {
		case PPARAM, PPARAMOUT:
			t := dm.defsFor(n)
			dm.Insert(t, b, ssa.AdjustBefore) // define param at entry block
			if n.Class == PPARAMOUT {
				dm.Use(t, po[0]) // Explicitly use PPARAMOUT at very last block
			}
		default:
		}
	}

	// Process memory variable.
	t := dm.defsFor(&memVar)
	dm.Insert(t, b, ssa.AdjustBefore) // define memory at entry block
	dm.Use(t, po[0])                  // Explicitly use memory at last block

	// Next load the map w/ basic definitions for ONames recorded per-block
	// Iterate over po to avoid unreachable blocks.
	for i := len(po) - 1; i >= 0; i-- {
		b := po[i]
		m := s.defvars[b.ID]
		for n := range m { // no specified order, but per-node trees are independent.
			t := dm.defsFor(n)
			dm.Insert(t, b, ssa.AdjustWithin)
		}
	}

	// Find last use of each variable
	for _, v := range s.fwdRefs {
		b := v.Block
		name := v.Aux.(*Node)
		t := dm.defsFor(name)
		dm.Use(t, b)
	}

	for _, t := range trees {
		// iterating over names in the outer loop
		for change := true; change; {
			change = false
			for i := t.firstdef; i >= t.lastuse; i-- {
				// Iterating in reverse of post-order reduces number of 'change' iterations;
				// all possible forward flow goes through each time.
				b := po[i]
				// Within tree t, would a use at b require a phi function to ensure a single definition?
				// TODO: perhaps more efficient to record specific use sites instead of range?
				if len(b.Preds) < 2 {
					continue // no phi possible
				}
				phi := t.stm.Find(b, ssa.AdjustWithin, helper) // Look for defs in earlier block or AdjustBefore in this one.
				if phi != nil && phi.(*ssa.Block) == b {
					continue // has a phi already in this block.
				}
				var defseen interface{}
				// Do preds see different definitions? if so, need a phi function.
				for _, e := range b.Preds {
					p := e.Block()
					dm.Use(t, p)                                // always count phi pred as "use"; no-op except for loop edges, which matter.
					x := t.stm.Find(p, ssa.AdjustAfter, helper) // Look for defs reaching or within predecessors.
					if x == nil {                               // nil def from a predecessor means a backedge that will be visited soon.
						continue
					}
					if defseen == nil {
						defseen = x
					}
					if defseen != x {
						// Need to insert a phi function here because predecessors's definitions differ.
						change = true
						// Phi insertion is at AdjustBefore, visible with find in same block at AdjustWithin or AdjustAfter.
						dm.Insert(t, b, ssa.AdjustBefore)
						break
					}
				}
			}
		}
	}
	return dm
}

// FindBetterDefiningBlock tries to find a better block for a definition of OName name
// reaching (or within) p than p itself.  If it cannot, it returns p instead.
// This aids in more efficient location of phi functions, since it can skip over
// branch code that might contain a definition of name if it actually does not.
func (m *sparseDefState) FindBetterDefiningBlock(name *Node, p *ssa.Block) *ssa.Block {
	if m == nil {
		return p
	}
	t := m.defmapForOname[name]
	// For now this is fail-soft, since the old algorithm still works using the unimproved block.
	if t == nil {
		return p
	}
	x := t.stm.Find(p, ssa.AdjustAfter, m.helper)
	if x == nil {
		return p
	}
	b := x.(*ssa.Block)
	if b == nil {
		return p
	}
	return b
}

func (d *onameDefs) String() string {
	return fmt.Sprintf("onameDefs:first=%d,last=%d,tree=%s", d.firstdef, d.lastuse, d.stm.String())
}
