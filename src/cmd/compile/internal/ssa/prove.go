// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "math"

type branch int

const (
	unknown = iota
	positive
	negative
)

// relation represents the set of possible relations between
// pairs of variables (v, w). Without a priori knowledge the
// mask is lt | eq | gt meaning v can be less than, equal to or
// greater than w. When the execution path branches on the condition
// `v op w` the set of relations is updated to exclude any
// relation not possible due to `v op w` being true (or false).
//
// E.g.
//
// r := relation(...)
//
// if v < w {
//   newR := r & lt
// }
// if v >= w {
//   newR := r & (eq|gt)
// }
// if v != w {
//   newR := r & (lt|gt)
// }
type relation uint

const (
	lt relation = 1 << iota
	eq
	gt
)

// domain represents the domain of a variable pair in which a set
// of relations is known.  For example, relations learned for unsigned
// pairs cannot be transferred to signed pairs because the same bit
// representation can mean something else.
type domain uint

const (
	signed domain = 1 << iota
	unsigned
	pointer
	boolean
)

type pair struct {
	v, w *Value // a pair of values, ordered by ID.
	// v can be nil, to mean the zero value.
	// for booleans the zero value (v == nil) is false.
	d domain
}

// fact is a pair plus a relation for that pair.
type fact struct {
	p pair
	r relation
}

// a limit records known upper and lower bounds for a value.
type limit struct {
	min, max   int64  // min <= value <= max, signed
	umin, umax uint64 // umin <= value <= umax, unsigned
}

var noLimit = limit{math.MinInt64, math.MaxInt64, 0, math.MaxUint64}

// a limitFact is a limit known for a particular value.
type limitFact struct {
	vid   ID
	limit limit
}

// factsTable keeps track of relations between pairs of values.
type factsTable struct {
	facts map[pair]relation // current known set of relation
	stack []fact            // previous sets of relations

	// known lower and upper bounds on individual values.
	limits     map[ID]limit
	limitStack []limitFact // previous entries
}

// checkpointFact is an invalid value used for checkpointing
// and restoring factsTable.
var checkpointFact = fact{}
var checkpointBound = limitFact{}

func newFactsTable() *factsTable {
	ft := &factsTable{}
	ft.facts = make(map[pair]relation)
	ft.stack = make([]fact, 4)
	ft.limits = make(map[ID]limit)
	ft.limitStack = make([]limitFact, 4)
	return ft
}

// get returns the known possible relations between v and w.
// If v and w are not in the map it returns lt|eq|gt, i.e. any order.
func (ft *factsTable) get(v, w *Value, d domain) relation {
	if v.isGenericIntConst() || w.isGenericIntConst() {
		reversed := false
		if v.isGenericIntConst() {
			v, w = w, v
			reversed = true
		}
		r := lt | eq | gt
		lim, ok := ft.limits[v.ID]
		if !ok {
			return r
		}
		c := w.AuxInt
		switch d {
		case signed:
			switch {
			case c < lim.min:
				r = gt
			case c > lim.max:
				r = lt
			case c == lim.min && c == lim.max:
				r = eq
			case c == lim.min:
				r = gt | eq
			case c == lim.max:
				r = lt | eq
			}
		case unsigned:
			// TODO: also use signed data if lim.min >= 0?
			var uc uint64
			switch w.Op {
			case OpConst64:
				uc = uint64(c)
			case OpConst32:
				uc = uint64(uint32(c))
			case OpConst16:
				uc = uint64(uint16(c))
			case OpConst8:
				uc = uint64(uint8(c))
			}
			switch {
			case uc < lim.umin:
				r = gt
			case uc > lim.umax:
				r = lt
			case uc == lim.umin && uc == lim.umax:
				r = eq
			case uc == lim.umin:
				r = gt | eq
			case uc == lim.umax:
				r = lt | eq
			}
		}
		if reversed {
			return reverseBits[r]
		}
		return r
	}

	reversed := false
	if lessByID(w, v) {
		v, w = w, v
		reversed = !reversed
	}

	p := pair{v, w, d}
	r, ok := ft.facts[p]
	if !ok {
		if p.v == p.w {
			r = eq
		} else {
			r = lt | eq | gt
		}
	}

	if reversed {
		return reverseBits[r]
	}
	return r
}

// update updates the set of relations between v and w in domain d
// restricting it to r.
func (ft *factsTable) update(v, w *Value, d domain, r relation) {
	if lessByID(w, v) {
		v, w = w, v
		r = reverseBits[r]
	}

	p := pair{v, w, d}
	oldR := ft.get(v, w, d)
	ft.stack = append(ft.stack, fact{p, oldR})
	ft.facts[p] = oldR & r

	// Extract bounds when comparing against constants
	if v.isGenericIntConst() {
		v, w = w, v
		r = reverseBits[r]
	}
	if v != nil && w.isGenericIntConst() {
		c := w.AuxInt
		// Note: all the +1/-1 below could overflow/underflow. Either will
		// still generate correct results, it will just lead to imprecision.
		// In fact if there is overflow/underflow, the corresponding
		// code is unreachable because the known range is outside the range
		// of the value's type.
		old, ok := ft.limits[v.ID]
		if !ok {
			old = noLimit
		}
		lim := old
		// Update lim with the new information we know.
		switch d {
		case signed:
			switch r {
			case lt:
				if c-1 < lim.max {
					lim.max = c - 1
				}
			case lt | eq:
				if c < lim.max {
					lim.max = c
				}
			case gt | eq:
				if c > lim.min {
					lim.min = c
				}
			case gt:
				if c+1 > lim.min {
					lim.min = c + 1
				}
			case lt | gt:
				if c == lim.min {
					lim.min++
				}
				if c == lim.max {
					lim.max--
				}
			case eq:
				lim.min = c
				lim.max = c
			}
		case unsigned:
			var uc uint64
			switch w.Op {
			case OpConst64:
				uc = uint64(c)
			case OpConst32:
				uc = uint64(uint32(c))
			case OpConst16:
				uc = uint64(uint16(c))
			case OpConst8:
				uc = uint64(uint8(c))
			}
			switch r {
			case lt:
				if uc-1 < lim.umax {
					lim.umax = uc - 1
				}
			case lt | eq:
				if uc < lim.umax {
					lim.umax = uc
				}
			case gt | eq:
				if uc > lim.umin {
					lim.umin = uc
				}
			case gt:
				if uc+1 > lim.umin {
					lim.umin = uc + 1
				}
			case lt | gt:
				if uc == lim.umin {
					lim.umin++
				}
				if uc == lim.umax {
					lim.umax--
				}
			case eq:
				lim.umin = uc
				lim.umax = uc
			}
		}
		ft.limitStack = append(ft.limitStack, limitFact{v.ID, old})
		ft.limits[v.ID] = lim
	}
}

// isNonNegative returns true if v is known to be non-negative.
func (ft *factsTable) isNonNegative(v *Value) bool {
	if isNonNegative(v) {
		return true
	}
	l, has := ft.limits[v.ID]
	return has && (l.min >= 0 || l.umax <= math.MaxInt64)
}

// checkpoint saves the current state of known relations.
// Called when descending on a branch.
func (ft *factsTable) checkpoint() {
	ft.stack = append(ft.stack, checkpointFact)
	ft.limitStack = append(ft.limitStack, checkpointBound)
}

// restore restores known relation to the state just
// before the previous checkpoint.
// Called when backing up on a branch.
func (ft *factsTable) restore() {
	for {
		old := ft.stack[len(ft.stack)-1]
		ft.stack = ft.stack[:len(ft.stack)-1]
		if old == checkpointFact {
			break
		}
		if old.r == lt|eq|gt {
			delete(ft.facts, old.p)
		} else {
			ft.facts[old.p] = old.r
		}
	}
	for {
		old := ft.limitStack[len(ft.limitStack)-1]
		ft.limitStack = ft.limitStack[:len(ft.limitStack)-1]
		if old.vid == 0 { // checkpointBound
			break
		}
		if old.limit == noLimit {
			delete(ft.limits, old.vid)
		} else {
			ft.limits[old.vid] = old.limit
		}
	}
}

func lessByID(v, w *Value) bool {
	if v == nil && w == nil {
		// Should not happen, but just in case.
		return false
	}
	if v == nil {
		return true
	}
	return w != nil && v.ID < w.ID
}

var (
	reverseBits = [...]relation{0, 4, 2, 6, 1, 5, 3, 7}

	// maps what we learn when the positive branch is taken.
	// For example:
	//      OpLess8:   {signed, lt},
	//	v1 = (OpLess8 v2 v3).
	// If v1 branch is taken than we learn that the rangeMaks
	// can be at most lt.
	domainRelationTable = map[Op]struct {
		d domain
		r relation
	}{
		OpEq8:   {signed | unsigned, eq},
		OpEq16:  {signed | unsigned, eq},
		OpEq32:  {signed | unsigned, eq},
		OpEq64:  {signed | unsigned, eq},
		OpEqPtr: {pointer, eq},

		OpNeq8:   {signed | unsigned, lt | gt},
		OpNeq16:  {signed | unsigned, lt | gt},
		OpNeq32:  {signed | unsigned, lt | gt},
		OpNeq64:  {signed | unsigned, lt | gt},
		OpNeqPtr: {pointer, lt | gt},

		OpLess8:   {signed, lt},
		OpLess8U:  {unsigned, lt},
		OpLess16:  {signed, lt},
		OpLess16U: {unsigned, lt},
		OpLess32:  {signed, lt},
		OpLess32U: {unsigned, lt},
		OpLess64:  {signed, lt},
		OpLess64U: {unsigned, lt},

		OpLeq8:   {signed, lt | eq},
		OpLeq8U:  {unsigned, lt | eq},
		OpLeq16:  {signed, lt | eq},
		OpLeq16U: {unsigned, lt | eq},
		OpLeq32:  {signed, lt | eq},
		OpLeq32U: {unsigned, lt | eq},
		OpLeq64:  {signed, lt | eq},
		OpLeq64U: {unsigned, lt | eq},

		OpGeq8:   {signed, eq | gt},
		OpGeq8U:  {unsigned, eq | gt},
		OpGeq16:  {signed, eq | gt},
		OpGeq16U: {unsigned, eq | gt},
		OpGeq32:  {signed, eq | gt},
		OpGeq32U: {unsigned, eq | gt},
		OpGeq64:  {signed, eq | gt},
		OpGeq64U: {unsigned, eq | gt},

		OpGreater8:   {signed, gt},
		OpGreater8U:  {unsigned, gt},
		OpGreater16:  {signed, gt},
		OpGreater16U: {unsigned, gt},
		OpGreater32:  {signed, gt},
		OpGreater32U: {unsigned, gt},
		OpGreater64:  {signed, gt},
		OpGreater64U: {unsigned, gt},

		// TODO: OpIsInBounds actually test 0 <= a < b. This means
		// that the positive branch learns signed/LT and unsigned/LT
		// but the negative branch only learns unsigned/GE.
		OpIsInBounds:      {unsigned, lt},
		OpIsSliceInBounds: {unsigned, lt | eq},
	}
)

// prove removes redundant BlockIf controls that can be inferred in a straight line.
//
// By far, the most common redundant pair are generated by bounds checking.
// For example for the code:
//
//    a[i] = 4
//    foo(a[i])
//
// The compiler will generate the following code:
//
//    if i >= len(a) {
//        panic("not in bounds")
//    }
//    a[i] = 4
//    if i >= len(a) {
//        panic("not in bounds")
//    }
//    foo(a[i])
//
// The second comparison i >= len(a) is clearly redundant because if the
// else branch of the first comparison is executed, we already know that i < len(a).
// The code for the second panic can be removed.
func prove(f *Func) {
	// current node state
	type walkState int
	const (
		descend walkState = iota
		simplify
	)
	// work maintains the DFS stack.
	type bp struct {
		block *Block    // current handled block
		state walkState // what's to do
	}
	work := make([]bp, 0, 256)
	work = append(work, bp{
		block: f.Entry,
		state: descend,
	})

	ft := newFactsTable()

	// DFS on the dominator tree.
	for len(work) > 0 {
		node := work[len(work)-1]
		work = work[:len(work)-1]
		parent := f.idom[node.block.ID]
		branch := getBranch(f.sdom, parent, node.block)

		switch node.state {
		case descend:
			if branch != unknown {
				ft.checkpoint()
				c := parent.Control
				updateRestrictions(ft, boolean, nil, c, lt|gt, branch)
				if tr, has := domainRelationTable[parent.Control.Op]; has {
					// When we branched from parent we learned a new set of
					// restrictions. Update the factsTable accordingly.
					updateRestrictions(ft, tr.d, c.Args[0], c.Args[1], tr.r, branch)
				}
			}

			work = append(work, bp{
				block: node.block,
				state: simplify,
			})
			for s := f.sdom.Child(node.block); s != nil; s = f.sdom.Sibling(s) {
				work = append(work, bp{
					block: s,
					state: descend,
				})
			}

		case simplify:
			succ := simplifyBlock(ft, node.block)
			if succ != unknown {
				b := node.block
				b.Kind = BlockFirst
				b.SetControl(nil)
				if succ == negative {
					b.swapSuccessors()
				}
			}

			if branch != unknown {
				ft.restore()
			}
		}
	}
}

// getBranch returns the range restrictions added by p
// when reaching b. p is the immediate dominator of b.
func getBranch(sdom SparseTree, p *Block, b *Block) branch {
	if p == nil || p.Kind != BlockIf {
		return unknown
	}
	// If p and p.Succs[0] are dominators it means that every path
	// from entry to b passes through p and p.Succs[0]. We care that
	// no path from entry to b passes through p.Succs[1]. If p.Succs[0]
	// has one predecessor then (apart from the degenerate case),
	// there is no path from entry that can reach b through p.Succs[1].
	// TODO: how about p->yes->b->yes, i.e. a loop in yes.
	if sdom.isAncestorEq(p.Succs[0].b, b) && len(p.Succs[0].b.Preds) == 1 {
		return positive
	}
	if sdom.isAncestorEq(p.Succs[1].b, b) && len(p.Succs[1].b.Preds) == 1 {
		return negative
	}
	return unknown
}

// updateRestrictions updates restrictions from the immediate
// dominating block (p) using r. r is adjusted according to the branch taken.
func updateRestrictions(ft *factsTable, t domain, v, w *Value, r relation, branch branch) {
	if t == 0 || branch == unknown {
		// Trivial case: nothing to do, or branch unknown.
		// Shoult not happen, but just in case.
		return
	}
	if branch == negative {
		// Negative branch taken, complement the relations.
		r = (lt | eq | gt) ^ r
	}
	for i := domain(1); i <= t; i <<= 1 {
		if t&i != 0 {
			ft.update(v, w, i, r)
		}
	}
}

// simplifyBlock simplifies block known the restrictions in ft.
// Returns which branch must always be taken.
func simplifyBlock(ft *factsTable, b *Block) branch {
	if b.Kind != BlockIf {
		return unknown
	}

	// First, checks if the condition itself is redundant.
	m := ft.get(nil, b.Control, boolean)
	if m == lt|gt {
		if b.Func.pass.debug > 0 {
			b.Func.Config.Warnl(b.Line, "Proved boolean %s", b.Control.Op)
		}
		return positive
	}
	if m == eq {
		if b.Func.pass.debug > 0 {
			b.Func.Config.Warnl(b.Line, "Disproved boolean %s", b.Control.Op)
		}
		return negative
	}

	// Next look check equalities.
	c := b.Control
	tr, has := domainRelationTable[c.Op]
	if !has {
		return unknown
	}

	a0, a1 := c.Args[0], c.Args[1]
	for d := domain(1); d <= tr.d; d <<= 1 {
		if d&tr.d == 0 {
			continue
		}

		// tr.r represents in which case the positive branch is taken.
		// m represents which cases are possible because of previous relations.
		// If the set of possible relations m is included in the set of relations
		// need to take the positive branch (or negative) then that branch will
		// always be taken.
		// For shortcut, if m == 0 then this block is dead code.
		m := ft.get(a0, a1, d)
		if m != 0 && tr.r&m == m {
			if b.Func.pass.debug > 0 {
				b.Func.Config.Warnl(b.Line, "Proved %s", c.Op)
			}
			return positive
		}
		if m != 0 && ((lt|eq|gt)^tr.r)&m == m {
			if b.Func.pass.debug > 0 {
				b.Func.Config.Warnl(b.Line, "Disproved %s", c.Op)
			}
			return negative
		}
	}

	// HACK: If the first argument of IsInBounds or IsSliceInBounds
	// is a constant and we already know that constant is smaller (or equal)
	// to the upper bound than this is proven. Most useful in cases such as:
	// if len(a) <= 1 { return }
	// do something with a[1]
	if (c.Op == OpIsInBounds || c.Op == OpIsSliceInBounds) && ft.isNonNegative(c.Args[0]) {
		m := ft.get(a0, a1, signed)
		if m != 0 && tr.r&m == m {
			if b.Func.pass.debug > 0 {
				b.Func.Config.Warnl(b.Line, "Proved non-negative bounds %s", c.Op)
			}
			return positive
		}
	}

	return unknown
}

// isNonNegative returns true is v is known to be greater or equal to zero.
func isNonNegative(v *Value) bool {
	switch v.Op {
	case OpConst64:
		return v.AuxInt >= 0

	case OpStringLen, OpSliceLen, OpSliceCap,
		OpZeroExt8to64, OpZeroExt16to64, OpZeroExt32to64:
		return true

	case OpRsh64x64:
		return isNonNegative(v.Args[0])
	}
	return false
}
