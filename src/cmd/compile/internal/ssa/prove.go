// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/internal/src"
	"fmt"
	"math"
)

type branch int

const (
	unknown branch = iota
	positive
	negative
	// The outedges from a jump table are jumpTable0,
	// jumpTable0+1, jumpTable0+2, etc. There could be an
	// arbitrary number so we can't list them all here.
	jumpTable0
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
//	r := relation(...)
//
//	if v < w {
//	  newR := r & lt
//	}
//	if v >= w {
//	  newR := r & (eq|gt)
//	}
//	if v != w {
//	  newR := r & (lt|gt)
//	}
type relation uint

const (
	lt relation = 1 << iota
	eq
	gt
)

var relationStrings = [...]string{
	0: "none", lt: "<", eq: "==", lt | eq: "<=",
	gt: ">", gt | lt: "!=", gt | eq: ">=", gt | eq | lt: "any",
}

func (r relation) String() string {
	if r < relation(len(relationStrings)) {
		return relationStrings[r]
	}
	return fmt.Sprintf("relation(%d)", uint(r))
}

// domain represents the domain of a variable pair in which a set
// of relations is known. For example, relations learned for unsigned
// pairs cannot be transferred to signed pairs because the same bit
// representation can mean something else.
type domain uint

const (
	signed domain = 1 << iota
	unsigned
	pointer
	boolean
)

var domainStrings = [...]string{
	"signed", "unsigned", "pointer", "boolean",
}

func (d domain) String() string {
	s := ""
	for i, ds := range domainStrings {
		if d&(1<<uint(i)) != 0 {
			if len(s) != 0 {
				s += "|"
			}
			s += ds
			d &^= 1 << uint(i)
		}
	}
	if d != 0 {
		if len(s) != 0 {
			s += "|"
		}
		s += fmt.Sprintf("0x%x", uint(d))
	}
	return s
}

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

func (l limit) String() string {
	return fmt.Sprintf("sm,SM,um,UM=%d,%d,%d,%d", l.min, l.max, l.umin, l.umax)
}

func (l limit) intersect(l2 limit) limit {
	if l.min < l2.min {
		l.min = l2.min
	}
	if l.umin < l2.umin {
		l.umin = l2.umin
	}
	if l.max > l2.max {
		l.max = l2.max
	}
	if l.umax > l2.umax {
		l.umax = l2.umax
	}
	return l
}

var noLimit = limit{math.MinInt64, math.MaxInt64, 0, math.MaxUint64}

// a limitFact is a limit known for a particular value.
type limitFact struct {
	vid   ID
	limit limit
}

// factsTable keeps track of relations between pairs of values.
//
// The fact table logic is sound, but incomplete. Outside of a few
// special cases, it performs no deduction or arithmetic. While there
// are known decision procedures for this, the ad hoc approach taken
// by the facts table is effective for real code while remaining very
// efficient.
type factsTable struct {
	// unsat is true if facts contains a contradiction.
	//
	// Note that the factsTable logic is incomplete, so if unsat
	// is false, the assertions in factsTable could be satisfiable
	// *or* unsatisfiable.
	unsat      bool // true if facts contains a contradiction
	unsatDepth int  // number of unsat checkpoints

	facts map[pair]relation // current known set of relation
	stack []fact            // previous sets of relations

	// order is a couple of partial order sets that record information
	// about relations between SSA values in the signed and unsigned
	// domain.
	orderS *poset
	orderU *poset

	// known lower and upper bounds on individual values.
	limits     map[ID]limit
	limitStack []limitFact // previous entries

	// For each slice s, a map from s to a len(s)/cap(s) value (if any)
	// TODO: check if there are cases that matter where we have
	// more than one len(s) for a slice. We could keep a list if necessary.
	lens map[ID]*Value
	caps map[ID]*Value

	// zero is a zero-valued constant
	zero *Value
}

// checkpointFact is an invalid value used for checkpointing
// and restoring factsTable.
var checkpointFact = fact{}
var checkpointBound = limitFact{}

func newFactsTable(f *Func) *factsTable {
	ft := &factsTable{}
	ft.orderS = f.newPoset()
	ft.orderU = f.newPoset()
	ft.orderS.SetUnsigned(false)
	ft.orderU.SetUnsigned(true)
	ft.facts = make(map[pair]relation)
	ft.stack = make([]fact, 4)
	ft.limits = make(map[ID]limit)
	ft.limitStack = make([]limitFact, 4)
	ft.zero = f.ConstInt64(f.Config.Types.Int64, 0)
	return ft
}

// update updates the set of relations between v and w in domain d
// restricting it to r.
func (ft *factsTable) update(parent *Block, v, w *Value, d domain, r relation) {
	if parent.Func.pass.debug > 2 {
		parent.Func.Warnl(parent.Pos, "parent=%s, update %s %s %s", parent, v, w, r)
	}
	// No need to do anything else if we already found unsat.
	if ft.unsat {
		return
	}

	// Self-fact. It's wasteful to register it into the facts
	// table, so just note whether it's satisfiable
	if v == w {
		if r&eq == 0 {
			ft.unsat = true
		}
		return
	}

	if d == signed || d == unsigned {
		var ok bool
		order := ft.orderS
		if d == unsigned {
			order = ft.orderU
		}
		switch r {
		case lt:
			ok = order.SetOrder(v, w)
		case gt:
			ok = order.SetOrder(w, v)
		case lt | eq:
			ok = order.SetOrderOrEqual(v, w)
		case gt | eq:
			ok = order.SetOrderOrEqual(w, v)
		case eq:
			ok = order.SetEqual(v, w)
		case lt | gt:
			ok = order.SetNonEqual(v, w)
		default:
			panic("unknown relation")
		}
		if !ok {
			if parent.Func.pass.debug > 2 {
				parent.Func.Warnl(parent.Pos, "unsat %s %s %s", v, w, r)
			}
			ft.unsat = true
			return
		}
	} else {
		if lessByID(w, v) {
			v, w = w, v
			r = reverseBits[r]
		}

		p := pair{v, w, d}
		oldR, ok := ft.facts[p]
		if !ok {
			if v == w {
				oldR = eq
			} else {
				oldR = lt | eq | gt
			}
		}
		// No changes compared to information already in facts table.
		if oldR == r {
			return
		}
		ft.stack = append(ft.stack, fact{p, oldR})
		ft.facts[p] = oldR & r
		// If this relation is not satisfiable, mark it and exit right away
		if oldR&r == 0 {
			if parent.Func.pass.debug > 2 {
				parent.Func.Warnl(parent.Pos, "unsat %s %s %s", v, w, r)
			}
			ft.unsat = true
			return
		}
	}

	// Extract bounds when comparing against constants
	if v.isGenericIntConst() {
		v, w = w, v
		r = reverseBits[r]
	}
	if v != nil && w.isGenericIntConst() {
		// Note: all the +1/-1 below could overflow/underflow. Either will
		// still generate correct results, it will just lead to imprecision.
		// In fact if there is overflow/underflow, the corresponding
		// code is unreachable because the known range is outside the range
		// of the value's type.
		old, ok := ft.limits[v.ID]
		if !ok {
			old = noLimit
			if v.isGenericIntConst() {
				switch d {
				case signed:
					old.min, old.max = v.AuxInt, v.AuxInt
					if v.AuxInt >= 0 {
						old.umin, old.umax = uint64(v.AuxInt), uint64(v.AuxInt)
					}
				case unsigned:
					old.umin = v.AuxUnsigned()
					old.umax = old.umin
					if int64(old.umin) >= 0 {
						old.min, old.max = int64(old.umin), int64(old.umin)
					}
				}
			}
		}
		lim := noLimit
		switch d {
		case signed:
			c := w.AuxInt
			switch r {
			case lt:
				lim.max = c - 1
			case lt | eq:
				lim.max = c
			case gt | eq:
				lim.min = c
			case gt:
				lim.min = c + 1
			case lt | gt:
				lim = old
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
			if lim.min >= 0 {
				// int(x) >= 0 && int(x) >= N  ⇒  uint(x) >= N
				lim.umin = uint64(lim.min)
			}
			if lim.max != noLimit.max && old.min >= 0 && lim.max >= 0 {
				// 0 <= int(x) <= N  ⇒  0 <= uint(x) <= N
				// This is for a max update, so the lower bound
				// comes from what we already know (old).
				lim.umax = uint64(lim.max)
			}
		case unsigned:
			uc := w.AuxUnsigned()
			switch r {
			case lt:
				lim.umax = uc - 1
			case lt | eq:
				lim.umax = uc
			case gt | eq:
				lim.umin = uc
			case gt:
				lim.umin = uc + 1
			case lt | gt:
				lim = old
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
			// We could use the contrapositives of the
			// signed implications to derive signed facts,
			// but it turns out not to matter.
		}
		ft.limitStack = append(ft.limitStack, limitFact{v.ID, old})
		lim = old.intersect(lim)
		ft.limits[v.ID] = lim
		if v.Block.Func.pass.debug > 2 {
			v.Block.Func.Warnl(parent.Pos, "parent=%s, new limits %s %s %s %s", parent, v, w, r, lim.String())
		}
		if lim.min > lim.max || lim.umin > lim.umax {
			ft.unsat = true
			return
		}
	}

	// Derived facts below here are only about numbers.
	if d != signed && d != unsigned {
		return
	}

	// Additional facts we know given the relationship between len and cap.
	//
	// TODO: Since prove now derives transitive relations, it
	// should be sufficient to learn that len(w) <= cap(w) at the
	// beginning of prove where we look for all len/cap ops.
	if v.Op == OpSliceLen && r&lt == 0 && ft.caps[v.Args[0].ID] != nil {
		// len(s) > w implies cap(s) > w
		// len(s) >= w implies cap(s) >= w
		// len(s) == w implies cap(s) >= w
		ft.update(parent, ft.caps[v.Args[0].ID], w, d, r|gt)
	}
	if w.Op == OpSliceLen && r&gt == 0 && ft.caps[w.Args[0].ID] != nil {
		// same, length on the RHS.
		ft.update(parent, v, ft.caps[w.Args[0].ID], d, r|lt)
	}
	if v.Op == OpSliceCap && r&gt == 0 && ft.lens[v.Args[0].ID] != nil {
		// cap(s) < w implies len(s) < w
		// cap(s) <= w implies len(s) <= w
		// cap(s) == w implies len(s) <= w
		ft.update(parent, ft.lens[v.Args[0].ID], w, d, r|lt)
	}
	if w.Op == OpSliceCap && r&lt == 0 && ft.lens[w.Args[0].ID] != nil {
		// same, capacity on the RHS.
		ft.update(parent, v, ft.lens[w.Args[0].ID], d, r|gt)
	}

	// Process fence-post implications.
	//
	// First, make the condition > or >=.
	if r == lt || r == lt|eq {
		v, w = w, v
		r = reverseBits[r]
	}
	switch r {
	case gt:
		if x, delta := isConstDelta(v); x != nil && delta == 1 {
			// x+1 > w  ⇒  x >= w
			//
			// This is useful for eliminating the
			// growslice branch of append.
			ft.update(parent, x, w, d, gt|eq)
		} else if x, delta := isConstDelta(w); x != nil && delta == -1 {
			// v > x-1  ⇒  v >= x
			ft.update(parent, v, x, d, gt|eq)
		}
	case gt | eq:
		if x, delta := isConstDelta(v); x != nil && delta == -1 {
			// x-1 >= w && x > min  ⇒  x > w
			//
			// Useful for i > 0; s[i-1].
			lim, ok := ft.limits[x.ID]
			if ok && ((d == signed && lim.min > opMin[v.Op]) || (d == unsigned && lim.umin > 0)) {
				ft.update(parent, x, w, d, gt)
			}
		} else if x, delta := isConstDelta(w); x != nil && delta == 1 {
			// v >= x+1 && x < max  ⇒  v > x
			lim, ok := ft.limits[x.ID]
			if ok && ((d == signed && lim.max < opMax[w.Op]) || (d == unsigned && lim.umax < opUMax[w.Op])) {
				ft.update(parent, v, x, d, gt)
			}
		}
	}

	// Process: x+delta > w (with delta constant)
	// Only signed domain for now (useful for accesses to slices in loops).
	if r == gt || r == gt|eq {
		if x, delta := isConstDelta(v); x != nil && d == signed {
			if parent.Func.pass.debug > 1 {
				parent.Func.Warnl(parent.Pos, "x+d %s w; x:%v %v delta:%v w:%v d:%v", r, x, parent.String(), delta, w.AuxInt, d)
			}
			underflow := true
			if l, has := ft.limits[x.ID]; has && delta < 0 {
				if (x.Type.Size() == 8 && l.min >= math.MinInt64-delta) ||
					(x.Type.Size() == 4 && l.min >= math.MinInt32-delta) {
					underflow = false
				}
			}
			if delta < 0 && !underflow {
				// If delta < 0 and x+delta cannot underflow then x > x+delta (that is, x > v)
				ft.update(parent, x, v, signed, gt)
			}
			if !w.isGenericIntConst() {
				// If we know that x+delta > w but w is not constant, we can derive:
				//    if delta < 0 and x+delta cannot underflow, then x > w
				// This is useful for loops with bounds "len(slice)-K" (delta = -K)
				if delta < 0 && !underflow {
					ft.update(parent, x, w, signed, r)
				}
			} else {
				// With w,delta constants, we want to derive: x+delta > w  ⇒  x > w-delta
				//
				// We compute (using integers of the correct size):
				//    min = w - delta
				//    max = MaxInt - delta
				//
				// And we prove that:
				//    if min<max: min < x AND x <= max
				//    if min>max: min < x OR  x <= max
				//
				// This is always correct, even in case of overflow.
				//
				// If the initial fact is x+delta >= w instead, the derived conditions are:
				//    if min<max: min <= x AND x <= max
				//    if min>max: min <= x OR  x <= max
				//
				// Notice the conditions for max are still <=, as they handle overflows.
				var min, max int64
				var vmin, vmax *Value
				switch x.Type.Size() {
				case 8:
					min = w.AuxInt - delta
					max = int64(^uint64(0)>>1) - delta

					vmin = parent.NewValue0I(parent.Pos, OpConst64, parent.Func.Config.Types.Int64, min)
					vmax = parent.NewValue0I(parent.Pos, OpConst64, parent.Func.Config.Types.Int64, max)

				case 4:
					min = int64(int32(w.AuxInt) - int32(delta))
					max = int64(int32(^uint32(0)>>1) - int32(delta))

					vmin = parent.NewValue0I(parent.Pos, OpConst32, parent.Func.Config.Types.Int32, min)
					vmax = parent.NewValue0I(parent.Pos, OpConst32, parent.Func.Config.Types.Int32, max)

				default:
					panic("unimplemented")
				}

				if min < max {
					// Record that x > min and max >= x
					ft.update(parent, x, vmin, d, r)
					ft.update(parent, vmax, x, d, r|eq)
				} else {
					// We know that either x>min OR x<=max. factsTable cannot record OR conditions,
					// so let's see if we can already prove that one of them is false, in which case
					// the other must be true
					if l, has := ft.limits[x.ID]; has {
						if l.max <= min {
							if r&eq == 0 || l.max < min {
								// x>min (x>=min) is impossible, so it must be x<=max
								ft.update(parent, vmax, x, d, r|eq)
							}
						} else if l.min > max {
							// x<=max is impossible, so it must be x>min
							ft.update(parent, x, vmin, d, r)
						}
					}
				}
			}
		}
	}

	// Look through value-preserving extensions.
	// If the domain is appropriate for the pre-extension Type,
	// repeat the update with the pre-extension Value.
	if isCleanExt(v) {
		switch {
		case d == signed && v.Args[0].Type.IsSigned():
			fallthrough
		case d == unsigned && !v.Args[0].Type.IsSigned():
			ft.update(parent, v.Args[0], w, d, r)
		}
	}
	if isCleanExt(w) {
		switch {
		case d == signed && w.Args[0].Type.IsSigned():
			fallthrough
		case d == unsigned && !w.Args[0].Type.IsSigned():
			ft.update(parent, v, w.Args[0], d, r)
		}
	}
}

var opMin = map[Op]int64{
	OpAdd64: math.MinInt64, OpSub64: math.MinInt64,
	OpAdd32: math.MinInt32, OpSub32: math.MinInt32,
}

var opMax = map[Op]int64{
	OpAdd64: math.MaxInt64, OpSub64: math.MaxInt64,
	OpAdd32: math.MaxInt32, OpSub32: math.MaxInt32,
}

var opUMax = map[Op]uint64{
	OpAdd64: math.MaxUint64, OpSub64: math.MaxUint64,
	OpAdd32: math.MaxUint32, OpSub32: math.MaxUint32,
}

// isNonNegative reports whether v is known to be non-negative.
func (ft *factsTable) isNonNegative(v *Value) bool {
	if isNonNegative(v) {
		return true
	}

	var max int64
	switch v.Type.Size() {
	case 1:
		max = math.MaxInt8
	case 2:
		max = math.MaxInt16
	case 4:
		max = math.MaxInt32
	case 8:
		max = math.MaxInt64
	default:
		panic("unexpected integer size")
	}

	// Check if the recorded limits can prove that the value is positive

	if l, has := ft.limits[v.ID]; has && (l.min >= 0 || l.umax <= uint64(max)) {
		return true
	}

	// Check if v = x+delta, and we can use x's limits to prove that it's positive
	if x, delta := isConstDelta(v); x != nil {
		if l, has := ft.limits[x.ID]; has {
			if delta > 0 && l.min >= -delta && l.max <= max-delta {
				return true
			}
			if delta < 0 && l.min >= -delta {
				return true
			}
		}
	}

	// Check if v is a value-preserving extension of a non-negative value.
	if isCleanExt(v) && ft.isNonNegative(v.Args[0]) {
		return true
	}

	// Check if the signed poset can prove that the value is >= 0
	return ft.orderS.OrderedOrEqual(ft.zero, v)
}

// checkpoint saves the current state of known relations.
// Called when descending on a branch.
func (ft *factsTable) checkpoint() {
	if ft.unsat {
		ft.unsatDepth++
	}
	ft.stack = append(ft.stack, checkpointFact)
	ft.limitStack = append(ft.limitStack, checkpointBound)
	ft.orderS.Checkpoint()
	ft.orderU.Checkpoint()
}

// restore restores known relation to the state just
// before the previous checkpoint.
// Called when backing up on a branch.
func (ft *factsTable) restore() {
	if ft.unsatDepth > 0 {
		ft.unsatDepth--
	} else {
		ft.unsat = false
	}
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
	ft.orderS.Undo()
	ft.orderU.Undo()
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
	// If v1 branch is taken then we learn that the rangeMask
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

		// For these ops, the negative branch is different: we can only
		// prove signed/GE (signed/GT) if we can prove that arg0 is non-negative.
		// See the special case in addBranchRestrictions.
		OpIsInBounds:      {signed | unsigned, lt},      // 0 <= arg0 < arg1
		OpIsSliceInBounds: {signed | unsigned, lt | eq}, // 0 <= arg0 <= arg1
	}
)

// cleanup returns the posets to the free list
func (ft *factsTable) cleanup(f *Func) {
	for _, po := range []*poset{ft.orderS, ft.orderU} {
		// Make sure it's empty as it should be. A non-empty poset
		// might cause errors and miscompilations if reused.
		if checkEnabled {
			if err := po.CheckEmpty(); err != nil {
				f.Fatalf("poset not empty after function %s: %v", f.Name, err)
			}
		}
		f.retPoset(po)
	}
}

// prove removes redundant BlockIf branches that can be inferred
// from previous dominating comparisons.
//
// By far, the most common redundant pair are generated by bounds checking.
// For example for the code:
//
//	a[i] = 4
//	foo(a[i])
//
// The compiler will generate the following code:
//
//	if i >= len(a) {
//	    panic("not in bounds")
//	}
//	a[i] = 4
//	if i >= len(a) {
//	    panic("not in bounds")
//	}
//	foo(a[i])
//
// The second comparison i >= len(a) is clearly redundant because if the
// else branch of the first comparison is executed, we already know that i < len(a).
// The code for the second panic can be removed.
//
// prove works by finding contradictions and trimming branches whose
// conditions are unsatisfiable given the branches leading up to them.
// It tracks a "fact table" of branch conditions. For each branching
// block, it asserts the branch conditions that uniquely dominate that
// block, and then separately asserts the block's branch condition and
// its negation. If either leads to a contradiction, it can trim that
// successor.
func prove(f *Func) {
	ft := newFactsTable(f)
	ft.checkpoint()

	var lensVars map[*Block][]*Value

	// Find length and capacity ops.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Uses == 0 {
				// We don't care about dead values.
				// (There can be some that are CSEd but not removed yet.)
				continue
			}
			switch v.Op {
			case OpStringLen:
				ft.update(b, v, ft.zero, signed, gt|eq)
			case OpSliceLen:
				if ft.lens == nil {
					ft.lens = map[ID]*Value{}
				}
				// Set all len Values for the same slice as equal in the poset.
				// The poset handles transitive relations, so Values related to
				// any OpSliceLen for this slice will be correctly related to others.
				if l, ok := ft.lens[v.Args[0].ID]; ok {
					ft.update(b, v, l, signed, eq)
				} else {
					ft.lens[v.Args[0].ID] = v
				}
				ft.update(b, v, ft.zero, signed, gt|eq)
				if v.Args[0].Op == OpSliceMake {
					if lensVars == nil {
						lensVars = make(map[*Block][]*Value)
					}
					lensVars[b] = append(lensVars[b], v)
				}
			case OpSliceCap:
				if ft.caps == nil {
					ft.caps = map[ID]*Value{}
				}
				// Same as case OpSliceLen above, but for slice cap.
				if c, ok := ft.caps[v.Args[0].ID]; ok {
					ft.update(b, v, c, signed, eq)
				} else {
					ft.caps[v.Args[0].ID] = v
				}
				ft.update(b, v, ft.zero, signed, gt|eq)
				if v.Args[0].Op == OpSliceMake {
					if lensVars == nil {
						lensVars = make(map[*Block][]*Value)
					}
					lensVars[b] = append(lensVars[b], v)
				}
			case OpCtz64, OpCtz32, OpCtz16, OpCtz8, OpBitLen64, OpBitLen32, OpBitLen16, OpBitLen8:
				ft.update(b, v, ft.zero, signed, gt|eq)
				// TODO: we could also do <= 64/32/16/8, if that helped.
			case OpAnd64, OpAnd32, OpAnd16, OpAnd8:
				ft.update(b, v, v.Args[1], unsigned, lt|eq)
				ft.update(b, v, v.Args[0], unsigned, lt|eq)
			case OpOr64, OpOr32, OpOr16, OpOr8:
				ft.update(b, v, v.Args[1], unsigned, gt|eq)
				ft.update(b, v, v.Args[0], unsigned, gt|eq)
			}
		}
	}

	// Find induction variables. Currently, findIndVars
	// is limited to one induction variable per block.
	var indVars map[*Block]indVar
	for _, v := range findIndVar(f) {
		if indVars == nil {
			indVars = make(map[*Block]indVar)
		}
		indVars[v.entry] = v
	}

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

	idom := f.Idom()
	sdom := f.Sdom()

	// DFS on the dominator tree.
	//
	// For efficiency, we consider only the dominator tree rather
	// than the entire flow graph. On the way down, we consider
	// incoming branches and accumulate conditions that uniquely
	// dominate the current block. If we discover a contradiction,
	// we can eliminate the entire block and all of its children.
	// On the way back up, we consider outgoing branches that
	// haven't already been considered. This way we consider each
	// branch condition only once.
	for len(work) > 0 {
		node := work[len(work)-1]
		work = work[:len(work)-1]
		parent := idom[node.block.ID]
		branch := getBranch(sdom, parent, node.block)

		switch node.state {
		case descend:
			ft.checkpoint()

			// Entering the block, add the block-depending facts that we collected
			// at the beginning: induction variables and lens/caps of slices.
			if iv, ok := indVars[node.block]; ok {
				addIndVarRestrictions(ft, parent, iv)
			}
			if lens, ok := lensVars[node.block]; ok {
				for _, v := range lens {
					switch v.Op {
					case OpSliceLen:
						ft.update(node.block, v, v.Args[0].Args[1], signed, eq)
					case OpSliceCap:
						ft.update(node.block, v, v.Args[0].Args[2], signed, eq)
					}
				}
			}

			if branch != unknown {
				addBranchRestrictions(ft, parent, branch)
				if ft.unsat {
					// node.block is unreachable.
					// Remove it and don't visit
					// its children.
					removeBranch(parent, branch)
					ft.restore()
					break
				}
				// Otherwise, we can now commit to
				// taking this branch. We'll restore
				// ft when we unwind.
			}

			// Add inductive facts for phis in this block.
			addLocalInductiveFacts(ft, node.block)

			work = append(work, bp{
				block: node.block,
				state: simplify,
			})
			for s := sdom.Child(node.block); s != nil; s = sdom.Sibling(s) {
				work = append(work, bp{
					block: s,
					state: descend,
				})
			}

		case simplify:
			simplifyBlock(sdom, ft, node.block)
			ft.restore()
		}
	}

	ft.restore()

	ft.cleanup(f)
}

// getBranch returns the range restrictions added by p
// when reaching b. p is the immediate dominator of b.
func getBranch(sdom SparseTree, p *Block, b *Block) branch {
	if p == nil {
		return unknown
	}
	switch p.Kind {
	case BlockIf:
		// If p and p.Succs[0] are dominators it means that every path
		// from entry to b passes through p and p.Succs[0]. We care that
		// no path from entry to b passes through p.Succs[1]. If p.Succs[0]
		// has one predecessor then (apart from the degenerate case),
		// there is no path from entry that can reach b through p.Succs[1].
		// TODO: how about p->yes->b->yes, i.e. a loop in yes.
		if sdom.IsAncestorEq(p.Succs[0].b, b) && len(p.Succs[0].b.Preds) == 1 {
			return positive
		}
		if sdom.IsAncestorEq(p.Succs[1].b, b) && len(p.Succs[1].b.Preds) == 1 {
			return negative
		}
	case BlockJumpTable:
		// TODO: this loop can lead to quadratic behavior, as
		// getBranch can be called len(p.Succs) times.
		for i, e := range p.Succs {
			if sdom.IsAncestorEq(e.b, b) && len(e.b.Preds) == 1 {
				return jumpTable0 + branch(i)
			}
		}
	}
	return unknown
}

// addIndVarRestrictions updates the factsTables ft with the facts
// learned from the induction variable indVar which drives the loop
// starting in Block b.
func addIndVarRestrictions(ft *factsTable, b *Block, iv indVar) {
	d := signed
	if ft.isNonNegative(iv.min) && ft.isNonNegative(iv.max) {
		d |= unsigned
	}

	if iv.flags&indVarMinExc == 0 {
		addRestrictions(b, ft, d, iv.min, iv.ind, lt|eq)
	} else {
		addRestrictions(b, ft, d, iv.min, iv.ind, lt)
	}

	if iv.flags&indVarMaxInc == 0 {
		addRestrictions(b, ft, d, iv.ind, iv.max, lt)
	} else {
		addRestrictions(b, ft, d, iv.ind, iv.max, lt|eq)
	}
}

// addBranchRestrictions updates the factsTables ft with the facts learned when
// branching from Block b in direction br.
func addBranchRestrictions(ft *factsTable, b *Block, br branch) {
	c := b.Controls[0]
	switch {
	case br == negative:
		addRestrictions(b, ft, boolean, nil, c, eq)
	case br == positive:
		addRestrictions(b, ft, boolean, nil, c, lt|gt)
	case br >= jumpTable0:
		idx := br - jumpTable0
		val := int64(idx)
		if v, off := isConstDelta(c); v != nil {
			// Establish the bound on the underlying value we're switching on,
			// not on the offset-ed value used as the jump table index.
			c = v
			val -= off
		}
		old, ok := ft.limits[c.ID]
		if !ok {
			old = noLimit
		}
		ft.limitStack = append(ft.limitStack, limitFact{c.ID, old})
		if val < old.min || val > old.max || uint64(val) < old.umin || uint64(val) > old.umax {
			ft.unsat = true
			if b.Func.pass.debug > 2 {
				b.Func.Warnl(b.Pos, "block=%s outedge=%d %s=%d unsat", b, idx, c, val)
			}
		} else {
			ft.limits[c.ID] = limit{val, val, uint64(val), uint64(val)}
			if b.Func.pass.debug > 2 {
				b.Func.Warnl(b.Pos, "block=%s outedge=%d %s=%d", b, idx, c, val)
			}
		}
	default:
		panic("unknown branch")
	}
	if tr, has := domainRelationTable[c.Op]; has {
		// When we branched from parent we learned a new set of
		// restrictions. Update the factsTable accordingly.
		d := tr.d
		if d == signed && ft.isNonNegative(c.Args[0]) && ft.isNonNegative(c.Args[1]) {
			d |= unsigned
		}
		switch c.Op {
		case OpIsInBounds, OpIsSliceInBounds:
			// 0 <= a0 < a1 (or 0 <= a0 <= a1)
			//
			// On the positive branch, we learn:
			//   signed: 0 <= a0 < a1 (or 0 <= a0 <= a1)
			//   unsigned:    a0 < a1 (or a0 <= a1)
			//
			// On the negative branch, we learn (0 > a0 ||
			// a0 >= a1). In the unsigned domain, this is
			// simply a0 >= a1 (which is the reverse of the
			// positive branch, so nothing surprising).
			// But in the signed domain, we can't express the ||
			// condition, so check if a0 is non-negative instead,
			// to be able to learn something.
			switch br {
			case negative:
				d = unsigned
				if ft.isNonNegative(c.Args[0]) {
					d |= signed
				}
				addRestrictions(b, ft, d, c.Args[0], c.Args[1], tr.r^(lt|gt|eq))
			case positive:
				addRestrictions(b, ft, signed, ft.zero, c.Args[0], lt|eq)
				addRestrictions(b, ft, d, c.Args[0], c.Args[1], tr.r)
			}
		default:
			switch br {
			case negative:
				addRestrictions(b, ft, d, c.Args[0], c.Args[1], tr.r^(lt|gt|eq))
			case positive:
				addRestrictions(b, ft, d, c.Args[0], c.Args[1], tr.r)
			}
		}

	}
}

// addRestrictions updates restrictions from the immediate
// dominating block (p) using r.
func addRestrictions(parent *Block, ft *factsTable, t domain, v, w *Value, r relation) {
	if t == 0 {
		// Trivial case: nothing to do.
		// Shoult not happen, but just in case.
		return
	}
	for i := domain(1); i <= t; i <<= 1 {
		if t&i == 0 {
			continue
		}
		ft.update(parent, v, w, i, r)
	}
}

// addLocalInductiveFacts adds inductive facts when visiting b, where
// b is a join point in a loop. In contrast with findIndVar, this
// depends on facts established for b, which is why it happens when
// visiting b.
//
// TODO: It would be nice to combine this with findIndVar.
func addLocalInductiveFacts(ft *factsTable, b *Block) {
	// This looks for a specific pattern of induction:
	//
	// 1. i1 = OpPhi(min, i2) in b
	// 2. i2 = i1 + 1
	// 3. i2 < max at exit from b.Preds[1]
	// 4. min < max
	//
	// If all of these conditions are true, then i1 < max and i1 >= min.

	// To ensure this is a loop header node.
	if len(b.Preds) != 2 {
		return
	}

	for _, i1 := range b.Values {
		if i1.Op != OpPhi {
			continue
		}

		// Check for conditions 1 and 2. This is easy to do
		// and will throw out most phis.
		min, i2 := i1.Args[0], i1.Args[1]
		if i1q, delta := isConstDelta(i2); i1q != i1 || delta != 1 {
			continue
		}

		// Try to prove condition 3. We can't just query the
		// fact table for this because we don't know what the
		// facts of b.Preds[1] are (in general, b.Preds[1] is
		// a loop-back edge, so we haven't even been there
		// yet). As a conservative approximation, we look for
		// this condition in the predecessor chain until we
		// hit a join point.
		uniquePred := func(b *Block) *Block {
			if len(b.Preds) == 1 {
				return b.Preds[0].b
			}
			return nil
		}
		pred, child := b.Preds[1].b, b
		for ; pred != nil; pred, child = uniquePred(pred), pred {
			if pred.Kind != BlockIf {
				continue
			}
			control := pred.Controls[0]

			br := unknown
			if pred.Succs[0].b == child {
				br = positive
			}
			if pred.Succs[1].b == child {
				if br != unknown {
					continue
				}
				br = negative
			}
			if br == unknown {
				continue
			}

			tr, has := domainRelationTable[control.Op]
			if !has {
				continue
			}
			r := tr.r
			if br == negative {
				// Negative branch taken to reach b.
				// Complement the relations.
				r = (lt | eq | gt) ^ r
			}

			// Check for i2 < max or max > i2.
			var max *Value
			if r == lt && control.Args[0] == i2 {
				max = control.Args[1]
			} else if r == gt && control.Args[1] == i2 {
				max = control.Args[0]
			} else {
				continue
			}

			// Check condition 4 now that we have a
			// candidate max. For this we can query the
			// fact table. We "prove" min < max by showing
			// that min >= max is unsat. (This may simply
			// compare two constants; that's fine.)
			ft.checkpoint()
			ft.update(b, min, max, tr.d, gt|eq)
			proved := ft.unsat
			ft.restore()

			if proved {
				// We know that min <= i1 < max.
				if b.Func.pass.debug > 0 {
					printIndVar(b, i1, min, max, 1, 0)
				}
				ft.update(b, min, i1, tr.d, lt|eq)
				ft.update(b, i1, max, tr.d, lt)
			}
		}
	}
}

var ctzNonZeroOp = map[Op]Op{OpCtz8: OpCtz8NonZero, OpCtz16: OpCtz16NonZero, OpCtz32: OpCtz32NonZero, OpCtz64: OpCtz64NonZero}
var mostNegativeDividend = map[Op]int64{
	OpDiv16: -1 << 15,
	OpMod16: -1 << 15,
	OpDiv32: -1 << 31,
	OpMod32: -1 << 31,
	OpDiv64: -1 << 63,
	OpMod64: -1 << 63}

// simplifyBlock simplifies some constant values in b and evaluates
// branches to non-uniquely dominated successors of b.
func simplifyBlock(sdom SparseTree, ft *factsTable, b *Block) {
	for _, v := range b.Values {
		switch v.Op {
		case OpSlicemask:
			// Replace OpSlicemask operations in b with constants where possible.
			x, delta := isConstDelta(v.Args[0])
			if x == nil {
				break
			}
			// slicemask(x + y)
			// if x is larger than -y (y is negative), then slicemask is -1.
			lim, ok := ft.limits[x.ID]
			if !ok {
				break
			}
			if lim.umin > uint64(-delta) {
				if v.Args[0].Op == OpAdd64 {
					v.reset(OpConst64)
				} else {
					v.reset(OpConst32)
				}
				if b.Func.pass.debug > 0 {
					b.Func.Warnl(v.Pos, "Proved slicemask not needed")
				}
				v.AuxInt = -1
			}
		case OpCtz8, OpCtz16, OpCtz32, OpCtz64:
			// On some architectures, notably amd64, we can generate much better
			// code for CtzNN if we know that the argument is non-zero.
			// Capture that information here for use in arch-specific optimizations.
			x := v.Args[0]
			lim, ok := ft.limits[x.ID]
			if !ok {
				break
			}
			if lim.umin > 0 || lim.min > 0 || lim.max < 0 {
				if b.Func.pass.debug > 0 {
					b.Func.Warnl(v.Pos, "Proved %v non-zero", v.Op)
				}
				v.Op = ctzNonZeroOp[v.Op]
			}
		case OpRsh8x8, OpRsh8x16, OpRsh8x32, OpRsh8x64,
			OpRsh16x8, OpRsh16x16, OpRsh16x32, OpRsh16x64,
			OpRsh32x8, OpRsh32x16, OpRsh32x32, OpRsh32x64,
			OpRsh64x8, OpRsh64x16, OpRsh64x32, OpRsh64x64:
			// Check whether, for a >> b, we know that a is non-negative
			// and b is all of a's bits except the MSB. If so, a is shifted to zero.
			bits := 8 * v.Type.Size()
			if v.Args[1].isGenericIntConst() && v.Args[1].AuxInt >= bits-1 && ft.isNonNegative(v.Args[0]) {
				if b.Func.pass.debug > 0 {
					b.Func.Warnl(v.Pos, "Proved %v shifts to zero", v.Op)
				}
				switch bits {
				case 64:
					v.reset(OpConst64)
				case 32:
					v.reset(OpConst32)
				case 16:
					v.reset(OpConst16)
				case 8:
					v.reset(OpConst8)
				default:
					panic("unexpected integer size")
				}
				v.AuxInt = 0
				break // Be sure not to fallthrough - this is no longer OpRsh.
			}
			// If the Rsh hasn't been replaced with 0, still check if it is bounded.
			fallthrough
		case OpLsh8x8, OpLsh8x16, OpLsh8x32, OpLsh8x64,
			OpLsh16x8, OpLsh16x16, OpLsh16x32, OpLsh16x64,
			OpLsh32x8, OpLsh32x16, OpLsh32x32, OpLsh32x64,
			OpLsh64x8, OpLsh64x16, OpLsh64x32, OpLsh64x64,
			OpRsh8Ux8, OpRsh8Ux16, OpRsh8Ux32, OpRsh8Ux64,
			OpRsh16Ux8, OpRsh16Ux16, OpRsh16Ux32, OpRsh16Ux64,
			OpRsh32Ux8, OpRsh32Ux16, OpRsh32Ux32, OpRsh32Ux64,
			OpRsh64Ux8, OpRsh64Ux16, OpRsh64Ux32, OpRsh64Ux64:
			// Check whether, for a << b, we know that b
			// is strictly less than the number of bits in a.
			by := v.Args[1]
			lim, ok := ft.limits[by.ID]
			if !ok {
				break
			}
			bits := 8 * v.Args[0].Type.Size()
			if lim.umax < uint64(bits) || (lim.max < bits && ft.isNonNegative(by)) {
				v.AuxInt = 1 // see shiftIsBounded
				if b.Func.pass.debug > 0 {
					b.Func.Warnl(v.Pos, "Proved %v bounded", v.Op)
				}
			}
		case OpDiv16, OpDiv32, OpDiv64, OpMod16, OpMod32, OpMod64:
			// On amd64 and 386 fix-up code can be avoided if we know
			//  the divisor is not -1 or the dividend > MinIntNN.
			// Don't modify AuxInt on other architectures,
			// as that can interfere with CSE.
			// TODO: add other architectures?
			if b.Func.Config.arch != "386" && b.Func.Config.arch != "amd64" {
				break
			}
			divr := v.Args[1]
			divrLim, divrLimok := ft.limits[divr.ID]
			divd := v.Args[0]
			divdLim, divdLimok := ft.limits[divd.ID]
			if (divrLimok && (divrLim.max < -1 || divrLim.min > -1)) ||
				(divdLimok && divdLim.min > mostNegativeDividend[v.Op]) {
				// See DivisionNeedsFixUp in rewrite.go.
				// v.AuxInt = 1 means we have proved both that the divisor is not -1
				// and that the dividend is not the most negative integer,
				// so we do not need to add fix-up code.
				v.AuxInt = 1
				if b.Func.pass.debug > 0 {
					b.Func.Warnl(v.Pos, "Proved %v does not need fix-up", v.Op)
				}
			}
		}
		// Fold provable constant results.
		// Helps in cases where we reuse a value after branching on its equality.
		for i, arg := range v.Args {
			switch arg.Op {
			case OpConst64, OpConst32, OpConst16, OpConst8:
				continue
			}
			lim, ok := ft.limits[arg.ID]
			if !ok {
				continue
			}

			var constValue int64
			typ := arg.Type
			bits := 8 * typ.Size()
			switch {
			case lim.min == lim.max:
				constValue = lim.min
			case lim.umin == lim.umax:
				// truncate then sign extand
				switch bits {
				case 64:
					constValue = int64(lim.umin)
				case 32:
					constValue = int64(int32(lim.umin))
				case 16:
					constValue = int64(int16(lim.umin))
				case 8:
					constValue = int64(int8(lim.umin))
				default:
					panic("unexpected integer size")
				}
			default:
				continue
			}
			var c *Value
			f := b.Func
			switch bits {
			case 64:
				c = f.ConstInt64(typ, constValue)
			case 32:
				c = f.ConstInt32(typ, int32(constValue))
			case 16:
				c = f.ConstInt16(typ, int16(constValue))
			case 8:
				c = f.ConstInt8(typ, int8(constValue))
			default:
				panic("unexpected integer size")
			}
			v.SetArg(i, c)
			if b.Func.pass.debug > 1 {
				b.Func.Warnl(v.Pos, "Proved %v's arg %d (%v) is constant %d", v, i, arg, constValue)
			}
		}
	}

	if b.Kind != BlockIf {
		return
	}

	// Consider outgoing edges from this block.
	parent := b
	for i, branch := range [...]branch{positive, negative} {
		child := parent.Succs[i].b
		if getBranch(sdom, parent, child) != unknown {
			// For edges to uniquely dominated blocks, we
			// already did this when we visited the child.
			continue
		}
		// For edges to other blocks, this can trim a branch
		// even if we couldn't get rid of the child itself.
		ft.checkpoint()
		addBranchRestrictions(ft, parent, branch)
		unsat := ft.unsat
		ft.restore()
		if unsat {
			// This branch is impossible, so remove it
			// from the block.
			removeBranch(parent, branch)
			// No point in considering the other branch.
			// (It *is* possible for both to be
			// unsatisfiable since the fact table is
			// incomplete. We could turn this into a
			// BlockExit, but it doesn't seem worth it.)
			break
		}
	}
}

func removeBranch(b *Block, branch branch) {
	c := b.Controls[0]
	if b.Func.pass.debug > 0 {
		verb := "Proved"
		if branch == positive {
			verb = "Disproved"
		}
		if b.Func.pass.debug > 1 {
			b.Func.Warnl(b.Pos, "%s %s (%s)", verb, c.Op, c)
		} else {
			b.Func.Warnl(b.Pos, "%s %s", verb, c.Op)
		}
	}
	if c != nil && c.Pos.IsStmt() == src.PosIsStmt && c.Pos.SameFileAndLine(b.Pos) {
		// attempt to preserve statement marker.
		b.Pos = b.Pos.WithIsStmt()
	}
	if branch == positive || branch == negative {
		b.Kind = BlockFirst
		b.ResetControls()
		if branch == positive {
			b.swapSuccessors()
		}
	} else {
		// TODO: figure out how to remove an entry from a jump table
	}
}

// isNonNegative reports whether v is known to be greater or equal to zero.
func isNonNegative(v *Value) bool {
	if !v.Type.IsInteger() {
		v.Fatalf("isNonNegative bad type: %v", v.Type)
	}
	// TODO: return true if !v.Type.IsSigned()
	// SSA isn't type-safe enough to do that now (issue 37753).
	// The checks below depend only on the pattern of bits.

	switch v.Op {
	case OpConst64:
		return v.AuxInt >= 0

	case OpConst32:
		return int32(v.AuxInt) >= 0

	case OpConst16:
		return int16(v.AuxInt) >= 0

	case OpConst8:
		return int8(v.AuxInt) >= 0

	case OpStringLen, OpSliceLen, OpSliceCap,
		OpZeroExt8to64, OpZeroExt16to64, OpZeroExt32to64,
		OpZeroExt8to32, OpZeroExt16to32, OpZeroExt8to16,
		OpCtz64, OpCtz32, OpCtz16, OpCtz8,
		OpCtz64NonZero, OpCtz32NonZero, OpCtz16NonZero, OpCtz8NonZero,
		OpBitLen64, OpBitLen32, OpBitLen16, OpBitLen8:
		return true

	case OpRsh64Ux64, OpRsh32Ux64:
		by := v.Args[1]
		return by.Op == OpConst64 && by.AuxInt > 0

	case OpRsh64x64, OpRsh32x64, OpRsh8x64, OpRsh16x64, OpRsh32x32, OpRsh64x32,
		OpSignExt32to64, OpSignExt16to64, OpSignExt8to64, OpSignExt16to32, OpSignExt8to32:
		return isNonNegative(v.Args[0])

	case OpAnd64, OpAnd32, OpAnd16, OpAnd8:
		return isNonNegative(v.Args[0]) || isNonNegative(v.Args[1])

	case OpMod64, OpMod32, OpMod16, OpMod8,
		OpDiv64, OpDiv32, OpDiv16, OpDiv8,
		OpOr64, OpOr32, OpOr16, OpOr8,
		OpXor64, OpXor32, OpXor16, OpXor8:
		return isNonNegative(v.Args[0]) && isNonNegative(v.Args[1])

		// We could handle OpPhi here, but the improvements from doing
		// so are very minor, and it is neither simple nor cheap.
	}
	return false
}

// isConstDelta returns non-nil if v is equivalent to w+delta (signed).
func isConstDelta(v *Value) (w *Value, delta int64) {
	cop := OpConst64
	switch v.Op {
	case OpAdd32, OpSub32:
		cop = OpConst32
	}
	switch v.Op {
	case OpAdd64, OpAdd32:
		if v.Args[0].Op == cop {
			return v.Args[1], v.Args[0].AuxInt
		}
		if v.Args[1].Op == cop {
			return v.Args[0], v.Args[1].AuxInt
		}
	case OpSub64, OpSub32:
		if v.Args[1].Op == cop {
			aux := v.Args[1].AuxInt
			if aux != -aux { // Overflow; too bad
				return v.Args[0], -aux
			}
		}
	}
	return nil, 0
}

// isCleanExt reports whether v is the result of a value-preserving
// sign or zero extension
func isCleanExt(v *Value) bool {
	switch v.Op {
	case OpSignExt8to16, OpSignExt8to32, OpSignExt8to64,
		OpSignExt16to32, OpSignExt16to64, OpSignExt32to64:
		// signed -> signed is the only value-preserving sign extension
		return v.Args[0].Type.IsSigned() && v.Type.IsSigned()

	case OpZeroExt8to16, OpZeroExt8to32, OpZeroExt8to64,
		OpZeroExt16to32, OpZeroExt16to64, OpZeroExt32to64:
		// unsigned -> signed/unsigned are value-preserving zero extensions
		return !v.Args[0].Type.IsSigned()
	}
	return false
}
