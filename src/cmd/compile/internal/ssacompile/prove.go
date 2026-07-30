// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"fmt"
	"math"
	"math/bits"
	"strings"

	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/block"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
	"cmd/internal/src"
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

func (b branch) String() string {
	switch b {
	case unknown:
		return "unk"
	case positive:
		return "pos"
	case negative:
		return "neg"
	default:
		return fmt.Sprintf("jmp%d", b-jumpTable0)
	}
}

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

// a limitFact is a limit known for a particular value.
type limitFact struct {
	vid   ssa.ID
	limit ssa.Limit
}

// An ordering encodes facts like v < w.
type ordering struct {
	next *ordering // linked list of all known orderings for v.
	// Note: v is implicit here, determined by which linked list it is in.
	w *ssa.Value
	d domain
	r relation // one of ==,!=,<,<=,>,>=
	// if d is boolean or pointer, r can only be ==, !=
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

	// order* is a couple of partial order sets that record information
	// about relations between SSA values in the signed and unsigned
	// domain.
	orderS *ssa.Poset
	orderU *ssa.Poset

	// orderings contains a list of known orderings between values.
	// These lists are indexed by v.ID.
	// We do not record transitive orderings. Only explicitly learned
	// orderings are recorded. Transitive orderings can be obtained
	// by walking along the individual orderings.
	orderings map[ssa.ID]*ordering
	// stack of IDs which have had an entry added in orderings.
	// In addition, ID==0 are checkpoint markers.
	orderingsStack []ssa.ID
	orderingCache  *ordering // unused ordering records

	// known lower and upper constant bounds on individual values.
	limits       []ssa.Limit // indexed by value ID
	limitStack   []limitFact // previous entries
	recurseCheck []bool      // recursion detector for limit propagation

	// For each slice s, a map from s to a len(s)/cap(s) value (if any)
	// TODO: check if there are cases that matter where we have
	// more than one len(s) for a slice. We could keep a list if necessary.
	lens map[ssa.ID]*ssa.Value
	caps map[ssa.ID]*ssa.Value

	// reusedTopoSortIDsToBlockIndexes recycle allocations for topo-sort
	reusedTopoSortIDsToBlockIndexes []uint
}

// checkpointBound is an invalid value used for checkpointing
// and restoring factsTable.
var checkpointBound = limitFact{}

func newFactsTable(f *ssa.Func) *factsTable {
	ft := &factsTable{}
	ft.orderS = f.NewPoset()
	ft.orderU = f.NewPoset()
	ft.orderings = make(map[ssa.ID]*ordering)
	ft.limits = f.Cache.AllocLimitSlice(f.NumValues())
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			ft.limits[v.ID] = ssa.InitLimit(v)
		}
	}
	ft.limitStack = make([]limitFact, 4)
	ft.recurseCheck = f.Cache.AllocBoolSlice(f.NumValues())
	return ft
}

// initLimitForNewValue initializes the limits for newly created values,
// possibly needing to expand the limits slice. Currently used by
// simplifyBlock when certain provably constant results are folded.
func (ft *factsTable) initLimitForNewValue(v *ssa.Value) {
	if int(v.ID) >= len(ft.limits) {
		f := v.Block.Func
		n := f.NumValues()
		if cap(ft.limits) >= n {
			ft.limits = ft.limits[:n]
		} else {
			old := ft.limits
			ft.limits = f.Cache.AllocLimitSlice(n)
			copy(ft.limits, old)
			f.Cache.FreeLimitSlice(old)
		}
	}
	ft.limits[v.ID] = ssa.InitLimit(v)
}

// signedMin records the fact that we know v is at least
// min in the signed domain.
func (ft *factsTable) signedMin(v *ssa.Value, min int64) {
	ft.newLimit(v, ssa.Limit{Min: min, Max: math.MaxInt64, Umin: 0, Umax: math.MaxUint64})
}

// signedMax records the fact that we know v is at most
// max in the signed domain.
func (ft *factsTable) signedMax(v *ssa.Value, max int64) {
	ft.newLimit(v, ssa.Limit{Min: math.MinInt64, Max: max, Umin: 0, Umax: math.MaxUint64})
}
func (ft *factsTable) signedMinMax(v *ssa.Value, min, max int64) {
	ft.newLimit(v, ssa.Limit{Min: min, Max: max, Umin: 0, Umax: math.MaxUint64})
}

// setNonNegative records the fact that v is known to be non-negative.
func (ft *factsTable) setNonNegative(v *ssa.Value) {
	ft.signedMin(v, 0)
}

// unsignedMin records the fact that we know v is at least
// min in the unsigned domain.
func (ft *factsTable) unsignedMin(v *ssa.Value, min uint64) {
	ft.newLimit(v, ssa.Limit{Min: math.MinInt64, Max: math.MaxInt64, Umin: min, Umax: math.MaxUint64})
}

// unsignedMax records the fact that we know v is at most
// max in the unsigned domain.
func (ft *factsTable) unsignedMax(v *ssa.Value, max uint64) {
	ft.newLimit(v, ssa.Limit{Min: math.MinInt64, Max: math.MaxInt64, Umin: 0, Umax: max})
}
func (ft *factsTable) unsignedMinMax(v *ssa.Value, min, max uint64) {
	ft.newLimit(v, ssa.Limit{Min: math.MinInt64, Max: math.MaxInt64, Umin: min, Umax: max})
}

func (ft *factsTable) booleanFalse(v *ssa.Value) {
	ft.newLimit(v, ssa.Limit{Min: 0, Max: 0, Umin: 0, Umax: 0})
}
func (ft *factsTable) booleanTrue(v *ssa.Value) {
	ft.newLimit(v, ssa.Limit{Min: 1, Max: 1, Umin: 1, Umax: 1})
}
func (ft *factsTable) pointerNil(v *ssa.Value) {
	ft.newLimit(v, ssa.Limit{Min: 0, Max: 0, Umin: 0, Umax: 0})
}
func (ft *factsTable) pointerNonNil(v *ssa.Value) {
	l := ssa.NoLimit()
	l.Umin = 1
	ft.newLimit(v, l)
}

// newLimit adds new limiting information for v.
func (ft *factsTable) newLimit(v *ssa.Value, newLim ssa.Limit) {
	oldLim := ft.limits[v.ID]

	// Merge old and new information.
	lim := oldLim.Intersect(newLim)

	// signed <-> unsigned propagation
	if lim.Min >= 0 {
		lim = lim.UnsignedMinMax(uint64(lim.Min), uint64(lim.Max))
	}
	if ssa.FitsInBitsU(lim.Umax, uint(8*v.Type.Size()-1)) {
		lim = lim.SignedMinMax(int64(lim.Umin), int64(lim.Umax))
	}

	if lim == oldLim {
		return // nothing new to record
	}

	if lim.Unsat() {
		ft.unsat = true
		return
	}

	// Check for recursion. This normally happens because in unsatisfiable
	// cases we have a < b < a, and every update to a's limits returns
	// here again with the limit increased by 2.
	// Normally this is caught early by the orderS/orderU posets, but in
	// cases where the comparisons jump between signed and unsigned domains,
	// the posets will not notice.
	if ft.recurseCheck[v.ID] {
		// This should only happen for unsatisfiable cases. TODO: check
		return
	}
	ft.recurseCheck[v.ID] = true
	defer func() {
		ft.recurseCheck[v.ID] = false
	}()

	// Record undo information.
	ft.limitStack = append(ft.limitStack, limitFact{v.ID, oldLim})
	// Record new information.
	ft.limits[v.ID] = lim
	if v.Block.Func.Pass.Debug > 2 {
		// TODO: pos is probably wrong. This is the position where v is defined,
		// not the position where we learned the fact about it (which was
		// probably some subsequent compare+branch).
		v.Block.Func.Warnl(v.Pos, "new limit %s %s unsat=%v", v, lim.String(), ft.unsat)
	}

	// Propagate this new constant range to other values
	// that we know are ordered with respect to this one.
	// Note overflow/underflow in the arithmetic below is ok,
	// it will just lead to imprecision (undetected unsatisfiability).
	for o := ft.orderings[v.ID]; o != nil; o = o.next {
		switch o.d {
		case signed:
			switch o.r {
			case eq: // v == w
				ft.signedMinMax(o.w, lim.Min, lim.Max)
			case lt | eq: // v <= w
				ft.signedMin(o.w, lim.Min)
			case lt: // v < w
				ft.signedMin(o.w, lim.Min+1)
			case gt | eq: // v >= w
				ft.signedMax(o.w, lim.Max)
			case gt: // v > w
				ft.signedMax(o.w, lim.Max-1)
			case lt | gt: // v != w
				if lim.Min == lim.Max { // v is a constant
					c := lim.Min
					if ft.limits[o.w.ID].Min == c {
						ft.signedMin(o.w, c+1)
					}
					if ft.limits[o.w.ID].Max == c {
						ft.signedMax(o.w, c-1)
					}
				}
			}
		case unsigned:
			switch o.r {
			case eq: // v == w
				ft.unsignedMinMax(o.w, lim.Umin, lim.Umax)
			case lt | eq: // v <= w
				ft.unsignedMin(o.w, lim.Umin)
			case lt: // v < w
				ft.unsignedMin(o.w, lim.Umin+1)
			case gt | eq: // v >= w
				ft.unsignedMax(o.w, lim.Umax)
			case gt: // v > w
				ft.unsignedMax(o.w, lim.Umax-1)
			case lt | gt: // v != w
				if lim.Umin == lim.Umax { // v is a constant
					c := lim.Umin
					if ft.limits[o.w.ID].Umin == c {
						ft.unsignedMin(o.w, c+1)
					}
					if ft.limits[o.w.ID].Umax == c {
						ft.unsignedMax(o.w, c-1)
					}
				}
			}
		case boolean:
			switch o.r {
			case eq:
				if lim.Min == 0 && lim.Max == 0 { // constant false
					ft.booleanFalse(o.w)
				}
				if lim.Min == 1 && lim.Max == 1 { // constant true
					ft.booleanTrue(o.w)
				}
			case lt | gt:
				if lim.Min == 0 && lim.Max == 0 { // constant false
					ft.booleanTrue(o.w)
				}
				if lim.Min == 1 && lim.Max == 1 { // constant true
					ft.booleanFalse(o.w)
				}
			}
		case pointer:
			switch o.r {
			case eq:
				if lim.Umax == 0 { // nil
					ft.pointerNil(o.w)
				}
				if lim.Umin > 0 { // non-nil
					ft.pointerNonNil(o.w)
				}
			case lt | gt:
				if lim.Umax == 0 { // nil
					ft.pointerNonNil(o.w)
				}
				// note: not equal to non-nil doesn't tell us anything.
			}
		}
	}

	// If this is new known constant for a boolean value,
	// extract relation between its args. For example, if
	// We learn v is false, and v is defined as a<b, then we learn a>=b.
	if v.Type.IsBoolean() {
		// If we reach here, it is because we have a more restrictive
		// value for v than the default. The only two such values
		// are constant true or constant false.
		if lim.Min != lim.Max {
			v.Block.Func.Fatalf("boolean not constant %v", v)
		}
		isTrue := lim.Min == 1
		if dr, ok := domainRelationTable[v.Op]; ok && v.Op != ssaop.OpIsInBounds && v.Op != ssaop.OpIsSliceInBounds {
			d := dr.d
			r := dr.r
			if d == signed && ft.isNonNegative(v.Args[0]) && ft.isNonNegative(v.Args[1]) {
				d |= unsigned
			}
			if !isTrue {
				r ^= lt | gt | eq
			}
			// TODO: v.Block is wrong?
			addRestrictions(v.Block, ft, d, v.Args[0], v.Args[1], r)
		}
		switch v.Op {
		case ssaop.OpIsNonNil:
			if isTrue {
				ft.pointerNonNil(v.Args[0])
			} else {
				ft.pointerNil(v.Args[0])
			}
		case ssaop.OpIsInBounds, ssaop.OpIsSliceInBounds:
			// 0 <= a0 < a1 (or 0 <= a0 <= a1)
			r := lt
			if v.Op == ssaop.OpIsSliceInBounds {
				r |= eq
			}
			if isTrue {
				// On the positive branch, we learn:
				//   signed: 0 <= a0 < a1 (or 0 <= a0 <= a1)
				//   unsigned:    a0 < a1 (or a0 <= a1)
				ft.setNonNegative(v.Args[0])
				ft.update(v.Block, v.Args[0], v.Args[1], signed, r)
				ft.update(v.Block, v.Args[0], v.Args[1], unsigned, r)
			} else {
				// On the negative branch, we learn (0 > a0 ||
				// a0 >= a1). In the unsigned domain, this is
				// simply a0 >= a1 (which is the reverse of the
				// positive branch, so nothing surprising).
				// But in the signed domain, we can't express the ||
				// condition, so check if a0 is non-negative instead,
				// to be able to learn something.
				r ^= lt | gt | eq // >= (index) or > (slice)
				if ft.isNonNegative(v.Args[0]) {
					ft.update(v.Block, v.Args[0], v.Args[1], signed, r)
				}
				ft.update(v.Block, v.Args[0], v.Args[1], unsigned, r)
				// TODO: v.Block is wrong here
			}
		}
	}
}

func (ft *factsTable) addOrdering(v, w *ssa.Value, d domain, r relation) {
	o := ft.orderingCache
	if o == nil {
		o = &ordering{}
	} else {
		ft.orderingCache = o.next
	}
	o.w = w
	o.d = d
	o.r = r
	o.next = ft.orderings[v.ID]
	ft.orderings[v.ID] = o
	ft.orderingsStack = append(ft.orderingsStack, v.ID)
}

// update updates the set of relations between v and w in domain d
// restricting it to r.
func (ft *factsTable) update(parent *ssa.Block, v, w *ssa.Value, d domain, r relation) {
	if parent.Func.Pass.Debug > 2 {
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
		ft.addOrdering(v, w, d, r)
		ft.addOrdering(w, v, d, reverseBits[r])

		if !ok {
			if parent.Func.Pass.Debug > 2 {
				parent.Func.Warnl(parent.Pos, "unsat %s %s %s", v, w, r)
			}
			ft.unsat = true
			return
		}
	}
	if d == boolean || d == pointer {
		for o := ft.orderings[v.ID]; o != nil; o = o.next {
			if o.d == d && o.w == w {
				// We already know a relationship between v and w.
				// Either it is a duplicate, or it is a contradiction,
				// as we only allow eq and lt|gt for these domains,
				if o.r != r {
					ft.unsat = true
				}
				return
			}
		}
		// TODO: this does not do transitive equality.
		// We could use a poset like above, but somewhat degenerate (==,!= only).
		ft.addOrdering(v, w, d, r)
		ft.addOrdering(w, v, d, r) // note: reverseBits unnecessary for eq and lt|gt.
	}

	// Extract new constant limits based on the comparison.
	vLimit := ft.limits[v.ID]
	wLimit := ft.limits[w.ID]
	// Note: all the +1/-1 below could overflow/underflow. Either will
	// still generate correct results, it will just lead to imprecision.
	// In fact if there is overflow/underflow, the corresponding
	// code is unreachable because the known range is outside the range
	// of the value's type.
	switch d {
	case signed:
		switch r {
		case eq: // v == w
			ft.signedMinMax(v, wLimit.Min, wLimit.Max)
			ft.signedMinMax(w, vLimit.Min, vLimit.Max)
		case lt: // v < w
			ft.signedMax(v, wLimit.Max-1)
			ft.signedMin(w, vLimit.Min+1)
		case lt | eq: // v <= w
			ft.signedMax(v, wLimit.Max)
			ft.signedMin(w, vLimit.Min)
		case gt: // v > w
			ft.signedMin(v, wLimit.Min+1)
			ft.signedMax(w, vLimit.Max-1)
		case gt | eq: // v >= w
			ft.signedMin(v, wLimit.Min)
			ft.signedMax(w, vLimit.Max)
		case lt | gt: // v != w
			if vLimit.Min == vLimit.Max { // v is a constant
				c := vLimit.Min
				if wLimit.Min == c {
					ft.signedMin(w, c+1)
				}
				if wLimit.Max == c {
					ft.signedMax(w, c-1)
				}
			}
			if wLimit.Min == wLimit.Max { // w is a constant
				c := wLimit.Min
				if vLimit.Min == c {
					ft.signedMin(v, c+1)
				}
				if vLimit.Max == c {
					ft.signedMax(v, c-1)
				}
			}
		}
	case unsigned:
		switch r {
		case eq: // v == w
			ft.unsignedMinMax(v, wLimit.Umin, wLimit.Umax)
			ft.unsignedMinMax(w, vLimit.Umin, vLimit.Umax)
		case lt: // v < w
			ft.unsignedMax(v, wLimit.Umax-1)
			ft.unsignedMin(w, vLimit.Umin+1)
		case lt | eq: // v <= w
			ft.unsignedMax(v, wLimit.Umax)
			ft.unsignedMin(w, vLimit.Umin)
		case gt: // v > w
			ft.unsignedMin(v, wLimit.Umin+1)
			ft.unsignedMax(w, vLimit.Umax-1)
		case gt | eq: // v >= w
			ft.unsignedMin(v, wLimit.Umin)
			ft.unsignedMax(w, vLimit.Umax)
		case lt | gt: // v != w
			if vLimit.Umin == vLimit.Umax { // v is a constant
				c := vLimit.Umin
				if wLimit.Umin == c {
					ft.unsignedMin(w, c+1)
				}
				if wLimit.Umax == c {
					ft.unsignedMax(w, c-1)
				}
			}
			if wLimit.Umin == wLimit.Umax { // w is a constant
				c := wLimit.Umin
				if vLimit.Umin == c {
					ft.unsignedMin(v, c+1)
				}
				if vLimit.Umax == c {
					ft.unsignedMax(v, c-1)
				}
			}
		}
	case boolean:
		switch r {
		case eq: // v == w
			if vLimit.Min == 1 { // v is true
				ft.booleanTrue(w)
			}
			if vLimit.Max == 0 { // v is false
				ft.booleanFalse(w)
			}
			if wLimit.Min == 1 { // w is true
				ft.booleanTrue(v)
			}
			if wLimit.Max == 0 { // w is false
				ft.booleanFalse(v)
			}
		case lt | gt: // v != w
			if vLimit.Min == 1 { // v is true
				ft.booleanFalse(w)
			}
			if vLimit.Max == 0 { // v is false
				ft.booleanTrue(w)
			}
			if wLimit.Min == 1 { // w is true
				ft.booleanFalse(v)
			}
			if wLimit.Max == 0 { // w is false
				ft.booleanTrue(v)
			}
		}
	case pointer:
		switch r {
		case eq: // v == w
			if vLimit.Umax == 0 { // v is nil
				ft.pointerNil(w)
			}
			if vLimit.Umin > 0 { // v is non-nil
				ft.pointerNonNil(w)
			}
			if wLimit.Umax == 0 { // w is nil
				ft.pointerNil(v)
			}
			if wLimit.Umin > 0 { // w is non-nil
				ft.pointerNonNil(v)
			}
		case lt | gt: // v != w
			if vLimit.Umax == 0 { // v is nil
				ft.pointerNonNil(w)
			}
			if wLimit.Umax == 0 { // w is nil
				ft.pointerNonNil(v)
			}
			// Note: the other direction doesn't work.
			// Being not equal to a non-nil pointer doesn't
			// make you (necessarily) a nil pointer.
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
	if v.Op == ssaop.OpSliceLen && r&lt == 0 && ft.caps[v.Args[0].ID] != nil {
		// len(s) > w implies cap(s) > w
		// len(s) >= w implies cap(s) >= w
		// len(s) == w implies cap(s) >= w
		ft.update(parent, ft.caps[v.Args[0].ID], w, d, r|gt)
	}
	if w.Op == ssaop.OpSliceLen && r&gt == 0 && ft.caps[w.Args[0].ID] != nil {
		// same, length on the RHS.
		ft.update(parent, v, ft.caps[w.Args[0].ID], d, r|lt)
	}
	if v.Op == ssaop.OpSliceCap && r&gt == 0 && ft.lens[v.Args[0].ID] != nil {
		// cap(s) < w implies len(s) < w
		// cap(s) <= w implies len(s) <= w
		// cap(s) == w implies len(s) <= w
		ft.update(parent, ft.lens[v.Args[0].ID], w, d, r|lt)
	}
	if w.Op == ssaop.OpSliceCap && r&lt == 0 && ft.lens[w.Args[0].ID] != nil {
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
			lim := ft.limits[x.ID]
			if (d == signed && lim.Min > opMin[v.Op]) || (d == unsigned && lim.Umin > 0) {
				ft.update(parent, x, w, d, gt)
			}
		} else if x, delta := isConstDelta(w); x != nil && delta == 1 {
			// v >= x+1 && x < max  ⇒  v > x
			lim := ft.limits[x.ID]
			if (d == signed && lim.Max < opMax[w.Op]) || (d == unsigned && lim.Umax < opUMax[w.Op]) {
				ft.update(parent, v, x, d, gt)
			}
		}
	}

	// Process: x+delta > w (with delta constant)
	// Only signed domain for now (useful for accesses to slices in loops).
	if r == gt || r == gt|eq {
		if x, delta := isConstDelta(v); x != nil && d == signed {
			if parent.Func.Pass.Debug > 1 {
				parent.Func.Warnl(parent.Pos, "x+d %s w; x:%v %v delta:%v w:%v d:%v", r, x, parent.String(), delta, w.AuxInt, d)
			}
			underflow := true
			if delta < 0 {
				l := ft.limits[x.ID]
				if (x.Type.Size() == 8 && l.Min >= math.MinInt64-delta) ||
					(x.Type.Size() == 4 && l.Min >= math.MinInt32-delta) {
					underflow = false
				}
			}
			if delta < 0 && !underflow {
				// If delta < 0 and x+delta cannot underflow then x > x+delta (that is, x > v)
				ft.update(parent, x, v, signed, gt)
			}
			if !w.IsGenericIntConst() {
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
				switch x.Type.Size() {
				case 8:
					min = w.AuxInt - delta
					max = int64(^uint64(0)>>1) - delta
				case 4:
					min = int64(int32(w.AuxInt) - int32(delta))
					max = int64(int32(^uint32(0)>>1) - int32(delta))
				case 2:
					min = int64(int16(w.AuxInt) - int16(delta))
					max = int64(int16(^uint16(0)>>1) - int16(delta))
				case 1:
					min = int64(int8(w.AuxInt) - int8(delta))
					max = int64(int8(^uint8(0)>>1) - int8(delta))
				default:
					panic("unimplemented")
				}

				if min < max {
					// Record that x > min and max >= x
					if r == gt {
						min++
					}
					ft.signedMinMax(x, min, max)
				} else {
					// We know that either x>min OR x<=max. factsTable cannot record OR conditions,
					// so let's see if we can already prove that one of them is false, in which case
					// the other must be true
					l := ft.limits[x.ID]
					if l.Max <= min {
						if r&eq == 0 || l.Max < min {
							// x>min (x>=min) is impossible, so it must be x<=max
							ft.signedMax(x, max)
						}
					} else if l.Min > max {
						// x<=max is impossible, so it must be x>min
						if r == gt {
							min++
						}
						ft.signedMin(x, min)
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

var opMin = map[ssaop.Op]int64{
	ssaop.OpAdd64: math.MinInt64, ssaop.OpSub64: math.MinInt64,
	ssaop.OpAdd32: math.MinInt32, ssaop.OpSub32: math.MinInt32,
}

var opMax = map[ssaop.Op]int64{
	ssaop.OpAdd64: math.MaxInt64, ssaop.OpSub64: math.MaxInt64,
	ssaop.OpAdd32: math.MaxInt32, ssaop.OpSub32: math.MaxInt32,
}

var opUMax = map[ssaop.Op]uint64{
	ssaop.OpAdd64: math.MaxUint64, ssaop.OpSub64: math.MaxUint64,
	ssaop.OpAdd32: math.MaxUint32, ssaop.OpSub32: math.MaxUint32,
}

// isNonNegative reports whether v is known to be non-negative.
func (ft *factsTable) isNonNegative(v *ssa.Value) bool {
	return ft.limits[v.ID].Min >= 0
}

// checkpoint saves the current state of known relations.
// Called when descending on a branch.
func (ft *factsTable) checkpoint() {
	if ft.unsat {
		ft.unsatDepth++
	}
	ft.limitStack = append(ft.limitStack, checkpointBound)
	ft.orderS.Checkpoint()
	ft.orderU.Checkpoint()
	ft.orderingsStack = append(ft.orderingsStack, 0)
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
		old := ft.limitStack[len(ft.limitStack)-1]
		ft.limitStack = ft.limitStack[:len(ft.limitStack)-1]
		if old.vid == 0 { // checkpointBound
			break
		}
		ft.limits[old.vid] = old.limit
	}
	ft.orderS.Undo()
	ft.orderU.Undo()
	for {
		id := ft.orderingsStack[len(ft.orderingsStack)-1]
		ft.orderingsStack = ft.orderingsStack[:len(ft.orderingsStack)-1]
		if id == 0 { // checkpoint marker
			break
		}
		o := ft.orderings[id]
		ft.orderings[id] = o.next
		o.next = ft.orderingCache
		ft.orderingCache = o
	}
}

var (
	reverseBits = [...]relation{0, 4, 2, 6, 1, 5, 3, 7}

	// maps what we learn when the positive branch is taken.
	// For example:
	//      OpLess8:   {signed, lt},
	//	v1 = (OpLess8 v2 v3).
	// If we learn that v1 is true, then we can deduce that v2<v3
	// in the signed domain.
	domainRelationTable = map[ssaop.Op]struct {
		d domain
		r relation
	}{
		ssaop.OpEq8:   {signed | unsigned, eq},
		ssaop.OpEq16:  {signed | unsigned, eq},
		ssaop.OpEq32:  {signed | unsigned, eq},
		ssaop.OpEq64:  {signed | unsigned, eq},
		ssaop.OpEqPtr: {pointer, eq},
		ssaop.OpEqB:   {boolean, eq},

		ssaop.OpNeq8:   {signed | unsigned, lt | gt},
		ssaop.OpNeq16:  {signed | unsigned, lt | gt},
		ssaop.OpNeq32:  {signed | unsigned, lt | gt},
		ssaop.OpNeq64:  {signed | unsigned, lt | gt},
		ssaop.OpNeqPtr: {pointer, lt | gt},
		ssaop.OpNeqB:   {boolean, lt | gt},

		ssaop.OpLess8:   {signed, lt},
		ssaop.OpLess8U:  {unsigned, lt},
		ssaop.OpLess16:  {signed, lt},
		ssaop.OpLess16U: {unsigned, lt},
		ssaop.OpLess32:  {signed, lt},
		ssaop.OpLess32U: {unsigned, lt},
		ssaop.OpLess64:  {signed, lt},
		ssaop.OpLess64U: {unsigned, lt},

		ssaop.OpLeq8:   {signed, lt | eq},
		ssaop.OpLeq8U:  {unsigned, lt | eq},
		ssaop.OpLeq16:  {signed, lt | eq},
		ssaop.OpLeq16U: {unsigned, lt | eq},
		ssaop.OpLeq32:  {signed, lt | eq},
		ssaop.OpLeq32U: {unsigned, lt | eq},
		ssaop.OpLeq64:  {signed, lt | eq},
		ssaop.OpLeq64U: {unsigned, lt | eq},
	}
)

// cleanup returns the posets to the free list
func (ft *factsTable) cleanup(f *ssa.Func) {
	for _, po := range []*ssa.Poset{ft.orderS, ft.orderU} {
		// Make sure it's empty as it should be. A non-empty poset
		// might cause errors and miscompilations if reused.
		if checkEnabled {
			if err := po.CheckEmpty(); err != nil {
				f.Fatalf("poset not empty after function %s: %v", f.Name, err)
			}
		}
		f.RetPoset(po)
	}
	f.Cache.FreeLimitSlice(ft.limits)
	f.Cache.FreeBoolSlice(ft.recurseCheck)
	if cap(ft.reusedTopoSortIDsToBlockIndexes) > 0 {
		f.Cache.FreeUintSlice(ft.reusedTopoSortIDsToBlockIndexes)
	}
}

// addSlicesOfSameLen finds the slices that are in the same block and whose Op
// is OpPhi and always have the same length, then add the equality relationship
// between them to ft. If two slices start out with the same length and decrease
// in length by the same amount on each round of the loop (or in the if block),
// then we think their lengths are always equal.
//
// See https://go.dev/issues/75144
//
// In fact, we are just propagating the equality
//
//	if len(a) == len(b) { // from here
//		for len(a) > 4 {
//			a = a[4:]
//			b = b[4:]
//		}
//		if len(a) == len(b) { // to here
//			return true
//		}
//	}
//
// or change the for to if:
//
//	if len(a) == len(b) { // from here
//		if len(a) > 4 {
//			a = a[4:]
//			b = b[4:]
//		}
//		if len(a) == len(b) { // to here
//			return true
//		}
//	}
func addSlicesOfSameLen(ft *factsTable, b *ssa.Block) {
	// Let w points to the first value we're interested in, and then we
	// only process those values ​​that appear to be the same length as w,
	// looping only once. This should be enough in most cases. And u is
	// similar to w, see comment for predIndex.
	var u, w *ssa.Value
	var i, j, k sliceInfo
	isInterested := func(v *ssa.Value) bool {
		j = getSliceInfo(v)
		return j.sliceWhere != sliceUnknown
	}
	for _, v := range b.Values {
		if v.Uses == 0 {
			continue
		}
		if v.Op == ssaop.OpPhi && len(v.Args) == 2 && ft.lens[v.ID] != nil && isInterested(v) {
			if j.predIndex == 1 && ft.lens[v.Args[0].ID] != nil {
				// found v = (Phi x (SliceMake _ (Add64 (Const64 [n]) (SliceLen x)) _))) or
				// v = (Phi x (SliceMake _ (Add64 (Const64 [n]) (SliceLen v)) _)))
				if w == nil {
					k = j
					w = v
					continue
				}
				// propagate the equality
				if j == k && ft.orderS.Equal(ft.lens[v.Args[0].ID], ft.lens[w.Args[0].ID]) {
					ft.update(b, ft.lens[v.ID], ft.lens[w.ID], signed, eq)
				}
			} else if j.predIndex == 0 && ft.lens[v.Args[1].ID] != nil {
				// found v = (Phi (SliceMake _ (Add64 (Const64 [n]) (SliceLen x)) _)) x) or
				// v = (Phi (SliceMake _ (Add64 (Const64 [n]) (SliceLen v)) _)) x)
				if u == nil {
					i = j
					u = v
					continue
				}
				// propagate the equality
				if j == i && ft.orderS.Equal(ft.lens[v.Args[1].ID], ft.lens[u.Args[1].ID]) {
					ft.update(b, ft.lens[v.ID], ft.lens[u.ID], signed, eq)
				}
			}
		}
	}
}

type sliceWhere int

const (
	sliceUnknown sliceWhere = iota
	sliceInFor
	sliceInIf
)

// predIndex is used to indicate the branch represented by the predecessor
// block in which the slicing operation occurs.
type predIndex int

type sliceInfo struct {
	lengthDiff int64
	sliceWhere
	predIndex
}

// getSliceInfo returns the negative increment of the slice length in a slice
// operation by examine the Phi node at the merge block. So, we only interest
// in the slice operation if it is inside a for block or an if block.
// Otherwise it returns sliceInfo{0, sliceUnknown, 0}.
//
// For the following for block:
//
//	for len(a) > 4 {
//	    a = a[4:]
//	}
//
// vp = (Phi v3 v9)
// v5 = (SliceLen vp)
// v7 = (Add64 (Const64 [-4]) v5)
// v9 = (SliceMake _ v7 _)
//
// returns sliceInfo{-4, sliceInFor, 1}
//
// For a subsequent merge block after an if block:
//
//	if len(a) > 4 {
//	    a = a[4:]
//	}
//	a // here
//
// vp = (Phi v3 v9)
// v5 = (SliceLen v3)
// v7 = (Add64 (Const64 [-4]) v5)
// v9 = (SliceMake _ v7 _)
//
// returns sliceInfo{-4, sliceInIf, 1}
//
// Returns sliceInfo{0, sliceUnknown, 0} if it is not the slice
// operation we are interested in.
func getSliceInfo(vp *ssa.Value) (inf sliceInfo) {
	if vp.Op != ssaop.OpPhi || len(vp.Args) != 2 {
		return
	}
	var i predIndex
	var l *ssa.Value // length for OpSliceMake
	if vp.Args[0].Op != ssaop.OpSliceMake && vp.Args[1].Op == ssaop.OpSliceMake {
		l = vp.Args[1].Args[1]
		i = 1
	} else if vp.Args[0].Op == ssaop.OpSliceMake && vp.Args[1].Op != ssaop.OpSliceMake {
		l = vp.Args[0].Args[1]
		i = 0
	} else {
		return
	}
	var op ssaop.Op
	switch l.Op {
	case ssaop.OpAdd64:
		op = ssaop.OpConst64
	case ssaop.OpAdd32:
		op = ssaop.OpConst32
	default:
		return
	}
	if l.Args[0].Op == op && l.Args[1].Op == ssaop.OpSliceLen && l.Args[1].Args[0] == vp {
		return sliceInfo{l.Args[0].AuxInt, sliceInFor, i}
	}
	if l.Args[1].Op == op && l.Args[0].Op == ssaop.OpSliceLen && l.Args[0].Args[0] == vp {
		return sliceInfo{l.Args[1].AuxInt, sliceInFor, i}
	}
	if l.Args[0].Op == op && l.Args[1].Op == ssaop.OpSliceLen && l.Args[1].Args[0] == vp.Args[1-i] {
		return sliceInfo{l.Args[0].AuxInt, sliceInIf, i}
	}
	if l.Args[1].Op == op && l.Args[0].Op == ssaop.OpSliceLen && l.Args[0].Args[0] == vp.Args[1-i] {
		return sliceInfo{l.Args[1].AuxInt, sliceInIf, i}
	}
	return
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
func prove(f *ssa.Func) {
	// Find induction variables.
	var indVars map[*ssa.Block][]indVar
	for _, v := range findIndVar(f) {
		ind := v.ind
		if len(ind.Args) != 2 {
			// the rewrite code assumes there is only ever two parents to loops
			panic("unexpected induction with too many parents")
		}

		nxt := v.nxt
		if !(ind.Uses == 2 && // 2 used by comparison and next
			nxt.Uses == 1) { // 1 used by induction
			// ind or nxt is used inside the loop, add it for the facts table
			if indVars == nil {
				indVars = make(map[*ssa.Block][]indVar)
			}
			indVars[v.entry] = append(indVars[v.entry], v)
			continue
		} else {
			// Since this induction variable is not used for anything but counting the iterations,
			// no point in putting it into the facts table.
		}

		maybeRewriteLoopToDownwardCountingLoop(f, v)
	}

	ft := newFactsTable(f)
	ft.checkpoint()

	// Find length and capacity ops.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Uses == 0 {
				// We don't care about dead values.
				// (There can be some that are CSEd but not removed yet.)
				continue
			}
			switch v.Op {
			case ssaop.OpSliceLen:
				if ft.lens == nil {
					ft.lens = map[ssa.ID]*ssa.Value{}
				}
				// Set all len Values for the same slice as equal in the poset.
				// The poset handles transitive relations, so Values related to
				// any OpSliceLen for this slice will be correctly related to others.
				if l, ok := ft.lens[v.Args[0].ID]; ok {
					ft.update(b, v, l, signed, eq)
				} else {
					ft.lens[v.Args[0].ID] = v
				}
			case ssaop.OpSliceCap:
				if ft.caps == nil {
					ft.caps = map[ssa.ID]*ssa.Value{}
				}
				// Same as case OpSliceLen above, but for slice cap.
				if c, ok := ft.caps[v.Args[0].ID]; ok {
					ft.update(b, v, c, signed, eq)
				} else {
					ft.caps[v.Args[0].ID] = v
				}
			}
		}
	}

	// current node state
	type walkState int
	const (
		descend walkState = iota
		restore
	)
	// work maintains the DFS stack.
	type bp struct {
		block *ssa.Block // current handled block
		state walkState  // what's to do
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

			// Entering the block, add facts about the induction variable
			// that is bound to this block.
			for _, iv := range indVars[node.block] {
				addIndVarRestrictions(ft, parent, iv)
			}

			// Add results of reaching this block via a branch from
			// its immediate dominator (if any).
			if branch != unknown {
				addBranchRestrictions(ft, parent, branch)
			}

			// Add slices of the same length start from current block.
			addSlicesOfSameLen(ft, node.block)

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

			ft.topoSortValuesInBlock(node.block)

			for _, v := range node.block.Values {
				ft.flowLimit(v)
				// constant fold arguments before addValueFact to avoid v's v.Args learned facts time traveling into v's arguments.
				// in other words if v teaches us something about it's arguments,
				// we can't use that to optimize v's arguments since v hasn't ran yet.
				ft.constantFoldArguments(v)
				ft.addValueFact(node.block, v)
				ft.simplifyValue(node.block, v)
			}

			ft.simplifyBlock(sdom, node.block)

			work = append(work, bp{
				block: node.block,
				state: restore,
			})
			for s := sdom.Child(node.block); s != nil; s = sdom.Sibling(s) {
				work = append(work, bp{
					block: s,
					state: descend,
				})
			}

		case restore:
			ft.restore()
		}
	}

	ft.restore()

	ft.cleanup(f)
}

// flowLimit updates the known limits of v in ft.
// flowLimit can use the ranges of input arguments.
//
// Note: this calculation only happens at the point the value is defined. We do not reevaluate
// it later. So for example:
//
//	v := x + y
//	if 0 <= x && x < 5 && 0 <= y && y < 5 { ... use v ... }
//
// we don't discover that the range of v is bounded in the conditioned
// block. We could recompute the range of v once we enter the block so
// we know that it is 0 <= v <= 8, but we don't have a mechanism to do
// that right now.
func (ft *factsTable) flowLimit(v *ssa.Value) {
	if !v.Type.IsInteger() {
		// TODO: boolean?
		return
	}

	// Additional limits based on opcode and argument.
	// No need to repeat things here already done in initLimit.
	switch v.Op {

	// extensions
	case ssaop.OpZeroExt8to64, ssaop.OpZeroExt8to32, ssaop.OpZeroExt8to16, ssaop.OpZeroExt16to64, ssaop.OpZeroExt16to32, ssaop.OpZeroExt32to64:
		a := ft.limits[v.Args[0].ID]
		ft.unsignedMinMax(v, a.Umin, a.Umax)
	case ssaop.OpSignExt8to64, ssaop.OpSignExt8to32, ssaop.OpSignExt8to16, ssaop.OpSignExt16to64, ssaop.OpSignExt16to32, ssaop.OpSignExt32to64:
		a := ft.limits[v.Args[0].ID]
		ft.signedMinMax(v, a.Min, a.Max)
	case ssaop.OpTrunc64to8, ssaop.OpTrunc64to16, ssaop.OpTrunc64to32, ssaop.OpTrunc32to8, ssaop.OpTrunc32to16, ssaop.OpTrunc16to8:
		a := ft.limits[v.Args[0].ID]
		if a.Umax <= 1<<(uint64(v.Type.Size())*8)-1 {
			ft.unsignedMinMax(v, a.Umin, a.Umax)
		}

	// math/bits
	case ssaop.OpCtz64, ssaop.OpCtz32, ssaop.OpCtz16, ssaop.OpCtz8:
		a := v.Args[0]
		al := ft.limits[a.ID]
		ft.newLimit(v, al.Ctz(uint(a.Type.Size())*8))

	case ssaop.OpPopCount64, ssaop.OpPopCount32, ssaop.OpPopCount16, ssaop.OpPopCount8:
		a := v.Args[0]
		al := ft.limits[a.ID]
		ft.newLimit(v, al.Popcount(uint(a.Type.Size())*8))

	case ssaop.OpBitLen64, ssaop.OpBitLen32, ssaop.OpBitLen16, ssaop.OpBitLen8:
		a := v.Args[0]
		al := ft.limits[a.ID]
		ft.newLimit(v, al.Bitlen(uint(a.Type.Size())*8))

	// Masks.

	// TODO: if y.umax and y.umin share a leading bit pattern, y also has that leading bit pattern.
	// we could compare the patterns of always set bits in a and b and learn more about minimum and maximum.
	// But I doubt this help any real world code.
	case ssaop.OpOr64, ssaop.OpOr32, ssaop.OpOr16, ssaop.OpOr8:
		// OR can only make the value bigger and can't flip bits proved to be zero in both inputs.
		a := ft.limits[v.Args[0].ID]
		b := ft.limits[v.Args[1].ID]
		ft.unsignedMinMax(v,
			max(a.Umin, b.Umin),
			1<<bits.Len64(a.Umax|b.Umax)-1)
	case ssaop.OpXor64, ssaop.OpXor32, ssaop.OpXor16, ssaop.OpXor8:
		// XOR can't flip bits that are proved to be zero in both inputs.
		a := ft.limits[v.Args[0].ID]
		b := ft.limits[v.Args[1].ID]
		ft.unsignedMax(v, 1<<bits.Len64(a.Umax|b.Umax)-1)
	case ssaop.OpCom64, ssaop.OpCom32, ssaop.OpCom16, ssaop.OpCom8:
		a := ft.limits[v.Args[0].ID]
		ft.newLimit(v, a.Com(uint(v.Type.Size())*8))

	// Arithmetic.
	case ssaop.OpAdd64, ssaop.OpAdd32, ssaop.OpAdd16, ssaop.OpAdd8:
		a := ft.limits[v.Args[0].ID]
		b := ft.limits[v.Args[1].ID]
		ft.newLimit(v, a.Add(b, uint(v.Type.Size())*8))
	case ssaop.OpSub64, ssaop.OpSub32, ssaop.OpSub16, ssaop.OpSub8:
		a := ft.limits[v.Args[0].ID]
		b := ft.limits[v.Args[1].ID]
		ft.newLimit(v, a.Sub(b, uint(v.Type.Size())*8))
		ft.detectMod(v)
		ft.detectSliceLenRelation(v)
		ft.detectSubRelations(v)
	case ssaop.OpNeg64, ssaop.OpNeg32, ssaop.OpNeg16, ssaop.OpNeg8:
		a := ft.limits[v.Args[0].ID]
		bitsize := uint(v.Type.Size()) * 8
		ft.newLimit(v, a.Neg(bitsize))
	case ssaop.OpMul64, ssaop.OpMul32, ssaop.OpMul16, ssaop.OpMul8:
		a := ft.limits[v.Args[0].ID]
		b := ft.limits[v.Args[1].ID]
		ft.newLimit(v, a.Mul(b, uint(v.Type.Size())*8))
	case ssaop.OpLsh64x64, ssaop.OpLsh64x32, ssaop.OpLsh64x16, ssaop.OpLsh64x8,
		ssaop.OpLsh32x64, ssaop.OpLsh32x32, ssaop.OpLsh32x16, ssaop.OpLsh32x8,
		ssaop.OpLsh16x64, ssaop.OpLsh16x32, ssaop.OpLsh16x16, ssaop.OpLsh16x8,
		ssaop.OpLsh8x64, ssaop.OpLsh8x32, ssaop.OpLsh8x16, ssaop.OpLsh8x8:
		a := ft.limits[v.Args[0].ID]
		b := ft.limits[v.Args[1].ID]
		bitsize := uint(v.Type.Size()) * 8
		ft.newLimit(v, a.Mul(b.Exp2(bitsize), bitsize))
	case ssaop.OpRsh64x64, ssaop.OpRsh64x32, ssaop.OpRsh64x16, ssaop.OpRsh64x8,
		ssaop.OpRsh32x64, ssaop.OpRsh32x32, ssaop.OpRsh32x16, ssaop.OpRsh32x8,
		ssaop.OpRsh16x64, ssaop.OpRsh16x32, ssaop.OpRsh16x16, ssaop.OpRsh16x8,
		ssaop.OpRsh8x64, ssaop.OpRsh8x32, ssaop.OpRsh8x16, ssaop.OpRsh8x8:
		a := ft.limits[v.Args[0].ID]
		b := ft.limits[v.Args[1].ID]
		if b.Min >= 0 {
			// Shift of negative makes a value closer to 0 (greater),
			// so if a.min is negative, v.min is a.min>>b.min instead of a.min>>b.max,
			// and similarly if a.max is negative, v.max is a.max>>b.max.
			// Easier to compute min and max of both than to write sign logic.
			vmin := min(a.Min>>b.Min, a.Min>>b.Max)
			vmax := max(a.Max>>b.Min, a.Max>>b.Max)
			ft.signedMinMax(v, vmin, vmax)
		}
	case ssaop.OpRsh64Ux64, ssaop.OpRsh64Ux32, ssaop.OpRsh64Ux16, ssaop.OpRsh64Ux8,
		ssaop.OpRsh32Ux64, ssaop.OpRsh32Ux32, ssaop.OpRsh32Ux16, ssaop.OpRsh32Ux8,
		ssaop.OpRsh16Ux64, ssaop.OpRsh16Ux32, ssaop.OpRsh16Ux16, ssaop.OpRsh16Ux8,
		ssaop.OpRsh8Ux64, ssaop.OpRsh8Ux32, ssaop.OpRsh8Ux16, ssaop.OpRsh8Ux8:
		a := ft.limits[v.Args[0].ID]
		b := ft.limits[v.Args[1].ID]
		if b.Min >= 0 {
			ft.unsignedMinMax(v, a.Umin>>b.Max, a.Umax>>b.Min)
		}
	case ssaop.OpDiv64, ssaop.OpDiv32, ssaop.OpDiv16, ssaop.OpDiv8:
		a := ft.limits[v.Args[0].ID]
		b := ft.limits[v.Args[1].ID]
		if !(a.Nonnegative() && b.Nonnegative()) {
			// TODO: we could handle signed limits but I didn't bother.
			break
		}
		fallthrough
	case ssaop.OpDiv64u, ssaop.OpDiv32u, ssaop.OpDiv16u, ssaop.OpDiv8u:
		a := ft.limits[v.Args[0].ID]
		b := ft.limits[v.Args[1].ID]
		lim := ssa.NoLimit()
		if b.Umax > 0 {
			lim = lim.UnsignedMin(a.Umin / b.Umax)
		}
		if b.Umin > 0 {
			lim = lim.UnsignedMax(a.Umax / b.Umin)
		}
		ft.newLimit(v, lim)
	case ssaop.OpMod64, ssaop.OpMod32, ssaop.OpMod16, ssaop.OpMod8:
		ft.modLimit(true, v, v.Args[0], v.Args[1])
	case ssaop.OpMod64u, ssaop.OpMod32u, ssaop.OpMod16u, ssaop.OpMod8u:
		ft.modLimit(false, v, v.Args[0], v.Args[1])

	case ssaop.OpPhi:
		// Compute the union of all the input phis.
		// Often this will convey no information, because the block
		// is not dominated by its predecessors and hence the
		// phi arguments might not have been processed yet. But if
		// the values are declared earlier, it may help. e.g., for
		//    v = phi(c3, c5)
		// where c3 = OpConst [3] and c5 = OpConst [5] are
		// defined in the entry block, we can derive [3,5]
		// as the limit for v.
		l := ft.limits[v.Args[0].ID]
		for _, a := range v.Args[1:] {
			l2 := ft.limits[a.ID]
			l.Min = min(l.Min, l2.Min)
			l.Max = max(l.Max, l2.Max)
			l.Umin = min(l.Umin, l2.Umin)
			l.Umax = max(l.Umax, l2.Umax)
		}
		ft.newLimit(v, l)
	}
}

// detectSliceLenRelation matches the pattern where
//  1. v := slicelen - index, OR v := slicecap - index
//     AND
//  2. index <= slicelen - K
//     THEN
//
// slicecap - index >= slicelen - index >= K
//
// Note that "index" is not used for indexing in this pattern, but
// in the motivating example (chunked slice iteration) it is.
func (ft *factsTable) detectSliceLenRelation(v *ssa.Value) {
	if v.Op != ssaop.OpSub64 {
		return
	}

	if !(v.Args[0].Op == ssaop.OpSliceLen || v.Args[0].Op == ssaop.OpStringLen || v.Args[0].Op == ssaop.OpSliceCap) {
		return
	}

	index := v.Args[1]
	if !ft.isNonNegative(index) {
		return
	}
	slice := v.Args[0].Args[0]

	for o := ft.orderings[index.ID]; o != nil; o = o.next {
		if o.d != signed {
			continue
		}
		or := o.r
		if or != lt && or != lt|eq {
			continue
		}
		ow := o.w
		if ow.Op != ssaop.OpAdd64 && ow.Op != ssaop.OpSub64 {
			continue
		}
		var lenOffset *ssa.Value
		if bound := ow.Args[0]; (bound.Op == ssaop.OpSliceLen || bound.Op == ssaop.OpStringLen) && bound.Args[0] == slice {
			lenOffset = ow.Args[1]
		} else if bound := ow.Args[1]; (bound.Op == ssaop.OpSliceLen || bound.Op == ssaop.OpStringLen) && bound.Args[0] == slice {
			// Do not infer K - slicelen, see issue #76709.
			if ow.Op == ssaop.OpAdd64 {
				lenOffset = ow.Args[0]
			}
		}
		if lenOffset == nil || lenOffset.Op != ssaop.OpConst64 {
			continue
		}
		K := lenOffset.AuxInt
		if ow.Op == ssaop.OpAdd64 {
			K = -K
		}
		if K < 0 {
			continue
		}
		if or == lt {
			K++
		}
		if K < 0 { // We hate thinking about overflow
			continue
		}
		ft.signedMin(v, K)
	}
}

// v must be Sub{64,32,16,8}.
func (ft *factsTable) detectSubRelations(v *ssa.Value) {
	// v = x-y
	x := v.Args[0]
	y := v.Args[1]
	if x == y {
		ft.signedMinMax(v, 0, 0)
		return
	}
	xLim := ft.limits[x.ID]
	yLim := ft.limits[y.ID]

	// Check if we might wrap around. If so, give up.
	width := uint(v.Type.Size()) * 8

	// v >= 1 in the signed domain?
	var vSignedMinOne bool

	// Signed optimizations
	if _, ok := ssa.SafeSub(xLim.Min, yLim.Max, width); ok {
		// Large abs negative y can also overflow
		if _, ok := ssa.SafeSub(xLim.Max, yLim.Min, width); ok {
			// x-y won't overflow

			// Subtracting a positive non-zero number only makes
			// things smaller. If it's positive or zero, it might
			// also do nothing (x-0 == v).
			if yLim.Min > 0 {
				ft.update(v.Block, v, x, signed, lt)
			} else if yLim.Min == 0 {
				ft.update(v.Block, v, x, signed, lt|eq)
			}

			// Subtracting a number from a bigger one
			// can't go below 1. If the numbers might be
			// equal, then it can't go below 0.
			//
			// This requires the overflow checks because
			// large negative y can cause an overflow.
			if ft.orderS.Ordered(y, x) {
				ft.signedMin(v, 1)
				vSignedMinOne = true
			} else if ft.orderS.OrderedOrEqual(y, x) {
				ft.setNonNegative(v)
			}
		}
	}

	// Unsigned optimizations
	if _, ok := ssa.SafeSubU(xLim.Umin, yLim.Umax, width); ok {
		if yLim.Umin > 0 {
			ft.update(v.Block, v, x, unsigned, lt)
		} else {
			ft.update(v.Block, v, x, unsigned, lt|eq)
		}
	}

	// Proving v >= 1 in the signed domain automatically
	// proves it in the unsigned domain, so we can skip it.
	//
	// We don't need overflow checks here, since if y < x,
	// then x-y can never overflow for uint.
	if !vSignedMinOne && ft.orderU.Ordered(y, x) {
		ft.unsignedMin(v, 1)
	}
}

// x%d has been rewritten to x - (x/d)*d.
func (ft *factsTable) detectMod(v *ssa.Value) {
	var opDiv, opDivU, opMul, opConst ssaop.Op
	switch v.Op {
	case ssaop.OpSub64:
		opDiv = ssaop.OpDiv64
		opDivU = ssaop.OpDiv64u
		opMul = ssaop.OpMul64
		opConst = ssaop.OpConst64
	case ssaop.OpSub32:
		opDiv = ssaop.OpDiv32
		opDivU = ssaop.OpDiv32u
		opMul = ssaop.OpMul32
		opConst = ssaop.OpConst32
	case ssaop.OpSub16:
		opDiv = ssaop.OpDiv16
		opDivU = ssaop.OpDiv16u
		opMul = ssaop.OpMul16
		opConst = ssaop.OpConst16
	case ssaop.OpSub8:
		opDiv = ssaop.OpDiv8
		opDivU = ssaop.OpDiv8u
		opMul = ssaop.OpMul8
		opConst = ssaop.OpConst8
	}

	mul := v.Args[1]
	if mul.Op != opMul {
		return
	}
	div, con := mul.Args[0], mul.Args[1]
	if div.Op == opConst {
		div, con = con, div
	}
	if con.Op != opConst || (div.Op != opDiv && div.Op != opDivU) || div.Args[0] != v.Args[0] || div.Args[1].Op != opConst || div.Args[1].AuxInt != con.AuxInt {
		return
	}
	ft.modLimit(div.Op == opDiv, v, v.Args[0], con)
}

// modLimit sets v with facts derived from v = p % q.
func (ft *factsTable) modLimit(signed bool, v, p, q *ssa.Value) {
	a := ft.limits[p.ID]
	b := ft.limits[q.ID]
	if signed {
		if a.Min < 0 && b.Min > 0 {
			ft.signedMinMax(v, -(b.Max - 1), b.Max-1)
			return
		}
		if !(a.Nonnegative() && b.Nonnegative()) {
			// TODO: we could handle signed limits but I didn't bother.
			return
		}
		if a.Min >= 0 && b.Min > 0 {
			ft.setNonNegative(v)
		}
	}
	// Underflow in the arithmetic below is ok, it gives to MaxUint64 which does nothing to the limit.
	ft.unsignedMax(v, min(a.Umax, b.Umax-1))
}

// getBranch returns the range restrictions added by p
// when reaching b. p is the immediate dominator of b.
func getBranch(sdom ssa.SparseTree, p *ssa.Block, b *ssa.Block) branch {
	if p == nil {
		return unknown
	}
	switch p.Kind {
	case block.BlockIf:
		// If p and p.Succs[0] are dominators it means that every path
		// from entry to b passes through p and p.Succs[0]. We care that
		// no path from entry to b passes through p.Succs[1]. If p.Succs[0]
		// has one predecessor then (apart from the degenerate case),
		// there is no path from entry that can reach b through p.Succs[1].
		// TODO: how about p->yes->b->yes, i.e. a loop in yes.
		if sdom.IsAncestorEq(p.Succs[0].B, b) && len(p.Succs[0].B.Preds) == 1 {
			return positive
		}
		if sdom.IsAncestorEq(p.Succs[1].B, b) && len(p.Succs[1].B.Preds) == 1 {
			return negative
		}
	case block.BlockJumpTable:
		// TODO: this loop can lead to quadratic behavior, as
		// getBranch can be called len(p.Succs) times.
		for i, e := range p.Succs {
			if sdom.IsAncestorEq(e.B, b) && len(e.B.Preds) == 1 {
				return jumpTable0 + branch(i)
			}
		}
	}
	return unknown
}

// addIndVarRestrictions updates the factsTables ft with the facts
// learned from the induction variable indVar which drives the loop
// starting in Block b.
func addIndVarRestrictions(ft *factsTable, b *ssa.Block, iv indVar) {
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
func addBranchRestrictions(ft *factsTable, b *ssa.Block, br branch) {
	c := b.Controls[0]
	switch {
	case br == negative:
		ft.booleanFalse(c)
	case br == positive:
		ft.booleanTrue(c)
	case br >= jumpTable0:
		idx := br - jumpTable0
		val := int64(idx)
		if v, off := isConstDelta(c); v != nil {
			// Establish the bound on the underlying value we're switching on,
			// not on the offset-ed value used as the jump table index.
			c = v
			val -= off
		}
		ft.newLimit(c, ssa.Limit{Min: val, Max: val, Umin: uint64(val), Umax: uint64(val)})
	default:
		panic("unknown branch")
	}
}

// addRestrictions updates restrictions from the immediate
// dominating block (p) using r.
func addRestrictions(parent *ssa.Block, ft *factsTable, t domain, v, w *ssa.Value, r relation) {
	if t == 0 {
		// Trivial case: nothing to do.
		// Should not happen, but just in case.
		return
	}
	for i := domain(1); i <= t; i <<= 1 {
		if t&i == 0 {
			continue
		}
		ft.update(parent, v, w, i, r)
	}
}

func unsignedAddOverflows(a, b uint64, t *types.Type) bool {
	switch t.Size() {
	case 8:
		return a+b < a
	case 4:
		return a+b > math.MaxUint32
	case 2:
		return a+b > math.MaxUint16
	case 1:
		return a+b > math.MaxUint8
	default:
		panic("unreachable")
	}
}

func signedAddOverflowsOrUnderflows(a, b int64, t *types.Type) bool {
	r := a + b
	switch t.Size() {
	case 8:
		return (a >= 0 && b >= 0 && r < 0) || (a < 0 && b < 0 && r >= 0)
	case 4:
		return r < math.MinInt32 || math.MaxInt32 < r
	case 2:
		return r < math.MinInt16 || math.MaxInt16 < r
	case 1:
		return r < math.MinInt8 || math.MaxInt8 < r
	default:
		panic("unreachable")
	}
}

func unsignedSubUnderflows(a, b uint64) bool {
	return a < b
}

// checkForChunkedIndexBounds looks for index expressions of the form
// A[i+delta] where delta < K and i <= len(A)-K.  That is, this is a chunked
// iteration where the index is not directly compared to the length.
// if isReslice, then delta can be equal to K.
func checkForChunkedIndexBounds(ft *factsTable, b *ssa.Block, index, bound *ssa.Value, isReslice bool) bool {
	if bound.Op != ssaop.OpSliceLen && bound.Op != ssaop.OpStringLen && bound.Op != ssaop.OpSliceCap {
		return false
	}

	// this is a slice bounds check against len or capacity,
	// and refers back to a prior check against length, which
	// will also work for the cap since that is not smaller
	// than the length.

	slice := bound.Args[0]
	lim := ft.limits[index.ID]
	if lim.Min < 0 {
		return false
	}
	i, delta := isConstDelta(index)
	if i == nil {
		return false
	}
	if delta < 0 {
		return false
	}
	// special case for blocked iteration over a slice.
	// slicelen > i + delta && <==== if clauses above
	// && index >= 0           <==== if clause above
	// delta >= 0 &&           <==== if clause above
	// slicelen-K >/>= x       <==== checked below
	// && K >=/> delta         <==== checked below
	// then v > w
	// example: i <=/< len - 4/3 means i+{0,1,2,3} are legal indices
	for o := ft.orderings[i.ID]; o != nil; o = o.next {
		if o.d != signed {
			continue
		}
		if ow := o.w; ow.Op == ssaop.OpAdd64 {
			var lenOffset *ssa.Value
			if bound := ow.Args[0]; (bound.Op == ssaop.OpSliceLen || bound.Op == ssaop.OpStringLen) && bound.Args[0] == slice {
				lenOffset = ow.Args[1]
			} else if bound := ow.Args[1]; (bound.Op == ssaop.OpSliceLen || bound.Op == ssaop.OpStringLen) && bound.Args[0] == slice {
				lenOffset = ow.Args[0]
			}
			if lenOffset == nil || lenOffset.Op != ssaop.OpConst64 {
				continue
			}
			if K := -lenOffset.AuxInt; K >= 0 {
				or := o.r
				if isReslice {
					K++
				}
				if or == lt {
					or = lt | eq
					K++
				}
				if K < 0 { // We hate thinking about overflow
					continue
				}

				if delta < K && or == lt|eq {
					return true
				}
			}
		}
	}
	return false
}

func (ft *factsTable) addValueFact(b *ssa.Block, v *ssa.Value) {
	switch v.Op {
	case ssaop.OpAdd64, ssaop.OpAdd32, ssaop.OpAdd16, ssaop.OpAdd8:
		x := ft.limits[v.Args[0].ID]
		y := ft.limits[v.Args[1].ID]
		if !unsignedAddOverflows(x.Umax, y.Umax, v.Type) {
			r := gt
			if x.MaybeZero() {
				r |= eq
			}
			ft.update(b, v, v.Args[1], unsigned, r)
			r = gt
			if y.MaybeZero() {
				r |= eq
			}
			ft.update(b, v, v.Args[0], unsigned, r)
		}
		if x.Min >= 0 && !signedAddOverflowsOrUnderflows(x.Max, y.Max, v.Type) {
			r := gt
			if x.MaybeZero() {
				r |= eq
			}
			ft.update(b, v, v.Args[1], signed, r)
		}
		if y.Min >= 0 && !signedAddOverflowsOrUnderflows(x.Max, y.Max, v.Type) {
			r := gt
			if y.MaybeZero() {
				r |= eq
			}
			ft.update(b, v, v.Args[0], signed, r)
		}
		if x.Max <= 0 && !signedAddOverflowsOrUnderflows(x.Min, y.Min, v.Type) {
			r := lt
			if x.MaybeZero() {
				r |= eq
			}
			ft.update(b, v, v.Args[1], signed, r)
		}
		if y.Max <= 0 && !signedAddOverflowsOrUnderflows(x.Min, y.Min, v.Type) {
			r := lt
			if y.MaybeZero() {
				r |= eq
			}
			ft.update(b, v, v.Args[0], signed, r)
		}
	case ssaop.OpSub64, ssaop.OpSub32, ssaop.OpSub16, ssaop.OpSub8:
		x := ft.limits[v.Args[0].ID]
		y := ft.limits[v.Args[1].ID]
		if !unsignedSubUnderflows(x.Umin, y.Umax) {
			r := lt
			if y.MaybeZero() {
				r |= eq
			}
			ft.update(b, v, v.Args[0], unsigned, r)
		}
		// FIXME: we could also do signed facts but the overflow checks are much trickier and I don't need it yet.
	case ssaop.OpAnd64, ssaop.OpAnd32, ssaop.OpAnd16, ssaop.OpAnd8:
		ft.update(b, v, v.Args[0], unsigned, lt|eq)
		ft.update(b, v, v.Args[1], unsigned, lt|eq)
		if ft.isNonNegative(v.Args[0]) {
			ft.update(b, v, v.Args[0], signed, lt|eq)
		}
		if ft.isNonNegative(v.Args[1]) {
			ft.update(b, v, v.Args[1], signed, lt|eq)
		}
	case ssaop.OpOr64, ssaop.OpOr32, ssaop.OpOr16, ssaop.OpOr8:
		// TODO: investigate how to always add facts without much slowdown, see issue #57959
		//ft.update(b, v, v.Args[0], unsigned, gt|eq)
		//ft.update(b, v, v.Args[1], unsigned, gt|eq)
	case ssaop.OpDiv64, ssaop.OpDiv32, ssaop.OpDiv16, ssaop.OpDiv8:
		if !ft.isNonNegative(v.Args[1]) {
			break
		}
		fallthrough
	case ssaop.OpRsh8x64, ssaop.OpRsh8x32, ssaop.OpRsh8x16, ssaop.OpRsh8x8,
		ssaop.OpRsh16x64, ssaop.OpRsh16x32, ssaop.OpRsh16x16, ssaop.OpRsh16x8,
		ssaop.OpRsh32x64, ssaop.OpRsh32x32, ssaop.OpRsh32x16, ssaop.OpRsh32x8,
		ssaop.OpRsh64x64, ssaop.OpRsh64x32, ssaop.OpRsh64x16, ssaop.OpRsh64x8:
		if !ft.isNonNegative(v.Args[0]) {
			break
		}
		fallthrough
	case ssaop.OpDiv64u, ssaop.OpDiv32u, ssaop.OpDiv16u, ssaop.OpDiv8u,
		ssaop.OpRsh8Ux64, ssaop.OpRsh8Ux32, ssaop.OpRsh8Ux16, ssaop.OpRsh8Ux8,
		ssaop.OpRsh16Ux64, ssaop.OpRsh16Ux32, ssaop.OpRsh16Ux16, ssaop.OpRsh16Ux8,
		ssaop.OpRsh32Ux64, ssaop.OpRsh32Ux32, ssaop.OpRsh32Ux16, ssaop.OpRsh32Ux8,
		ssaop.OpRsh64Ux64, ssaop.OpRsh64Ux32, ssaop.OpRsh64Ux16, ssaop.OpRsh64Ux8:
		switch add := v.Args[0]; add.Op {
		// round-up division pattern; given:
		// v = (x + y) / z
		// if y < z then v <= x
		case ssaop.OpAdd64, ssaop.OpAdd32, ssaop.OpAdd16, ssaop.OpAdd8:
			z := v.Args[1]
			zl := ft.limits[z.ID]
			var uminDivisor uint64
			switch v.Op {
			case ssaop.OpDiv64u, ssaop.OpDiv32u, ssaop.OpDiv16u, ssaop.OpDiv8u,
				ssaop.OpDiv64, ssaop.OpDiv32, ssaop.OpDiv16, ssaop.OpDiv8:
				uminDivisor = zl.Umin
			case ssaop.OpRsh8Ux64, ssaop.OpRsh8Ux32, ssaop.OpRsh8Ux16, ssaop.OpRsh8Ux8,
				ssaop.OpRsh16Ux64, ssaop.OpRsh16Ux32, ssaop.OpRsh16Ux16, ssaop.OpRsh16Ux8,
				ssaop.OpRsh32Ux64, ssaop.OpRsh32Ux32, ssaop.OpRsh32Ux16, ssaop.OpRsh32Ux8,
				ssaop.OpRsh64Ux64, ssaop.OpRsh64Ux32, ssaop.OpRsh64Ux16, ssaop.OpRsh64Ux8,
				ssaop.OpRsh8x64, ssaop.OpRsh8x32, ssaop.OpRsh8x16, ssaop.OpRsh8x8,
				ssaop.OpRsh16x64, ssaop.OpRsh16x32, ssaop.OpRsh16x16, ssaop.OpRsh16x8,
				ssaop.OpRsh32x64, ssaop.OpRsh32x32, ssaop.OpRsh32x16, ssaop.OpRsh32x8,
				ssaop.OpRsh64x64, ssaop.OpRsh64x32, ssaop.OpRsh64x16, ssaop.OpRsh64x8:
				uminDivisor = 1 << zl.Umin
			default:
				panic("unreachable")
			}

			x := add.Args[0]
			xl := ft.limits[x.ID]
			y := add.Args[1]
			yl := ft.limits[y.ID]
			if !unsignedAddOverflows(xl.Umax, yl.Umax, add.Type) {
				if xl.Umax < uminDivisor {
					ft.update(b, v, y, unsigned, lt|eq)
				}
				if yl.Umax < uminDivisor {
					ft.update(b, v, x, unsigned, lt|eq)
				}
			}
		}
		ft.update(b, v, v.Args[0], unsigned, lt|eq)
	case ssaop.OpMod64, ssaop.OpMod32, ssaop.OpMod16, ssaop.OpMod8:
		if !ft.isNonNegative(v.Args[0]) || !ft.isNonNegative(v.Args[1]) {
			break
		}
		fallthrough
	case ssaop.OpMod64u, ssaop.OpMod32u, ssaop.OpMod16u, ssaop.OpMod8u:
		ft.update(b, v, v.Args[0], unsigned, lt|eq)
		// Note: we have to be careful that this doesn't imply
		// that the modulus is >0, which isn't true until *after*
		// the mod instruction executes (and thus panics if the
		// modulus is 0). See issue 67625.
		ft.update(b, v, v.Args[1], unsigned, lt)
	case ssaop.OpStringLen:
		if v.Args[0].Op == ssaop.OpStringMake {
			ft.update(b, v, v.Args[0].Args[1], signed, eq)
		}
	case ssaop.OpSliceLen:
		if v.Args[0].Op == ssaop.OpSliceMake {
			ft.update(b, v, v.Args[0].Args[1], signed, eq)
		}
	case ssaop.OpSliceCap:
		if v.Args[0].Op == ssaop.OpSliceMake {
			ft.update(b, v, v.Args[0].Args[2], signed, eq)
		}
	case ssaop.OpIsInBounds:
		if checkForChunkedIndexBounds(ft, b, v.Args[0], v.Args[1], false) {
			if b.Func.Pass.Debug > 0 {
				b.Func.Warnl(v.Pos, "Proved %s for blocked indexing", v.Op)
			}
			ft.booleanTrue(v)
		}
	case ssaop.OpIsSliceInBounds:
		if checkForChunkedIndexBounds(ft, b, v.Args[0], v.Args[1], true) {
			if b.Func.Pass.Debug > 0 {
				b.Func.Warnl(v.Pos, "Proved %s for blocked reslicing", v.Op)
			}
			ft.booleanTrue(v)
		}
	case ssaop.OpPhi:
		addLocalFactsPhi(ft, v)
	}
}

func addLocalFactsPhi(ft *factsTable, v *ssa.Value) {
	// Look for phis that implement min/max.
	//   z:
	//      c = Less64 x y (or other Less/Leq operation)
	//      If c -> bx by
	//   bx: <- z
	//       -> b ...
	//   by: <- z
	//      -> b ...
	//   b: <- bx by
	//      v = Phi x y
	// Then v is either min or max of x,y.
	// If it is the min, then we deduce v <= x && v <= y.
	// If it is the max, then we deduce v >= x && v >= y.
	// The min case is useful for the copy builtin, see issue 16833.
	if len(v.Args) != 2 {
		return
	}
	b := v.Block
	x := v.Args[0]
	y := v.Args[1]
	bx := b.Preds[0].B
	by := b.Preds[1].B
	var z *ssa.Block // branch point
	switch {
	case bx == by: // bx == by == z case
		z = bx
	case by.UniquePred() == bx: // bx == z case
		z = bx
	case bx.UniquePred() == by: // by == z case
		z = by
	case bx.UniquePred() == by.UniquePred():
		z = bx.UniquePred()
	}
	if z == nil || z.Kind != block.BlockIf {
		return
	}
	c := z.Controls[0]
	if len(c.Args) != 2 {
		return
	}
	var isMin bool // if c, a less-than comparison, is true, phi chooses x.
	if bx == z {
		isMin = b.Preds[0].I == 0
	} else {
		isMin = bx.Preds[0].I == 0
	}
	if c.Args[0] == x && c.Args[1] == y {
		// ok
	} else if c.Args[0] == y && c.Args[1] == x {
		// Comparison is reversed from how the values are listed in the Phi.
		isMin = !isMin
	} else {
		// Not comparing x and y.
		return
	}
	var dom domain
	switch c.Op {
	case ssaop.OpLess64, ssaop.OpLess32, ssaop.OpLess16, ssaop.OpLess8, ssaop.OpLeq64, ssaop.OpLeq32, ssaop.OpLeq16, ssaop.OpLeq8:
		dom = signed
	case ssaop.OpLess64U, ssaop.OpLess32U, ssaop.OpLess16U, ssaop.OpLess8U, ssaop.OpLeq64U, ssaop.OpLeq32U, ssaop.OpLeq16U, ssaop.OpLeq8U:
		dom = unsigned
	default:
		return
	}
	var rel relation
	if isMin {
		rel = lt | eq
	} else {
		rel = gt | eq
	}
	ft.update(b, v, x, dom, rel)
	ft.update(b, v, y, dom, rel)
}

var ctzNonZeroOp = map[ssaop.Op]ssaop.Op{
	ssaop.OpCtz8:  ssaop.OpCtz8NonZero,
	ssaop.OpCtz16: ssaop.OpCtz16NonZero,
	ssaop.OpCtz32: ssaop.OpCtz32NonZero,
	ssaop.OpCtz64: ssaop.OpCtz64NonZero,
}
var mostNegativeDividend = map[ssaop.Op]int64{
	ssaop.OpDiv16: -1 << 15,
	ssaop.OpMod16: -1 << 15,
	ssaop.OpDiv32: -1 << 31,
	ssaop.OpMod32: -1 << 31,
	ssaop.OpDiv64: -1 << 63,
	ssaop.OpMod64: -1 << 63,
}
var unsignedOp = map[ssaop.Op]ssaop.Op{
	ssaop.OpDiv8:     ssaop.OpDiv8u,
	ssaop.OpDiv16:    ssaop.OpDiv16u,
	ssaop.OpDiv32:    ssaop.OpDiv32u,
	ssaop.OpDiv64:    ssaop.OpDiv64u,
	ssaop.OpMod8:     ssaop.OpMod8u,
	ssaop.OpMod16:    ssaop.OpMod16u,
	ssaop.OpMod32:    ssaop.OpMod32u,
	ssaop.OpMod64:    ssaop.OpMod64u,
	ssaop.OpRsh8x8:   ssaop.OpRsh8Ux8,
	ssaop.OpRsh8x16:  ssaop.OpRsh8Ux16,
	ssaop.OpRsh8x32:  ssaop.OpRsh8Ux32,
	ssaop.OpRsh8x64:  ssaop.OpRsh8Ux64,
	ssaop.OpRsh16x8:  ssaop.OpRsh16Ux8,
	ssaop.OpRsh16x16: ssaop.OpRsh16Ux16,
	ssaop.OpRsh16x32: ssaop.OpRsh16Ux32,
	ssaop.OpRsh16x64: ssaop.OpRsh16Ux64,
	ssaop.OpRsh32x8:  ssaop.OpRsh32Ux8,
	ssaop.OpRsh32x16: ssaop.OpRsh32Ux16,
	ssaop.OpRsh32x32: ssaop.OpRsh32Ux32,
	ssaop.OpRsh32x64: ssaop.OpRsh32Ux64,
	ssaop.OpRsh64x8:  ssaop.OpRsh64Ux8,
	ssaop.OpRsh64x16: ssaop.OpRsh64Ux16,
	ssaop.OpRsh64x32: ssaop.OpRsh64Ux32,
	ssaop.OpRsh64x64: ssaop.OpRsh64Ux64,
}

var bytesizeToConst = [...]ssaop.Op{
	8 / 8:  ssaop.OpConst8,
	16 / 8: ssaop.OpConst16,
	32 / 8: ssaop.OpConst32,
	64 / 8: ssaop.OpConst64,
}
var bytesizeToNeq = [...]ssaop.Op{
	8 / 8:  ssaop.OpNeq8,
	16 / 8: ssaop.OpNeq16,
	32 / 8: ssaop.OpNeq32,
	64 / 8: ssaop.OpNeq64,
}
var bytesizeToAnd = [...]ssaop.Op{
	8 / 8:  ssaop.OpAnd8,
	16 / 8: ssaop.OpAnd16,
	32 / 8: ssaop.OpAnd32,
	64 / 8: ssaop.OpAnd64,
}

var invertEqNeqOp = map[ssaop.Op]ssaop.Op{
	ssaop.OpEq8:  ssaop.OpNeq8,
	ssaop.OpNeq8: ssaop.OpEq8,

	ssaop.OpEq16:  ssaop.OpNeq16,
	ssaop.OpNeq16: ssaop.OpEq16,

	ssaop.OpEq32:  ssaop.OpNeq32,
	ssaop.OpNeq32: ssaop.OpEq32,

	ssaop.OpEq64:  ssaop.OpNeq64,
	ssaop.OpNeq64: ssaop.OpEq64,
}

func (ft *factsTable) simplifyValue(b *ssa.Block, v *ssa.Value) {
	switch v.Op {
	case ssaop.OpStaticLECall:
		if b.Func.Pass.Debug > 0 && len(v.Args) == 2 {
			fn := AuxToCall(v.Aux).Fn
			if fn != nil && strings.Contains(fn.String(), "prove") {
				// Print bounds of any argument to single-arg function with "prove" in name,
				// for debugging and especially for test/prove.go.
				// (v.Args[1] is mem).
				x := v.Args[0]
				b.Func.Warnl(v.Pos, "Proved %v (%v)", ft.limits[x.ID], x)
			}
		}
	case ssaop.OpSlicemask:
		// Replace OpSlicemask operations in b with constants where possible.
		cap := v.Args[0]
		x, delta := isConstDelta(cap)
		if x != nil {
			// slicemask(x + y)
			// if x is larger than -y (y is negative), then slicemask is -1.
			lim := ft.limits[x.ID]
			if lim.Umin > uint64(-delta) {
				if v.Type.Size() == 8 {
					v.Reset(ssaop.OpConst64)
				} else {
					v.Reset(ssaop.OpConst32)
				}
				if b.Func.Pass.Debug > 0 {
					b.Func.Warnl(v.Pos, "Proved slicemask not needed")
				}
				v.AuxInt = -1
			}
			break
		}
		lim := ft.limits[cap.ID]
		if lim.Umin > 0 {
			if v.Type.Size() == 8 {
				v.Reset(ssaop.OpConst64)
			} else {
				v.Reset(ssaop.OpConst32)
			}
			if b.Func.Pass.Debug > 0 {
				b.Func.Warnl(v.Pos, "Proved slicemask not needed (by limit)")
			}
			v.AuxInt = -1
		}

	case ssaop.OpCtz8, ssaop.OpCtz16, ssaop.OpCtz32, ssaop.OpCtz64:
		// On some architectures, notably amd64, we can generate much better
		// code for CtzNN if we know that the argument is non-zero.
		// Capture that information here for use in arch-specific optimizations.
		x := v.Args[0]
		lim := ft.limits[x.ID]
		if lim.Umin > 0 || lim.Min > 0 || lim.Max < 0 {
			if b.Func.Pass.Debug > 0 {
				b.Func.Warnl(v.Pos, "Proved %v non-zero", v.Op)
			}
			v.Op = ctzNonZeroOp[v.Op]
		}
	case ssaop.OpRsh8x8, ssaop.OpRsh8x16, ssaop.OpRsh8x32, ssaop.OpRsh8x64,
		ssaop.OpRsh16x8, ssaop.OpRsh16x16, ssaop.OpRsh16x32, ssaop.OpRsh16x64,
		ssaop.OpRsh32x8, ssaop.OpRsh32x16, ssaop.OpRsh32x32, ssaop.OpRsh32x64,
		ssaop.OpRsh64x8, ssaop.OpRsh64x16, ssaop.OpRsh64x32, ssaop.OpRsh64x64:
		if ft.isNonNegative(v.Args[0]) {
			if b.Func.Pass.Debug > 0 {
				b.Func.Warnl(v.Pos, "Proved %v is unsigned", v.Op)
			}
			v.Op = unsignedOp[v.Op]
		}
		fallthrough
	case ssaop.OpLsh8x8, ssaop.OpLsh8x16, ssaop.OpLsh8x32, ssaop.OpLsh8x64,
		ssaop.OpLsh16x8, ssaop.OpLsh16x16, ssaop.OpLsh16x32, ssaop.OpLsh16x64,
		ssaop.OpLsh32x8, ssaop.OpLsh32x16, ssaop.OpLsh32x32, ssaop.OpLsh32x64,
		ssaop.OpLsh64x8, ssaop.OpLsh64x16, ssaop.OpLsh64x32, ssaop.OpLsh64x64,
		ssaop.OpRsh8Ux8, ssaop.OpRsh8Ux16, ssaop.OpRsh8Ux32, ssaop.OpRsh8Ux64,
		ssaop.OpRsh16Ux8, ssaop.OpRsh16Ux16, ssaop.OpRsh16Ux32, ssaop.OpRsh16Ux64,
		ssaop.OpRsh32Ux8, ssaop.OpRsh32Ux16, ssaop.OpRsh32Ux32, ssaop.OpRsh32Ux64,
		ssaop.OpRsh64Ux8, ssaop.OpRsh64Ux16, ssaop.OpRsh64Ux32, ssaop.OpRsh64Ux64:
		// Check whether, for a << b, we know that b
		// is strictly less than the number of bits in a.
		by := v.Args[1]
		lim := ft.limits[by.ID]
		bits := 8 * v.Args[0].Type.Size()
		if lim.Umax < uint64(bits) || (lim.Max < bits && ft.isNonNegative(by)) {
			v.AuxInt = 1 // see shiftIsBounded
			if b.Func.Pass.Debug > 0 && !by.IsGenericIntConst() {
				b.Func.Warnl(v.Pos, "Proved %v bounded", v.Op)
			}
		}
	case ssaop.OpDiv8, ssaop.OpDiv16, ssaop.OpDiv32, ssaop.OpDiv64, ssaop.OpMod8, ssaop.OpMod16, ssaop.OpMod32, ssaop.OpMod64:
		p, q := ft.limits[v.Args[0].ID], ft.limits[v.Args[1].ID] // p/q
		if p.Nonnegative() && q.Nonnegative() {
			if b.Func.Pass.Debug > 0 {
				b.Func.Warnl(v.Pos, "Proved %v is unsigned", v.Op)
			}
			v.Op = unsignedOp[v.Op]
			v.AuxInt = 0
			break
		}
		// Fixup code can be avoided on x86 if we know
		//  the divisor is not -1 or the dividend > MinIntNN.
		if v.Op != ssaop.OpDiv8 && v.Op != ssaop.OpMod8 && (q.Max < -1 || q.Min > -1 || p.Min > mostNegativeDividend[v.Op]) {
			// See DivisionNeedsFixUp in rewrite.go.
			// v.AuxInt = 1 means we have proved that the divisor is not -1
			// or that the dividend is not the most negative integer,
			// so we do not need to add fix-up code.
			if b.Func.Pass.Debug > 0 {
				b.Func.Warnl(v.Pos, "Proved %v does not need fix-up", v.Op)
			}
			// Only usable on amd64 and 386, and only for ≥ 16-bit ops.
			// Don't modify AuxInt on other architectures, as that can interfere with CSE.
			// (Print the debug info above always, so that test/prove.go can be
			// checked on non-x86 systems.)
			// TODO: add other architectures?
			if b.Func.Config.Arch == "386" || b.Func.Config.Arch == "amd64" {
				v.AuxInt = 1
			}
		}
	case ssaop.OpMul64, ssaop.OpMul32, ssaop.OpMul16, ssaop.OpMul8:
		if vl := ft.limits[v.ID]; vl.Min == vl.Max || vl.Umin == vl.Umax {
			// v is going to be constant folded away; don't "optimize" it.
			break
		}
		x := v.Args[0]
		xl := ft.limits[x.ID]
		y := v.Args[1]
		yl := ft.limits[y.ID]
		if xl.Umin == xl.Umax && IsPowerOfTwo(xl.Umin) ||
			xl.Min == xl.Max && IsPowerOfTwo(xl.Min) ||
			yl.Umin == yl.Umax && IsPowerOfTwo(yl.Umin) ||
			yl.Min == yl.Max && IsPowerOfTwo(yl.Min) {
			// 0,1 * a power of two is better done as a shift
			break
		}
		switch xOne, yOne := xl.Umax <= 1, yl.Umax <= 1; {
		case xOne && yOne:
			v.Op = bytesizeToAnd[v.Type.Size()]
			if b.Func.Pass.Debug > 0 {
				b.Func.Warnl(v.Pos, "Rewrote Mul %v into And", v)
			}
		case yOne && b.Func.Config.HaveCondSelect:
			x, y = y, x
			fallthrough
		case xOne && b.Func.Config.HaveCondSelect:
			if !canCondSelect(v, b.Func.Config.Arch, nil) {
				break
			}
			zero := b.Func.ConstVal(bytesizeToConst[v.Type.Size()], v.Type, 0, true)
			ft.initLimitForNewValue(zero)
			check := b.NewValue2(v.Pos, bytesizeToNeq[v.Type.Size()], types.Types[types.TBOOL], zero, x)
			ft.initLimitForNewValue(check)
			v.Reset(ssaop.OpCondSelect)
			v.AddArg3(y, zero, check)

			if b.Func.Pass.Debug > 0 {
				b.Func.Warnl(v.Pos, "Rewrote Mul %v into CondSelect; %v is bool", v, x)
			}
		}
	case ssaop.OpEq64, ssaop.OpEq32, ssaop.OpEq16, ssaop.OpEq8,
		ssaop.OpNeq64, ssaop.OpNeq32, ssaop.OpNeq16, ssaop.OpNeq8:
		// Canonicalize:
		// [0,1] != 1 → [0,1] == 0
		// [0,1] == 1 → [0,1] != 0
		// Comparison with zero often encode smaller.
		xPos, yPos := 0, 1
		x, y := v.Args[xPos], v.Args[yPos]
		xl, yl := ft.limits[x.ID], ft.limits[y.ID]
		xConst, xIsConst := xl.ConstValue()
		yConst, yIsConst := yl.ConstValue()
		switch {
		case xIsConst && yIsConst:
		case xIsConst:
			xPos, yPos = yPos, xPos
			x, y = y, x
			xl, yl = yl, xl
			xConst, yConst = yConst, xConst
			fallthrough
		case yIsConst:
			if yConst != 1 ||
				xl.Umax > 1 {
				break
			}
			zero := b.Func.ConstVal(bytesizeToConst[x.Type.Size()], x.Type, 0, true)
			ft.initLimitForNewValue(zero)
			oldOp := v.Op
			v.Op = invertEqNeqOp[v.Op]
			v.SetArg(yPos, zero)
			if b.Func.Pass.Debug > 0 {
				b.Func.Warnl(v.Pos, "Rewrote %v (%v) %v argument is boolean-like; rewrote to %v against 0", v, oldOp, x, v.Op)
			}
		}
	case ssaop.OpAnd64, ssaop.OpAnd32, ssaop.OpAnd16, ssaop.OpAnd8:
		x, y := v.Args[0], v.Args[1]
		xl, yl := ft.limits[x.ID], ft.limits[y.ID]
		xConst, xIsConst := xl.ConstValue()
		yConst, yIsConst := yl.ConstValue()
		// Remove no-op Ands
		switch {
		case xIsConst && yIsConst:
		case xIsConst:
			x, y = y, x
			xl, yl = yl, xl
			xConst, yConst = yConst, xConst
			fallthrough
		case yIsConst:
			knownBits, fixedLen := xl.UnsignedFixedLeadingBits()
			varyingLen := 64 - fixedLen
			wantBits := knownBits | (uint64(1)<<varyingLen - 1)
			// wantBits has the fixed bits and the worst case bits (set) for the varying bits
			// if after anding it with y it isn't modified we know the and is always a no-op.
			if wantBits&uint64(yConst) != wantBits {
				break
			}

			oldOp := v.Op
			v.CopyOf(x)
			if b.Func.Pass.Debug > 0 {
				b.Func.Warnl(v.Pos, "Proved %v is a no-op %v", v, oldOp)
			}
		}
	case ssaop.OpOr64, ssaop.OpOr32, ssaop.OpOr16, ssaop.OpOr8:
		x, y := v.Args[0], v.Args[1]
		xl, yl := ft.limits[x.ID], ft.limits[y.ID]
		xConst, xIsConst := xl.ConstValue()
		yConst, yIsConst := yl.ConstValue()
		// Remove no-op Ors
		switch {
		case xIsConst && yIsConst:
		case xIsConst:
			x, y = y, x
			xl, yl = yl, xl
			xConst, yConst = yConst, xConst
			fallthrough
		case yIsConst:
			wantBits, _ := xl.UnsignedFixedLeadingBits()
			// wantBits has the fixed bits and the worst case bits (unset) for the varying bits
			// if after oring it with y it isn't modified we know the or is always a no-op.
			if wantBits|uint64(yConst) != wantBits {
				break
			}

			oldOp := v.Op
			v.CopyOf(x)
			if b.Func.Pass.Debug > 0 {
				b.Func.Warnl(v.Pos, "Proved %v is a no-op %v", v, oldOp)
			}
		}
	}
}

func (ft *factsTable) constantFoldArguments(v *ssa.Value) {
	for i, arg := range v.Args {
		lim := ft.limits[arg.ID]
		constValue, ok := lim.ConstValue()
		if !ok {
			continue
		}
		switch arg.Op {
		case ssaop.OpConst64, ssaop.OpConst32, ssaop.OpConst16, ssaop.OpConst8, ssaop.OpConstBool, ssaop.OpConstNil:
			continue
		}
		typ := arg.Type
		f := v.Block.Func
		var c *ssa.Value
		switch {
		case typ.IsBoolean():
			c = f.ConstBool(typ, constValue != 0)
		case typ.IsInteger() && typ.Size() == 1:
			c = f.ConstInt8(typ, int8(constValue))
		case typ.IsInteger() && typ.Size() == 2:
			c = f.ConstInt16(typ, int16(constValue))
		case typ.IsInteger() && typ.Size() == 4:
			c = f.ConstInt32(typ, int32(constValue))
		case typ.IsInteger() && typ.Size() == 8:
			c = f.ConstInt64(typ, constValue)
		case typ.IsPtrShaped():
			if constValue == 0 {
				c = f.ConstNil(typ)
			} else {
				// Not sure how this might happen, but if it
				// does, just skip it.
				continue
			}
		default:
			// Not sure how this might happen, but if it
			// does, just skip it.
			continue
		}
		v.SetArg(i, c)
		ft.initLimitForNewValue(c)
		if f.Pass.Debug > 1 {
			f.Warnl(v.Pos, "Proved %v's arg %d (%v) is constant %d", v, i, arg, constValue)
		}
	}
}

func (ft *factsTable) simplifyBlock(sdom ssa.SparseTree, b *ssa.Block) {
	if b.Kind != block.BlockIf {
		return
	}

	// Consider outgoing edges from this block.
	parent := b
	for i, branch := range [...]branch{positive, negative} {
		child := parent.Succs[i].B
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

func removeBranch(b *ssa.Block, branch branch) {
	c := b.Controls[0]
	if c != nil && b.Func.Pass.Debug > 0 {
		verb := "Proved"
		if branch == positive {
			verb = "Disproved"
		}
		if b.Func.Pass.Debug > 1 {
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
		b.Kind = block.BlockFirst
		b.ResetControls()
		if branch == positive {
			b.SwapSuccessors()
		}
	} else {
		// TODO: figure out how to remove an entry from a jump table
	}
}

// isConstDelta returns non-nil if v is equivalent to w+delta (signed).
func isConstDelta(v *ssa.Value) (w *ssa.Value, delta int64) {
	cop := ssaop.OpConst64
	switch v.Op {
	case ssaop.OpAdd32, ssaop.OpSub32:
		cop = ssaop.OpConst32
	case ssaop.OpAdd16, ssaop.OpSub16:
		cop = ssaop.OpConst16
	case ssaop.OpAdd8, ssaop.OpSub8:
		cop = ssaop.OpConst8
	}
	switch v.Op {
	case ssaop.OpAdd64, ssaop.OpAdd32, ssaop.OpAdd16, ssaop.OpAdd8:
		if v.Args[0].Op == cop {
			return v.Args[1], v.Args[0].AuxInt
		}
		if v.Args[1].Op == cop {
			return v.Args[0], v.Args[1].AuxInt
		}
	case ssaop.OpSub64, ssaop.OpSub32, ssaop.OpSub16, ssaop.OpSub8:
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
// sign or zero extension.
func isCleanExt(v *ssa.Value) bool {
	switch v.Op {
	case ssaop.OpSignExt8to16, ssaop.OpSignExt8to32, ssaop.OpSignExt8to64,
		ssaop.OpSignExt16to32, ssaop.OpSignExt16to64, ssaop.OpSignExt32to64:
		// signed -> signed is the only value-preserving sign extension
		return v.Args[0].Type.IsSigned() && v.Type.IsSigned()

	case ssaop.OpZeroExt8to16, ssaop.OpZeroExt8to32, ssaop.OpZeroExt8to64,
		ssaop.OpZeroExt16to32, ssaop.OpZeroExt16to64, ssaop.OpZeroExt32to64:
		// unsigned -> signed/unsigned are value-preserving zero extensions
		return !v.Args[0].Type.IsSigned()
	}
	return false
}

// topoSortValue works with an outside loop to implements an O(V + E) toposort.
// Practically E = O(1) so it's practically O(V).
// The algorithm works by maintaining two partitions inside b.Values:
// the first one is sorted, the second one is unsorted. (spos index the first unsorted value).
// Then we run DFS on the graph, once we reach a value that has no unsorted dependencies we
// swap it from the unsorted partition to the end of the sorted partition.
func topoSortValue(b *ssa.Block, positions []uint, spos uint, v *ssa.Value) uint {
	if v.Op == ssaop.OpPhi {
		// phis have no dependencies as far as we care, so they are always sorted
	} else {
		for _, arg := range v.Args {
			if arg.Block != b {
				continue // skip dependencies with other blocks
			}
			argIndex := positions[arg.ID]
			if argIndex < spos {
				continue // the argument is sorted so skip it
			}
			spos = topoSortValue(b, positions, spos, arg)
		}
	}

	vpos := positions[v.ID]
	sv := b.Values[spos]

	b.Values[vpos], b.Values[spos] = sv, v
	positions[v.ID], positions[sv.ID] = spos, vpos

	return spos + 1
}

// topoSortValuesInBlock ensure ranging over b.Values visit values before they are being used.
// It does not consider dependencies with other blocks; thus Phi nodes are considered to not have any dependencies.
func (ft *factsTable) topoSortValuesInBlock(b *ssa.Block) {
	f := b.Func
	want := f.NumValues()

	positions := ft.reusedTopoSortIDsToBlockIndexes
	if want <= cap(positions) {
		positions = positions[:want]
	} else {
		if cap(positions) > 0 {
			f.Cache.FreeUintSlice(positions)
		}
		positions = f.Cache.AllocUintSlice(want)
		ft.reusedTopoSortIDsToBlockIndexes = positions
	}

	for i, v := range b.Values {
		positions[v.ID] = uint(i)
	}

	var sorted uint
	for sorted < uint(len(b.Values)) {
		sorted = topoSortValue(b, positions, sorted, b.Values[sorted])
	}
}
