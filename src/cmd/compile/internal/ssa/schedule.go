// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"container/heap"
	"slices"
	"sort"
)

const (
	ScorePhi       = iota // towards top of block
	ScoreArg              // must occur at the top of the entry block
	ScoreInitMem          // after the args - used as mark by debug info generation
	ScoreReadTuple        // must occur immediately after tuple-generating insn (or call)
	ScoreNilCheck
	ScoreMemory
	ScoreReadFlags
	ScoreDefault
	ScoreFlags
	ScoreControl // towards bottom of block
)

type ValHeap struct {
	a           []*Value
	score       []int8
	inBlockUses []bool
}

func (h ValHeap) Len() int      { return len(h.a) }
func (h ValHeap) Swap(i, j int) { a := h.a; a[i], a[j] = a[j], a[i] }

func (h *ValHeap) Push(x interface{}) {
	// Push and Pop use pointer receivers because they modify the slice's length,
	// not just its contents.
	v := x.(*Value)
	h.a = append(h.a, v)
}
func (h *ValHeap) Pop() interface{} {
	old := h.a
	n := len(old)
	x := old[n-1]
	h.a = old[0 : n-1]
	return x
}
func (h ValHeap) Less(i, j int) bool {
	x := h.a[i]
	y := h.a[j]
	sx := h.score[x.ID]
	sy := h.score[y.ID]
	if c := sx - sy; c != 0 {
		return c < 0 // lower scores come earlier.
	}
	// Note: only scores are required for correct scheduling.
	// Everything else is just heuristics.

	ix := h.inBlockUses[x.ID]
	iy := h.inBlockUses[y.ID]
	if ix != iy {
		return ix // values with in-block uses come earlier
	}

	if x.Pos != y.Pos { // Favor in-order line stepping
		return x.Pos.Before(y.Pos)
	}
	if x.Op != OpPhi {
		if c := len(x.Args) - len(y.Args); c != 0 {
			return c > 0 // smaller args come later
		}
	}
	if c := x.Uses - y.Uses; c != 0 {
		return c > 0 // smaller uses come later
	}
	// These comparisons are fairly arbitrary.
	// The goal here is stability in the face
	// of unrelated changes elsewhere in the compiler.
	if c := x.AuxInt - y.AuxInt; c != 0 {
		return c < 0
	}
	if cmp := x.Type.Compare(y.Type); cmp != types.CMPeq {
		return cmp == types.CMPlt
	}
	return x.ID < y.ID
}

func (op Op) isLoweredGetClosurePtr() bool {
	switch op {
	case OpAMD64LoweredGetClosurePtr, OpPPC64LoweredGetClosurePtr, OpARMLoweredGetClosurePtr, OpARM64LoweredGetClosurePtr,
		Op386LoweredGetClosurePtr, OpMIPS64LoweredGetClosurePtr, OpLOONG64LoweredGetClosurePtr, OpS390XLoweredGetClosurePtr, OpMIPSLoweredGetClosurePtr,
		OpRISCV64LoweredGetClosurePtr, OpWasmLoweredGetClosurePtr:
		return true
	}
	return false
}

// Schedule the Values in each Block. After this phase returns, the
// order of b.Values matters and is the order in which those values
// will appear in the assembly output. For now it generates a
// reasonable valid schedule using a priority queue. TODO(khr):
// schedule smarter.
func schedule(f *Func) {
	// reusable priority queue
	priq := new(ValHeap)

	// "priority" for a value
	score := f.Cache.allocInt8Slice(f.NumValues())
	defer f.Cache.freeInt8Slice(score)

	// maps mem values to the next live memory value
	nextMem := f.Cache.allocValueSlice(f.NumValues())
	defer f.Cache.freeValueSlice(nextMem)

	// inBlockUses records whether a value is used in the block
	// in which it lives. (block control values don't count as uses.)
	inBlockUses := f.Cache.allocBoolSlice(f.NumValues())
	defer f.Cache.freeBoolSlice(inBlockUses)
	if f.Config.optimize {
		for _, b := range f.Blocks {
			for _, v := range b.Values {
				for _, a := range v.Args {
					if a.Block == b {
						inBlockUses[a.ID] = true
					}
				}
			}
		}
	}
	priq.inBlockUses = inBlockUses

	for _, b := range f.Blocks {
		// Compute score. Larger numbers are scheduled closer to the end of the block.
		for _, v := range b.Values {
			switch {
			case v.Op.isLoweredGetClosurePtr():
				// We also score GetLoweredClosurePtr as early as possible to ensure that the
				// context register is not stomped. GetLoweredClosurePtr should only appear
				// in the entry block where there are no phi functions, so there is no
				// conflict or ambiguity here.
				if b != f.Entry {
					f.Fatalf("LoweredGetClosurePtr appeared outside of entry block, b=%s", b.String())
				}
				score[v.ID] = ScorePhi
			case opcodeTable[v.Op].nilCheck:
				// Nil checks must come before loads from the same address.
				score[v.ID] = ScoreNilCheck
			case v.Op == OpPhi:
				// We want all the phis first.
				score[v.ID] = ScorePhi
			case v.Op == OpArgIntReg || v.Op == OpArgFloatReg:
				// In-register args must be scheduled as early as possible to ensure that they
				// are not stomped (similar to the closure pointer above).
				// In particular, they need to come before regular OpArg operations because
				// of how regalloc places spill code (see regalloc.go:placeSpills:mustBeFirst).
				if b != f.Entry {
					f.Fatalf("%s appeared outside of entry block, b=%s", v.Op, b.String())
				}
				score[v.ID] = ScorePhi
			case v.Op == OpArg || v.Op == OpSP || v.Op == OpSB:
				// We want all the args as early as possible, for better debugging.
				score[v.ID] = ScoreArg
			case v.Op == OpInitMem:
				// Early, but after args. See debug.go:buildLocationLists
				score[v.ID] = ScoreInitMem
			case v.Type.IsMemory():
				// Schedule stores as early as possible. This tends to
				// reduce register pressure.
				score[v.ID] = ScoreMemory
			case v.Op == OpSelect0 || v.Op == OpSelect1 || v.Op == OpSelectN:
				// Tuple selectors need to appear immediately after the instruction
				// that generates the tuple.
				score[v.ID] = ScoreReadTuple
			case v.hasFlagInput():
				// Schedule flag-reading ops earlier, to minimize the lifetime
				// of flag values.
				score[v.ID] = ScoreReadFlags
			case v.isFlagOp():
				// Schedule flag register generation as late as possible.
				// This makes sure that we only have one live flags
				// value at a time.
				// Note that this case is after the case above, so values
				// which both read and generate flags are given ScoreReadFlags.
				score[v.ID] = ScoreFlags
			default:
				score[v.ID] = ScoreDefault
				// If we're reading flags, schedule earlier to keep flag lifetime short.
				for _, a := range v.Args {
					if a.isFlagOp() {
						score[v.ID] = ScoreReadFlags
					}
				}
			}
		}
		for _, c := range b.ControlValues() {
			// Force the control values to be scheduled at the end,
			// unless they have other special priority.
			if c.Block != b || score[c.ID] < ScoreReadTuple {
				continue
			}
			if score[c.ID] == ScoreReadTuple {
				score[c.Args[0].ID] = ScoreControl
				continue
			}
			score[c.ID] = ScoreControl
		}
	}
	priq.score = score

	// An edge represents a scheduling constraint that x must appear before y in the schedule.
	type edge struct {
		x, y *Value
	}
	edges := make([]edge, 0, 64)

	// inEdges is the number of scheduling edges incoming from values that haven't been scheduled yet.
	// i.e. inEdges[y.ID] = |e in edges where e.y == y and e.x is not in the schedule yet|.
	inEdges := f.Cache.allocInt32Slice(f.NumValues())
	defer f.Cache.freeInt32Slice(inEdges)

	for _, b := range f.Blocks {
		edges = edges[:0]
		// Standard edges: from the argument of a value to that value.
		for _, v := range b.Values {
			if v.Op == OpPhi {
				// If a value is used by a phi, it does not induce
				// a scheduling edge because that use is from the
				// previous iteration.
				continue
			}
			for _, a := range v.Args {
				if a.Block == b {
					edges = append(edges, edge{a, v})
				}
			}
		}

		// Find store chain for block.
		// Store chains for different blocks overwrite each other, so
		// the calculated store chain is good only for this block.
		for _, v := range b.Values {
			if v.Op != OpPhi && v.Op != OpInitMem && v.Type.IsMemory() {
				nextMem[v.MemoryArg().ID] = v
			}
		}

		// Add edges to enforce that any load must come before the following store.
		for _, v := range b.Values {
			if v.Op == OpPhi || v.Type.IsMemory() {
				continue
			}
			w := v.MemoryArg()
			if w == nil {
				continue
			}
			if s := nextMem[w.ID]; s != nil && s.Block == b {
				edges = append(edges, edge{v, s})
			}
		}

		// Sort all the edges by source Value ID.
		slices.SortFunc(edges, func(i, j edge) int {
			return int(i.x.ID - j.x.ID)
		})
		// Compute inEdges for values in this block.
		for _, e := range edges {
			inEdges[e.y.ID]++
		}

		// Initialize priority queue with schedulable values.
		priq.a = priq.a[:0]
		for _, v := range b.Values {
			if inEdges[v.ID] == 0 {
				heap.Push(priq, v)
			}
		}

		// Produce the schedule. Pick the highest priority scheduleable value,
		// add it to the schedule, add any of its uses that are now scheduleable
		// to the queue, and repeat.
		nv := len(b.Values)
		b.Values = b.Values[:0]
		for priq.Len() > 0 {
			// Schedule the next schedulable value in priority order.
			v := heap.Pop(priq).(*Value)
			b.Values = append(b.Values, v)

			// Find all the scheduling edges out from this value.
			i := sort.Search(len(edges), func(i int) bool {
				return edges[i].x.ID >= v.ID
			})
			j := sort.Search(len(edges), func(i int) bool {
				return edges[i].x.ID > v.ID
			})
			// Decrement inEdges for each target of edges from v.
			for _, e := range edges[i:j] {
				inEdges[e.y.ID]--
				if inEdges[e.y.ID] == 0 {
					heap.Push(priq, e.y)
				}
			}
		}
		if len(b.Values) != nv {
			f.Fatalf("schedule does not include all values in block %s", b)
		}
	}

	// Remove SPanchored now that we've scheduled.
	// Also unlink nil checks now that ordering is assured
	// between the nil check and the uses of the nil-checked pointer.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			for i, a := range v.Args {
				if a.Op == OpSPanchored || opcodeTable[a.Op].nilCheck {
					v.SetArg(i, a.Args[0])
				}
			}
		}
		for i, c := range b.ControlValues() {
			if c.Op == OpSPanchored || opcodeTable[c.Op].nilCheck {
				b.ReplaceControl(i, c.Args[0])
			}
		}
	}
	for _, b := range f.Blocks {
		i := 0
		for _, v := range b.Values {
			if v.Op == OpSPanchored {
				// Free this value
				if v.Uses != 0 {
					base.Fatalf("SPAnchored still has %d uses", v.Uses)
				}
				v.resetArgs()
				f.freeValue(v)
			} else {
				if opcodeTable[v.Op].nilCheck {
					if v.Uses != 0 {
						base.Fatalf("nilcheck still has %d uses", v.Uses)
					}
					// We can't delete the nil check, but we mark
					// it as having void type so regalloc won't
					// try to allocate a register for it.
					v.Type = types.TypeVoid
				}
				b.Values[i] = v
				i++
			}
		}
		b.truncateValues(i)
	}

	f.scheduled = true
}

// storeOrder orders values with respect to stores. That is,
// if v transitively depends on store s, v is ordered after s,
// otherwise v is ordered before s.
// Specifically, values are ordered like
//
//	store1
//	NilCheck that depends on store1
//	other values that depends on store1
//	store2
//	NilCheck that depends on store2
//	other values that depends on store2
//	...
//
// The order of non-store and non-NilCheck values are undefined
// (not necessarily dependency order). This should be cheaper
// than a full scheduling as done above.
// Note that simple dependency order won't work: there is no
// dependency between NilChecks and values like IsNonNil.
// Auxiliary data structures are passed in as arguments, so
// that they can be allocated in the caller and be reused.
// This function takes care of reset them.
func storeOrder(values []*Value, sset *sparseSet, storeNumber []int32) []*Value {
	if len(values) == 0 {
		return values
	}

	f := values[0].Block.Func

	// find all stores

	// Members of values that are store values.
	// A constant bound allows this to be stack-allocated. 64 is
	// enough to cover almost every storeOrder call.
	stores := make([]*Value, 0, 64)
	hasNilCheck := false
	sset.clear() // sset is the set of stores that are used in other values
	for _, v := range values {
		if v.Type.IsMemory() {
			stores = append(stores, v)
			if v.Op == OpInitMem || v.Op == OpPhi {
				continue
			}
			sset.add(v.MemoryArg().ID) // record that v's memory arg is used
		}
		if v.Op == OpNilCheck {
			hasNilCheck = true
		}
	}
	if len(stores) == 0 || !hasNilCheck && f.pass.name == "nilcheckelim" {
		// there is no store, the order does not matter
		return values
	}

	// find last store, which is the one that is not used by other stores
	var last *Value
	for _, v := range stores {
		if !sset.contains(v.ID) {
			if last != nil {
				f.Fatalf("two stores live simultaneously: %v and %v", v, last)
			}
			last = v
		}
	}

	// We assign a store number to each value. Store number is the
	// index of the latest store that this value transitively depends.
	// The i-th store in the current block gets store number 3*i. A nil
	// check that depends on the i-th store gets store number 3*i+1.
	// Other values that depends on the i-th store gets store number 3*i+2.
	// Special case: 0 -- unassigned, 1 or 2 -- the latest store it depends
	// is in the previous block (or no store at all, e.g. value is Const).
	// First we assign the number to all stores by walking back the store chain,
	// then assign the number to other values in DFS order.
	count := make([]int32, 3*(len(stores)+1))
	sset.clear() // reuse sparse set to ensure that a value is pushed to stack only once
	for n, w := len(stores), last; n > 0; n-- {
		storeNumber[w.ID] = int32(3 * n)
		count[3*n]++
		sset.add(w.ID)
		if w.Op == OpInitMem || w.Op == OpPhi {
			if n != 1 {
				f.Fatalf("store order is wrong: there are stores before %v", w)
			}
			break
		}
		w = w.MemoryArg()
	}
	var stack []*Value
	for _, v := range values {
		if sset.contains(v.ID) {
			// in sset means v is a store, or already pushed to stack, or already assigned a store number
			continue
		}
		stack = append(stack, v)
		sset.add(v.ID)

		for len(stack) > 0 {
			w := stack[len(stack)-1]
			if storeNumber[w.ID] != 0 {
				stack = stack[:len(stack)-1]
				continue
			}
			if w.Op == OpPhi {
				// Phi value doesn't depend on store in the current block.
				// Do this early to avoid dependency cycle.
				storeNumber[w.ID] = 2
				count[2]++
				stack = stack[:len(stack)-1]
				continue
			}

			max := int32(0) // latest store dependency
			argsdone := true
			for _, a := range w.Args {
				if a.Block != w.Block {
					continue
				}
				if !sset.contains(a.ID) {
					stack = append(stack, a)
					sset.add(a.ID)
					argsdone = false
					break
				}
				if storeNumber[a.ID]/3 > max {
					max = storeNumber[a.ID] / 3
				}
			}
			if !argsdone {
				continue
			}

			n := 3*max + 2
			if w.Op == OpNilCheck {
				n = 3*max + 1
			}
			storeNumber[w.ID] = n
			count[n]++
			stack = stack[:len(stack)-1]
		}
	}

	// convert count to prefix sum of counts: count'[i] = sum_{j<=i} count[i]
	for i := range count {
		if i == 0 {
			continue
		}
		count[i] += count[i-1]
	}
	if count[len(count)-1] != int32(len(values)) {
		f.Fatalf("storeOrder: value is missing, total count = %d, values = %v", count[len(count)-1], values)
	}

	// place values in count-indexed bins, which are in the desired store order
	order := make([]*Value, len(values))
	for _, v := range values {
		s := storeNumber[v.ID]
		order[count[s-1]] = v
		count[s-1]++
	}

	// Order nil checks in source order. We want the first in source order to trigger.
	// If two are on the same line, we don't really care which happens first.
	// See issue 18169.
	if hasNilCheck {
		start := -1
		for i, v := range order {
			if v.Op == OpNilCheck {
				if start == -1 {
					start = i
				}
			} else {
				if start != -1 {
					slices.SortFunc(order[start:i], valuePosCmp)
					start = -1
				}
			}
		}
		if start != -1 {
			slices.SortFunc(order[start:], valuePosCmp)
		}
	}

	return order
}

// isFlagOp reports if v is an OP with the flag type.
func (v *Value) isFlagOp() bool {
	if v.Type.IsFlags() || v.Type.IsTuple() && v.Type.FieldType(1).IsFlags() {
		return true
	}
	// PPC64 carry generators put their carry in a non-flag-typed register
	// in their output.
	switch v.Op {
	case OpPPC64SUBC, OpPPC64ADDC, OpPPC64SUBCconst, OpPPC64ADDCconst:
		return true
	}
	return false
}

// hasFlagInput reports whether v has a flag value as any of its inputs.
func (v *Value) hasFlagInput() bool {
	for _, a := range v.Args {
		if a.isFlagOp() {
			return true
		}
	}
	// PPC64 carry dependencies are conveyed through their final argument,
	// so we treat those operations as taking flags as well.
	switch v.Op {
	case OpPPC64SUBE, OpPPC64ADDE, OpPPC64SUBZEzero, OpPPC64ADDZE, OpPPC64ADDZEzero:
		return true
	}
	return false
}

func valuePosCmp(a, b *Value) int {
	if a.Pos.Before(b.Pos) {
		return -1
	}
	if a.Pos.After(b.Pos) {
		return +1
	}
	return 0
}
