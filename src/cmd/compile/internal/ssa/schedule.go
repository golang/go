// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"container/heap"
	"sort"
)

const (
	ScorePhi = iota // towards top of block
	ScoreArg
	ScoreNilCheck
	ScoreReadTuple
	ScoreVarDef
	ScoreMemory
	ScoreReadFlags
	ScoreDefault
	ScoreFlags
	ScoreControl // towards bottom of block
)

type ValHeap struct {
	a     []*Value
	score []int8
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
		return c > 0 // higher score comes later.
	}
	if x.Pos != y.Pos { // Favor in-order line stepping
		return x.Pos.After(y.Pos)
	}
	if x.Op != OpPhi {
		if c := len(x.Args) - len(y.Args); c != 0 {
			return c < 0 // smaller args comes later
		}
	}
	if c := x.Uses - y.Uses; c != 0 {
		return c < 0 // smaller uses come later
	}
	// These comparisons are fairly arbitrary.
	// The goal here is stability in the face
	// of unrelated changes elsewhere in the compiler.
	if c := x.AuxInt - y.AuxInt; c != 0 {
		return c > 0
	}
	if cmp := x.Type.Compare(y.Type); cmp != types.CMPeq {
		return cmp == types.CMPgt
	}
	return x.ID > y.ID
}

func (op Op) isLoweredGetClosurePtr() bool {
	switch op {
	case OpAMD64LoweredGetClosurePtr, OpPPC64LoweredGetClosurePtr, OpARMLoweredGetClosurePtr, OpARM64LoweredGetClosurePtr,
		Op386LoweredGetClosurePtr, OpMIPS64LoweredGetClosurePtr, OpS390XLoweredGetClosurePtr, OpMIPSLoweredGetClosurePtr,
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
	// For each value, the number of times it is used in the block
	// by values that have not been scheduled yet.
	uses := make([]int32, f.NumValues())

	// reusable priority queue
	priq := new(ValHeap)

	// "priority" for a value
	score := make([]int8, f.NumValues())

	// scheduling order. We queue values in this list in reverse order.
	// A constant bound allows this to be stack-allocated. 64 is
	// enough to cover almost every schedule call.
	order := make([]*Value, 0, 64)

	// maps mem values to the next live memory value
	nextMem := make([]*Value, f.NumValues())
	// additional pretend arguments for each Value. Used to enforce load/store ordering.
	additionalArgs := make([][]*Value, f.NumValues())

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
			case v.Op == OpAMD64LoweredNilCheck || v.Op == OpPPC64LoweredNilCheck ||
				v.Op == OpARMLoweredNilCheck || v.Op == OpARM64LoweredNilCheck ||
				v.Op == Op386LoweredNilCheck || v.Op == OpMIPS64LoweredNilCheck ||
				v.Op == OpS390XLoweredNilCheck || v.Op == OpMIPSLoweredNilCheck ||
				v.Op == OpRISCV64LoweredNilCheck || v.Op == OpWasmLoweredNilCheck:
				// Nil checks must come before loads from the same address.
				score[v.ID] = ScoreNilCheck
			case v.Op == OpPhi:
				// We want all the phis first.
				score[v.ID] = ScorePhi
			case v.Op == OpVarDef:
				// We want all the vardefs next.
				score[v.ID] = ScoreVarDef
			case v.Op == OpArg:
				// We want all the args as early as possible, for better debugging.
				score[v.ID] = ScoreArg
			case v.Type.IsMemory():
				// Schedule stores as early as possible. This tends to
				// reduce register pressure. It also helps make sure
				// VARDEF ops are scheduled before the corresponding LEA.
				score[v.ID] = ScoreMemory
			case v.Op == OpSelect0 || v.Op == OpSelect1:
				// Schedule the pseudo-op of reading part of a tuple
				// immediately after the tuple-generating op, since
				// this value is already live. This also removes its
				// false dependency on the other part of the tuple.
				// Also ensures tuple is never spilled.
				score[v.ID] = ScoreReadTuple
			case v.Type.IsFlags() || v.Type.IsTuple() && v.Type.FieldType(1).IsFlags():
				// Schedule flag register generation as late as possible.
				// This makes sure that we only have one live flags
				// value at a time.
				score[v.ID] = ScoreFlags
			default:
				score[v.ID] = ScoreDefault
				// If we're reading flags, schedule earlier to keep flag lifetime short.
				for _, a := range v.Args {
					if a.Type.IsFlags() {
						score[v.ID] = ScoreReadFlags
					}
				}
			}
		}
	}

	for _, b := range f.Blocks {
		// Find store chain for block.
		// Store chains for different blocks overwrite each other, so
		// the calculated store chain is good only for this block.
		for _, v := range b.Values {
			if v.Op != OpPhi && v.Type.IsMemory() {
				for _, w := range v.Args {
					if w.Type.IsMemory() {
						nextMem[w.ID] = v
					}
				}
			}
		}

		// Compute uses.
		for _, v := range b.Values {
			if v.Op == OpPhi {
				// If a value is used by a phi, it does not induce
				// a scheduling edge because that use is from the
				// previous iteration.
				continue
			}
			for _, w := range v.Args {
				if w.Block == b {
					uses[w.ID]++
				}
				// Any load must come before the following store.
				if !v.Type.IsMemory() && w.Type.IsMemory() {
					// v is a load.
					s := nextMem[w.ID]
					if s == nil || s.Block != b {
						continue
					}
					additionalArgs[s.ID] = append(additionalArgs[s.ID], v)
					uses[v.ID]++
				}
			}
		}

		for _, c := range b.ControlValues() {
			// Force the control values to be scheduled at the end,
			// unless they are phi values (which must be first).
			// OpArg also goes first -- if it is stack it register allocates
			// to a LoadReg, if it is register it is from the beginning anyway.
			if c.Op == OpPhi || c.Op == OpArg {
				continue
			}
			score[c.ID] = ScoreControl

			// Schedule values dependent on the control values at the end.
			// This reduces the number of register spills. We don't find
			// all values that depend on the controls, just values with a
			// direct dependency. This is cheaper and in testing there
			// was no difference in the number of spills.
			for _, v := range b.Values {
				if v.Op != OpPhi {
					for _, a := range v.Args {
						if a == c {
							score[v.ID] = ScoreControl
						}
					}
				}
			}

		}

		// To put things into a priority queue
		// The values that should come last are least.
		priq.score = score
		priq.a = priq.a[:0]

		// Initialize priority queue with schedulable values.
		for _, v := range b.Values {
			if uses[v.ID] == 0 {
				heap.Push(priq, v)
			}
		}

		// Schedule highest priority value, update use counts, repeat.
		order = order[:0]
		tuples := make(map[ID][]*Value)
		for priq.Len() > 0 {
			// Find highest priority schedulable value.
			// Note that schedule is assembled backwards.

			v := heap.Pop(priq).(*Value)

			// Add it to the schedule.
			// Do not emit tuple-reading ops until we're ready to emit the tuple-generating op.
			//TODO: maybe remove ReadTuple score above, if it does not help on performance
			switch {
			case v.Op == OpSelect0:
				if tuples[v.Args[0].ID] == nil {
					tuples[v.Args[0].ID] = make([]*Value, 2)
				}
				tuples[v.Args[0].ID][0] = v
			case v.Op == OpSelect1:
				if tuples[v.Args[0].ID] == nil {
					tuples[v.Args[0].ID] = make([]*Value, 2)
				}
				tuples[v.Args[0].ID][1] = v
			case v.Type.IsTuple() && tuples[v.ID] != nil:
				if tuples[v.ID][1] != nil {
					order = append(order, tuples[v.ID][1])
				}
				if tuples[v.ID][0] != nil {
					order = append(order, tuples[v.ID][0])
				}
				delete(tuples, v.ID)
				fallthrough
			default:
				order = append(order, v)
			}

			// Update use counts of arguments.
			for _, w := range v.Args {
				if w.Block != b {
					continue
				}
				uses[w.ID]--
				if uses[w.ID] == 0 {
					// All uses scheduled, w is now schedulable.
					heap.Push(priq, w)
				}
			}
			for _, w := range additionalArgs[v.ID] {
				uses[w.ID]--
				if uses[w.ID] == 0 {
					// All uses scheduled, w is now schedulable.
					heap.Push(priq, w)
				}
			}
		}
		if len(order) != len(b.Values) {
			f.Fatalf("schedule does not include all values in block %s", b)
		}
		for i := 0; i < len(b.Values); i++ {
			b.Values[i] = order[len(b.Values)-1-i]
		}
	}

	f.scheduled = true
}

// storeOrder orders values with respect to stores. That is,
// if v transitively depends on store s, v is ordered after s,
// otherwise v is ordered before s.
// Specifically, values are ordered like
//   store1
//   NilCheck that depends on store1
//   other values that depends on store1
//   store2
//   NilCheck that depends on store2
//   other values that depends on store2
//   ...
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
					sort.Sort(bySourcePos(order[start:i]))
					start = -1
				}
			}
		}
		if start != -1 {
			sort.Sort(bySourcePos(order[start:]))
		}
	}

	return order
}

type bySourcePos []*Value

func (s bySourcePos) Len() int           { return len(s) }
func (s bySourcePos) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s bySourcePos) Less(i, j int) bool { return s[i].Pos.Before(s[j].Pos) }
