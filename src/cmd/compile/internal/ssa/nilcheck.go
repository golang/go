// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// nilcheckelim eliminates unnecessary nil checks.
// runs on machine-independent code.
func nilcheckelim(f *Func) {
	// A nil check is redundant if the same nil check was successful in a
	// dominating block. The efficacy of this pass depends heavily on the
	// efficacy of the cse pass.
	sdom := f.sdom()

	// TODO: Eliminate more nil checks.
	// We can recursively remove any chain of fixed offset calculations,
	// i.e. struct fields and array elements, even with non-constant
	// indices: x is non-nil iff x.a.b[i].c is.

	type walkState int
	const (
		Work     walkState = iota // process nil checks and traverse to dominees
		ClearPtr                  // forget the fact that ptr is nil
	)

	type bp struct {
		block *Block // block, or nil in ClearPtr state
		ptr   *Value // if non-nil, ptr that is to be cleared in ClearPtr state
		op    walkState
	}

	work := make([]bp, 0, 256)
	work = append(work, bp{block: f.Entry})

	// map from value ID to bool indicating if value is known to be non-nil
	// in the current dominator path being walked. This slice is updated by
	// walkStates to maintain the known non-nil values.
	nonNilValues := make([]bool, f.NumValues())

	// make an initial pass identifying any non-nil values
	for _, b := range f.Blocks {
		// a value resulting from taking the address of a
		// value, or a value constructed from an offset of a
		// non-nil ptr (OpAddPtr) implies it is non-nil
		for _, v := range b.Values {
			if v.Op == OpAddr || v.Op == OpAddPtr {
				nonNilValues[v.ID] = true
			} else if v.Op == OpPhi {
				// phis whose arguments are all non-nil
				// are non-nil
				argsNonNil := true
				for _, a := range v.Args {
					if !nonNilValues[a.ID] {
						argsNonNil = false
					}
				}
				if argsNonNil {
					nonNilValues[v.ID] = true
				}
			}
		}
	}

	// allocate auxiliary date structures for computing store order
	sset := f.newSparseSet(f.NumValues())
	defer f.retSparseSet(sset)
	storeNumber := make([]int32, f.NumValues())

	// perform a depth first walk of the dominee tree
	for len(work) > 0 {
		node := work[len(work)-1]
		work = work[:len(work)-1]

		switch node.op {
		case Work:
			b := node.block

			// First, see if we're dominated by an explicit nil check.
			if len(b.Preds) == 1 {
				p := b.Preds[0].b
				if p.Kind == BlockIf && p.Control.Op == OpIsNonNil && p.Succs[0].b == b {
					ptr := p.Control.Args[0]
					if !nonNilValues[ptr.ID] {
						nonNilValues[ptr.ID] = true
						work = append(work, bp{op: ClearPtr, ptr: ptr})
					}
				}
			}

			// Next, order values in the current block w.r.t. stores.
			b.Values = storeOrder(b.Values, sset, storeNumber)

			// Next, process values in the block.
			i := 0
			for _, v := range b.Values {
				b.Values[i] = v
				i++
				switch v.Op {
				case OpIsNonNil:
					ptr := v.Args[0]
					if nonNilValues[ptr.ID] {
						// This is a redundant explicit nil check.
						v.reset(OpConstBool)
						v.AuxInt = 1 // true
					}
				case OpNilCheck:
					ptr := v.Args[0]
					if nonNilValues[ptr.ID] {
						// This is a redundant implicit nil check.
						// Logging in the style of the former compiler -- and omit line 1,
						// which is usually in generated code.
						if f.Config.Debug_checknil() && v.Pos.Line() > 1 {
							f.Config.Warnl(v.Pos, "removed nil check")
						}
						v.reset(OpUnknown)
						// TODO: f.freeValue(v)
						i--
						continue
					}
					// Record the fact that we know ptr is non nil, and remember to
					// undo that information when this dominator subtree is done.
					nonNilValues[ptr.ID] = true
					work = append(work, bp{op: ClearPtr, ptr: ptr})
				}
			}
			for j := i; j < len(b.Values); j++ {
				b.Values[j] = nil
			}
			b.Values = b.Values[:i]

			// Add all dominated blocks to the work list.
			for w := sdom[node.block.ID].child; w != nil; w = sdom[w.ID].sibling {
				work = append(work, bp{op: Work, block: w})
			}

		case ClearPtr:
			nonNilValues[node.ptr.ID] = false
			continue
		}
	}
}

// All platforms are guaranteed to fault if we load/store to anything smaller than this address.
//
// This should agree with minLegalPointer in the runtime.
const minZeroPage = 4096

// nilcheckelim2 eliminates unnecessary nil checks.
// Runs after lowering and scheduling.
func nilcheckelim2(f *Func) {
	unnecessary := f.newSparseSet(f.NumValues())
	defer f.retSparseSet(unnecessary)
	for _, b := range f.Blocks {
		// Walk the block backwards. Find instructions that will fault if their
		// input pointer is nil. Remove nil checks on those pointers, as the
		// faulting instruction effectively does the nil check for free.
		unnecessary.clear()
		for i := len(b.Values) - 1; i >= 0; i-- {
			v := b.Values[i]
			if opcodeTable[v.Op].nilCheck && unnecessary.contains(v.Args[0].ID) {
				if f.Config.Debug_checknil() && v.Pos.Line() > 1 {
					f.Config.Warnl(v.Pos, "removed nil check")
				}
				v.reset(OpUnknown)
				continue
			}
			if v.Type.IsMemory() || v.Type.IsTuple() && v.Type.FieldType(1).IsMemory() {
				if v.Op == OpVarDef || v.Op == OpVarKill || v.Op == OpVarLive {
					// These ops don't really change memory.
					continue
				}
				// This op changes memory.  Any faulting instruction after v that
				// we've recorded in the unnecessary map is now obsolete.
				unnecessary.clear()
			}

			// Find any pointers that this op is guaranteed to fault on if nil.
			var ptrstore [2]*Value
			ptrs := ptrstore[:0]
			if opcodeTable[v.Op].faultOnNilArg0 {
				ptrs = append(ptrs, v.Args[0])
			}
			if opcodeTable[v.Op].faultOnNilArg1 {
				ptrs = append(ptrs, v.Args[1])
			}
			for _, ptr := range ptrs {
				// Check to make sure the offset is small.
				switch opcodeTable[v.Op].auxType {
				case auxSymOff:
					if v.Aux != nil || v.AuxInt < 0 || v.AuxInt >= minZeroPage {
						continue
					}
				case auxSymValAndOff:
					off := ValAndOff(v.AuxInt).Off()
					if v.Aux != nil || off < 0 || off >= minZeroPage {
						continue
					}
				case auxInt32:
					// Mips uses this auxType for atomic add constant. It does not affect the effective address.
				case auxInt64:
					// ARM uses this auxType for duffcopy/duffzero/alignment info.
					// It does not affect the effective address.
				case auxNone:
					// offset is zero.
				default:
					v.Fatalf("can't handle aux %s (type %d) yet\n", v.auxString(), int(opcodeTable[v.Op].auxType))
				}
				// This instruction is guaranteed to fault if ptr is nil.
				// Any previous nil check op is unnecessary.
				unnecessary.add(ptr.ID)
			}
		}
		// Remove values we've clobbered with OpUnknown.
		i := 0
		for _, v := range b.Values {
			if v.Op != OpUnknown {
				b.Values[i] = v
				i++
			}
		}
		for j := i; j < len(b.Values); j++ {
			b.Values[j] = nil
		}
		b.Values = b.Values[:i]

		// TODO: if b.Kind == BlockPlain, start the analysis in the subsequent block to find
		// more unnecessary nil checks.  Would fix test/nilptr3_ssa.go:157.
	}
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
// than a full scheduling as done in schedule.go.
// Note that simple dependency order won't work: there is no
// dependency between NilChecks and values like IsNonNil.
// Auxiliary data structures are passed in as arguments, so
// that they can be allocated in the caller and be reused.
// This function takes care of reset them.
func storeOrder(values []*Value, sset *sparseSet, storeNumber []int32) []*Value {
	// find all stores
	var stores []*Value // members of values that are store values
	hasNilCheck := false
	sset.clear() // sset is the set of stores that are used in other values
	for _, v := range values {
		if v.Type.IsMemory() {
			stores = append(stores, v)
			if v.Op == OpInitMem || v.Op == OpPhi {
				continue
			}
			a := v.Args[len(v.Args)-1]
			if v.Op == OpSelect1 {
				a = a.Args[len(a.Args)-1]
			}
			sset.add(a.ID) // record that a is used
		}
		if v.Op == OpNilCheck {
			hasNilCheck = true
		}
	}
	if len(stores) == 0 || !hasNilCheck {
		// there is no store or nilcheck, the order does not matter
		return values
	}

	f := stores[0].Block.Func

	// find last store, which is the one that is not used by other stores
	var last *Value
	for _, v := range stores {
		if !sset.contains(v.ID) {
			if last != nil {
				f.Fatalf("two stores live simutaneously: %v and %v", v, last)
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
		if w.Op == OpSelect1 {
			w = w.Args[0]
		}
		w = w.Args[len(w.Args)-1]
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
					continue
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

	return order
}
