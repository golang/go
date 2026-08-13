// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/block"
	"cmd/compile/internal/ssa/ssaop"
)

// tighten moves Values closer to the Blocks in which they are used.
// This can reduce the amount of register spilling required,
// if it doesn't also create more live values.
// A Value can be moved to any block that
// dominates all blocks in which it is used.
func tighten(f *ssa.Func) {
	if base.Flag.N != 0 && len(f.Blocks) < 10000 {
		// Skip the optimization in -N mode, except for huge functions.
		// Too many values live across blocks can cause pathological
		// behavior in the register allocator (see issue 52180).
		return
	}

	canMove := f.Cache.AllocBoolSlice(f.NumValues())
	defer f.Cache.FreeBoolSlice(canMove)

	// Compute the memory states of each block.
	startMem := memStartState(f)
	defer f.Cache.FreeValueSlice(startMem)

	distinctArgs := f.NewSparseSet(f.NumValues())
	defer f.RetSparseSet(distinctArgs)

	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op.IsLoweredGetClosurePtr() {
				// Must stay in the entry block.
				continue
			}
			switch v.Op {
			case ssaop.OpPhi, ssaop.OpArg, ssaop.OpArgIntReg, ssaop.OpArgFloatReg, ssaop.OpSelect0, ssaop.OpSelect1, ssaop.OpSelectN, ssaop.OpInitMem:
				// Phis need to stay in their block.
				// Arg must stay in the entry block.
				// Tuple selectors must stay with the tuple generator.
				// SelectN is typically, ultimately, a register.
				// InitMem must stay in the first block.
				continue
			}
			if ssaop.OpcodeTable[v.Op].NilCheck {
				// Nil checks need to stay in their block. See issue 72860.
				continue
			}
			// Count distinct arguments which will need a register.
			distinctArgs.Clear()

			for _, a := range v.Args {
				// SP and SB are special registers and have no effect on
				// the allocation of general-purpose registers.
				if a.NeedRegister() && a.Op != ssaop.OpSB && a.Op != ssaop.OpSP {
					distinctArgs.Add(a.ID)
				}
			}

			if distinctArgs.Size() >= 2 && !v.Type.IsFlags() {
				// Don't move values with more than one input, as that may
				// increase register pressure.
				// We make an exception for flags, as we want flag generators
				// moved next to uses (because we only have 1 flag register).
				continue
			}
			canMove[v.ID] = true
		}
	}

	// Build data structure for fast least-common-ancestor queries.
	lca := makeLCArange(f)

	// For each moveable value, record the block that dominates all uses found so far.
	target := f.Cache.AllocBlockSlice(f.NumValues())
	defer f.Cache.FreeBlockSlice(target)

	// Grab loop information.
	// We use this to make sure we don't tighten a value into a (deeper) loop.
	idom := f.Idom()
	loops := f.Loopnest()

	changed := true
	for changed {
		changed = false

		// Reset target
		clear(target)

		// Compute target locations (for moveable values only).
		// target location = the least common ancestor of all uses in the dominator tree.
		for _, b := range f.Blocks {
			for _, v := range b.Values {
				for i, a := range v.Args {
					if !canMove[a.ID] {
						continue
					}
					use := b
					if v.Op == ssaop.OpPhi {
						use = b.Preds[i].B
					}
					if target[a.ID] == nil {
						target[a.ID] = use
					} else {
						target[a.ID] = lca.find(target[a.ID], use)
					}
				}
			}
			for _, c := range b.ControlValues() {
				if !canMove[c.ID] {
					continue
				}
				if target[c.ID] == nil {
					target[c.ID] = b
				} else {
					target[c.ID] = lca.find(target[c.ID], b)
				}
			}
		}

		// If the target location is inside a loop,
		// move the target location up to just before the loop head.
		if !loops.HasIrreducible {
			// Loop info might not be correct for irreducible loops. See issue 75569.
			for _, b := range f.Blocks {
				origloop := loops.B2L[b.ID]
				for _, v := range b.Values {
					t := target[v.ID]
					if t == nil {
						continue
					}
					targetloop := loops.B2L[t.ID]
					for targetloop != nil && (origloop == nil || targetloop.Depth > origloop.Depth) {
						t = idom[targetloop.Header.ID]
						target[v.ID] = t
						targetloop = loops.B2L[t.ID]
					}
				}
			}
		}

		// Move values to target locations.
		for _, b := range f.Blocks {
			for i := 0; i < len(b.Values); i++ {
				v := b.Values[i]
				t := target[v.ID]
				if t == nil || t == b {
					// v is not moveable, or is already in correct place.
					continue
				}
				if mem := v.MemoryArg(); mem != nil {
					if startMem[t.ID] != mem {
						// We can't move a value with a memory arg unless the target block
						// has that memory arg as its starting memory.
						continue
					}
				}
				if f.Pass.Debug > 0 {
					b.Func.Warnl(v.Pos, "%v is moved", v.Op)
				}
				// Move v to the block which dominates its uses.
				t.Values = append(t.Values, v)
				v.Block = t
				last := len(b.Values) - 1
				b.Values[i] = b.Values[last]
				b.Values[last] = nil
				b.Values = b.Values[:last]
				changed = true
				i--
			}
		}
	}
}

// phiTighten moves constants closer to phi users.
// This pass avoids having lots of constants live for lots of the program.
// See issue 16407.
func phiTighten(f *ssa.Func) {
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op != ssaop.OpPhi {
				continue
			}
			for i, a := range v.Args {
				if !a.Rematerializeable() {
					continue // not a constant we can move around
				}
				if a.Block == b.Preds[i].B {
					continue // already in the right place
				}
				// Make a copy of a, put in predecessor block.
				v.SetArg(i, a.CopyInto(b.Preds[i].B))
			}
		}
	}
}

// Returns the memory state that block starts with
// and the memory state at the end of the block.
//
// If the block doesn't use nor produce any memory value
// this function returns nil and nil. Otherwise it returns
// the live memories at entry and exit of the block.
//
// Entry, exit and return blocks are handled by the same logic:
// their memory states can be inferred from memory operations
// (such as OpInitMemory and OpMakeResult).
//
// Note: This is mainly implementation detail of memState, better use it instead.
//
// cache is the bitset that can be indexed by Value's ID.
// The bitset must not be set for values inside the block.
func blockState(b *ssa.Block, cache ssa.Bitset) (start, end *ssa.Value) {
	// usedMem bitset is used to record if memory value was used
	// by another one inside the block.
	// Indexed by Value's ID, and for all memory values in the block
	// the usedMem mustn't be set beforehand.
	if cache.Cap() < b.Func.NumValues() {
		b.Fatalf("blockState(%v) cache is too small %v < %v", b, cache.Cap(), b.Func.NumValues())
	}
	usedMem := cache

	// captures start and b
	setStart := func(v *ssa.Value) {
		if start != nil && start != v {
			b.Fatalf("blockState(%v): conflicting start values %v and %v", b, start, v)
		}
		start = v
	}

	// 1. Scan all values in block to find start and fill the usedMem
	for _, v := range b.Values {
		// Check for start.
		// The starting memory of the block is from the block itself
		// (InitMemory or Phi<mem>), or memory state from predecessor block.
		if v.Op == ssaop.OpInitMem || (v.Op == ssaop.OpPhi && v.Type.IsMemory()) {
			setStart(v)
			continue
		}

		m := v.MemoryArg()
		if m == nil {
			continue
		}

		if m.Block != b {
			setStart(m)
		} else {
			if v.Type.IsMemory() {
				// Record that m was used as an argument,
				// meaning it can't be an end value
				//
				// Note: We clobber bitset only for values
				// from current block
				usedMem.Set(uint32(m.ID))
			}
		}
	}

	// 2. Scan values to find the end using the usedMem info.
	//
	// The memory value is the end iff it is not used as the argument of another
	// value in this block.
	//
	// FIXME(ArsenySamoylov): This doesn't always work as intended because
	// the single-live-memory invariant is not maintained: stale memory values
	// with zero uses appear to diverge the memory chain.
	// Noticeable example include OpCopy left after lower pass and dead OpStore after dse.
	for _, v := range b.Values {
		if v.Type.IsMemory() && !usedMem.Test(uint32(v.ID)) {
			end = v
			break
		}
	}

	if start != nil && end == nil {
		// This could happen only when there is no memory values in the
		// block but there are value that has memory argument (e.g. Load)
		// from which we found the start.
		if start.Block == b {
			b.Fatalf("blockState(%v): unexpected, end is nil, while the start (%v) is found", b, start)
		}

		end = start
	}

	if start == nil && end != nil {
		b.Fatalf("blockState(%v): unexpected, start is nil, while the end (%v) is found", b, end)
	}
	return
}

// Simple wrapper around memState
// that return only start memory state.
// The result must be freed explicitly by calling to f.Cache.freeValueSlice.
func memStartState(f *ssa.Func) []*ssa.Value {
	startMem, endMem := memState(f)
	f.Cache.FreeValueSlice(endMem)
	return startMem
}

// Simple wrapper around memState
// that return only end memory state.
// The result must be freed explicitly by calling to f.Cache.freeValueSlice.
func memEndState(f *ssa.Func) []*ssa.Value {
	startMem, endMem := memState(f)
	f.Cache.FreeValueSlice(startMem)
	return endMem
}

// memState computes the live memory at the beginning and end of each block of
// the function. The memory state is represented by a value of mem type.
//
// The memory state is undefined for invalid (`b.Kind == block.BlockInvalid`) blocks and blocks
// that can't be reached from entry block through successors relation; state may be nil or non-nil.
//
// For a valid block that can be reached from entry the state could be nil iff the whole function
// have no memory values at all (this is extremely rare and happens only in some synthetic edge cases).
//
// The result is returned in two slices startMem and endMem.
// These slices must be explicitly freed by calling to f.Cache.freeValueSlice.
func memState(f *ssa.Func) (startMem, endMem []*ssa.Value) {
	startMem = f.Cache.AllocValueSlice(f.NumBlocks())
	endMem = f.Cache.AllocValueSlice(f.NumBlocks())

	// First step will not find memory state for blocks
	// that don't have memory operations.
	// Save blocks that do to propagate it.
	haveState := f.Cache.AllocBlockSlice(len(f.Blocks))[:0]
	// FIXME(@ArsenySamoylov): haveState could be empty by the end of the analysis, leading to Free panicking
	// thats why we need to set len == cap
	defer f.Cache.FreeBlockSlice(haveState[:cap(haveState)])

	// For blockState
	cache := f.Cache.AllocBitset(f.NumValues())
	defer f.Cache.FreeBitset(cache)

	for _, b := range f.Blocks {
		start, end := blockState(b, cache)
		if start == nil && end == nil {
			continue
		}
		endMem[b.ID] = end
		startMem[b.ID] = start
		haveState = append(haveState, b)
	}

	// Second step, floods the known memory state
	// from blocks with known state to their successors.
	for i := 0; i < len(haveState); i++ {
		b := haveState[i]
		bEndMem := endMem[b.ID]
		if bEndMem == nil {
			b.Fatalf("block %v must have had the state", b)
		}

		for _, succEdge := range b.Succs {
			succ := succEdge.Block()
			m := startMem[succ.ID]
			if m != nil {
				continue
			}

			startMem[succ.ID] = bEndMem
			endMem[succ.ID] = bEndMem

			haveState = append(haveState, succ)
		}
	}

	if checkEnabled {
		if len(haveState) == 0 {
			// There are functions that have no ssa instructions (or just a few but non memory ones).
			// For example see test/fixedbugs/bug076.go func f or test/escape2.go func foo120.
			//
			// So, there is no memory state for the whole function and nothing to check.
			return
		}

		// Sanity check
		for _, b := range f.Blocks {
			start := startMem[b.ID]
			end := endMem[b.ID]

			if start != nil && end != nil {
				// Check that we can reach start from end
				for curMem := end; curMem != start; curMem = curMem.MemoryArg() {
					if curMem.Block != b {
						b.Fatalf("memState: can't reach start from end")
					}
				}

				// b has consistent memory state
				continue
			}

			// BlockInvalid blocks can be disconnected and have nil state.
			//
			// Also there are cases of blocks (and even chains of blocks) that
			// can't be reached from entry block. Hence they are invalid as well
			// (but are not marked as such).
			//
			// Notably, this seems to come after late fuse pass.
			//
			// Ps: We could change the flooding logic from step 2 to also propagate
			// information to predecessors. This way we could infer information
			// for these blocks, yet these blocks are in inconsistent state
			// so better not to have a proper state for them.
			//
			// Pss: This logic could loop infinitely on cycle. Since this is just a sanity check
			// it should stay as simple as possible. So this avoids cycle detection in favor of simplicity.
			unreachable := false
			for currB := b; ; currB = currB.Preds[0].Block() {
				if currB.Kind == block.BlockInvalid {
					unreachable = true
					break
				}

				if len(currB.Preds) != 1 {
					// (@ArsenySamoylov): Currently only a chain of blocks that starts from single BlockInvalid block
					// has been observed. In theory we could have a subgraph where "invalid" blocks have more than
					// one incoming edge. But instead of overengineering the check here, I guess, it would be better
					// to address the root problem of why we have invalid subgraphs in the first place
					// if (or when) such case would be observed.
					break
				}
			}

			if unreachable {
				// It is okay to have nil state
				continue
			}

			f.Fatalf("memState: sanity check failed, nil mem state for %v", b)
		}
	}
	return
}
