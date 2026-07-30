// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ssa/ssacore"
	"cmd/compile/internal/ssa/ssaop"
)

// tighten moves Values closer to the Blocks in which they are used.
// This can reduce the amount of register spilling required,
// if it doesn't also create more live values.
// A Value can be moved to any block that
// dominates all blocks in which it is used.
func tighten(f *ssacore.Func) {
	if base.Flag.N != 0 && len(f.Blocks) < 10000 {
		// Skip the optimization in -N mode, except for huge functions.
		// Too many values live across blocks can cause pathological
		// behavior in the register allocator (see issue 52180).
		return
	}

	canMove := f.Cache.AllocBoolSlice(f.NumValues())
	defer f.Cache.FreeBoolSlice(canMove)

	// Compute the memory states of each block.
	startMem := f.Cache.AllocValueSlice(f.NumBlocks())
	defer f.Cache.FreeValueSlice(startMem)
	endMem := f.Cache.AllocValueSlice(f.NumBlocks())
	defer f.Cache.FreeValueSlice(endMem)
	distinctArgs := f.NewSparseSet(f.NumValues())
	defer f.RetSparseSet(distinctArgs)
	memState(f, startMem, endMem)

	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op.IsLoweredGetClosurePtr() {
				// Must stay in the entry block.
				continue
			}
			switch v.Op {
			case ssaop.OpPhi, ssaop.OpArg, ssaop.OpArgIntReg, ssaop.OpArgFloatReg, ssaop.OpSelect0, ssaop.OpSelect1, ssaop.OpSelectN:
				// Phis need to stay in their block.
				// Arg must stay in the entry block.
				// Tuple selectors must stay with the tuple generator.
				// SelectN is typically, ultimately, a register.
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
func phiTighten(f *ssacore.Func) {
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

// memState computes the memory state at the beginning and end of each block of
// the function. The memory state is represented by a value of mem type.
// The returned result is stored in startMem and endMem, and endMem is nil for
// blocks with no successors (Exit,Ret,RetJmp blocks). This algorithm is not
// suitable for infinite loop blocks that do not contain any mem operations.
// For example:
// b1:
//
//	(some values)
//
// plain -> b2
// b2: <- b1 b2
// Plain -> b2
//
// Algorithm introduction:
//  1. The start memory state of a block is InitMem, a Phi node of type mem or
//     an incoming memory value.
//  2. The start memory state of a block is consistent with the end memory state
//     of its parent nodes. If the start memory state of a block is a Phi value,
//     then the end memory state of its parent nodes is consistent with the
//     corresponding argument value of the Phi node.
//  3. The algorithm first obtains the memory state of some blocks in the tree
//     in the first step. Then floods the known memory state to other nodes in
//     the second step.
func memState(f *ssacore.Func, startMem, endMem []*ssacore.Value) {
	// This slice contains the set of blocks that have had their startMem set but this
	// startMem value has not yet been propagated to the endMem of its predecessors
	changed := make([]*ssacore.Block, 0)
	// First step, init the memory state of some blocks.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			var mem *ssacore.Value
			if v.Op == ssaop.OpPhi {
				if v.Type.IsMemory() {
					mem = v
				}
			} else if v.Op == ssaop.OpInitMem {
				mem = v // This is actually not needed.
			} else if a := v.MemoryArg(); a != nil && a.Block != b {
				// The only incoming memory value doesn't belong to this block.
				mem = a
			}
			if mem != nil {
				if old := startMem[b.ID]; old != nil {
					if old == mem {
						continue
					}
					f.Fatalf("func %s, startMem[%v] has different values, old %v, new %v", f.Name, b, old, mem)
				}
				startMem[b.ID] = mem
				changed = append(changed, b)
			}
		}
	}

	// Second step, floods the known memory state of some blocks to others.
	for len(changed) != 0 {
		top := changed[0]
		changed = changed[1:]
		mem := startMem[top.ID]
		for i, p := range top.Preds {
			pb := p.B
			if endMem[pb.ID] != nil {
				continue
			}
			if mem.Op == ssaop.OpPhi && mem.Block == top {
				endMem[pb.ID] = mem.Args[i]
			} else {
				endMem[pb.ID] = mem
			}
			if startMem[pb.ID] == nil {
				startMem[pb.ID] = endMem[pb.ID]
				changed = append(changed, pb)
			}
		}
	}
}
