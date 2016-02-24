// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Register allocation.
//
// We use a version of a linear scan register allocator.  We treat the
// whole function as a single long basic block and run through
// it using a greedy register allocator.  Then all merge edges
// (those targeting a block with len(Preds)>1) are processed to
// shuffle data into the place that the target of the edge expects.
//
// The greedy allocator moves values into registers just before they
// are used, spills registers only when necessary, and spills the
// value whose next use is farthest in the future.
//
// The register allocator requires that a block is not scheduled until
// at least one of its predecessors have been scheduled.  The most recent
// such predecessor provides the starting register state for a block.
//
// It also requires that there are no critical edges (critical =
// comes from a block with >1 successor and goes to a block with >1
// predecessor).  This makes it easy to add fixup code on merge edges -
// the source of a merge edge has only one successor, so we can add
// fixup code to the end of that block.

// Spilling
//
// For every value, we generate a spill immediately after the value itself.
//     x = Op y z    : AX
//     x2 = StoreReg x
// While AX still holds x, any uses of x will use that value.  When AX is needed
// for another value, we simply reuse AX.  Spill code has already been generated
// so there is no code generated at "spill" time.  When x is referenced
// subsequently, we issue a load to restore x to a register using x2 as
//  its argument:
//    x3 = Restore x2 : CX
// x3 can then be used wherever x is referenced again.
// If the spill (x2) is never used, it will be removed at the end of regalloc.
//
// Phi values are special, as always.  We define two kinds of phis, those
// where the merge happens in a register (a "register" phi) and those where
// the merge happens in a stack location (a "stack" phi).
//
// A register phi must have the phi and all of its inputs allocated to the
// same register.  Register phis are spilled similarly to regular ops:
//     b1: y = ... : AX        b2: z = ... : AX
//         goto b3                 goto b3
//     b3: x = phi(y, z) : AX
//         x2 = StoreReg x
//
// A stack phi must have the phi and all of its inputs allocated to the same
// stack location.  Stack phis start out life already spilled - each phi
// input must be a store (using StoreReg) at the end of the corresponding
// predecessor block.
//     b1: y = ... : AX        b2: z = ... : BX
//         y2 = StoreReg y         z2 = StoreReg z
//         goto b3                 goto b3
//     b3: x = phi(y2, z2)
// The stack allocator knows that StoreReg args of stack-allocated phis
// must be allocated to the same stack slot as the phi that uses them.
// x is now a spilled value and a restore must appear before its first use.

// TODO

// Use an affinity graph to mark two values which should use the
// same register.  This affinity graph will be used to prefer certain
// registers for allocation.  This affinity helps eliminate moves that
// are required for phi implementations and helps generate allocations
// for 2-register architectures.

// Note: regalloc generates a not-quite-SSA output.  If we have:
//
//             b1: x = ... : AX
//                 x2 = StoreReg x
//                 ... AX gets reused for something else ...
//                 if ... goto b3 else b4
//
//   b3: x3 = LoadReg x2 : BX       b4: x4 = LoadReg x2 : CX
//       ... use x3 ...                 ... use x4 ...
//
//             b2: ... use x3 ...
//
// If b3 is the primary predecessor of b2, then we use x3 in b2 and
// add a x4:CX->BX copy at the end of b4.
// But the definition of x3 doesn't dominate b2.  We should really
// insert a dummy phi at the start of b2 (x5=phi(x3,x4):BX) to keep
// SSA form.  For now, we ignore this problem as remaining in strict
// SSA form isn't needed after regalloc.  We'll just leave the use
// of x3 not dominated by the definition of x3, and the CX->BX copy
// will have no use (so don't run deadcode after regalloc!).
// TODO: maybe we should introduce these extra phis?

package ssa

import (
	"cmd/internal/obj"
	"fmt"
	"unsafe"
)

const regDebug = false // TODO: compiler flag
const logSpills = false

// regalloc performs register allocation on f.  It sets f.RegAlloc
// to the resulting allocation.
func regalloc(f *Func) {
	var s regAllocState
	s.init(f)
	s.regalloc(f)
}

type register uint8

const noRegister register = 255

type regMask uint64

func (m regMask) String() string {
	s := ""
	for r := register(0); r < numRegs; r++ {
		if m>>r&1 == 0 {
			continue
		}
		if s != "" {
			s += " "
		}
		s += fmt.Sprintf("r%d", r)
	}
	return s
}

// TODO: make arch-dependent
var numRegs register = 64

var registers = [...]Register{
	Register{0, "AX"},
	Register{1, "CX"},
	Register{2, "DX"},
	Register{3, "BX"},
	Register{4, "SP"},
	Register{5, "BP"},
	Register{6, "SI"},
	Register{7, "DI"},
	Register{8, "R8"},
	Register{9, "R9"},
	Register{10, "R10"},
	Register{11, "R11"},
	Register{12, "R12"},
	Register{13, "R13"},
	Register{14, "R14"},
	Register{15, "R15"},
	Register{16, "X0"},
	Register{17, "X1"},
	Register{18, "X2"},
	Register{19, "X3"},
	Register{20, "X4"},
	Register{21, "X5"},
	Register{22, "X6"},
	Register{23, "X7"},
	Register{24, "X8"},
	Register{25, "X9"},
	Register{26, "X10"},
	Register{27, "X11"},
	Register{28, "X12"},
	Register{29, "X13"},
	Register{30, "X14"},
	Register{31, "X15"},
	Register{32, "SB"}, // pseudo-register for global base pointer (aka %rip)

	// TODO: make arch-dependent
}

// countRegs returns the number of set bits in the register mask.
func countRegs(r regMask) int {
	n := 0
	for r != 0 {
		n += int(r & 1)
		r >>= 1
	}
	return n
}

// pickReg picks an arbitrary register from the register mask.
func pickReg(r regMask) register {
	// pick the lowest one
	if r == 0 {
		panic("can't pick a register from an empty set")
	}
	for i := register(0); ; i++ {
		if r&1 != 0 {
			return i
		}
		r >>= 1
	}
}

type use struct {
	dist int32 // distance from start of the block to a use of a value
	next *use  // linked list of uses of a value in nondecreasing dist order
}

type valState struct {
	regs              regMask // the set of registers holding a Value (usually just one)
	uses              *use    // list of uses in this block
	spill             *Value  // spilled copy of the Value
	spillUsed         bool
	needReg           bool     // cached value of !v.Type.IsMemory() && !v.Type.IsVoid() && !.v.Type.IsFlags()
	rematerializeable bool     // cached value of v.rematerializeable()
	desired           register // register we want value to be in, if any
	avoid             regMask  // registers to avoid if we can
}

type regState struct {
	v *Value // Original (preregalloc) Value stored in this register.
	c *Value // A Value equal to v which is currently in a register.  Might be v or a copy of it.
	// If a register is unused, v==c==nil
}

type regAllocState struct {
	f *Func

	// for each block, its primary predecessor.
	// A predecessor of b is primary if it is the closest
	// predecessor that appears before b in the layout order.
	// We record the index in the Preds list where the primary predecessor sits.
	primary []int32

	// live values at the end of each block.  live[b.ID] is a list of value IDs
	// which are live at the end of b, together with a count of how many instructions
	// forward to the next use.
	live [][]liveInfo

	// current state of each (preregalloc) Value
	values []valState

	// For each Value, map from its value ID back to the
	// preregalloc Value it was derived from.
	orig []*Value

	// current state of each register
	regs []regState

	// registers that contain values which can't be kicked out
	nospill regMask

	// mask of registers currently in use
	used regMask

	// current block we're working on
	curBlock *Block

	// cache of use records
	freeUseRecords *use

	// endRegs[blockid] is the register state at the end of each block.
	// encoded as a set of endReg records.
	endRegs [][]endReg

	// startRegs[blockid] is the register state at the start of merge blocks.
	// saved state does not include the state of phi ops in the block.
	startRegs [][]startReg

	// spillLive[blockid] is the set of live spills at the end of each block
	spillLive [][]ID
}

type endReg struct {
	r register
	v *Value // pre-regalloc value held in this register (TODO: can we use ID here?)
	c *Value // cached version of the value
}

type startReg struct {
	r   register
	vid ID // pre-regalloc value needed in this register
}

// freeReg frees up register r.  Any current user of r is kicked out.
func (s *regAllocState) freeReg(r register) {
	v := s.regs[r].v
	if v == nil {
		s.f.Fatalf("tried to free an already free register %d\n", r)
	}

	// Mark r as unused.
	if regDebug {
		fmt.Printf("freeReg %s (dump %s/%s)\n", registers[r].Name(), v, s.regs[r].c)
	}
	s.regs[r] = regState{}
	s.values[v.ID].regs &^= regMask(1) << r
	s.used &^= regMask(1) << r
}

// freeRegs frees up all registers listed in m.
func (s *regAllocState) freeRegs(m regMask) {
	for m&s.used != 0 {
		s.freeReg(pickReg(m & s.used))
	}
}

// setOrig records that c's original value is the same as
// v's original value.
func (s *regAllocState) setOrig(c *Value, v *Value) {
	for int(c.ID) >= len(s.orig) {
		s.orig = append(s.orig, nil)
	}
	if s.orig[c.ID] != nil {
		s.f.Fatalf("orig value set twice %s %s", c, v)
	}
	s.orig[c.ID] = s.orig[v.ID]
}

// assignReg assigns register r to hold c, a copy of v.
// r must be unused.
func (s *regAllocState) assignReg(r register, v *Value, c *Value) {
	if regDebug {
		fmt.Printf("assignReg %s %s/%s\n", registers[r].Name(), v, c)
	}
	if s.regs[r].v != nil {
		s.f.Fatalf("tried to assign register %d to %s/%s but it is already used by %s", r, v, c, s.regs[r].v)
	}

	// Update state.
	s.regs[r] = regState{v, c}
	s.values[v.ID].regs |= regMask(1) << r
	s.used |= regMask(1) << r
	s.f.setHome(c, &registers[r])
}

// allocReg chooses a register for v from the set of registers in mask.
// If there is no unused register, a Value will be kicked out of
// a register to make room.
func (s *regAllocState) allocReg(v *Value, mask regMask) register {
	mask &^= s.nospill
	if mask == 0 {
		s.f.Fatalf("no register available")
	}

	// Pick an unused register if one is available.
	if mask&^s.used != 0 {
		mask &^= s.used

		// Use desired register if we can.
		d := s.values[v.ID].desired
		if d != noRegister && mask>>d&1 != 0 {
			mask = regMask(1) << d
		}

		// Avoid avoidable registers if we can.
		if mask&^s.values[v.ID].avoid != 0 {
			mask &^= s.values[v.ID].avoid
		}

		return pickReg(mask)
	}

	// Pick a value to spill.  Spill the value with the
	// farthest-in-the-future use.
	// TODO: Prefer registers with already spilled Values?
	// TODO: Modify preference using affinity graph.
	// TODO: if a single value is in multiple registers, spill one of them
	// before spilling a value in just a single register.

	// SP and SB are allocated specially.  No regular value should
	// be allocated to them.
	mask &^= 1<<4 | 1<<32

	// Find a register to spill.  We spill the register containing the value
	// whose next use is as far in the future as possible.
	// https://en.wikipedia.org/wiki/Page_replacement_algorithm#The_theoretically_optimal_page_replacement_algorithm
	var r register
	maxuse := int32(-1)
	for t := register(0); t < numRegs; t++ {
		if mask>>t&1 == 0 {
			continue
		}
		v := s.regs[t].v
		if n := s.values[v.ID].uses.dist; n > maxuse {
			// v's next use is farther in the future than any value
			// we've seen so far.  A new best spill candidate.
			r = t
			maxuse = n
		}
	}
	if maxuse == -1 {
		s.f.Unimplementedf("couldn't find register to spill")
	}
	s.freeReg(r)
	return r
}

// allocValToReg allocates v to a register selected from regMask and
// returns the register copy of v. Any previous user is kicked out and spilled
// (if necessary). Load code is added at the current pc. If nospill is set the
// allocated register is marked nospill so the assignment cannot be
// undone until the caller allows it by clearing nospill. Returns a
// *Value which is either v or a copy of v allocated to the chosen register.
func (s *regAllocState) allocValToReg(v *Value, mask regMask, nospill bool, line int32) *Value {
	vi := &s.values[v.ID]

	// Check if v is already in a requested register.
	if mask&vi.regs != 0 {
		r := pickReg(mask & vi.regs)
		if s.regs[r].v != v || s.regs[r].c == nil {
			panic("bad register state")
		}
		if nospill {
			s.nospill |= regMask(1) << r
		}
		return s.regs[r].c
	}

	if v.Op != OpSP {
		mask &^= 1 << 4 // dont' spill SP
	}
	if v.Op != OpSB {
		mask &^= 1 << 32 // don't spill SB
	}
	mask &^= s.reserved()

	// Allocate a register.
	r := s.allocReg(v, mask)

	// Allocate v to the new register.
	var c *Value
	if vi.regs != 0 {
		// Copy from a register that v is already in.
		r2 := pickReg(vi.regs)
		if s.regs[r2].v != v {
			panic("bad register state")
		}
		c = s.curBlock.NewValue1(line, OpCopy, v.Type, s.regs[r2].c)
	} else if v.rematerializeable() {
		// Rematerialize instead of loading from the spill location.
		c = v.copyInto(s.curBlock)
	} else {
		switch {
		// Load v from its spill location.
		case vi.spill != nil:
			if logSpills {
				fmt.Println("regalloc: load spill")
			}
			c = s.curBlock.NewValue1(line, OpLoadReg, v.Type, vi.spill)
			vi.spillUsed = true
		default:
			s.f.Fatalf("attempt to load unspilled value %v", v.LongString())
		}
	}
	s.setOrig(c, v)
	s.assignReg(r, v, c)
	if nospill {
		s.nospill |= regMask(1) << r
	}
	return c
}

func (s *regAllocState) init(f *Func) {
	if numRegs > noRegister || numRegs > register(unsafe.Sizeof(regMask(0))*8) {
		panic("too many registers")
	}

	s.f = f
	s.regs = make([]regState, numRegs)
	s.values = make([]valState, f.NumValues())
	s.orig = make([]*Value, f.NumValues())
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if !v.Type.IsMemory() && !v.Type.IsVoid() && !v.Type.IsFlags() {
				s.values[v.ID].needReg = true
				s.values[v.ID].rematerializeable = v.rematerializeable()
				s.values[v.ID].desired = noRegister
				s.orig[v.ID] = v
			}
		}
	}
	s.computeLive()

	// Compute block order.  This array allows us to distinguish forward edges
	// from backward edges and compute how far they go.
	blockOrder := make([]int32, f.NumBlocks())
	for i, b := range f.Blocks {
		blockOrder[b.ID] = int32(i)
	}

	// Compute primary predecessors.
	s.primary = make([]int32, f.NumBlocks())
	for _, b := range f.Blocks {
		best := -1
		for i, p := range b.Preds {
			if blockOrder[p.ID] >= blockOrder[b.ID] {
				continue // backward edge
			}
			if best == -1 || blockOrder[p.ID] > blockOrder[b.Preds[best].ID] {
				best = i
			}
		}
		s.primary[b.ID] = int32(best)
	}

	s.endRegs = make([][]endReg, f.NumBlocks())
	s.startRegs = make([][]startReg, f.NumBlocks())
	s.spillLive = make([][]ID, f.NumBlocks())
}

// Adds a use record for id at distance dist from the start of the block.
// All calls to addUse must happen with nonincreasing dist.
func (s *regAllocState) addUse(id ID, dist int32) {
	r := s.freeUseRecords
	if r != nil {
		s.freeUseRecords = r.next
	} else {
		r = &use{}
	}
	r.dist = dist
	r.next = s.values[id].uses
	s.values[id].uses = r
	if r.next != nil && dist > r.next.dist {
		s.f.Fatalf("uses added in wrong order")
	}
}

// advanceUses advances the uses of v's args from the state before v to the state after v.
// Any values which have no more uses are deallocated from registers.
func (s *regAllocState) advanceUses(v *Value) {
	for _, a := range v.Args {
		if !s.values[a.ID].needReg {
			continue
		}
		ai := &s.values[a.ID]
		r := ai.uses
		ai.uses = r.next
		if r.next == nil {
			// Value is dead, free all registers that hold it.
			s.freeRegs(ai.regs)
		}
		r.next = s.freeUseRecords
		s.freeUseRecords = r
	}
}

// Sets the state of the registers to that encoded in regs.
func (s *regAllocState) setState(regs []endReg) {
	s.freeRegs(s.used)
	for _, x := range regs {
		s.assignReg(x.r, x.v, x.c)
	}
}

// compatRegs returns the set of registers which can store a type t.
func (s *regAllocState) compatRegs(t Type) regMask {
	var m regMask
	if t.IsFloat() {
		m = 0xffff << 16 // X0-X15
	} else {
		m = 0xffef << 0 // AX-R15, except SP
	}
	return m &^ s.reserved()
}

func (s *regAllocState) regalloc(f *Func) {
	liveSet := f.newSparseSet(f.NumValues())
	defer f.retSparseSet(liveSet)
	var oldSched []*Value
	var phis []*Value
	var phiRegs []register
	var args []*Value

	if f.Entry != f.Blocks[0] {
		f.Fatalf("entry block must be first")
	}

	for _, b := range f.Blocks {
		s.curBlock = b

		// Initialize liveSet and uses fields for this block.
		// Walk backwards through the block doing liveness analysis.
		liveSet.clear()
		for _, e := range s.live[b.ID] {
			s.addUse(e.ID, int32(len(b.Values))+e.dist) // pseudo-uses from beyond end of block
			liveSet.add(e.ID)
		}
		if v := b.Control; v != nil && s.values[v.ID].needReg {
			s.addUse(v.ID, int32(len(b.Values))) // psuedo-use by control value
			liveSet.add(v.ID)
		}
		for i := len(b.Values) - 1; i >= 0; i-- {
			v := b.Values[i]
			liveSet.remove(v.ID)
			if v.Op == OpPhi {
				// Remove v from the live set, but don't add
				// any inputs.  This is the state the len(b.Preds)>1
				// case below desires; it wants to process phis specially.
				continue
			}
			for _, a := range v.Args {
				if !s.values[a.ID].needReg {
					continue
				}
				s.addUse(a.ID, int32(i))
				liveSet.add(a.ID)
			}
		}
		if regDebug {
			fmt.Printf("uses for %s:%s\n", s.f.Name, b)
			for i := range s.values {
				vi := &s.values[i]
				u := vi.uses
				if u == nil {
					continue
				}
				fmt.Printf("  v%d:", i)
				for u != nil {
					fmt.Printf(" %d", u.dist)
					u = u.next
				}
				fmt.Println()
			}
		}

		// Make a copy of the block schedule so we can generate a new one in place.
		// We make a separate copy for phis and regular values.
		nphi := 0
		for _, v := range b.Values {
			if v.Op != OpPhi {
				break
			}
			nphi++
		}
		phis = append(phis[:0], b.Values[:nphi]...)
		oldSched = append(oldSched[:0], b.Values[nphi:]...)
		b.Values = b.Values[:0]

		// Initialize start state of block.
		if b == f.Entry {
			// Regalloc state is empty to start.
			if nphi > 0 {
				f.Fatalf("phis in entry block")
			}
		} else if len(b.Preds) == 1 {
			// Start regalloc state with the end state of the previous block.
			s.setState(s.endRegs[b.Preds[0].ID])
			if nphi > 0 {
				f.Fatalf("phis in single-predecessor block")
			}
			// Drop any values which are no longer live.
			// This may happen because at the end of p, a value may be
			// live but only used by some other successor of p.
			for r := register(0); r < numRegs; r++ {
				v := s.regs[r].v
				if v != nil && !liveSet.contains(v.ID) {
					s.freeReg(r)
				}
			}
		} else {
			// This is the complicated case.  We have more than one predecessor,
			// which means we may have Phi ops.

			// Copy phi ops into new schedule.
			b.Values = append(b.Values, phis...)

			// Start with the final register state of the primary predecessor
			idx := s.primary[b.ID]
			if idx < 0 {
				f.Fatalf("block with no primary predecessor %s", b)
			}
			p := b.Preds[idx]
			s.setState(s.endRegs[p.ID])

			if regDebug {
				fmt.Printf("starting merge block %s with end state of %s:\n", b, p)
				for _, x := range s.endRegs[p.ID] {
					fmt.Printf("  %s: orig:%s cache:%s\n", registers[x.r].Name(), x.v, x.c)
				}
			}

			// Decide on registers for phi ops.  Use the registers determined
			// by the primary predecessor if we can.
			// TODO: pick best of (already processed) predecessors?
			// Majority vote?  Deepest nesting level?
			phiRegs = phiRegs[:0]
			var phiUsed regMask
			for _, v := range phis {
				if !s.values[v.ID].needReg {
					phiRegs = append(phiRegs, noRegister)
					continue
				}
				a := v.Args[idx]
				m := s.values[a.ID].regs &^ phiUsed
				var r register
				if m != 0 {
					r = pickReg(m)
					s.freeReg(r)
					phiUsed |= regMask(1) << r
					phiRegs = append(phiRegs, r)
				} else {
					phiRegs = append(phiRegs, noRegister)
				}
			}

			// Second pass - deallocate any phi inputs which are now dead.
			for _, v := range phis {
				if !s.values[v.ID].needReg {
					continue
				}
				a := v.Args[idx]
				if !liveSet.contains(a.ID) {
					// Input is dead beyond the phi, deallocate
					// anywhere else it might live.
					s.freeRegs(s.values[a.ID].regs)
				}
			}

			// Third pass - pick registers for phis whose inputs
			// were not in a register.
			for i, v := range phis {
				if !s.values[v.ID].needReg {
					continue
				}
				if phiRegs[i] != noRegister {
					continue
				}
				m := s.compatRegs(v.Type) &^ phiUsed &^ s.used
				if m != 0 {
					r := pickReg(m)
					phiRegs[i] = r
					phiUsed |= regMask(1) << r
				}
			}

			// Set registers for phis.  Add phi spill code.
			for i, v := range phis {
				if !s.values[v.ID].needReg {
					continue
				}
				r := phiRegs[i]
				if r == noRegister {
					// stack-based phi
					// Spills will be inserted in all the predecessors below.
					s.values[v.ID].spill = v        // v starts life spilled
					s.values[v.ID].spillUsed = true // use is guaranteed
					continue
				}
				// register-based phi
				s.assignReg(r, v, v)
				// Spill the phi in case we need to restore it later.
				spill := b.NewValue1(v.Line, OpStoreReg, v.Type, v)
				s.setOrig(spill, v)
				s.values[v.ID].spill = spill
				s.values[v.ID].spillUsed = false
			}

			// Save the starting state for use by merge edges.
			var regList []startReg
			for r := register(0); r < numRegs; r++ {
				v := s.regs[r].v
				if v == nil {
					continue
				}
				if phiUsed>>r&1 != 0 {
					// Skip registers that phis used, we'll handle those
					// specially during merge edge processing.
					continue
				}
				regList = append(regList, startReg{r, v.ID})
			}
			s.startRegs[b.ID] = regList

			if regDebug {
				fmt.Printf("after phis\n")
				for _, x := range s.startRegs[b.ID] {
					fmt.Printf("  %s: v%d\n", registers[x.r].Name(), x.vid)
				}
			}
		}

		// Compute preferred registers for each value using a backwards pass.
		// Note that we do this phase after startRegs is set above, so that
		// we get the right behavior for a block which branches to itself.
		for _, succ := range b.Succs {
			// TODO: prioritize likely successor.
			for _, x := range s.startRegs[succ.ID] {
				v := s.orig[x.vid]
				s.values[v.ID].desired = x.r
			}
			// Process phi ops in succ
			i := -1
			for j, p := range succ.Preds {
				if p == b {
					i = j
					break
				}
			}
			if i == -1 {
				s.f.Fatalf("can't find predecssor %s of %s\n", b, succ)
			}
			for _, v := range succ.Values {
				if v.Op != OpPhi {
					break
				}
				if !s.values[v.ID].needReg {
					continue
				}
				r, ok := s.f.getHome(v.ID).(*Register)
				if !ok {
					continue
				}
				a := s.orig[v.Args[i].ID]
				s.values[a.ID].desired = register(r.Num)
			}
		}

		// Set avoid fields to help desired register availability.
		liveSet.clear()
		for _, e := range s.live[b.ID] {
			liveSet.add(e.ID)
		}
		if v := b.Control; v != nil && s.values[v.ID].needReg {
			liveSet.add(v.ID)
		}
		for i := len(oldSched) - 1; i >= 0; i-- {
			v := oldSched[i]
			liveSet.remove(v.ID)

			r := s.values[v.ID].desired
			if r != noRegister {
				m := regMask(1) << r
				// All live values should avoid this register so
				// it will be available at this point.
				for _, w := range liveSet.contents() {
					s.values[w].avoid |= m
				}
			}

			for _, a := range v.Args {
				if !s.values[a.ID].needReg {
					continue
				}
				liveSet.add(a.ID)
			}
		}

		// Process all the non-phi values.
		for _, v := range oldSched {
			if regDebug {
				fmt.Printf("  processing %s\n", v.LongString())
			}
			if v.Op == OpPhi {
				f.Fatalf("phi %s not at start of block", v)
			}
			if v.Op == OpSP {
				s.assignReg(4, v, v) // TODO: arch-dependent
				b.Values = append(b.Values, v)
				s.advanceUses(v)
				continue
			}
			if v.Op == OpSB {
				s.assignReg(32, v, v) // TODO: arch-dependent
				b.Values = append(b.Values, v)
				s.advanceUses(v)
				continue
			}
			if v.Op == OpArg {
				// Args are "pre-spilled" values.  We don't allocate
				// any register here.  We just set up the spill pointer to
				// point at itself and any later user will restore it to use it.
				s.values[v.ID].spill = v
				s.values[v.ID].spillUsed = true // use is guaranteed
				b.Values = append(b.Values, v)
				s.advanceUses(v)
				continue
			}
			regspec := opcodeTable[v.Op].reg
			if len(regspec.inputs) == 0 && len(regspec.outputs) == 0 {
				// No register allocation required (or none specified yet)
				s.freeRegs(regspec.clobbers)
				b.Values = append(b.Values, v)
				continue
			}

			if s.values[v.ID].rematerializeable {
				// Value is rematerializeable, don't issue it here.
				// It will get issued just before each use (see
				// allocValueToReg).
				s.advanceUses(v)
				continue
			}

			// Move arguments to registers.  Process in an ordering defined
			// by the register specification (most constrained first).
			args = append(args[:0], v.Args...)
			for _, i := range regspec.inputs {
				if i.regs == flagRegMask {
					// TODO: remove flag input from regspec.inputs.
					continue
				}
				args[i.idx] = s.allocValToReg(v.Args[i.idx], i.regs, true, v.Line)
			}

			// Now that all args are in regs, we're ready to issue the value itself.
			// Before we pick a register for the output value, allow input registers
			// to be deallocated. We do this here so that the output can use the
			// same register as a dying input.
			s.nospill = 0
			s.advanceUses(v) // frees any registers holding args that are no longer live

			// Dump any registers which will be clobbered
			s.freeRegs(regspec.clobbers)

			// Pick register for output.
			var mask regMask
			if s.values[v.ID].needReg {
				mask = regspec.outputs[0] &^ s.reserved()
				if mask>>33&1 != 0 {
					s.f.Fatalf("bad mask %s\n", v.LongString())
				}
			}
			if mask != 0 {
				r := s.allocReg(v, mask)
				s.assignReg(r, v, v)
			}

			// Issue the Value itself.
			for i, a := range args {
				v.Args[i] = a // use register version of arguments
			}
			b.Values = append(b.Values, v)

			// Issue a spill for this value.  We issue spills unconditionally,
			// then at the end of regalloc delete the ones we never use.
			// TODO: schedule the spill at a point that dominates all restores.
			// The restore may be off in an unlikely branch somewhere and it
			// would be better to have the spill in that unlikely branch as well.
			// v := ...
			// if unlikely {
			//     f()
			// }
			// It would be good to have both spill and restore inside the IF.
			if s.values[v.ID].needReg {
				spill := b.NewValue1(v.Line, OpStoreReg, v.Type, v)
				s.setOrig(spill, v)
				s.values[v.ID].spill = spill
				s.values[v.ID].spillUsed = false
			}
		}

		if v := b.Control; v != nil && s.values[v.ID].needReg {
			if regDebug {
				fmt.Printf("  processing control %s\n", v.LongString())
			}
			// Load control value into reg.
			// TODO: regspec for block control values, instead of using
			// register set from the control op's output.
			s.allocValToReg(v, opcodeTable[v.Op].reg.outputs[0], false, b.Line)
			// Remove this use from the uses list.
			vi := &s.values[v.ID]
			u := vi.uses
			vi.uses = u.next
			if u.next == nil {
				s.freeRegs(vi.regs) // value is dead
			}
			u.next = s.freeUseRecords
			s.freeUseRecords = u
		}

		// Save end-of-block register state.
		// First count how many, this cuts allocations in half.
		k := 0
		for r := register(0); r < numRegs; r++ {
			v := s.regs[r].v
			if v == nil {
				continue
			}
			k++
		}
		regList := make([]endReg, 0, k)
		for r := register(0); r < numRegs; r++ {
			v := s.regs[r].v
			if v == nil {
				continue
			}
			regList = append(regList, endReg{r, v, s.regs[r].c})
		}
		s.endRegs[b.ID] = regList

		// Check. TODO: remove
		{
			liveSet.clear()
			for _, x := range s.live[b.ID] {
				liveSet.add(x.ID)
			}
			for r := register(0); r < numRegs; r++ {
				v := s.regs[r].v
				if v == nil {
					continue
				}
				if !liveSet.contains(v.ID) {
					s.f.Fatalf("val %s is in reg but not live at end of %s", v, b)
				}
			}
		}

		// If a value is live at the end of the block and
		// isn't in a register, remember that its spill location
		// is live.  We need to remember this information so that
		// the liveness analysis in stackalloc is correct.
		for _, e := range s.live[b.ID] {
			if s.values[e.ID].regs != 0 {
				// in a register, we'll use that source for the merge.
				continue
			}
			spill := s.values[e.ID].spill
			if spill == nil {
				// rematerializeable values will have spill==nil.
				continue
			}
			s.spillLive[b.ID] = append(s.spillLive[b.ID], spill.ID)
			s.values[e.ID].spillUsed = true
		}

		// Clear any final uses.
		// All that is left should be the pseudo-uses added for values which
		// are live at the end of b.
		for _, e := range s.live[b.ID] {
			u := s.values[e.ID].uses
			if u == nil {
				f.Fatalf("live at end, no uses v%d", e.ID)
			}
			if u.next != nil {
				f.Fatalf("live at end, too many uses v%d", e.ID)
			}
			s.values[e.ID].uses = nil
			u.next = s.freeUseRecords
			s.freeUseRecords = u
		}
	}

	// Erase any spills we never used
	for i := range s.values {
		vi := s.values[i]
		if vi.spillUsed {
			if logSpills {
				fmt.Println("regalloc: spilled value")
			}
			continue
		}
		spill := vi.spill
		if spill == nil {
			// Constants, SP, SB, ...
			continue
		}
		f.freeValue(spill)
	}
	for _, b := range f.Blocks {
		i := 0
		for _, v := range b.Values {
			if v.Op == OpInvalid {
				continue
			}
			b.Values[i] = v
			i++
		}
		b.Values = b.Values[:i]
		// TODO: zero b.Values[i:], recycle Values
		// Not important now because this is the last phase that manipulates Values
	}

	// Anything that didn't get a register gets a stack location here.
	// (StoreReg, stack-based phis, inputs, ...)
	stacklive := stackalloc(s.f, s.spillLive)

	// Fix up all merge edges.
	s.shuffle(stacklive)
}

// shuffle fixes up all the merge edges (those going into blocks of indegree > 1).
func (s *regAllocState) shuffle(stacklive [][]ID) {
	var e edgeState
	e.s = s
	e.cache = map[ID][]*Value{}
	e.contents = map[Location]contentRecord{}
	if regDebug {
		fmt.Printf("shuffle %s\n", s.f.Name)
		fmt.Println(s.f.String())
	}

	for _, b := range s.f.Blocks {
		if len(b.Preds) <= 1 {
			continue
		}
		e.b = b
		for i, p := range b.Preds {
			e.p = p
			e.setup(i, s.endRegs[p.ID], s.startRegs[b.ID], stacklive[p.ID])
			e.process()
		}
	}
}

type edgeState struct {
	s    *regAllocState
	p, b *Block // edge goes from p->b.

	// for each pre-regalloc value, a list of equivalent cached values
	cache map[ID][]*Value

	// map from location to the value it contains
	contents map[Location]contentRecord

	// desired destination locations
	destinations []dstRecord
	extra        []dstRecord

	usedRegs   regMask // registers currently holding something
	uniqueRegs regMask // registers holding the only copy of a value
	finalRegs  regMask // registers holding final target
}

type contentRecord struct {
	vid   ID     // pre-regalloc value
	c     *Value // cached value
	final bool   // this is a satisfied destination
}

type dstRecord struct {
	loc    Location // register or stack slot
	vid    ID       // pre-regalloc value it should contain
	splice **Value  // place to store reference to the generating instruction
}

// setup initializes the edge state for shuffling.
func (e *edgeState) setup(idx int, srcReg []endReg, dstReg []startReg, stacklive []ID) {
	if regDebug {
		fmt.Printf("edge %s->%s\n", e.p, e.b)
	}

	// Clear state.
	for k := range e.cache {
		delete(e.cache, k)
	}
	for k := range e.contents {
		delete(e.contents, k)
	}
	e.usedRegs = 0
	e.uniqueRegs = 0
	e.finalRegs = 0

	// Live registers can be sources.
	for _, x := range srcReg {
		e.set(&registers[x.r], x.v.ID, x.c, false)
	}
	// So can all of the spill locations.
	for _, spillID := range stacklive {
		v := e.s.orig[spillID]
		spill := e.s.values[v.ID].spill
		e.set(e.s.f.getHome(spillID), v.ID, spill, false)
	}

	// Figure out all the destinations we need.
	dsts := e.destinations[:0]
	for _, x := range dstReg {
		dsts = append(dsts, dstRecord{&registers[x.r], x.vid, nil})
	}
	// Phis need their args to end up in a specific location.
	for _, v := range e.b.Values {
		if v.Op != OpPhi {
			break
		}
		loc := e.s.f.getHome(v.ID)
		if loc == nil {
			continue
		}
		dsts = append(dsts, dstRecord{loc, v.Args[idx].ID, &v.Args[idx]})
	}
	e.destinations = dsts

	if regDebug {
		for vid, a := range e.cache {
			for _, c := range a {
				fmt.Printf("src %s: v%d cache=%s\n", e.s.f.getHome(c.ID).Name(), vid, c)
			}
		}
		for _, d := range e.destinations {
			fmt.Printf("dst %s: v%d\n", d.loc.Name(), d.vid)
		}
	}
}

// process generates code to move all the values to the right destination locations.
func (e *edgeState) process() {
	dsts := e.destinations

	// Process the destinations until they are all satisfied.
	for len(dsts) > 0 {
		i := 0
		for _, d := range dsts {
			if !e.processDest(d.loc, d.vid, d.splice) {
				// Failed - save for next iteration.
				dsts[i] = d
				i++
			}
		}
		if i < len(dsts) {
			// Made some progress.  Go around again.
			dsts = dsts[:i]

			// Append any extras destinations we generated.
			dsts = append(dsts, e.extra...)
			e.extra = e.extra[:0]
			continue
		}

		// We made no progress.  That means that any
		// remaining unsatisfied moves are in simple cycles.
		// For example, A -> B -> C -> D -> A.
		//   A ----> B
		//   ^       |
		//   |       |
		//   |       v
		//   D <---- C

		// To break the cycle, we pick an unused register, say R,
		// and put a copy of B there.
		//   A ----> B
		//   ^       |
		//   |       |
		//   |       v
		//   D <---- C <---- R=copyofB
		// When we resume the outer loop, the A->B move can now proceed,
		// and eventually the whole cycle completes.

		// Copy any cycle location to a temp register.  This duplicates
		// one of the cycle entries, allowing the just duplicated value
		// to be overwritten and the cycle to proceed.
		loc := dsts[0].loc
		vid := e.contents[loc].vid
		c := e.contents[loc].c
		r := e.findRegFor(c.Type)
		if regDebug {
			fmt.Printf("breaking cycle with v%d in %s:%s\n", vid, loc.Name(), c)
		}
		if _, isReg := loc.(*Register); isReg {
			c = e.p.NewValue1(c.Line, OpCopy, c.Type, c)
		} else {
			c = e.p.NewValue1(c.Line, OpLoadReg, c.Type, c)
		}
		e.set(r, vid, c, false)
	}
}

// processDest generates code to put value vid into location loc.  Returns true
// if progress was made.
func (e *edgeState) processDest(loc Location, vid ID, splice **Value) bool {
	occupant := e.contents[loc]
	if occupant.vid == vid {
		// Value is already in the correct place.
		e.contents[loc] = contentRecord{vid, occupant.c, true}
		if splice != nil {
			*splice = occupant.c
		}
		// Note: if splice==nil then c will appear dead.  This is
		// non-SSA formed code, so be careful after this pass not to run
		// deadcode elimination.
		return true
	}

	// Check if we're allowed to clobber the destination location.
	if len(e.cache[occupant.vid]) == 1 && !e.s.values[occupant.vid].rematerializeable {
		// We can't overwrite the last copy
		// of a value that needs to survive.
		return false
	}

	// Copy from a source of v, register preferred.
	v := e.s.orig[vid]
	var c *Value
	var src Location
	if regDebug {
		fmt.Printf("moving v%d to %s\n", vid, loc.Name())
		fmt.Printf("sources of v%d:", vid)
	}
	for _, w := range e.cache[vid] {
		h := e.s.f.getHome(w.ID)
		if regDebug {
			fmt.Printf(" %s:%s", h.Name(), w)
		}
		_, isreg := h.(*Register)
		if src == nil || isreg {
			c = w
			src = h
		}
	}
	if regDebug {
		if src != nil {
			fmt.Printf(" [use %s]\n", src.Name())
		} else {
			fmt.Printf(" [no source]\n")
		}
	}
	_, dstReg := loc.(*Register)
	var x *Value
	if c == nil {
		if !e.s.values[vid].rematerializeable {
			e.s.f.Fatalf("can't find source for %s->%s: v%d\n", e.p, e.b, vid)
		}
		if dstReg {
			x = v.copyInto(e.p)
		} else {
			// Rematerialize into stack slot.  Need a free
			// register to accomplish this.
			e.erase(loc) // see pre-clobber comment below
			r := e.findRegFor(v.Type)
			x = v.copyInto(e.p)
			e.set(r, vid, x, false)
			// Make sure we spill with the size of the slot, not the
			// size of x (which might be wider due to our dropping
			// of narrowing conversions).
			x = e.p.NewValue1(x.Line, OpStoreReg, loc.(LocalSlot).Type, x)
		}
	} else {
		// Emit move from src to dst.
		_, srcReg := src.(*Register)
		if srcReg {
			if dstReg {
				x = e.p.NewValue1(c.Line, OpCopy, c.Type, c)
			} else {
				x = e.p.NewValue1(c.Line, OpStoreReg, loc.(LocalSlot).Type, c)
			}
		} else {
			if dstReg {
				x = e.p.NewValue1(c.Line, OpLoadReg, c.Type, c)
			} else {
				// mem->mem.  Use temp register.

				// Pre-clobber destination.  This avoids the
				// following situation:
				//   - v is currently held in R0 and stacktmp0.
				//   - We want to copy stacktmp1 to stacktmp0.
				//   - We choose R0 as the temporary register.
				// During the copy, both R0 and stacktmp0 are
				// clobbered, losing both copies of v.  Oops!
				// Erasing the destination early means R0 will not
				// be chosen as the temp register, as it will then
				// be the last copy of v.
				e.erase(loc)

				r := e.findRegFor(c.Type)
				t := e.p.NewValue1(c.Line, OpLoadReg, c.Type, c)
				e.set(r, vid, t, false)
				x = e.p.NewValue1(c.Line, OpStoreReg, loc.(LocalSlot).Type, t)
			}
		}
	}
	e.set(loc, vid, x, true)
	if splice != nil {
		*splice = x
	}
	return true
}

// set changes the contents of location loc to hold the given value and its cached representative.
func (e *edgeState) set(loc Location, vid ID, c *Value, final bool) {
	e.s.f.setHome(c, loc)
	e.erase(loc)
	e.contents[loc] = contentRecord{vid, c, final}
	a := e.cache[vid]
	a = append(a, c)
	e.cache[vid] = a
	if r, ok := loc.(*Register); ok {
		e.usedRegs |= regMask(1) << uint(r.Num)
		if final {
			e.finalRegs |= regMask(1) << uint(r.Num)
		}
		if len(a) == 1 {
			e.uniqueRegs |= regMask(1) << uint(r.Num)
		}
		if len(a) == 2 {
			if t, ok := e.s.f.getHome(a[0].ID).(*Register); ok {
				e.uniqueRegs &^= regMask(1) << uint(t.Num)
			}
		}
	}
	if regDebug {
		fmt.Printf("%s\n", c.LongString())
		fmt.Printf("v%d now available in %s:%s\n", vid, loc.Name(), c)
	}
}

// erase removes any user of loc.
func (e *edgeState) erase(loc Location) {
	cr := e.contents[loc]
	if cr.c == nil {
		return
	}
	vid := cr.vid

	if cr.final {
		// Add a destination to move this value back into place.
		// Make sure it gets added to the tail of the destination queue
		// so we make progress on other moves first.
		e.extra = append(e.extra, dstRecord{loc, cr.vid, nil})
	}

	// Remove c from the list of cached values.
	a := e.cache[vid]
	for i, c := range a {
		if e.s.f.getHome(c.ID) == loc {
			if regDebug {
				fmt.Printf("v%d no longer available in %s:%s\n", vid, loc.Name(), c)
			}
			a[i], a = a[len(a)-1], a[:len(a)-1]
			break
		}
	}
	e.cache[vid] = a

	// Update register masks.
	if r, ok := loc.(*Register); ok {
		e.usedRegs &^= regMask(1) << uint(r.Num)
		if cr.final {
			e.finalRegs &^= regMask(1) << uint(r.Num)
		}
	}
	if len(a) == 1 {
		if r, ok := e.s.f.getHome(a[0].ID).(*Register); ok {
			e.uniqueRegs |= regMask(1) << uint(r.Num)
		}
	}
}

// findRegFor finds a register we can use to make a temp copy of type typ.
func (e *edgeState) findRegFor(typ Type) Location {
	// Which registers are possibilities.
	var m regMask
	if typ.IsFloat() {
		m = e.s.compatRegs(e.s.f.Config.fe.TypeFloat64())
	} else {
		m = e.s.compatRegs(e.s.f.Config.fe.TypeInt64())
	}

	// Pick a register.  In priority order:
	// 1) an unused register
	// 2) a non-unique register not holding a final value
	// 3) a non-unique register
	x := m &^ e.usedRegs
	if x != 0 {
		return &registers[pickReg(x)]
	}
	x = m &^ e.uniqueRegs &^ e.finalRegs
	if x != 0 {
		return &registers[pickReg(x)]
	}
	x = m &^ e.uniqueRegs
	if x != 0 {
		return &registers[pickReg(x)]
	}

	// No register is available.  Allocate a temp location to spill a register to.
	// The type of the slot is immaterial - it will not be live across
	// any safepoint.  Just use a type big enough to hold any register.
	typ = e.s.f.Config.fe.TypeInt64()
	t := LocalSlot{e.s.f.Config.fe.Auto(typ), typ, 0}
	// TODO: reuse these slots.

	// Pick a register to spill.
	for vid, a := range e.cache {
		for _, c := range a {
			if r, ok := e.s.f.getHome(c.ID).(*Register); ok && m>>uint(r.Num)&1 != 0 {
				x := e.p.NewValue1(c.Line, OpStoreReg, c.Type, c)
				e.set(t, vid, x, false)
				if regDebug {
					fmt.Printf("  SPILL %s->%s %s\n", r.Name(), t.Name(), x.LongString())
				}
				// r will now be overwritten by the caller.  At some point
				// later, the newly saved value will be moved back to its
				// final destination in processDest.
				return r
			}
		}
	}

	fmt.Printf("m:%d unique:%d final:%d\n", m, e.uniqueRegs, e.finalRegs)
	for vid, a := range e.cache {
		for _, c := range a {
			fmt.Printf("v%d: %s %s\n", vid, c, e.s.f.getHome(c.ID).Name())
		}
	}
	e.s.f.Fatalf("can't find empty register on edge %s->%s", e.p, e.b)
	return nil
}

func (v *Value) rematerializeable() bool {
	if !opcodeTable[v.Op].rematerializeable {
		return false
	}
	for _, a := range v.Args {
		// SP and SB (generated by OpSP and OpSB) are always available.
		if a.Op != OpSP && a.Op != OpSB {
			return false
		}
	}
	return true
}

type liveInfo struct {
	ID   ID    // ID of variable
	dist int32 // # of instructions before next use
}

// computeLive computes a map from block ID to a list of value IDs live at the end
// of that block.  Together with the value ID is a count of how many instructions
// to the next use of that value.  The resulting map is stored at s.live.
// TODO: this could be quadratic if lots of variables are live across lots of
// basic blocks.  Figure out a way to make this function (or, more precisely, the user
// of this function) require only linear size & time.
func (s *regAllocState) computeLive() {
	f := s.f
	s.live = make([][]liveInfo, f.NumBlocks())
	var phis []*Value

	live := newSparseMap(f.NumValues())
	t := newSparseMap(f.NumValues())

	// Instead of iterating over f.Blocks, iterate over their postordering.
	// Liveness information flows backward, so starting at the end
	// increases the probability that we will stabilize quickly.
	// TODO: Do a better job yet. Here's one possibility:
	// Calculate the dominator tree and locate all strongly connected components.
	// If a value is live in one block of an SCC, it is live in all.
	// Walk the dominator tree from end to beginning, just once, treating SCC
	// components as single blocks, duplicated calculated liveness information
	// out to all of them.
	po := postorder(f)
	for {
		changed := false

		for _, b := range po {
			// Start with known live values at the end of the block.
			// Add len(b.Values) to adjust from end-of-block distance
			// to beginning-of-block distance.
			live.clear()
			for _, e := range s.live[b.ID] {
				live.set(e.ID, e.dist+int32(len(b.Values)))
			}

			// Mark control value as live
			if b.Control != nil && s.values[b.Control.ID].needReg {
				live.set(b.Control.ID, int32(len(b.Values)))
			}

			// Propagate backwards to the start of the block
			// Assumes Values have been scheduled.
			phis := phis[:0]
			for i := len(b.Values) - 1; i >= 0; i-- {
				v := b.Values[i]
				live.remove(v.ID)
				if v.Op == OpPhi {
					// save phi ops for later
					phis = append(phis, v)
					continue
				}
				for _, a := range v.Args {
					if s.values[a.ID].needReg {
						live.set(a.ID, int32(i))
					}
				}
			}

			// For each predecessor of b, expand its list of live-at-end values.
			// invariant: live contains the values live at the start of b (excluding phi inputs)
			for i, p := range b.Preds {
				// Compute additional distance for the edge.
				const normalEdge = 10
				const likelyEdge = 1
				const unlikelyEdge = 100
				// Note: delta must be at least 1 to distinguish the control
				// value use from the first user in a successor block.
				delta := int32(normalEdge)
				if len(p.Succs) == 2 {
					if p.Succs[0] == b && p.Likely == BranchLikely ||
						p.Succs[1] == b && p.Likely == BranchUnlikely {
						delta = likelyEdge
					}
					if p.Succs[0] == b && p.Likely == BranchUnlikely ||
						p.Succs[1] == b && p.Likely == BranchLikely {
						delta = unlikelyEdge
					}
				}

				// Start t off with the previously known live values at the end of p.
				t.clear()
				for _, e := range s.live[p.ID] {
					t.set(e.ID, e.dist)
				}
				update := false

				// Add new live values from scanning this block.
				for _, e := range live.contents() {
					d := e.val + delta
					if !t.contains(e.key) || d < t.get(e.key) {
						update = true
						t.set(e.key, d)
					}
				}
				// Also add the correct arg from the saved phi values.
				// All phis are at distance delta (we consider them
				// simultaneously happening at the start of the block).
				for _, v := range phis {
					id := v.Args[i].ID
					if s.values[id].needReg && !t.contains(id) || delta < t.get(id) {
						update = true
						t.set(id, delta)
					}
				}

				if !update {
					continue
				}
				// The live set has changed, update it.
				l := s.live[p.ID][:0]
				if cap(l) < t.size() {
					l = make([]liveInfo, 0, t.size())
				}
				for _, e := range t.contents() {
					l = append(l, liveInfo{e.key, e.val})
				}
				s.live[p.ID] = l
				changed = true
			}
		}

		if !changed {
			break
		}
	}
	if regDebug {
		fmt.Println("live values at end of each block")
		for _, b := range f.Blocks {
			fmt.Printf("  %s:", b)
			for _, x := range s.live[b.ID] {
				fmt.Printf(" v%d", x.ID)
			}
			fmt.Println()
		}
	}
}

// reserved returns a mask of reserved registers.
func (s *regAllocState) reserved() regMask {
	var m regMask
	if obj.Framepointer_enabled != 0 {
		m |= 1 << 5 // BP
	}
	if s.f.Config.ctxt.Flag_dynlink {
		m |= 1 << 15 // R15
	}
	return m
}
