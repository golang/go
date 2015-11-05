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
// Flags values are special. Instead of attempting to spill and restore the flags
// register, we recalculate it if needed.
// There are more efficient schemes (see the discussion in CL 13844),
// but flag restoration is empirically rare, and this approach is simple
// and architecture-independent.
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

const regDebug = false
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
	Register{33, "FLAGS"},

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
	regs       regMask // the set of registers holding a Value (usually just one)
	uses       *use    // list of uses in this block
	spill      *Value  // spilled copy of the Value
	spill2     *Value  // special alternate spill location used for phi resolution
	spillUsed  bool
	spill2used bool
}

type regState struct {
	v *Value // Original (preregalloc) Value stored in this register.
	c *Value // A Value equal to v which is currently in a register.  Might be v or a copy of it.
	// If a register is unused, v==c==nil
}

type regAllocState struct {
	f *Func

	// For each value, whether it needs a register or not.
	// Cached value of !v.Type.IsMemory() && !v.Type.IsVoid().
	needReg []bool

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

	// Home locations (registers) for Values
	home []Location

	// current block we're working on
	curBlock *Block

	// cache of use records
	freeUseRecords *use
}

// freeReg frees up register r.  Any current user of r is kicked out.
func (s *regAllocState) freeReg(r register) {
	v := s.regs[r].v
	if v == nil {
		s.f.Fatalf("tried to free an already free register %d\n", r)
	}

	// Mark r as unused.
	if regDebug {
		fmt.Printf("freeReg %d (dump %s/%s)\n", r, v, s.regs[r].c)
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

func (s *regAllocState) setHome(v *Value, r register) {
	// Remember assignment.
	for int(v.ID) >= len(s.home) {
		s.home = append(s.home, nil)
		s.home = s.home[:cap(s.home)]
	}
	s.home[v.ID] = &registers[r]
}
func (s *regAllocState) getHome(v *Value) register {
	if int(v.ID) >= len(s.home) || s.home[v.ID] == nil {
		return noRegister
	}
	return register(s.home[v.ID].(*Register).Num)
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
		fmt.Printf("assignReg %d %s/%s\n", r, v, c)
	}
	if s.regs[r].v != nil {
		s.f.Fatalf("tried to assign register %d to %s/%s but it is already used by %s", r, v, c, s.regs[r].v)
	}

	// Update state.
	s.regs[r] = regState{v, c}
	s.values[v.ID].regs |= regMask(1) << r
	s.used |= regMask(1) << r
	s.setHome(c, r)
}

// allocReg picks an unused register from regmask.  If there is no unused register,
// a Value will be kicked out of a register to make room.
func (s *regAllocState) allocReg(mask regMask) register {
	// Pick a register to use.
	mask &^= s.nospill
	if mask == 0 {
		s.f.Fatalf("no register available")
	}

	var r register
	if unused := mask & ^s.used; unused != 0 {
		// Pick an unused register.
		return pickReg(unused)
		// TODO: use affinity graph to pick a good register
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
	maxuse := int32(-1)
	for t := register(0); t < numRegs; t++ {
		if mask>>t&1 == 0 {
			continue
		}
		v := s.regs[t].v

		if s.values[v.ID].uses == nil {
			// No subsequent use.
			// This can happen when fixing up merge blocks at the end.
			// We've already run through the use lists so they are empty.
			// Any register would be ok at this point.
			r = t
			maxuse = 0
			break
		}
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
func (s *regAllocState) allocValToReg(v *Value, mask regMask, nospill bool) *Value {
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
	r := s.allocReg(mask)

	// Allocate v to the new register.
	var c *Value
	if vi.regs != 0 {
		// Copy from a register that v is already in.
		r2 := pickReg(vi.regs)
		if s.regs[r2].v != v {
			panic("bad register state")
		}
		c = s.curBlock.NewValue1(v.Line, OpCopy, v.Type, s.regs[r2].c)
	} else if v.rematerializeable() {
		// Rematerialize instead of loading from the spill location.
		c = s.curBlock.NewValue0(v.Line, v.Op, v.Type)
		c.Aux = v.Aux
		c.AuxInt = v.AuxInt
		c.AddArgs(v.Args...)
	} else {
		switch {
		// It is difficult to spill and reload flags on many architectures.
		// Instead, we regenerate the flags register by issuing the same instruction again.
		// This requires (possibly) spilling and reloading that instruction's args.
		case v.Type.IsFlags():
			if logSpills {
				fmt.Println("regalloc: regenerating flags")
			}
			ns := s.nospill
			// Place v's arguments in registers, spilling and loading as needed
			args := make([]*Value, 0, len(v.Args))
			regspec := opcodeTable[v.Op].reg
			for _, i := range regspec.inputs {
				// Extract the original arguments to v
				a := s.orig[v.Args[i.idx].ID]
				if a.Type.IsFlags() {
					s.f.Fatalf("cannot load flags value with flags arg: %v has unwrapped arg %v", v.LongString(), a.LongString())
				}
				cc := s.allocValToReg(a, i.regs, true)
				args = append(args, cc)
			}
			s.nospill = ns
			// Recalculate v
			c = s.curBlock.NewValue0(v.Line, v.Op, v.Type)
			c.Aux = v.Aux
			c.AuxInt = v.AuxInt
			c.resetArgs()
			c.AddArgs(args...)

		// Load v from its spill location.
		case vi.spill2 != nil:
			if logSpills {
				fmt.Println("regalloc: load spill2")
			}
			c = s.curBlock.NewValue1(v.Line, OpLoadReg, v.Type, vi.spill2)
			vi.spill2used = true
		case vi.spill != nil:
			if logSpills {
				fmt.Println("regalloc: load spill")
			}
			c = s.curBlock.NewValue1(v.Line, OpLoadReg, v.Type, vi.spill)
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
	s.needReg = make([]bool, f.NumValues())
	s.regs = make([]regState, numRegs)
	s.values = make([]valState, f.NumValues())
	s.orig = make([]*Value, f.NumValues())
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Type.IsMemory() || v.Type.IsVoid() {
				continue
			}
			s.needReg[v.ID] = true
			s.orig[v.ID] = v
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
		if !s.needReg[a.ID] {
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

// Sets the state of the registers to that encoded in state.
func (s *regAllocState) setState(state []regState) {
	s.freeRegs(s.used)
	for r, x := range state {
		if x.c == nil {
			continue
		}
		s.assignReg(register(r), x.v, x.c)
	}
}

// compatRegs returns the set of registers which can store v.
func (s *regAllocState) compatRegs(v *Value) regMask {
	var m regMask
	if v.Type.IsFloat() {
		m = 0xffff << 16 // X0-X15
	} else {
		m = 0xffef << 0 // AX-R15, except SP
	}
	return m &^ s.reserved()
}

func (s *regAllocState) regalloc(f *Func) {
	liveSet := newSparseSet(f.NumValues())
	argset := newSparseSet(f.NumValues())
	var oldSched []*Value
	var phis []*Value
	var stackPhis []*Value
	var regPhis []*Value
	var phiRegs []register
	var args []*Value

	if f.Entry != f.Blocks[0] {
		f.Fatalf("entry block must be first")
	}

	// For each merge block, we record the starting register state (after phi ops)
	// for that merge block.  Indexed by blockid/regnum.
	startRegs := make([][]*Value, f.NumBlocks())
	// end state of registers for each block, idexed by blockid/regnum.
	endRegs := make([][]regState, f.NumBlocks())
	for _, b := range f.Blocks {
		s.curBlock = b

		// Initialize liveSet and uses fields for this block.
		// Walk backwards through the block doing liveness analysis.
		liveSet.clear()
		for _, e := range s.live[b.ID] {
			s.addUse(e.ID, int32(len(b.Values))+e.dist) // pseudo-uses from beyond end of block
			liveSet.add(e.ID)
		}
		if c := b.Control; c != nil && s.needReg[c.ID] {
			s.addUse(c.ID, int32(len(b.Values))) // psuedo-use by control value
			liveSet.add(c.ID)
		}
		for i := len(b.Values) - 1; i >= 0; i-- {
			v := b.Values[i]
			if v.Op == OpPhi {
				break // Don't process phi ops.
			}
			liveSet.remove(v.ID)
			for _, a := range v.Args {
				if !s.needReg[a.ID] {
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
				fmt.Printf("v%d:", i)
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
			s.setState(endRegs[b.Preds[0].ID])
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
			s.setState(endRegs[p.ID])

			// Decide on registers for phi ops.  Use the registers determined
			// by the primary predecessor if we can.
			// TODO: pick best of (already processed) predecessors?
			// Majority vote?  Deepest nesting level?
			phiRegs = phiRegs[:0]
			var used regMask
			for _, v := range phis {
				if v.Type.IsMemory() {
					phiRegs = append(phiRegs, noRegister)
					continue
				}
				regs := s.values[v.Args[idx].ID].regs
				m := regs &^ used
				var r register
				if m != 0 {
					r = pickReg(m)
					used |= regMask(1) << r
				} else {
					r = noRegister
				}
				phiRegs = append(phiRegs, r)
			}
			// Change register user from phi input to phi.  Add phi spill code.
			for i, v := range phis {
				if v.Type.IsMemory() {
					continue
				}
				r := phiRegs[i]
				if r == noRegister {
					m := s.compatRegs(v) & ^s.used
					if m == 0 {
						// stack-based phi
						// Spills will be inserted in all the predecessors below.
						s.values[v.ID].spill = v        // v starts life spilled
						s.values[v.ID].spillUsed = true // use is guaranteed
						continue
					}
					// Allocate phi to an unused register.
					r = pickReg(m)
				} else {
					s.freeReg(r)
				}
				// register-based phi
				// Transfer ownership of register from input arg to phi.
				s.assignReg(r, v, v)
				// Spill the phi in case we need to restore it later.
				spill := b.NewValue1(v.Line, OpStoreReg, v.Type, v)
				s.setOrig(spill, v)
				s.values[v.ID].spill = spill
				s.values[v.ID].spillUsed = false
			}

			// Save the starting state for use by incoming edges below.
			startRegs[b.ID] = make([]*Value, numRegs)
			for r := register(0); r < numRegs; r++ {
				startRegs[b.ID][r] = s.regs[r].v
			}
		}

		// Process all the non-phi values.
		for idx, v := range oldSched {
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
			if regDebug {
				fmt.Printf("%d: working on %s %s %v\n", idx, v, v.LongString(), regspec)
			}
			if len(regspec.inputs) == 0 && len(regspec.outputs) == 0 {
				// No register allocation required (or none specified yet)
				s.freeRegs(regspec.clobbers)
				b.Values = append(b.Values, v)
				continue
			}

			if v.rematerializeable() {
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
				args[i.idx] = s.allocValToReg(v.Args[i.idx], i.regs, true)
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
			var r register
			var mask regMask
			if len(regspec.outputs) > 0 {
				mask = regspec.outputs[0] &^ s.reserved()
			}
			if mask != 0 {
				r = s.allocReg(mask)
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
			if !v.Type.IsFlags() {
				spill := b.NewValue1(v.Line, OpStoreReg, v.Type, v)
				s.setOrig(spill, v)
				s.values[v.ID].spill = spill
				s.values[v.ID].spillUsed = false
			}
		}

		if c := b.Control; c != nil && s.needReg[c.ID] {
			// Load control value into reg.
			// TODO: regspec for block control values, instead of using
			// register set from the control op's output.
			s.allocValToReg(c, opcodeTable[c.Op].reg.outputs[0], false)
			// Remove this use from the uses list.
			u := s.values[c.ID].uses
			s.values[c.ID].uses = u.next
			u.next = s.freeUseRecords
			s.freeUseRecords = u
		}

		// Record endRegs
		endRegs[b.ID] = make([]regState, numRegs)
		copy(endRegs[b.ID], s.regs)

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

	// Process merge block input edges.  They are the tricky ones.
	dst := make([]*Value, numRegs)
	for _, b := range f.Blocks {
		if len(b.Preds) <= 1 {
			continue
		}
		for i, p := range b.Preds {
			if regDebug {
				fmt.Printf("processing %s->%s\n", p, b)
			}

			// Find phis, separate them into stack & register classes.
			stackPhis = stackPhis[:0]
			regPhis = regPhis[:0]
			for _, v := range b.Values {
				if v.Op != OpPhi {
					break
				}
				if v.Type.IsMemory() {
					continue
				}
				if s.getHome(v) != noRegister {
					regPhis = append(regPhis, v)
				} else {
					stackPhis = append(stackPhis, v)
				}
			}

			// Start with the state that exists at the end of the
			// predecessor block.  We'll be adding instructions here
			// to shuffle registers & stack phis into the right spot.
			s.setState(endRegs[p.ID])
			s.curBlock = p

			// Handle stack-based phi ops first.  We need to handle them
			// first because we need a register with which to copy them.

			// We must be careful not to overwrite any stack phis which are
			// themselves args of other phis.  For example:
			//  v1 = phi(v2, v3) : 8(SP)
			//  v2 = phi(v4, v5) : 16(SP)
			// Here we must not write v2 until v2 is read and written to v1.
			// The situation could be even more complicated, with cycles, etc.
			// So in the interest of being simple, we find all the phis which
			// are arguments of other phis and copy their values to a temporary
			// location first.  This temporary location is called "spill2" and
			// represents a higher-priority but temporary spill location for the value.
			// Note this is not a problem for register-based phis because
			// if needed we will use the spilled location as the source, and
			// the spill location is not clobbered by the code generated here.
			argset.clear()
			for _, v := range stackPhis {
				argset.add(v.Args[i].ID)
			}
			for _, v := range regPhis {
				argset.add(v.Args[i].ID)
			}
			for _, v := range stackPhis {
				if !argset.contains(v.ID) {
					continue
				}

				// This stack-based phi is the argument of some other
				// phi in this block.  We must make a copy of its
				// value so that we don't clobber it prematurely.
				c := s.allocValToReg(v, s.compatRegs(v), false)
				d := p.NewValue1(v.Line, OpStoreReg, v.Type, c)
				s.setOrig(d, v)
				s.values[v.ID].spill2 = d
			}

			// Assign to stack-based phis.  We do stack phis first because
			// we might need a register to do the assignment.
			for _, v := range stackPhis {
				// Load phi arg into a register, then store it with a StoreReg.
				// If already in a register, use that.  If not, pick a compatible
				// register.
				w := v.Args[i]
				c := s.allocValToReg(w, s.compatRegs(w), false)
				v.Args[i] = p.NewValue1(v.Line, OpStoreReg, v.Type, c)
				s.setOrig(v.Args[i], w)
			}
			// Figure out what value goes in each register.
			for r := register(0); r < numRegs; r++ {
				dst[r] = startRegs[b.ID][r]
			}
			// Handle register-based phi ops.
			for _, v := range regPhis {
				r := s.getHome(v)
				if dst[r] != v {
					f.Fatalf("dst not right")
				}
				v.Args[i] = s.allocValToReg(v.Args[i], regMask(1)<<r, false)
				dst[r] = nil // we've handled this one
			}
			// Move other non-phi register values to the right register.
			for r := register(0); r < numRegs; r++ {
				if dst[r] == nil {
					continue
				}
				if s.regs[r].v == dst[r] {
					continue
				}
				mv := s.allocValToReg(dst[r], regMask(1)<<r, false)
				// TODO: ssa form is probably violated by this step.
				// I don't know how to splice in the new value because
				// I need to potentially make a phi and replace all uses.
				_ = mv
			}
			// Reset spill2 fields
			for _, v := range stackPhis {
				spill2 := s.values[v.ID].spill2
				if spill2 == nil {
					continue
				}
				if !s.values[v.ID].spill2used {
					spill2.Op = OpInvalid
					spill2.Type = TypeInvalid
					spill2.resetArgs()
				} else if logSpills {
					fmt.Println("regalloc: spilled phi")
				}
				s.values[v.ID].spill2 = nil
				s.values[v.ID].spill2used = false
			}
		}
	}
	// TODO: be smarter about the order in which to shuffle registers around.
	// if we need to do AX->CX and CX->DX, do the latter first.  Now if we do the
	// former first then the latter must be a restore instead of a register move.

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
		spill.Op = OpInvalid
		spill.Type = TypeInvalid
		spill.resetArgs()
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

	// Set final regalloc result.
	f.RegAlloc = s.home
}

func (v *Value) rematerializeable() bool {
	// TODO: add a flags field to opInfo for this test?

	// rematerializeable ops must be able to fill any register.
	outputs := opcodeTable[v.Op].reg.outputs
	if len(outputs) == 0 || countRegs(outputs[0]) <= 1 {
		// Note: this case handles OpAMD64LoweredGetClosurePtr
		// which can't be moved.
		return false
	}
	if len(v.Args) == 0 {
		return true
	}
	if len(v.Args) == 1 && (v.Args[0].Op == OpSP || v.Args[0].Op == OpSB) {
		return true
	}
	return false
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
		for _, b := range po {
			f.Logf("live %s %v\n", b, s.live[b.ID])
		}
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
			if b.Control != nil && s.needReg[b.Control.ID] {
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
					if s.needReg[a.ID] {
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
					if s.needReg[id] && !t.contains(id) || delta < t.get(id) {
						update = true
						t.set(id, delta)
					}
				}

				if !update {
					continue
				}
				// The live set has changed, update it.
				l := s.live[p.ID][:0]
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
