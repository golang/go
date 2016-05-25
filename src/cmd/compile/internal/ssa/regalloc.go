// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Register allocation.
//
// We use a version of a linear scan register allocator. We treat the
// whole function as a single long basic block and run through
// it using a greedy register allocator. Then all merge edges
// (those targeting a block with len(Preds)>1) are processed to
// shuffle data into the place that the target of the edge expects.
//
// The greedy allocator moves values into registers just before they
// are used, spills registers only when necessary, and spills the
// value whose next use is farthest in the future.
//
// The register allocator requires that a block is not scheduled until
// at least one of its predecessors have been scheduled. The most recent
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
// While AX still holds x, any uses of x will use that value. When AX is needed
// for another value, we simply reuse AX.  Spill code has already been generated
// so there is no code generated at "spill" time. When x is referenced
// subsequently, we issue a load to restore x to a register using x2 as
//  its argument:
//    x3 = Restore x2 : CX
// x3 can then be used wherever x is referenced again.
// If the spill (x2) is never used, it will be removed at the end of regalloc.
//
// Phi values are special, as always. We define two kinds of phis, those
// where the merge happens in a register (a "register" phi) and those where
// the merge happens in a stack location (a "stack" phi).
//
// A register phi must have the phi and all of its inputs allocated to the
// same register. Register phis are spilled similarly to regular ops:
//     b1: y = ... : AX        b2: z = ... : AX
//         goto b3                 goto b3
//     b3: x = phi(y, z) : AX
//         x2 = StoreReg x
//
// A stack phi must have the phi and all of its inputs allocated to the same
// stack location. Stack phis start out life already spilled - each phi
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
// same register. This affinity graph will be used to prefer certain
// registers for allocation. This affinity helps eliminate moves that
// are required for phi implementations and helps generate allocations
// for 2-register architectures.

// Note: regalloc generates a not-quite-SSA output. If we have:
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
// SSA form. For now, we ignore this problem as remaining in strict
// SSA form isn't needed after regalloc. We'll just leave the use
// of x3 not dominated by the definition of x3, and the CX->BX copy
// will have no use (so don't run deadcode after regalloc!).
// TODO: maybe we should introduce these extra phis?

// Additional not-quite-SSA output occurs when spills are sunk out
// of loops to the targets of exit edges from the loop.  Before sinking,
// there is one spill site (one StoreReg) targeting stack slot X, after
// sinking there may be multiple spill sites targeting stack slot X,
// with no phi functions at any join points reachable by the multiple
// spill sites.  In addition, uses of the spill from copies of the original
// will not name the copy in their reference; instead they will name
// the original, though both will have the same spill location.  The
// first sunk spill will be the original, but moved, to an exit block,
// thus ensuring that there is a definition somewhere corresponding to
// the original spill's uses.

package ssa

import (
	"fmt"
	"unsafe"
)

const (
	moveSpills = iota
	logSpills
	regDebug
	stackDebug
)

// distance is a measure of how far into the future values are used.
// distance is measured in units of instructions.
const (
	likelyDistance   = 1
	normalDistance   = 10
	unlikelyDistance = 100
)

// regalloc performs register allocation on f. It sets f.RegAlloc
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
	for r := register(0); m != 0; r++ {
		if m>>r&1 == 0 {
			continue
		}
		m &^= regMask(1) << r
		if s != "" {
			s += " "
		}
		s += fmt.Sprintf("r%d", r)
	}
	return s
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
	spillUsedShuffle  bool // true if used in shuffling, after ordinary uses
	needReg           bool // cached value of !v.Type.IsMemory() && !v.Type.IsVoid() && !.v.Type.IsFlags()
	rematerializeable bool // cached value of v.rematerializeable()
}

type regState struct {
	v *Value // Original (preregalloc) Value stored in this register.
	c *Value // A Value equal to v which is currently in a register.  Might be v or a copy of it.
	// If a register is unused, v==c==nil
}

type regAllocState struct {
	f *Func

	registers   []Register
	numRegs     register
	SPReg       register
	SBReg       register
	allocatable regMask

	// for each block, its primary predecessor.
	// A predecessor of b is primary if it is the closest
	// predecessor that appears before b in the layout order.
	// We record the index in the Preds list where the primary predecessor sits.
	primary []int32

	// live values at the end of each block.  live[b.ID] is a list of value IDs
	// which are live at the end of b, together with a count of how many instructions
	// forward to the next use.
	live [][]liveInfo
	// desired register assignments at the end of each block.
	// Note that this is a static map computed before allocation occurs. Dynamic
	// register desires (from partially completed allocations) will trump
	// this information.
	desired []desiredState

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

	loopnest *loopnest
}

type spillToSink struct {
	spill *Value // Spill instruction to move (a StoreReg)
	dests int32  // Bitmask indicating exit blocks from loop in which spill/val is defined. 1<<i set means val is live into loop.exitBlocks[i]
}

func (sts *spillToSink) spilledValue() *Value {
	return sts.spill.Args[0]
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

// freeReg frees up register r. Any current user of r is kicked out.
func (s *regAllocState) freeReg(r register) {
	v := s.regs[r].v
	if v == nil {
		s.f.Fatalf("tried to free an already free register %d\n", r)
	}

	// Mark r as unused.
	if s.f.pass.debug > regDebug {
		fmt.Printf("freeReg %s (dump %s/%s)\n", s.registers[r].Name(), v, s.regs[r].c)
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
	if s.f.pass.debug > regDebug {
		fmt.Printf("assignReg %s %s/%s\n", s.registers[r].Name(), v, c)
	}
	if s.regs[r].v != nil {
		s.f.Fatalf("tried to assign register %d to %s/%s but it is already used by %s", r, v, c, s.regs[r].v)
	}

	// Update state.
	s.regs[r] = regState{v, c}
	s.values[v.ID].regs |= regMask(1) << r
	s.used |= regMask(1) << r
	s.f.setHome(c, &s.registers[r])
}

// allocReg chooses a register for v from the set of registers in mask.
// If there is no unused register, a Value will be kicked out of
// a register to make room.
func (s *regAllocState) allocReg(v *Value, mask regMask) register {
	mask &= s.allocatable
	mask &^= s.nospill
	if mask == 0 {
		s.f.Fatalf("no register available")
	}

	// Pick an unused register if one is available.
	if mask&^s.used != 0 {
		return pickReg(mask &^ s.used)
	}

	// Pick a value to spill. Spill the value with the
	// farthest-in-the-future use.
	// TODO: Prefer registers with already spilled Values?
	// TODO: Modify preference using affinity graph.
	// TODO: if a single value is in multiple registers, spill one of them
	// before spilling a value in just a single register.

	// Find a register to spill. We spill the register containing the value
	// whose next use is as far in the future as possible.
	// https://en.wikipedia.org/wiki/Page_replacement_algorithm#The_theoretically_optimal_page_replacement_algorithm
	var r register
	maxuse := int32(-1)
	for t := register(0); t < s.numRegs; t++ {
		if mask>>t&1 == 0 {
			continue
		}
		v := s.regs[t].v
		if n := s.values[v.ID].uses.dist; n > maxuse {
			// v's next use is farther in the future than any value
			// we've seen so far. A new best spill candidate.
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
			if s.f.pass.debug > logSpills {
				s.f.Config.Warnl(vi.spill.Line, "load spill for %v from %v", v, vi.spill)
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
	s.f = f
	s.registers = f.Config.registers
	s.numRegs = register(len(s.registers))
	if s.numRegs > noRegister || s.numRegs > register(unsafe.Sizeof(regMask(0))*8) {
		panic("too many registers")
	}
	for r := register(0); r < s.numRegs; r++ {
		if s.registers[r].Name() == "SP" {
			s.SPReg = r
		}
		if s.registers[r].Name() == "SB" {
			s.SBReg = r
		}
	}

	// Figure out which registers we're allowed to use.
	s.allocatable = regMask(1)<<s.numRegs - 1
	s.allocatable &^= 1 << s.SPReg
	s.allocatable &^= 1 << s.SBReg
	if s.f.Config.ctxt.Framepointer_enabled {
		s.allocatable &^= 1 << 5 // BP
	}
	if s.f.Config.ctxt.Flag_dynlink {
		s.allocatable &^= 1 << 15 // R15
	}

	s.regs = make([]regState, s.numRegs)
	s.values = make([]valState, f.NumValues())
	s.orig = make([]*Value, f.NumValues())
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if !v.Type.IsMemory() && !v.Type.IsVoid() && !v.Type.IsFlags() {
				s.values[v.ID].needReg = true
				s.values[v.ID].rematerializeable = v.rematerializeable()
				s.orig[v.ID] = v
			}
		}
	}
	s.computeLive()

	// Compute block order. This array allows us to distinguish forward edges
	// from backward edges and compute how far they go.
	blockOrder := make([]int32, f.NumBlocks())
	for i, b := range f.Blocks {
		blockOrder[b.ID] = int32(i)
	}

	// Compute primary predecessors.
	s.primary = make([]int32, f.NumBlocks())
	for _, b := range f.Blocks {
		best := -1
		for i, e := range b.Preds {
			p := e.b
			if blockOrder[p.ID] >= blockOrder[b.ID] {
				continue // backward edge
			}
			if best == -1 || blockOrder[p.ID] > blockOrder[b.Preds[best].b.ID] {
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

// liveAfterCurrentInstruction reports whether v is live after
// the current instruction is completed.  v must be used by the
// current instruction.
func (s *regAllocState) liveAfterCurrentInstruction(v *Value) bool {
	u := s.values[v.ID].uses
	d := u.dist
	for u != nil && u.dist == d {
		u = u.next
	}
	return u != nil && u.dist > d
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
	if t.IsFloat() || t == TypeInt128 {
		m = 0xffff << 16 // X0-X15
	} else {
		m = 0xffff << 0 // AX-R15
	}
	return m & s.allocatable
}

// loopForBlock returns the loop containing block b,
// provided that the loop is "interesting" for purposes
// of improving register allocation (= is inner, and does
// not contain a call)
func (s *regAllocState) loopForBlock(b *Block) *loop {
	loop := s.loopnest.b2l[b.ID]

	// Minor for-the-time-being optimization: nothing happens
	// unless a loop is both inner and call-free, therefore
	// don't bother with other loops.
	if loop != nil && (loop.containsCall || !loop.isInner) {
		loop = nil
	}
	return loop
}

func (s *regAllocState) regalloc(f *Func) {
	liveSet := f.newSparseSet(f.NumValues())
	defer f.retSparseSet(liveSet)
	var oldSched []*Value
	var phis []*Value
	var phiRegs []register
	var args []*Value

	// statistics
	var nSpills int               // # of spills remaining
	var nSpillsInner int          // # of spills remaining in inner loops
	var nSpillsSunk int           // # of sunk spills remaining
	var nSpillsChanged int        // # of sunk spills lost because of register use change
	var nSpillsSunkUnused int     // # of spills not sunk because they were removed completely
	var nSpillsNotSunkLateUse int // # of spills not sunk because of very late use (in shuffle)

	// Data structure used for computing desired registers.
	var desired desiredState

	// Desired registers for inputs & outputs for each instruction in the block.
	type dentry struct {
		out [4]register    // desired output registers
		in  [3][4]register // desired input registers (for inputs 0,1, and 2)
	}
	var dinfo []dentry

	if f.Entry != f.Blocks[0] {
		f.Fatalf("entry block must be first")
	}

	// Get loop nest so that spills in inner loops can be
	// tracked.  When the last block of a loop is processed,
	// attempt to move spills out of the loop.
	s.loopnest.findExits()

	// Spills are moved from one block's slice of values to another's.
	// This confuses register allocation if it occurs before it is
	// complete, so candidates are recorded, then rechecked and
	// moved after all allocation (register and stack) is complete.
	// Because movement is only within a stack slot's lifetime, it
	// is safe to do this.
	var toSink []spillToSink
	// Will be used to figure out live inputs to exit blocks of inner loops.
	entryCandidates := newSparseMap(f.NumValues())

	for _, b := range f.Blocks {
		s.curBlock = b
		loop := s.loopForBlock(b)

		// Initialize liveSet and uses fields for this block.
		// Walk backwards through the block doing liveness analysis.
		liveSet.clear()
		d := int32(len(b.Values))
		if b.Kind == BlockCall || b.Kind == BlockDefer {
			d += unlikelyDistance
		}
		for _, e := range s.live[b.ID] {
			s.addUse(e.ID, d+e.dist) // pseudo-uses from beyond end of block
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
				// any inputs. This is the state the len(b.Preds)>1
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
		if s.f.pass.debug > regDebug {
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
			s.setState(s.endRegs[b.Preds[0].b.ID])
			if nphi > 0 {
				f.Fatalf("phis in single-predecessor block")
			}
			// Drop any values which are no longer live.
			// This may happen because at the end of p, a value may be
			// live but only used by some other successor of p.
			for r := register(0); r < s.numRegs; r++ {
				v := s.regs[r].v
				if v != nil && !liveSet.contains(v.ID) {
					s.freeReg(r)
				}
			}
		} else {
			// This is the complicated case. We have more than one predecessor,
			// which means we may have Phi ops.

			// Copy phi ops into new schedule.
			b.Values = append(b.Values, phis...)

			// Start with the final register state of the primary predecessor
			idx := s.primary[b.ID]
			if idx < 0 {
				f.Fatalf("block with no primary predecessor %s", b)
			}
			p := b.Preds[idx].b
			s.setState(s.endRegs[p.ID])

			if s.f.pass.debug > regDebug {
				fmt.Printf("starting merge block %s with end state of %s:\n", b, p)
				for _, x := range s.endRegs[p.ID] {
					fmt.Printf("  %s: orig:%s cache:%s\n", s.registers[x.r].Name(), x.v, x.c)
				}
			}

			// Decide on registers for phi ops. Use the registers determined
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
				if m != 0 {
					r := pickReg(m)
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

			// Set registers for phis. Add phi spill code.
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
				if loop != nil {
					loop.spills = append(loop.spills, v)
					nSpillsInner++
				}
				nSpills++
			}

			// Save the starting state for use by merge edges.
			var regList []startReg
			for r := register(0); r < s.numRegs; r++ {
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

			if s.f.pass.debug > regDebug {
				fmt.Printf("after phis\n")
				for _, x := range s.startRegs[b.ID] {
					fmt.Printf("  %s: v%d\n", s.registers[x.r].Name(), x.vid)
				}
			}
		}

		// Allocate space to record the desired registers for each value.
		dinfo = dinfo[:0]
		for i := 0; i < len(oldSched); i++ {
			dinfo = append(dinfo, dentry{})
		}

		// Load static desired register info at the end of the block.
		desired.copy(&s.desired[b.ID])

		// Check actual assigned registers at the start of the next block(s).
		// Dynamically assigned registers will trump the static
		// desired registers computed during liveness analysis.
		// Note that we do this phase after startRegs is set above, so that
		// we get the right behavior for a block which branches to itself.
		for _, e := range b.Succs {
			succ := e.b
			// TODO: prioritize likely successor?
			for _, x := range s.startRegs[succ.ID] {
				desired.add(x.vid, x.r)
			}
			// Process phi ops in succ.
			pidx := e.i
			for _, v := range succ.Values {
				if v.Op != OpPhi {
					break
				}
				if !s.values[v.ID].needReg {
					continue
				}
				rp, ok := s.f.getHome(v.ID).(*Register)
				if !ok {
					continue
				}
				desired.add(v.Args[pidx].ID, register(rp.Num))
			}
		}
		// Walk values backwards computing desired register info.
		// See computeLive for more comments.
		for i := len(oldSched) - 1; i >= 0; i-- {
			v := oldSched[i]
			prefs := desired.remove(v.ID)
			desired.clobber(opcodeTable[v.Op].reg.clobbers)
			for _, j := range opcodeTable[v.Op].reg.inputs {
				if countRegs(j.regs) != 1 {
					continue
				}
				desired.clobber(j.regs)
				desired.add(v.Args[j.idx].ID, pickReg(j.regs))
			}
			if opcodeTable[v.Op].resultInArg0 {
				if opcodeTable[v.Op].commutative {
					desired.addList(v.Args[1].ID, prefs)
				}
				desired.addList(v.Args[0].ID, prefs)
			}
			// Save desired registers for this value.
			dinfo[i].out = prefs
			for j, a := range v.Args {
				if j >= len(dinfo[i].in) {
					break
				}
				dinfo[i].in[j] = desired.get(a.ID)
			}
		}

		// Process all the non-phi values.
		for idx, v := range oldSched {
			if s.f.pass.debug > regDebug {
				fmt.Printf("  processing %s\n", v.LongString())
			}
			if v.Op == OpPhi {
				f.Fatalf("phi %s not at start of block", v)
			}
			if v.Op == OpSP {
				s.assignReg(s.SPReg, v, v)
				b.Values = append(b.Values, v)
				s.advanceUses(v)
				continue
			}
			if v.Op == OpSB {
				s.assignReg(s.SBReg, v, v)
				b.Values = append(b.Values, v)
				s.advanceUses(v)
				continue
			}
			if v.Op == OpArg {
				// Args are "pre-spilled" values. We don't allocate
				// any register here. We just set up the spill pointer to
				// point at itself and any later user will restore it to use it.
				s.values[v.ID].spill = v
				s.values[v.ID].spillUsed = true // use is guaranteed
				b.Values = append(b.Values, v)
				s.advanceUses(v)
				continue
			}
			if v.Op == OpKeepAlive {
				// Make sure the argument to v is still live here.
				s.advanceUses(v)
				vi := &s.values[v.Args[0].ID]
				if vi.spillUsed {
					// Use the spill location.
					v.SetArg(0, vi.spill)
				} else {
					// No need to keep unspilled values live.
					// These are typically rematerializeable constants like nil,
					// or values of a variable that were modified since the last call.
					v.Op = OpCopy
					v.SetArgs1(v.Args[1])
				}
				b.Values = append(b.Values, v)
				continue
			}
			regspec := opcodeTable[v.Op].reg
			if len(regspec.inputs) == 0 && len(regspec.outputs) == 0 {
				// No register allocation required (or none specified yet)
				s.freeRegs(regspec.clobbers)
				b.Values = append(b.Values, v)
				s.advanceUses(v)
				continue
			}

			if s.values[v.ID].rematerializeable {
				// Value is rematerializeable, don't issue it here.
				// It will get issued just before each use (see
				// allocValueToReg).
				for _, a := range v.Args {
					a.Uses--
				}
				s.advanceUses(v)
				continue
			}

			if s.f.pass.debug > regDebug {
				fmt.Printf("value %s\n", v.LongString())
				fmt.Printf("  out:")
				for _, r := range dinfo[idx].out {
					if r != noRegister {
						fmt.Printf(" %s", s.registers[r].Name())
					}
				}
				fmt.Println()
				for i := 0; i < len(v.Args) && i < 3; i++ {
					fmt.Printf("  in%d:", i)
					for _, r := range dinfo[idx].in[i] {
						if r != noRegister {
							fmt.Printf(" %s", s.registers[r].Name())
						}
					}
					fmt.Println()
				}
			}

			// Move arguments to registers. Process in an ordering defined
			// by the register specification (most constrained first).
			args = append(args[:0], v.Args...)
			for _, i := range regspec.inputs {
				mask := i.regs
				if mask == flagRegMask {
					// TODO: remove flag input from regspec.inputs.
					continue
				}
				if mask&s.values[args[i.idx].ID].regs == 0 {
					// Need a new register for the input.
					mask &= s.allocatable
					mask &^= s.nospill
					// Used desired register if available.
					if i.idx < 3 {
						for _, r := range dinfo[idx].in[i.idx] {
							if r != noRegister && (mask&^s.used)>>r&1 != 0 {
								// Desired register is allowed and unused.
								mask = regMask(1) << r
								break
							}
						}
					}
					// Avoid registers we're saving for other values.
					if mask&^desired.avoid != 0 {
						mask &^= desired.avoid
					}
				}
				args[i.idx] = s.allocValToReg(args[i.idx], mask, true, v.Line)
			}

			// If the output clobbers the input register, make sure we have
			// at least two copies of the input register so we don't
			// have to reload the value from the spill location.
			if opcodeTable[v.Op].resultInArg0 {
				var m regMask
				if !s.liveAfterCurrentInstruction(v.Args[0]) {
					// arg0 is dead.  We can clobber its register.
					goto ok
				}
				if countRegs(s.values[v.Args[0].ID].regs) >= 2 {
					// we have at least 2 copies of arg0.  We can afford to clobber one.
					goto ok
				}
				if opcodeTable[v.Op].commutative {
					if !s.liveAfterCurrentInstruction(v.Args[1]) {
						args[0], args[1] = args[1], args[0]
						goto ok
					}
					if countRegs(s.values[v.Args[1].ID].regs) >= 2 {
						args[0], args[1] = args[1], args[0]
						goto ok
					}
				}

				// We can't overwrite arg0 (or arg1, if commutative).  So we
				// need to make a copy of an input so we have a register we can modify.

				// Possible new registers to copy into.
				m = s.compatRegs(v.Args[0].Type) &^ s.used
				if m == 0 {
					// No free registers.  In this case we'll just clobber
					// an input and future uses of that input must use a restore.
					// TODO(khr): We should really do this like allocReg does it,
					// spilling the value with the most distant next use.
					goto ok
				}

				// Try to move an input to the desired output.
				for _, r := range dinfo[idx].out {
					if r != noRegister && m>>r&1 != 0 {
						m = regMask(1) << r
						args[0] = s.allocValToReg(v.Args[0], m, true, v.Line)
						// Note: we update args[0] so the instruction will
						// use the register copy we just made.
						goto ok
					}
				}
				// Try to copy input to its desired location & use its old
				// location as the result register.
				for _, r := range dinfo[idx].in[0] {
					if r != noRegister && m>>r&1 != 0 {
						m = regMask(1) << r
						s.allocValToReg(v.Args[0], m, true, v.Line)
						// Note: no update to args[0] so the instruction will
						// use the original copy.
						goto ok
					}
				}
				if opcodeTable[v.Op].commutative {
					for _, r := range dinfo[idx].in[1] {
						if r != noRegister && m>>r&1 != 0 {
							m = regMask(1) << r
							s.allocValToReg(v.Args[1], m, true, v.Line)
							args[0], args[1] = args[1], args[0]
							goto ok
						}
					}
				}
				// Avoid future fixed uses if we can.
				if m&^desired.avoid != 0 {
					m &^= desired.avoid
				}
				// Save input 0 to a new register so we can clobber it.
				s.allocValToReg(v.Args[0], m, true, v.Line)
			ok:
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
			if s.values[v.ID].needReg {
				mask := regspec.outputs[0] & s.allocatable
				if opcodeTable[v.Op].resultInArg0 {
					if !opcodeTable[v.Op].commutative {
						// Output must use the same register as input 0.
						r := register(s.f.getHome(args[0].ID).(*Register).Num)
						mask = regMask(1) << r
					} else {
						// Output must use the same register as input 0 or 1.
						r0 := register(s.f.getHome(args[0].ID).(*Register).Num)
						r1 := register(s.f.getHome(args[1].ID).(*Register).Num)
						// Check r0 and r1 for desired output register.
						found := false
						for _, r := range dinfo[idx].out {
							if (r == r0 || r == r1) && (mask&^s.used)>>r&1 != 0 {
								mask = regMask(1) << r
								found = true
								if r == r1 {
									args[0], args[1] = args[1], args[0]
								}
								break
							}
						}
						if !found {
							// Neither are desired, pick r0.
							mask = regMask(1) << r0
						}
					}
				}
				for _, r := range dinfo[idx].out {
					if r != noRegister && (mask&^s.used)>>r&1 != 0 {
						// Desired register is allowed and unused.
						mask = regMask(1) << r
						break
					}
				}
				// Avoid registers we're saving for other values.
				if mask&^desired.avoid != 0 {
					mask &^= desired.avoid
				}
				r := s.allocReg(v, mask)
				s.assignReg(r, v, v)
			}

			// Issue the Value itself.
			for i, a := range args {
				v.SetArg(i, a) // use register version of arguments
			}
			b.Values = append(b.Values, v)

			// Issue a spill for this value. We issue spills unconditionally,
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
				if loop != nil {
					loop.spills = append(loop.spills, v)
					nSpillsInner++
				}
				nSpills++
			}
		}

		// Load control value into reg.
		if v := b.Control; v != nil && s.values[v.ID].needReg {
			if s.f.pass.debug > regDebug {
				fmt.Printf("  processing control %s\n", v.LongString())
			}
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

		// If we are approaching a merge point and we are the primary
		// predecessor of it, find live values that we use soon after
		// the merge point and promote them to registers now.
		if len(b.Succs) == 1 {
			// For this to be worthwhile, the loop must have no calls in it.
			top := b.Succs[0].b
			loop := s.loopnest.b2l[top.ID]
			if loop == nil || loop.header != top || loop.containsCall {
				goto badloop
			}

			// TODO: sort by distance, pick the closest ones?
			for _, live := range s.live[b.ID] {
				if live.dist >= unlikelyDistance {
					// Don't preload anything live after the loop.
					continue
				}
				vid := live.ID
				vi := &s.values[vid]
				if vi.regs != 0 {
					continue
				}
				v := s.orig[vid]
				m := s.compatRegs(v.Type) &^ s.used
				if m&^desired.avoid != 0 {
					m &^= desired.avoid
				}
				if m != 0 {
					s.allocValToReg(v, m, false, b.Line)
				}
			}
		}
	badloop:
		;

		// Save end-of-block register state.
		// First count how many, this cuts allocations in half.
		k := 0
		for r := register(0); r < s.numRegs; r++ {
			v := s.regs[r].v
			if v == nil {
				continue
			}
			k++
		}
		regList := make([]endReg, 0, k)
		for r := register(0); r < s.numRegs; r++ {
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
			for r := register(0); r < s.numRegs; r++ {
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
		// is live. We need to remember this information so that
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

		// Keep track of values that are spilled in the loop, but whose spill
		// is not used in the loop.  It may be possible to move ("sink") the
		// spill out of the loop into one or more exit blocks.
		if loop != nil {
			loop.scratch++                    // increment count of blocks in this loop that have been processed
			if loop.scratch == loop.nBlocks { // just processed last block of loop, if it is an inner loop.
				// This check is redundant with code at the top of the loop.
				// This is definitive; the one at the top of the loop is an optimization.
				if loop.isInner && // Common case, easier, most likely to be profitable
					!loop.containsCall && // Calls force spills, also lead to puzzling spill info.
					len(loop.exits) <= 32 { // Almost no inner loops have more than 32 exits,
					// and this allows use of a bitvector and a sparseMap.

					// TODO: exit calculation is messed up for non-inner loops
					// because of multilevel exits that are not part of the "exit"
					// count.

					// Compute the set of spill-movement candidates live at entry to exit blocks.
					// isLoopSpillCandidate filters for
					// (1) defined in appropriate loop
					// (2) needs a register
					// (3) spill not already used (in the loop)
					// Condition (3) === "in a register at all loop exits"

					entryCandidates.clear()

					for whichExit, ss := range loop.exits {
						// Start with live at end.
						for _, li := range s.live[ss.ID] {
							if s.isLoopSpillCandidate(loop, s.orig[li.ID]) {
								// s.live contains original IDs, use s.orig above to map back to *Value
								entryCandidates.setBit(li.ID, uint(whichExit))
							}
						}
						// Control can also be live.
						if ss.Control != nil && s.orig[ss.Control.ID] != nil && s.isLoopSpillCandidate(loop, s.orig[ss.Control.ID]) {
							entryCandidates.setBit(s.orig[ss.Control.ID].ID, uint(whichExit))
						}
						// Walk backwards, filling in locally live values, removing those defined.
						for i := len(ss.Values) - 1; i >= 0; i-- {
							v := ss.Values[i]
							vorig := s.orig[v.ID]
							if vorig != nil {
								entryCandidates.remove(vorig.ID) // Cannot be an issue, only keeps the sets smaller.
							}
							for _, a := range v.Args {
								aorig := s.orig[a.ID]
								if aorig != nil && s.isLoopSpillCandidate(loop, aorig) {
									entryCandidates.setBit(aorig.ID, uint(whichExit))
								}
							}
						}
					}

					for _, e := range loop.spills {
						whichblocks := entryCandidates.get(e.ID)
						oldSpill := s.values[e.ID].spill
						if whichblocks != 0 && whichblocks != -1 { // -1 = not in map.
							toSink = append(toSink, spillToSink{spill: oldSpill, dests: whichblocks})
						}
					}

				} // loop is inner etc
				loop.scratch = 0 // Don't leave a mess, just in case.
				loop.spills = nil
			} // if scratch == nBlocks
		} // if loop is not nil

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
			if s.f.pass.debug > logSpills {
				s.f.Config.Warnl(vi.spill.Line, "spilled value at %v remains", vi.spill)
			}
			continue
		}
		spill := vi.spill
		if spill == nil {
			// Constants, SP, SB, ...
			continue
		}
		loop := s.loopForBlock(spill.Block)
		if loop != nil {
			nSpillsInner--
		}

		spill.Args[0].Uses--
		f.freeValue(spill)
		nSpills--
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

	// Must clear these out before any potential recycling, though that's
	// not currently implemented.
	for i, ts := range toSink {
		vsp := ts.spill
		if vsp.Op == OpInvalid { // This spill was completely eliminated
			toSink[i].spill = nil
		}
	}

	// Anything that didn't get a register gets a stack location here.
	// (StoreReg, stack-based phis, inputs, ...)
	stacklive := stackalloc(s.f, s.spillLive)

	// Fix up all merge edges.
	s.shuffle(stacklive)

	// Insert moved spills (that have not been marked invalid above)
	// at start of appropriate block and remove the originals from their
	// location within loops.  Notice that this can break SSA form;
	// if a spill is sunk to multiple exits, there will be no phi for that
	// spill at a join point downstream of those two exits, though the
	// two spills will target the same stack slot.  Notice also that this
	// takes place after stack allocation, so the stack allocator does
	// not need to process these malformed flow graphs.
sinking:
	for _, ts := range toSink {
		vsp := ts.spill
		if vsp == nil { // This spill was completely eliminated
			nSpillsSunkUnused++
			continue sinking
		}
		e := ts.spilledValue()
		if s.values[e.ID].spillUsedShuffle {
			nSpillsNotSunkLateUse++
			continue sinking
		}

		// move spills to a better (outside of loop) block.
		// This would be costly if it occurred very often, but it doesn't.
		b := vsp.Block
		loop := s.loopnest.b2l[b.ID]
		dests := ts.dests

		// Pre-check to be sure that spilled value is still in expected register on all exits where live.
	check_val_still_in_reg:
		for i := uint(0); i < 32 && dests != 0; i++ {

			if dests&(1<<i) == 0 {
				continue
			}
			dests ^= 1 << i
			d := loop.exits[i]
			if len(d.Preds) > 1 {
				panic("Should be impossible given critical edges removed")
			}
			p := d.Preds[0].b // block in loop exiting to d.

			endregs := s.endRegs[p.ID]
			for _, regrec := range endregs {
				if regrec.v == e && regrec.r != noRegister && regrec.c == e { // TODO: regrec.c != e implies different spill possible.
					continue check_val_still_in_reg
				}
			}
			// If here, the register assignment was lost down at least one exit and it can't be sunk
			if s.f.pass.debug > moveSpills {
				s.f.Config.Warnl(e.Line, "lost register assignment for spill %v in %v at exit %v to %v",
					vsp, b, p, d)
			}
			nSpillsChanged++
			continue sinking
		}

		nSpillsSunk++
		nSpillsInner--
		// don't update nSpills, since spill is only moved, and if it is duplicated, the spills-on-a-path is not increased.

		dests = ts.dests

		// remove vsp from b.Values
		i := 0
		for _, w := range b.Values {
			if vsp == w {
				continue
			}
			b.Values[i] = w
			i++
		}
		b.Values = b.Values[:i]

		first := true
		for i := uint(0); i < 32 && dests != 0; i++ {

			if dests&(1<<i) == 0 {
				continue
			}

			dests ^= 1 << i

			d := loop.exits[i]
			vspnew := vsp // reuse original for first sunk spill, saves tracking down and renaming uses
			if !first {   // any sunk spills after first must make a copy
				vspnew = d.NewValue1(e.Line, OpStoreReg, e.Type, e)
				f.setHome(vspnew, f.getHome(vsp.ID)) // copy stack home
				if s.f.pass.debug > moveSpills {
					s.f.Config.Warnl(e.Line, "copied spill %v in %v for %v to %v in %v",
						vsp, b, e, vspnew, d)
				}
			} else {
				first = false
				vspnew.Block = d
				d.Values = append(d.Values, vspnew)
				if s.f.pass.debug > moveSpills {
					s.f.Config.Warnl(e.Line, "moved spill %v in %v for %v to %v in %v",
						vsp, b, e, vspnew, d)
				}
			}

			// shuffle vspnew to the beginning of its block
			copy(d.Values[1:], d.Values[0:len(d.Values)-1])
			d.Values[0] = vspnew

		}
	}

	if f.pass.stats > 0 {
		f.LogStat("spills_info",
			nSpills, "spills", nSpillsInner, "inner_spills_remaining", nSpillsSunk, "inner_spills_sunk", nSpillsSunkUnused, "inner_spills_unused", nSpillsNotSunkLateUse, "inner_spills_shuffled", nSpillsChanged, "inner_spills_changed")
	}
}

// isLoopSpillCandidate indicates whether the spill for v satisfies preliminary
// spill-sinking conditions just after the last block of loop has been processed.
// In particular:
//   v needs a register.
//   v's spill is not (YET) used.
//   v's definition is within loop.
// The spill may be used in the future, either by an outright use
// in the code, or by shuffling code inserted after stack allocation.
// Outright uses cause sinking; shuffling (within the loop) inhibits it.
func (s *regAllocState) isLoopSpillCandidate(loop *loop, v *Value) bool {
	return s.values[v.ID].needReg && !s.values[v.ID].spillUsed && s.loopnest.b2l[v.Block.ID] == loop
}

// lateSpillUse notes a late (after stack allocation) use of the spill of value with ID vid.
// This will inhibit spill sinking.
func (s *regAllocState) lateSpillUse(vid ID) {
	// TODO investigate why this is necessary.
	// It appears that an outside-the-loop use of
	// an otherwise sinkable spill makes the spill
	// a candidate for shuffling, when it would not
	// otherwise have been the case (spillUsed was not
	// true when isLoopSpillCandidate was called, yet
	// it was shuffled).  Such shuffling cuts the amount
	// of spill sinking by more than half (in make.bash)
	s.values[vid].spillUsedShuffle = true
}

// shuffle fixes up all the merge edges (those going into blocks of indegree > 1).
func (s *regAllocState) shuffle(stacklive [][]ID) {
	var e edgeState
	e.s = s
	e.cache = map[ID][]*Value{}
	e.contents = map[Location]contentRecord{}
	if s.f.pass.debug > regDebug {
		fmt.Printf("shuffle %s\n", s.f.Name)
		fmt.Println(s.f.String())
	}

	for _, b := range s.f.Blocks {
		if len(b.Preds) <= 1 {
			continue
		}
		e.b = b
		for i, edge := range b.Preds {
			p := edge.b
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
	cache      map[ID][]*Value
	cachedVals []ID // (superset of) keys of the above map, for deterministic iteration

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
	if e.s.f.pass.debug > regDebug {
		fmt.Printf("edge %s->%s\n", e.p, e.b)
	}

	// Clear state.
	for _, vid := range e.cachedVals {
		delete(e.cache, vid)
	}
	e.cachedVals = e.cachedVals[:0]
	for k := range e.contents {
		delete(e.contents, k)
	}
	e.usedRegs = 0
	e.uniqueRegs = 0
	e.finalRegs = 0

	// Live registers can be sources.
	for _, x := range srcReg {
		e.set(&e.s.registers[x.r], x.v.ID, x.c, false)
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
		dsts = append(dsts, dstRecord{&e.s.registers[x.r], x.vid, nil})
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

	if e.s.f.pass.debug > regDebug {
		for _, vid := range e.cachedVals {
			a := e.cache[vid]
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
			// Made some progress. Go around again.
			dsts = dsts[:i]

			// Append any extras destinations we generated.
			dsts = append(dsts, e.extra...)
			e.extra = e.extra[:0]
			continue
		}

		// We made no progress. That means that any
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

		// Copy any cycle location to a temp register. This duplicates
		// one of the cycle entries, allowing the just duplicated value
		// to be overwritten and the cycle to proceed.
		loc := dsts[0].loc
		vid := e.contents[loc].vid
		c := e.contents[loc].c
		r := e.findRegFor(c.Type)
		if e.s.f.pass.debug > regDebug {
			fmt.Printf("breaking cycle with v%d in %s:%s\n", vid, loc.Name(), c)
		}
		if _, isReg := loc.(*Register); isReg {
			c = e.p.NewValue1(c.Line, OpCopy, c.Type, c)
		} else {
			e.s.lateSpillUse(vid)
			c = e.p.NewValue1(c.Line, OpLoadReg, c.Type, c)
		}
		e.set(r, vid, c, false)
	}
}

// processDest generates code to put value vid into location loc. Returns true
// if progress was made.
func (e *edgeState) processDest(loc Location, vid ID, splice **Value) bool {
	occupant := e.contents[loc]
	if occupant.vid == vid {
		// Value is already in the correct place.
		e.contents[loc] = contentRecord{vid, occupant.c, true}
		if splice != nil {
			(*splice).Uses--
			*splice = occupant.c
			occupant.c.Uses++
		}
		// Note: if splice==nil then c will appear dead. This is
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
	if e.s.f.pass.debug > regDebug {
		fmt.Printf("moving v%d to %s\n", vid, loc.Name())
		fmt.Printf("sources of v%d:", vid)
	}
	for _, w := range e.cache[vid] {
		h := e.s.f.getHome(w.ID)
		if e.s.f.pass.debug > regDebug {
			fmt.Printf(" %s:%s", h.Name(), w)
		}
		_, isreg := h.(*Register)
		if src == nil || isreg {
			c = w
			src = h
		}
	}
	if e.s.f.pass.debug > regDebug {
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
			// Rematerialize into stack slot. Need a free
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
				e.s.lateSpillUse(vid)
				x = e.p.NewValue1(c.Line, OpLoadReg, c.Type, c)
			} else {
				// mem->mem. Use temp register.

				// Pre-clobber destination. This avoids the
				// following situation:
				//   - v is currently held in R0 and stacktmp0.
				//   - We want to copy stacktmp1 to stacktmp0.
				//   - We choose R0 as the temporary register.
				// During the copy, both R0 and stacktmp0 are
				// clobbered, losing both copies of v. Oops!
				// Erasing the destination early means R0 will not
				// be chosen as the temp register, as it will then
				// be the last copy of v.
				e.erase(loc)

				r := e.findRegFor(c.Type)
				e.s.lateSpillUse(vid)
				t := e.p.NewValue1(c.Line, OpLoadReg, c.Type, c)
				e.set(r, vid, t, false)
				x = e.p.NewValue1(c.Line, OpStoreReg, loc.(LocalSlot).Type, t)
			}
		}
	}
	e.set(loc, vid, x, true)
	if splice != nil {
		(*splice).Uses--
		*splice = x
		x.Uses++
	}
	return true
}

// set changes the contents of location loc to hold the given value and its cached representative.
func (e *edgeState) set(loc Location, vid ID, c *Value, final bool) {
	e.s.f.setHome(c, loc)
	e.erase(loc)
	e.contents[loc] = contentRecord{vid, c, final}
	a := e.cache[vid]
	if len(a) == 0 {
		e.cachedVals = append(e.cachedVals, vid)
	}
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
	if e.s.f.pass.debug > regDebug {
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
			if e.s.f.pass.debug > regDebug {
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

	// Pick a register. In priority order:
	// 1) an unused register
	// 2) a non-unique register not holding a final value
	// 3) a non-unique register
	x := m &^ e.usedRegs
	if x != 0 {
		return &e.s.registers[pickReg(x)]
	}
	x = m &^ e.uniqueRegs &^ e.finalRegs
	if x != 0 {
		return &e.s.registers[pickReg(x)]
	}
	x = m &^ e.uniqueRegs
	if x != 0 {
		return &e.s.registers[pickReg(x)]
	}

	// No register is available. Allocate a temp location to spill a register to.
	// The type of the slot is immaterial - it will not be live across
	// any safepoint. Just use a type big enough to hold any register.
	typ = e.s.f.Config.fe.TypeInt64()
	t := LocalSlot{e.s.f.Config.fe.Auto(typ), typ, 0}
	// TODO: reuse these slots.

	// Pick a register to spill.
	for _, vid := range e.cachedVals {
		a := e.cache[vid]
		for _, c := range a {
			if r, ok := e.s.f.getHome(c.ID).(*Register); ok && m>>uint(r.Num)&1 != 0 {
				x := e.p.NewValue1(c.Line, OpStoreReg, c.Type, c)
				e.set(t, vid, x, false)
				if e.s.f.pass.debug > regDebug {
					fmt.Printf("  SPILL %s->%s %s\n", r.Name(), t.Name(), x.LongString())
				}
				// r will now be overwritten by the caller. At some point
				// later, the newly saved value will be moved back to its
				// final destination in processDest.
				return r
			}
		}
	}

	fmt.Printf("m:%d unique:%d final:%d\n", m, e.uniqueRegs, e.finalRegs)
	for _, vid := range e.cachedVals {
		a := e.cache[vid]
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
	ID   ID    // ID of value
	dist int32 // # of instructions before next use
}

// dblock contains information about desired & avoid registers at the end of a block.
type dblock struct {
	prefers []desiredStateEntry
	avoid   regMask
}

// computeLive computes a map from block ID to a list of value IDs live at the end
// of that block. Together with the value ID is a count of how many instructions
// to the next use of that value. The resulting map is stored in s.live.
// computeLive also computes the desired register information at the end of each block.
// This desired register information is stored in s.desired.
// TODO: this could be quadratic if lots of variables are live across lots of
// basic blocks. Figure out a way to make this function (or, more precisely, the user
// of this function) require only linear size & time.
func (s *regAllocState) computeLive() {
	f := s.f
	s.live = make([][]liveInfo, f.NumBlocks())
	s.desired = make([]desiredState, f.NumBlocks())
	var phis []*Value

	live := newSparseMap(f.NumValues())
	t := newSparseMap(f.NumValues())

	// Keep track of which value we want in each register.
	var desired desiredState

	// Instead of iterating over f.Blocks, iterate over their postordering.
	// Liveness information flows backward, so starting at the end
	// increases the probability that we will stabilize quickly.
	// TODO: Do a better job yet. Here's one possibility:
	// Calculate the dominator tree and locate all strongly connected components.
	// If a value is live in one block of an SCC, it is live in all.
	// Walk the dominator tree from end to beginning, just once, treating SCC
	// components as single blocks, duplicated calculated liveness information
	// out to all of them.
	s.loopnest = loopnestfor(f)
	po := s.loopnest.po
	for {
		changed := false

		for _, b := range po {
			// Start with known live values at the end of the block.
			// Add len(b.Values) to adjust from end-of-block distance
			// to beginning-of-block distance.
			live.clear()
			d := int32(len(b.Values))
			if b.Kind == BlockCall || b.Kind == BlockDefer {
				// Because we keep no values in registers across a call,
				// make every use past a call appear very far away.
				d += unlikelyDistance
			}
			for _, e := range s.live[b.ID] {
				live.set(e.ID, e.dist+d)
			}

			// Mark control value as live
			if b.Control != nil && s.values[b.Control.ID].needReg {
				live.set(b.Control.ID, int32(len(b.Values)))
			}

			// Propagate backwards to the start of the block
			// Assumes Values have been scheduled.
			phis = phis[:0]
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
			// Propagate desired registers backwards.
			desired.copy(&s.desired[b.ID])
			for i := len(b.Values) - 1; i >= 0; i-- {
				v := b.Values[i]
				prefs := desired.remove(v.ID)
				if v.Op == OpPhi {
					// TODO: if v is a phi, save desired register for phi inputs.
					// For now, we just drop it and don't propagate
					// desired registers back though phi nodes.
					continue
				}
				// Cancel desired registers if they get clobbered.
				desired.clobber(opcodeTable[v.Op].reg.clobbers)
				// Update desired registers if there are any fixed register inputs.
				for _, j := range opcodeTable[v.Op].reg.inputs {
					if countRegs(j.regs) != 1 {
						continue
					}
					desired.clobber(j.regs)
					desired.add(v.Args[j.idx].ID, pickReg(j.regs))
				}
				// Set desired register of input 0 if this is a 2-operand instruction.
				if opcodeTable[v.Op].resultInArg0 {
					if opcodeTable[v.Op].commutative {
						desired.addList(v.Args[1].ID, prefs)
					}
					desired.addList(v.Args[0].ID, prefs)
				}
			}

			// For each predecessor of b, expand its list of live-at-end values.
			// invariant: live contains the values live at the start of b (excluding phi inputs)
			for i, e := range b.Preds {
				p := e.b
				// Compute additional distance for the edge.
				// Note: delta must be at least 1 to distinguish the control
				// value use from the first user in a successor block.
				delta := int32(normalDistance)
				if len(p.Succs) == 2 {
					if p.Succs[0].b == b && p.Likely == BranchLikely ||
						p.Succs[1].b == b && p.Likely == BranchUnlikely {
						delta = likelyDistance
					}
					if p.Succs[0].b == b && p.Likely == BranchUnlikely ||
						p.Succs[1].b == b && p.Likely == BranchLikely {
						delta = unlikelyDistance
					}
				}

				// Update any desired registers at the end of p.
				s.desired[p.ID].merge(&desired)

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
					if s.values[id].needReg && (!t.contains(id) || delta < t.get(id)) {
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
	if f.pass.debug > regDebug {
		fmt.Println("live values at end of each block")
		for _, b := range f.Blocks {
			fmt.Printf("  %s:", b)
			for _, x := range s.live[b.ID] {
				fmt.Printf(" v%d", x.ID)
				for _, e := range s.desired[b.ID].entries {
					if e.ID != x.ID {
						continue
					}
					fmt.Printf("[")
					first := true
					for _, r := range e.regs {
						if r == noRegister {
							continue
						}
						if !first {
							fmt.Printf(",")
						}
						fmt.Print(s.registers[r].Name())
						first = false
					}
					fmt.Printf("]")
				}
			}
			fmt.Printf(" avoid=%x", int64(s.desired[b.ID].avoid))
			fmt.Println()
		}
	}
}

// A desiredState represents desired register assignments.
type desiredState struct {
	// Desired assignments will be small, so we just use a list
	// of valueID+registers entries.
	entries []desiredStateEntry
	// Registers that other values want to be in.  This value will
	// contain at least the union of the regs fields of entries, but
	// may contain additional entries for values that were once in
	// this data structure but are no longer.
	avoid regMask
}
type desiredStateEntry struct {
	// (pre-regalloc) value
	ID ID
	// Registers it would like to be in, in priority order.
	// Unused slots are filled with noRegister.
	regs [4]register
}

func (d *desiredState) clear() {
	d.entries = d.entries[:0]
	d.avoid = 0
}

// get returns a list of desired registers for value vid.
func (d *desiredState) get(vid ID) [4]register {
	for _, e := range d.entries {
		if e.ID == vid {
			return e.regs
		}
	}
	return [4]register{noRegister, noRegister, noRegister, noRegister}
}

// add records that we'd like value vid to be in register r.
func (d *desiredState) add(vid ID, r register) {
	d.avoid |= regMask(1) << r
	for i := range d.entries {
		e := &d.entries[i]
		if e.ID != vid {
			continue
		}
		if e.regs[0] == r {
			// Already known and highest priority
			return
		}
		for j := 1; j < len(e.regs); j++ {
			if e.regs[j] == r {
				// Move from lower priority to top priority
				copy(e.regs[1:], e.regs[:j])
				e.regs[0] = r
				return
			}
		}
		copy(e.regs[1:], e.regs[:])
		e.regs[0] = r
		return
	}
	d.entries = append(d.entries, desiredStateEntry{vid, [4]register{r, noRegister, noRegister, noRegister}})
}

func (d *desiredState) addList(vid ID, regs [4]register) {
	// regs is in priority order, so iterate in reverse order.
	for i := len(regs) - 1; i >= 0; i-- {
		r := regs[i]
		if r != noRegister {
			d.add(vid, r)
		}
	}
}

// clobber erases any desired registers in the set m.
func (d *desiredState) clobber(m regMask) {
	for i := 0; i < len(d.entries); {
		e := &d.entries[i]
		j := 0
		for _, r := range e.regs {
			if r != noRegister && m>>r&1 == 0 {
				e.regs[j] = r
				j++
			}
		}
		if j == 0 {
			// No more desired registers for this value.
			d.entries[i] = d.entries[len(d.entries)-1]
			d.entries = d.entries[:len(d.entries)-1]
			continue
		}
		for ; j < len(e.regs); j++ {
			e.regs[j] = noRegister
		}
		i++
	}
	d.avoid &^= m
}

// copy copies a desired state from another desiredState x.
func (d *desiredState) copy(x *desiredState) {
	d.entries = append(d.entries[:0], x.entries...)
	d.avoid = x.avoid
}

// remove removes the desired registers for vid and returns them.
func (d *desiredState) remove(vid ID) [4]register {
	for i := range d.entries {
		if d.entries[i].ID == vid {
			regs := d.entries[i].regs
			d.entries[i] = d.entries[len(d.entries)-1]
			d.entries = d.entries[:len(d.entries)-1]
			return regs
		}
	}
	return [4]register{noRegister, noRegister, noRegister, noRegister}
}

// merge merges another desired state x into d.
func (d *desiredState) merge(x *desiredState) {
	d.avoid |= x.avoid
	// There should only be a few desired registers, so
	// linear insert is ok.
	for _, e := range x.entries {
		d.addList(e.ID, e.regs)
	}
}
