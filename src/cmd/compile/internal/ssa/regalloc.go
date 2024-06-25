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
// During the normal course of the allocator, we might throw a still-live
// value out of all registers. When that value is subsequently used, we must
// load it from a slot on the stack. We must also issue an instruction to
// initialize that stack location with a copy of v.
//
// pre-regalloc:
//   (1) v = Op ...
//   (2) x = Op ...
//   (3) ... = Op v ...
//
// post-regalloc:
//   (1) v = Op ...    : AX // computes v, store result in AX
//       s = StoreReg v     // spill v to a stack slot
//   (2) x = Op ...    : AX // some other op uses AX
//       c = LoadReg s : CX // restore v from stack slot
//   (3) ... = Op c ...     // use the restored value
//
// Allocation occurs normally until we reach (3) and we realize we have
// a use of v and it isn't in any register. At that point, we allocate
// a spill (a StoreReg) for v. We can't determine the correct place for
// the spill at this point, so we allocate the spill as blockless initially.
// The restore is then generated to load v back into a register so it can
// be used. Subsequent uses of v will use the restored value c instead.
//
// What remains is the question of where to schedule the spill.
// During allocation, we keep track of the dominator of all restores of v.
// The spill of v must dominate that block. The spill must also be issued at
// a point where v is still in a register.
//
// To find the right place, start at b, the block which dominates all restores.
//  - If b is v.Block, then issue the spill right after v.
//    It is known to be in a register at that point, and dominates any restores.
//  - Otherwise, if v is in a register at the start of b,
//    put the spill of v at the start of b.
//  - Otherwise, set b = immediate dominator of b, and repeat.
//
// Phi values are special, as always. We define two kinds of phis, those
// where the merge happens in a register (a "register" phi) and those where
// the merge happens in a stack location (a "stack" phi).
//
// A register phi must have the phi and all of its inputs allocated to the
// same register. Register phis are spilled similarly to regular ops.
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
// insert an extra phi at the start of b2 (x5=phi(x3,x4):BX) to keep
// SSA form. For now, we ignore this problem as remaining in strict
// SSA form isn't needed after regalloc. We'll just leave the use
// of x3 not dominated by the definition of x3, and the CX->BX copy
// will have no use (so don't run deadcode after regalloc!).
// TODO: maybe we should introduce these extra phis?

package ssa

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"cmd/internal/sys"
	"fmt"
	"internal/buildcfg"
	"math/bits"
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
	s.close()
}

type register uint8

const noRegister register = 255

// For bulk initializing
var noRegisters [32]register = [32]register{
	noRegister, noRegister, noRegister, noRegister, noRegister, noRegister, noRegister, noRegister,
	noRegister, noRegister, noRegister, noRegister, noRegister, noRegister, noRegister, noRegister,
	noRegister, noRegister, noRegister, noRegister, noRegister, noRegister, noRegister, noRegister,
	noRegister, noRegister, noRegister, noRegister, noRegister, noRegister, noRegister, noRegister,
}

// A regMask encodes a set of machine registers.
// TODO: regMask -> regSet?
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

func (s *regAllocState) RegMaskString(m regMask) string {
	str := ""
	for r := register(0); m != 0; r++ {
		if m>>r&1 == 0 {
			continue
		}
		m &^= regMask(1) << r
		if str != "" {
			str += " "
		}
		str += s.registers[r].String()
	}
	return str
}

// countRegs returns the number of set bits in the register mask.
func countRegs(r regMask) int {
	return bits.OnesCount64(uint64(r))
}

// pickReg picks an arbitrary register from the register mask.
func pickReg(r regMask) register {
	if r == 0 {
		panic("can't pick a register from an empty set")
	}
	// pick the lowest one
	return register(bits.TrailingZeros64(uint64(r)))
}

type use struct {
	dist int32    // distance from start of the block to a use of a value
	pos  src.XPos // source position of the use
	next *use     // linked list of uses of a value in nondecreasing dist order
}

// A valState records the register allocation state for a (pre-regalloc) value.
type valState struct {
	regs              regMask // the set of registers holding a Value (usually just one)
	uses              *use    // list of uses in this block
	spill             *Value  // spilled copy of the Value (if any)
	restoreMin        int32   // minimum of all restores' blocks' sdom.entry
	restoreMax        int32   // maximum of all restores' blocks' sdom.exit
	needReg           bool    // cached value of !v.Type.IsMemory() && !v.Type.IsVoid() && !.v.Type.IsFlags()
	rematerializeable bool    // cached value of v.rematerializeable()
}

type regState struct {
	v *Value // Original (preregalloc) Value stored in this register.
	c *Value // A Value equal to v which is currently in a register.  Might be v or a copy of it.
	// If a register is unused, v==c==nil
}

type regAllocState struct {
	f *Func

	sdom        SparseTree
	registers   []Register
	numRegs     register
	SPReg       register
	SBReg       register
	GReg        register
	allocatable regMask

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

	// ID of SP, SB values
	sp, sb ID

	// For each Value, map from its value ID back to the
	// preregalloc Value it was derived from.
	orig []*Value

	// current state of each register
	regs []regState

	// registers that contain values which can't be kicked out
	nospill regMask

	// mask of registers currently in use
	used regMask

	// mask of registers used since the start of the current block
	usedSinceBlockStart regMask

	// mask of registers used in the current instruction
	tmpused regMask

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

	// startRegsMask is a mask of the registers in startRegs[curBlock.ID].
	// Registers dropped from startRegsMask are later synchronoized back to
	// startRegs by dropping from there as well.
	startRegsMask regMask

	// spillLive[blockid] is the set of live spills at the end of each block
	spillLive [][]ID

	// a set of copies we generated to move things around, and
	// whether it is used in shuffle. Unused copies will be deleted.
	copies map[*Value]bool

	loopnest *loopnest

	// choose a good order in which to visit blocks for allocation purposes.
	visitOrder []*Block

	// blockOrder[b.ID] corresponds to the index of block b in visitOrder.
	blockOrder []int32

	// whether to insert instructions that clobber dead registers at call sites
	doClobber bool
}

type endReg struct {
	r register
	v *Value // pre-regalloc value held in this register (TODO: can we use ID here?)
	c *Value // cached version of the value
}

type startReg struct {
	r   register
	v   *Value   // pre-regalloc value needed in this register
	c   *Value   // cached version of the value
	pos src.XPos // source position of use of this register
}

// freeReg frees up register r. Any current user of r is kicked out.
func (s *regAllocState) freeReg(r register) {
	v := s.regs[r].v
	if v == nil {
		s.f.Fatalf("tried to free an already free register %d\n", r)
	}

	// Mark r as unused.
	if s.f.pass.debug > regDebug {
		fmt.Printf("freeReg %s (dump %s/%s)\n", &s.registers[r], v, s.regs[r].c)
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

// clobberRegs inserts instructions that clobber registers listed in m.
func (s *regAllocState) clobberRegs(m regMask) {
	m &= s.allocatable & s.f.Config.gpRegMask // only integer register can contain pointers, only clobber them
	for m != 0 {
		r := pickReg(m)
		m &^= 1 << r
		x := s.curBlock.NewValue0(src.NoXPos, OpClobberReg, types.TypeVoid)
		s.f.setHome(x, &s.registers[r])
	}
}

// setOrig records that c's original value is the same as
// v's original value.
func (s *regAllocState) setOrig(c *Value, v *Value) {
	if int(c.ID) >= cap(s.orig) {
		x := s.f.Cache.allocValueSlice(int(c.ID) + 1)
		copy(x, s.orig)
		s.f.Cache.freeValueSlice(s.orig)
		s.orig = x
	}
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
		fmt.Printf("assignReg %s %s/%s\n", &s.registers[r], v, c)
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

// allocReg chooses a register from the set of registers in mask.
// If there is no unused register, a Value will be kicked out of
// a register to make room.
func (s *regAllocState) allocReg(mask regMask, v *Value) register {
	if v.OnWasmStack {
		return noRegister
	}

	mask &= s.allocatable
	mask &^= s.nospill
	if mask == 0 {
		s.f.Fatalf("no register available for %s", v.LongString())
	}

	// Pick an unused register if one is available.
	if mask&^s.used != 0 {
		r := pickReg(mask &^ s.used)
		s.usedSinceBlockStart |= regMask(1) << r
		return r
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
		s.f.Fatalf("couldn't find register to spill")
	}

	if s.f.Config.ctxt.Arch.Arch == sys.ArchWasm {
		// TODO(neelance): In theory this should never happen, because all wasm registers are equal.
		// So if there is still a free register, the allocation should have picked that one in the first place instead of
		// trying to kick some other value out. In practice, this case does happen and it breaks the stack optimization.
		s.freeReg(r)
		return r
	}

	// Try to move it around before kicking out, if there is a free register.
	// We generate a Copy and record it. It will be deleted if never used.
	v2 := s.regs[r].v
	m := s.compatRegs(v2.Type) &^ s.used &^ s.tmpused &^ (regMask(1) << r)
	if m != 0 && !s.values[v2.ID].rematerializeable && countRegs(s.values[v2.ID].regs) == 1 {
		s.usedSinceBlockStart |= regMask(1) << r
		r2 := pickReg(m)
		c := s.curBlock.NewValue1(v2.Pos, OpCopy, v2.Type, s.regs[r].c)
		s.copies[c] = false
		if s.f.pass.debug > regDebug {
			fmt.Printf("copy %s to %s : %s\n", v2, c, &s.registers[r2])
		}
		s.setOrig(c, v2)
		s.assignReg(r2, v2, c)
	}

	// If the evicted register isn't used between the start of the block
	// and now then there is no reason to even request it on entry. We can
	// drop from startRegs in that case.
	if s.usedSinceBlockStart&(regMask(1)<<r) == 0 {
		if s.startRegsMask&(regMask(1)<<r) == 1 {
			if s.f.pass.debug > regDebug {
				fmt.Printf("dropped from startRegs: %s\n", &s.registers[r])
			}
			s.startRegsMask &^= regMask(1) << r
		}
	}

	s.freeReg(r)
	s.usedSinceBlockStart |= regMask(1) << r
	return r
}

// makeSpill returns a Value which represents the spilled value of v.
// b is the block in which the spill is used.
func (s *regAllocState) makeSpill(v *Value, b *Block) *Value {
	vi := &s.values[v.ID]
	if vi.spill != nil {
		// Final block not known - keep track of subtree where restores reside.
		vi.restoreMin = min32(vi.restoreMin, s.sdom[b.ID].entry)
		vi.restoreMax = max32(vi.restoreMax, s.sdom[b.ID].exit)
		return vi.spill
	}
	// Make a spill for v. We don't know where we want
	// to put it yet, so we leave it blockless for now.
	spill := s.f.newValueNoBlock(OpStoreReg, v.Type, v.Pos)
	// We also don't know what the spill's arg will be.
	// Leave it argless for now.
	s.setOrig(spill, v)
	vi.spill = spill
	vi.restoreMin = s.sdom[b.ID].entry
	vi.restoreMax = s.sdom[b.ID].exit
	return spill
}

// allocValToReg allocates v to a register selected from regMask and
// returns the register copy of v. Any previous user is kicked out and spilled
// (if necessary). Load code is added at the current pc. If nospill is set the
// allocated register is marked nospill so the assignment cannot be
// undone until the caller allows it by clearing nospill. Returns a
// *Value which is either v or a copy of v allocated to the chosen register.
func (s *regAllocState) allocValToReg(v *Value, mask regMask, nospill bool, pos src.XPos) *Value {
	if s.f.Config.ctxt.Arch.Arch == sys.ArchWasm && v.rematerializeable() {
		c := v.copyIntoWithXPos(s.curBlock, pos)
		c.OnWasmStack = true
		s.setOrig(c, v)
		return c
	}
	if v.OnWasmStack {
		return v
	}

	vi := &s.values[v.ID]
	pos = pos.WithNotStmt()
	// Check if v is already in a requested register.
	if mask&vi.regs != 0 {
		r := pickReg(mask & vi.regs)
		if s.regs[r].v != v || s.regs[r].c == nil {
			panic("bad register state")
		}
		if nospill {
			s.nospill |= regMask(1) << r
		}
		s.usedSinceBlockStart |= regMask(1) << r
		return s.regs[r].c
	}

	var r register
	// If nospill is set, the value is used immediately, so it can live on the WebAssembly stack.
	onWasmStack := nospill && s.f.Config.ctxt.Arch.Arch == sys.ArchWasm
	if !onWasmStack {
		// Allocate a register.
		r = s.allocReg(mask, v)
	}

	// Allocate v to the new register.
	var c *Value
	if vi.regs != 0 {
		// Copy from a register that v is already in.
		r2 := pickReg(vi.regs)
		if s.regs[r2].v != v {
			panic("bad register state")
		}
		s.usedSinceBlockStart |= regMask(1) << r2
		c = s.curBlock.NewValue1(pos, OpCopy, v.Type, s.regs[r2].c)
	} else if v.rematerializeable() {
		// Rematerialize instead of loading from the spill location.
		c = v.copyIntoWithXPos(s.curBlock, pos)
	} else {
		// Load v from its spill location.
		spill := s.makeSpill(v, s.curBlock)
		if s.f.pass.debug > logSpills {
			s.f.Warnl(vi.spill.Pos, "load spill for %v from %v", v, spill)
		}
		c = s.curBlock.NewValue1(pos, OpLoadReg, v.Type, spill)
	}

	s.setOrig(c, v)

	if onWasmStack {
		c.OnWasmStack = true
		return c
	}

	s.assignReg(r, v, c)
	if c.Op == OpLoadReg && s.isGReg(r) {
		s.f.Fatalf("allocValToReg.OpLoadReg targeting g: " + c.LongString())
	}
	if nospill {
		s.nospill |= regMask(1) << r
	}
	return c
}

// isLeaf reports whether f performs any calls.
func isLeaf(f *Func) bool {
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op.IsCall() && !v.Op.IsTailCall() {
				// tail call is not counted as it does not save the return PC or need a frame
				return false
			}
		}
	}
	return true
}

// needRegister reports whether v needs a register.
func (v *Value) needRegister() bool {
	return !v.Type.IsMemory() && !v.Type.IsVoid() && !v.Type.IsFlags() && !v.Type.IsTuple()
}

func (s *regAllocState) init(f *Func) {
	s.f = f
	s.f.RegAlloc = s.f.Cache.locs[:0]
	s.registers = f.Config.registers
	if nr := len(s.registers); nr == 0 || nr > int(noRegister) || nr > int(unsafe.Sizeof(regMask(0))*8) {
		s.f.Fatalf("bad number of registers: %d", nr)
	} else {
		s.numRegs = register(nr)
	}
	// Locate SP, SB, and g registers.
	s.SPReg = noRegister
	s.SBReg = noRegister
	s.GReg = noRegister
	for r := register(0); r < s.numRegs; r++ {
		switch s.registers[r].String() {
		case "SP":
			s.SPReg = r
		case "SB":
			s.SBReg = r
		case "g":
			s.GReg = r
		}
	}
	// Make sure we found all required registers.
	switch noRegister {
	case s.SPReg:
		s.f.Fatalf("no SP register found")
	case s.SBReg:
		s.f.Fatalf("no SB register found")
	case s.GReg:
		if f.Config.hasGReg {
			s.f.Fatalf("no g register found")
		}
	}

	// Figure out which registers we're allowed to use.
	s.allocatable = s.f.Config.gpRegMask | s.f.Config.fpRegMask | s.f.Config.specialRegMask
	s.allocatable &^= 1 << s.SPReg
	s.allocatable &^= 1 << s.SBReg
	if s.f.Config.hasGReg {
		s.allocatable &^= 1 << s.GReg
	}
	if buildcfg.FramePointerEnabled && s.f.Config.FPReg >= 0 {
		s.allocatable &^= 1 << uint(s.f.Config.FPReg)
	}
	if s.f.Config.LinkReg != -1 {
		if isLeaf(f) {
			// Leaf functions don't save/restore the link register.
			s.allocatable &^= 1 << uint(s.f.Config.LinkReg)
		}
	}
	if s.f.Config.ctxt.Flag_dynlink {
		switch s.f.Config.arch {
		case "386":
			// nothing to do.
			// Note that for Flag_shared (position independent code)
			// we do need to be careful, but that carefulness is hidden
			// in the rewrite rules so we always have a free register
			// available for global load/stores. See _gen/386.rules (search for Flag_shared).
		case "amd64":
			s.allocatable &^= 1 << 15 // R15
		case "arm":
			s.allocatable &^= 1 << 9 // R9
		case "arm64":
			// nothing to do
		case "loong64": // R2 (aka TP) already reserved.
			// nothing to do
		case "ppc64le": // R2 already reserved.
			// nothing to do
		case "riscv64": // X3 (aka GP) and X4 (aka TP) already reserved.
			// nothing to do
		case "s390x":
			s.allocatable &^= 1 << 11 // R11
		default:
			s.f.fe.Fatalf(src.NoXPos, "arch %s not implemented", s.f.Config.arch)
		}
	}

	// Linear scan register allocation can be influenced by the order in which blocks appear.
	// Decouple the register allocation order from the generated block order.
	// This also creates an opportunity for experiments to find a better order.
	s.visitOrder = layoutRegallocOrder(f)

	// Compute block order. This array allows us to distinguish forward edges
	// from backward edges and compute how far they go.
	s.blockOrder = make([]int32, f.NumBlocks())
	for i, b := range s.visitOrder {
		s.blockOrder[b.ID] = int32(i)
	}

	s.regs = make([]regState, s.numRegs)
	nv := f.NumValues()
	if cap(s.f.Cache.regallocValues) >= nv {
		s.f.Cache.regallocValues = s.f.Cache.regallocValues[:nv]
	} else {
		s.f.Cache.regallocValues = make([]valState, nv)
	}
	s.values = s.f.Cache.regallocValues
	s.orig = s.f.Cache.allocValueSlice(nv)
	s.copies = make(map[*Value]bool)
	for _, b := range s.visitOrder {
		for _, v := range b.Values {
			if v.needRegister() {
				s.values[v.ID].needReg = true
				s.values[v.ID].rematerializeable = v.rematerializeable()
				s.orig[v.ID] = v
			}
			// Note: needReg is false for values returning Tuple types.
			// Instead, we mark the corresponding Selects as needReg.
		}
	}
	s.computeLive()

	s.endRegs = make([][]endReg, f.NumBlocks())
	s.startRegs = make([][]startReg, f.NumBlocks())
	s.spillLive = make([][]ID, f.NumBlocks())
	s.sdom = f.Sdom()

	// wasm: Mark instructions that can be optimized to have their values only on the WebAssembly stack.
	if f.Config.ctxt.Arch.Arch == sys.ArchWasm {
		canLiveOnStack := f.newSparseSet(f.NumValues())
		defer f.retSparseSet(canLiveOnStack)
		for _, b := range f.Blocks {
			// New block. Clear candidate set.
			canLiveOnStack.clear()
			for _, c := range b.ControlValues() {
				if c.Uses == 1 && !opcodeTable[c.Op].generic {
					canLiveOnStack.add(c.ID)
				}
			}
			// Walking backwards.
			for i := len(b.Values) - 1; i >= 0; i-- {
				v := b.Values[i]
				if canLiveOnStack.contains(v.ID) {
					v.OnWasmStack = true
				} else {
					// Value can not live on stack. Values are not allowed to be reordered, so clear candidate set.
					canLiveOnStack.clear()
				}
				for _, arg := range v.Args {
					// Value can live on the stack if:
					// - it is only used once
					// - it is used in the same basic block
					// - it is not a "mem" value
					// - it is a WebAssembly op
					if arg.Uses == 1 && arg.Block == v.Block && !arg.Type.IsMemory() && !opcodeTable[arg.Op].generic {
						canLiveOnStack.add(arg.ID)
					}
				}
			}
		}
	}

	// The clobberdeadreg experiment inserts code to clobber dead registers
	// at call sites.
	// Ignore huge functions to avoid doing too much work.
	if base.Flag.ClobberDeadReg && len(s.f.Blocks) <= 10000 {
		// TODO: honor GOCLOBBERDEADHASH, or maybe GOSSAHASH.
		s.doClobber = true
	}
}

func (s *regAllocState) close() {
	s.f.Cache.freeValueSlice(s.orig)
}

// Adds a use record for id at distance dist from the start of the block.
// All calls to addUse must happen with nonincreasing dist.
func (s *regAllocState) addUse(id ID, dist int32, pos src.XPos) {
	r := s.freeUseRecords
	if r != nil {
		s.freeUseRecords = r.next
	} else {
		r = &use{}
	}
	r.dist = dist
	r.pos = pos
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
	if u == nil {
		panic(fmt.Errorf("u is nil, v = %s, s.values[v.ID] = %v", v.LongString(), s.values[v.ID]))
	}
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
func (s *regAllocState) compatRegs(t *types.Type) regMask {
	var m regMask
	if t.IsTuple() || t.IsFlags() {
		return 0
	}
	if t.IsFloat() || t == types.TypeInt128 {
		if t.Kind() == types.TFLOAT32 && s.f.Config.fp32RegMask != 0 {
			m = s.f.Config.fp32RegMask
		} else if t.Kind() == types.TFLOAT64 && s.f.Config.fp64RegMask != 0 {
			m = s.f.Config.fp64RegMask
		} else {
			m = s.f.Config.fpRegMask
		}
	} else {
		m = s.f.Config.gpRegMask
	}
	return m & s.allocatable
}

// regspec returns the regInfo for operation op.
func (s *regAllocState) regspec(v *Value) regInfo {
	op := v.Op
	if op == OpConvert {
		// OpConvert is a generic op, so it doesn't have a
		// register set in the static table. It can use any
		// allocatable integer register.
		m := s.allocatable & s.f.Config.gpRegMask
		return regInfo{inputs: []inputInfo{{regs: m}}, outputs: []outputInfo{{regs: m}}}
	}
	if op == OpArgIntReg {
		reg := v.Block.Func.Config.intParamRegs[v.AuxInt8()]
		return regInfo{outputs: []outputInfo{{regs: 1 << uint(reg)}}}
	}
	if op == OpArgFloatReg {
		reg := v.Block.Func.Config.floatParamRegs[v.AuxInt8()]
		return regInfo{outputs: []outputInfo{{regs: 1 << uint(reg)}}}
	}
	if op.IsCall() {
		if ac, ok := v.Aux.(*AuxCall); ok && ac.reg != nil {
			return *ac.Reg(&opcodeTable[op].reg, s.f.Config)
		}
	}
	if op == OpMakeResult && s.f.OwnAux.reg != nil {
		return *s.f.OwnAux.ResultReg(s.f.Config)
	}
	return opcodeTable[op].reg
}

func (s *regAllocState) isGReg(r register) bool {
	return s.f.Config.hasGReg && s.GReg == r
}

// Dummy value used to represent the value being held in a temporary register.
var tmpVal Value

func (s *regAllocState) regalloc(f *Func) {
	regValLiveSet := f.newSparseSet(f.NumValues()) // set of values that may be live in register
	defer f.retSparseSet(regValLiveSet)
	var oldSched []*Value
	var phis []*Value
	var phiRegs []register
	var args []*Value

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

	for _, b := range s.visitOrder {
		if s.f.pass.debug > regDebug {
			fmt.Printf("Begin processing block %v\n", b)
		}
		s.curBlock = b
		s.startRegsMask = 0
		s.usedSinceBlockStart = 0

		// Initialize regValLiveSet and uses fields for this block.
		// Walk backwards through the block doing liveness analysis.
		regValLiveSet.clear()
		for _, e := range s.live[b.ID] {
			s.addUse(e.ID, int32(len(b.Values))+e.dist, e.pos) // pseudo-uses from beyond end of block
			regValLiveSet.add(e.ID)
		}
		for _, v := range b.ControlValues() {
			if s.values[v.ID].needReg {
				s.addUse(v.ID, int32(len(b.Values)), b.Pos) // pseudo-use by control values
				regValLiveSet.add(v.ID)
			}
		}
		for i := len(b.Values) - 1; i >= 0; i-- {
			v := b.Values[i]
			regValLiveSet.remove(v.ID)
			if v.Op == OpPhi {
				// Remove v from the live set, but don't add
				// any inputs. This is the state the len(b.Preds)>1
				// case below desires; it wants to process phis specially.
				continue
			}
			if opcodeTable[v.Op].call {
				// Function call clobbers all the registers but SP and SB.
				regValLiveSet.clear()
				if s.sp != 0 && s.values[s.sp].uses != nil {
					regValLiveSet.add(s.sp)
				}
				if s.sb != 0 && s.values[s.sb].uses != nil {
					regValLiveSet.add(s.sb)
				}
			}
			for _, a := range v.Args {
				if !s.values[a.ID].needReg {
					continue
				}
				s.addUse(a.ID, int32(i), v.Pos)
				regValLiveSet.add(a.ID)
			}
		}
		if s.f.pass.debug > regDebug {
			fmt.Printf("use distances for %s\n", b)
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
				if v != nil && !regValLiveSet.contains(v.ID) {
					s.freeReg(r)
				}
			}
		} else {
			// This is the complicated case. We have more than one predecessor,
			// which means we may have Phi ops.

			// Start with the final register state of the predecessor with least spill values.
			// This is based on the following points:
			// 1, The less spill value indicates that the register pressure of this path is smaller,
			//    so the values of this block are more likely to be allocated to registers.
			// 2, Avoid the predecessor that contains the function call, because the predecessor that
			//    contains the function call usually generates a lot of spills and lose the previous
			//    allocation state.
			// TODO: Improve this part. At least the size of endRegs of the predecessor also has
			// an impact on the code size and compiler speed. But it is not easy to find a simple
			// and efficient method that combines multiple factors.
			idx := -1
			for i, p := range b.Preds {
				// If the predecessor has not been visited yet, skip it because its end state
				// (redRegs and spillLive) has not been computed yet.
				pb := p.b
				if s.blockOrder[pb.ID] >= s.blockOrder[b.ID] {
					continue
				}
				if idx == -1 {
					idx = i
					continue
				}
				pSel := b.Preds[idx].b
				if len(s.spillLive[pb.ID]) < len(s.spillLive[pSel.ID]) {
					idx = i
				} else if len(s.spillLive[pb.ID]) == len(s.spillLive[pSel.ID]) {
					// Use a bit of likely information. After critical pass, pb and pSel must
					// be plain blocks, so check edge pb->pb.Preds instead of edge pb->b.
					// TODO: improve the prediction of the likely predecessor. The following
					// method is only suitable for the simplest cases. For complex cases,
					// the prediction may be inaccurate, but this does not affect the
					// correctness of the program.
					// According to the layout algorithm, the predecessor with the
					// smaller blockOrder is the true branch, and the test results show
					// that it is better to choose the predecessor with a smaller
					// blockOrder than no choice.
					if pb.likelyBranch() && !pSel.likelyBranch() || s.blockOrder[pb.ID] < s.blockOrder[pSel.ID] {
						idx = i
					}
				}
			}
			if idx < 0 {
				f.Fatalf("bad visitOrder, no predecessor of %s has been visited before it", b)
			}
			p := b.Preds[idx].b
			s.setState(s.endRegs[p.ID])

			if s.f.pass.debug > regDebug {
				fmt.Printf("starting merge block %s with end state of %s:\n", b, p)
				for _, x := range s.endRegs[p.ID] {
					fmt.Printf("  %s: orig:%s cache:%s\n", &s.registers[x.r], x.v, x.c)
				}
			}

			// Decide on registers for phi ops. Use the registers determined
			// by the primary predecessor if we can.
			// TODO: pick best of (already processed) predecessors?
			// Majority vote? Deepest nesting level?
			phiRegs = phiRegs[:0]
			var phiUsed regMask

			for _, v := range phis {
				if !s.values[v.ID].needReg {
					phiRegs = append(phiRegs, noRegister)
					continue
				}
				a := v.Args[idx]
				// Some instructions target not-allocatable registers.
				// They're not suitable for further (phi-function) allocation.
				m := s.values[a.ID].regs &^ phiUsed & s.allocatable
				if m != 0 {
					r := pickReg(m)
					phiUsed |= regMask(1) << r
					phiRegs = append(phiRegs, r)
				} else {
					phiRegs = append(phiRegs, noRegister)
				}
			}

			// Second pass - deallocate all in-register phi inputs.
			for i, v := range phis {
				if !s.values[v.ID].needReg {
					continue
				}
				a := v.Args[idx]
				r := phiRegs[i]
				if r == noRegister {
					continue
				}
				if regValLiveSet.contains(a.ID) {
					// Input value is still live (it is used by something other than Phi).
					// Try to move it around before kicking out, if there is a free register.
					// We generate a Copy in the predecessor block and record it. It will be
					// deleted later if never used.
					//
					// Pick a free register. At this point some registers used in the predecessor
					// block may have been deallocated. Those are the ones used for Phis. Exclude
					// them (and they are not going to be helpful anyway).
					m := s.compatRegs(a.Type) &^ s.used &^ phiUsed
					if m != 0 && !s.values[a.ID].rematerializeable && countRegs(s.values[a.ID].regs) == 1 {
						r2 := pickReg(m)
						c := p.NewValue1(a.Pos, OpCopy, a.Type, s.regs[r].c)
						s.copies[c] = false
						if s.f.pass.debug > regDebug {
							fmt.Printf("copy %s to %s : %s\n", a, c, &s.registers[r2])
						}
						s.setOrig(c, a)
						s.assignReg(r2, a, c)
						s.endRegs[p.ID] = append(s.endRegs[p.ID], endReg{r2, a, c})
					}
				}
				s.freeReg(r)
			}

			// Copy phi ops into new schedule.
			b.Values = append(b.Values, phis...)

			// Third pass - pick registers for phis whose input
			// was not in a register in the primary predecessor.
			for i, v := range phis {
				if !s.values[v.ID].needReg {
					continue
				}
				if phiRegs[i] != noRegister {
					continue
				}
				m := s.compatRegs(v.Type) &^ phiUsed &^ s.used
				// If one of the other inputs of v is in a register, and the register is available,
				// select this register, which can save some unnecessary copies.
				for i, pe := range b.Preds {
					if i == idx {
						continue
					}
					ri := noRegister
					for _, er := range s.endRegs[pe.b.ID] {
						if er.v == s.orig[v.Args[i].ID] {
							ri = er.r
							break
						}
					}
					if ri != noRegister && m>>ri&1 != 0 {
						m = regMask(1) << ri
						break
					}
				}
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
					s.values[v.ID].spill = v // v starts life spilled
					continue
				}
				// register-based phi
				s.assignReg(r, v, v)
			}

			// Deallocate any values which are no longer live. Phis are excluded.
			for r := register(0); r < s.numRegs; r++ {
				if phiUsed>>r&1 != 0 {
					continue
				}
				v := s.regs[r].v
				if v != nil && !regValLiveSet.contains(v.ID) {
					s.freeReg(r)
				}
			}

			// Save the starting state for use by merge edges.
			// We append to a stack allocated variable that we'll
			// later copy into s.startRegs in one fell swoop, to save
			// on allocations.
			regList := make([]startReg, 0, 32)
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
				regList = append(regList, startReg{r, v, s.regs[r].c, s.values[v.ID].uses.pos})
				s.startRegsMask |= regMask(1) << r
			}
			s.startRegs[b.ID] = make([]startReg, len(regList))
			copy(s.startRegs[b.ID], regList)

			if s.f.pass.debug > regDebug {
				fmt.Printf("after phis\n")
				for _, x := range s.startRegs[b.ID] {
					fmt.Printf("  %s: v%d\n", &s.registers[x.r], x.v.ID)
				}
			}
		}

		// Allocate space to record the desired registers for each value.
		if l := len(oldSched); cap(dinfo) < l {
			dinfo = make([]dentry, l)
		} else {
			dinfo = dinfo[:l]
			for i := range dinfo {
				dinfo[i] = dentry{}
			}
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
				desired.add(x.v.ID, x.r)
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
					// If v is not assigned a register, pick a register assigned to one of v's inputs.
					// Hopefully v will get assigned that register later.
					// If the inputs have allocated register information, add it to desired,
					// which may reduce spill or copy operations when the register is available.
					for _, a := range v.Args {
						rp, ok = s.f.getHome(a.ID).(*Register)
						if ok {
							break
						}
					}
					if !ok {
						continue
					}
				}
				desired.add(v.Args[pidx].ID, register(rp.num))
			}
		}
		// Walk values backwards computing desired register info.
		// See computeLive for more comments.
		for i := len(oldSched) - 1; i >= 0; i-- {
			v := oldSched[i]
			prefs := desired.remove(v.ID)
			regspec := s.regspec(v)
			desired.clobber(regspec.clobbers)
			for _, j := range regspec.inputs {
				if countRegs(j.regs) != 1 {
					continue
				}
				desired.clobber(j.regs)
				desired.add(v.Args[j.idx].ID, pickReg(j.regs))
			}
			if opcodeTable[v.Op].resultInArg0 || v.Op == OpAMD64ADDQconst || v.Op == OpAMD64ADDLconst || v.Op == OpSelect0 {
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
			tmpReg := noRegister
			if s.f.pass.debug > regDebug {
				fmt.Printf("  processing %s\n", v.LongString())
			}
			regspec := s.regspec(v)
			if v.Op == OpPhi {
				f.Fatalf("phi %s not at start of block", v)
			}
			if v.Op == OpSP {
				s.assignReg(s.SPReg, v, v)
				b.Values = append(b.Values, v)
				s.advanceUses(v)
				s.sp = v.ID
				continue
			}
			if v.Op == OpSB {
				s.assignReg(s.SBReg, v, v)
				b.Values = append(b.Values, v)
				s.advanceUses(v)
				s.sb = v.ID
				continue
			}
			if v.Op == OpSelect0 || v.Op == OpSelect1 || v.Op == OpSelectN {
				if s.values[v.ID].needReg {
					if v.Op == OpSelectN {
						s.assignReg(register(s.f.getHome(v.Args[0].ID).(LocResults)[int(v.AuxInt)].(*Register).num), v, v)
					} else {
						var i = 0
						if v.Op == OpSelect1 {
							i = 1
						}
						s.assignReg(register(s.f.getHome(v.Args[0].ID).(LocPair)[i].(*Register).num), v, v)
					}
				}
				b.Values = append(b.Values, v)
				s.advanceUses(v)
				continue
			}
			if v.Op == OpGetG && s.f.Config.hasGReg {
				// use hardware g register
				if s.regs[s.GReg].v != nil {
					s.freeReg(s.GReg) // kick out the old value
				}
				s.assignReg(s.GReg, v, v)
				b.Values = append(b.Values, v)
				s.advanceUses(v)
				continue
			}
			if v.Op == OpArg {
				// Args are "pre-spilled" values. We don't allocate
				// any register here. We just set up the spill pointer to
				// point at itself and any later user will restore it to use it.
				s.values[v.ID].spill = v
				b.Values = append(b.Values, v)
				s.advanceUses(v)
				continue
			}
			if v.Op == OpKeepAlive {
				// Make sure the argument to v is still live here.
				s.advanceUses(v)
				a := v.Args[0]
				vi := &s.values[a.ID]
				if vi.regs == 0 && !vi.rematerializeable {
					// Use the spill location.
					// This forces later liveness analysis to make the
					// value live at this point.
					v.SetArg(0, s.makeSpill(a, b))
				} else if _, ok := a.Aux.(*ir.Name); ok && vi.rematerializeable {
					// Rematerializeable value with a gc.Node. This is the address of
					// a stack object (e.g. an LEAQ). Keep the object live.
					// Change it to VarLive, which is what plive expects for locals.
					v.Op = OpVarLive
					v.SetArgs1(v.Args[1])
					v.Aux = a.Aux
				} else {
					// In-register and rematerializeable values are already live.
					// These are typically rematerializeable constants like nil,
					// or values of a variable that were modified since the last call.
					v.Op = OpCopy
					v.SetArgs1(v.Args[1])
				}
				b.Values = append(b.Values, v)
				continue
			}
			if len(regspec.inputs) == 0 && len(regspec.outputs) == 0 {
				// No register allocation required (or none specified yet)
				if s.doClobber && v.Op.IsCall() {
					s.clobberRegs(regspec.clobbers)
				}
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
						fmt.Printf(" %s", &s.registers[r])
					}
				}
				fmt.Println()
				for i := 0; i < len(v.Args) && i < 3; i++ {
					fmt.Printf("  in%d:", i)
					for _, r := range dinfo[idx].in[i] {
						if r != noRegister {
							fmt.Printf(" %s", &s.registers[r])
						}
					}
					fmt.Println()
				}
			}

			// Move arguments to registers.
			// First, if an arg must be in a specific register and it is already
			// in place, keep it.
			args = append(args[:0], make([]*Value, len(v.Args))...)
			for i, a := range v.Args {
				if !s.values[a.ID].needReg {
					args[i] = a
				}
			}
			for _, i := range regspec.inputs {
				mask := i.regs
				if countRegs(mask) == 1 && mask&s.values[v.Args[i.idx].ID].regs != 0 {
					args[i.idx] = s.allocValToReg(v.Args[i.idx], mask, true, v.Pos)
				}
			}
			// Then, if an arg must be in a specific register and that
			// register is free, allocate that one. Otherwise when processing
			// another input we may kick a value into the free register, which
			// then will be kicked out again.
			// This is a common case for passing-in-register arguments for
			// function calls.
			for {
				freed := false
				for _, i := range regspec.inputs {
					if args[i.idx] != nil {
						continue // already allocated
					}
					mask := i.regs
					if countRegs(mask) == 1 && mask&^s.used != 0 {
						args[i.idx] = s.allocValToReg(v.Args[i.idx], mask, true, v.Pos)
						// If the input is in other registers that will be clobbered by v,
						// or the input is dead, free the registers. This may make room
						// for other inputs.
						oldregs := s.values[v.Args[i.idx].ID].regs
						if oldregs&^regspec.clobbers == 0 || !s.liveAfterCurrentInstruction(v.Args[i.idx]) {
							s.freeRegs(oldregs &^ mask &^ s.nospill)
							freed = true
						}
					}
				}
				if !freed {
					break
				}
			}
			// Last, allocate remaining ones, in an ordering defined
			// by the register specification (most constrained first).
			for _, i := range regspec.inputs {
				if args[i.idx] != nil {
					continue // already allocated
				}
				mask := i.regs
				if mask&s.values[v.Args[i.idx].ID].regs == 0 {
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
				args[i.idx] = s.allocValToReg(v.Args[i.idx], mask, true, v.Pos)
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
				if opcodeTable[v.Op].commutative && !s.liveAfterCurrentInstruction(v.Args[1]) {
					args[0], args[1] = args[1], args[0]
					goto ok
				}
				if s.values[v.Args[0].ID].rematerializeable {
					// We can rematerialize the input, don't worry about clobbering it.
					goto ok
				}
				if opcodeTable[v.Op].commutative && s.values[v.Args[1].ID].rematerializeable {
					args[0], args[1] = args[1], args[0]
					goto ok
				}
				if countRegs(s.values[v.Args[0].ID].regs) >= 2 {
					// we have at least 2 copies of arg0.  We can afford to clobber one.
					goto ok
				}
				if opcodeTable[v.Op].commutative && countRegs(s.values[v.Args[1].ID].regs) >= 2 {
					args[0], args[1] = args[1], args[0]
					goto ok
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

				// Try to move an input to the desired output, if allowed.
				for _, r := range dinfo[idx].out {
					if r != noRegister && (m&regspec.outputs[0].regs)>>r&1 != 0 {
						m = regMask(1) << r
						args[0] = s.allocValToReg(v.Args[0], m, true, v.Pos)
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
						c := s.allocValToReg(v.Args[0], m, true, v.Pos)
						s.copies[c] = false
						// Note: no update to args[0] so the instruction will
						// use the original copy.
						goto ok
					}
				}
				if opcodeTable[v.Op].commutative {
					for _, r := range dinfo[idx].in[1] {
						if r != noRegister && m>>r&1 != 0 {
							m = regMask(1) << r
							c := s.allocValToReg(v.Args[1], m, true, v.Pos)
							s.copies[c] = false
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
				c := s.allocValToReg(v.Args[0], m, true, v.Pos)
				s.copies[c] = false

				// Normally we use the register of the old copy of input 0 as the target.
				// However, if input 0 is already in its desired register then we use
				// the register of the new copy instead.
				if regspec.outputs[0].regs>>s.f.getHome(c.ID).(*Register).num&1 != 0 {
					if rp, ok := s.f.getHome(args[0].ID).(*Register); ok {
						r := register(rp.num)
						for _, r2 := range dinfo[idx].in[0] {
							if r == r2 {
								args[0] = c
								break
							}
						}
					}
				}
			}

		ok:
			// Pick a temporary register if needed.
			// It should be distinct from all the input registers, so we
			// allocate it after all the input registers, but before
			// the input registers are freed via advanceUses below.
			// (Not all instructions need that distinct part, but it is conservative.)
			// We also ensure it is not any of the single-choice output registers.
			if opcodeTable[v.Op].needIntTemp {
				m := s.allocatable & s.f.Config.gpRegMask
				for _, out := range regspec.outputs {
					if countRegs(out.regs) == 1 {
						m &^= out.regs
					}
				}
				if m&^desired.avoid&^s.nospill != 0 {
					m &^= desired.avoid
				}
				tmpReg = s.allocReg(m, &tmpVal)
				s.nospill |= regMask(1) << tmpReg
			}

			// Now that all args are in regs, we're ready to issue the value itself.
			// Before we pick a register for the output value, allow input registers
			// to be deallocated. We do this here so that the output can use the
			// same register as a dying input.
			if !opcodeTable[v.Op].resultNotInArgs {
				s.tmpused = s.nospill
				s.nospill = 0
				s.advanceUses(v) // frees any registers holding args that are no longer live
			}

			// Dump any registers which will be clobbered
			if s.doClobber && v.Op.IsCall() {
				// clobber registers that are marked as clobber in regmask, but
				// don't clobber inputs.
				s.clobberRegs(regspec.clobbers &^ s.tmpused &^ s.nospill)
			}
			s.freeRegs(regspec.clobbers)
			s.tmpused |= regspec.clobbers

			// Pick registers for outputs.
			{
				outRegs := noRegisters // TODO if this is costly, hoist and clear incrementally below.
				maxOutIdx := -1
				var used regMask
				if tmpReg != noRegister {
					// Ensure output registers are distinct from the temporary register.
					// (Not all instructions need that distinct part, but it is conservative.)
					used |= regMask(1) << tmpReg
				}
				for _, out := range regspec.outputs {
					if out.regs == 0 {
						continue
					}
					mask := out.regs & s.allocatable &^ used
					if mask == 0 {
						s.f.Fatalf("can't find any output register %s", v.LongString())
					}
					if opcodeTable[v.Op].resultInArg0 && out.idx == 0 {
						if !opcodeTable[v.Op].commutative {
							// Output must use the same register as input 0.
							r := register(s.f.getHome(args[0].ID).(*Register).num)
							if mask>>r&1 == 0 {
								s.f.Fatalf("resultInArg0 value's input %v cannot be an output of %s", s.f.getHome(args[0].ID).(*Register), v.LongString())
							}
							mask = regMask(1) << r
						} else {
							// Output must use the same register as input 0 or 1.
							r0 := register(s.f.getHome(args[0].ID).(*Register).num)
							r1 := register(s.f.getHome(args[1].ID).(*Register).num)
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
					if out.idx == 0 { // desired registers only apply to the first element of a tuple result
						for _, r := range dinfo[idx].out {
							if r != noRegister && (mask&^s.used)>>r&1 != 0 {
								// Desired register is allowed and unused.
								mask = regMask(1) << r
								break
							}
						}
					}
					// Avoid registers we're saving for other values.
					if mask&^desired.avoid&^s.nospill&^s.used != 0 {
						mask &^= desired.avoid
					}
					r := s.allocReg(mask, v)
					if out.idx > maxOutIdx {
						maxOutIdx = out.idx
					}
					outRegs[out.idx] = r
					used |= regMask(1) << r
					s.tmpused |= regMask(1) << r
				}
				// Record register choices
				if v.Type.IsTuple() {
					var outLocs LocPair
					if r := outRegs[0]; r != noRegister {
						outLocs[0] = &s.registers[r]
					}
					if r := outRegs[1]; r != noRegister {
						outLocs[1] = &s.registers[r]
					}
					s.f.setHome(v, outLocs)
					// Note that subsequent SelectX instructions will do the assignReg calls.
				} else if v.Type.IsResults() {
					// preallocate outLocs to the right size, which is maxOutIdx+1
					outLocs := make(LocResults, maxOutIdx+1, maxOutIdx+1)
					for i := 0; i <= maxOutIdx; i++ {
						if r := outRegs[i]; r != noRegister {
							outLocs[i] = &s.registers[r]
						}
					}
					s.f.setHome(v, outLocs)
				} else {
					if r := outRegs[0]; r != noRegister {
						s.assignReg(r, v, v)
					}
				}
				if tmpReg != noRegister {
					// Remember the temp register allocation, if any.
					if s.f.tempRegs == nil {
						s.f.tempRegs = map[ID]*Register{}
					}
					s.f.tempRegs[v.ID] = &s.registers[tmpReg]
				}
			}

			// deallocate dead args, if we have not done so
			if opcodeTable[v.Op].resultNotInArgs {
				s.nospill = 0
				s.advanceUses(v) // frees any registers holding args that are no longer live
			}
			s.tmpused = 0

			// Issue the Value itself.
			for i, a := range args {
				v.SetArg(i, a) // use register version of arguments
			}
			b.Values = append(b.Values, v)
		}

		// Copy the control values - we need this so we can reduce the
		// uses property of these values later.
		controls := append(make([]*Value, 0, 2), b.ControlValues()...)

		// Load control values into registers.
		for i, v := range b.ControlValues() {
			if !s.values[v.ID].needReg {
				continue
			}
			if s.f.pass.debug > regDebug {
				fmt.Printf("  processing control %s\n", v.LongString())
			}
			// We assume that a control input can be passed in any
			// type-compatible register. If this turns out not to be true,
			// we'll need to introduce a regspec for a block's control value.
			b.ReplaceControl(i, s.allocValToReg(v, s.compatRegs(v.Type), false, b.Pos))
		}

		// Reduce the uses of the control values once registers have been loaded.
		// This loop is equivalent to the advanceUses method.
		for _, v := range controls {
			vi := &s.values[v.ID]
			if !vi.needReg {
				continue
			}
			// Remove this use from the uses list.
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
			if s.f.Config.hasGReg && s.regs[s.GReg].v != nil {
				s.freeReg(s.GReg) // Spill value in G register before any merge.
			}
			// For this to be worthwhile, the loop must have no calls in it.
			top := b.Succs[0].b
			loop := s.loopnest.b2l[top.ID]
			if loop == nil || loop.header != top || loop.containsUnavoidableCall {
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
				if vi.rematerializeable {
					continue
				}
				v := s.orig[vid]
				m := s.compatRegs(v.Type) &^ s.used
				// Used desired register if available.
			outerloop:
				for _, e := range desired.entries {
					if e.ID != v.ID {
						continue
					}
					for _, r := range e.regs {
						if r != noRegister && m>>r&1 != 0 {
							m = regMask(1) << r
							break outerloop
						}
					}
				}
				if m&^desired.avoid != 0 {
					m &^= desired.avoid
				}
				if m != 0 {
					s.allocValToReg(v, m, false, b.Pos)
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

		if checkEnabled {
			regValLiveSet.clear()
			for _, x := range s.live[b.ID] {
				regValLiveSet.add(x.ID)
			}
			for r := register(0); r < s.numRegs; r++ {
				v := s.regs[r].v
				if v == nil {
					continue
				}
				if !regValLiveSet.contains(v.ID) {
					s.f.Fatalf("val %s is in reg but not live at end of %s", v, b)
				}
			}
		}

		// If a value is live at the end of the block and
		// isn't in a register, generate a use for the spill location.
		// We need to remember this information so that
		// the liveness analysis in stackalloc is correct.
		for _, e := range s.live[b.ID] {
			vi := &s.values[e.ID]
			if vi.regs != 0 {
				// in a register, we'll use that source for the merge.
				continue
			}
			if vi.rematerializeable {
				// we'll rematerialize during the merge.
				continue
			}
			if s.f.pass.debug > regDebug {
				fmt.Printf("live-at-end spill for %s at %s\n", s.orig[e.ID], b)
			}
			spill := s.makeSpill(s.orig[e.ID], b)
			s.spillLive[b.ID] = append(s.spillLive[b.ID], spill.ID)
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

		// allocReg may have dropped registers from startRegsMask that
		// aren't actually needed in startRegs. Synchronize back to
		// startRegs.
		//
		// This must be done before placing spills, which will look at
		// startRegs to decide if a block is a valid block for a spill.
		if c := countRegs(s.startRegsMask); c != len(s.startRegs[b.ID]) {
			regs := make([]startReg, 0, c)
			for _, sr := range s.startRegs[b.ID] {
				if s.startRegsMask&(regMask(1)<<sr.r) == 0 {
					continue
				}
				regs = append(regs, sr)
			}
			s.startRegs[b.ID] = regs
		}
	}

	// Decide where the spills we generated will go.
	s.placeSpills()

	// Anything that didn't get a register gets a stack location here.
	// (StoreReg, stack-based phis, inputs, ...)
	stacklive := stackalloc(s.f, s.spillLive)

	// Fix up all merge edges.
	s.shuffle(stacklive)

	// Erase any copies we never used.
	// Also, an unused copy might be the only use of another copy,
	// so continue erasing until we reach a fixed point.
	for {
		progress := false
		for c, used := range s.copies {
			if !used && c.Uses == 0 {
				if s.f.pass.debug > regDebug {
					fmt.Printf("delete copied value %s\n", c.LongString())
				}
				c.resetArgs()
				f.freeValue(c)
				delete(s.copies, c)
				progress = true
			}
		}
		if !progress {
			break
		}
	}

	for _, b := range s.visitOrder {
		i := 0
		for _, v := range b.Values {
			if v.Op == OpInvalid {
				continue
			}
			b.Values[i] = v
			i++
		}
		b.Values = b.Values[:i]
	}
}

func (s *regAllocState) placeSpills() {
	mustBeFirst := func(op Op) bool {
		return op.isLoweredGetClosurePtr() || op == OpPhi || op == OpArgIntReg || op == OpArgFloatReg
	}

	// Start maps block IDs to the list of spills
	// that go at the start of the block (but after any phis).
	start := map[ID][]*Value{}
	// After maps value IDs to the list of spills
	// that go immediately after that value ID.
	after := map[ID][]*Value{}

	for i := range s.values {
		vi := s.values[i]
		spill := vi.spill
		if spill == nil {
			continue
		}
		if spill.Block != nil {
			// Some spills are already fully set up,
			// like OpArgs and stack-based phis.
			continue
		}
		v := s.orig[i]

		// Walk down the dominator tree looking for a good place to
		// put the spill of v.  At the start "best" is the best place
		// we have found so far.
		// TODO: find a way to make this O(1) without arbitrary cutoffs.
		if v == nil {
			panic(fmt.Errorf("nil v, s.orig[%d], vi = %v, spill = %s", i, vi, spill.LongString()))
		}
		best := v.Block
		bestArg := v
		var bestDepth int16
		if l := s.loopnest.b2l[best.ID]; l != nil {
			bestDepth = l.depth
		}
		b := best
		const maxSpillSearch = 100
		for i := 0; i < maxSpillSearch; i++ {
			// Find the child of b in the dominator tree which
			// dominates all restores.
			p := b
			b = nil
			for c := s.sdom.Child(p); c != nil && i < maxSpillSearch; c, i = s.sdom.Sibling(c), i+1 {
				if s.sdom[c.ID].entry <= vi.restoreMin && s.sdom[c.ID].exit >= vi.restoreMax {
					// c also dominates all restores.  Walk down into c.
					b = c
					break
				}
			}
			if b == nil {
				// Ran out of blocks which dominate all restores.
				break
			}

			var depth int16
			if l := s.loopnest.b2l[b.ID]; l != nil {
				depth = l.depth
			}
			if depth > bestDepth {
				// Don't push the spill into a deeper loop.
				continue
			}

			// If v is in a register at the start of b, we can
			// place the spill here (after the phis).
			if len(b.Preds) == 1 {
				for _, e := range s.endRegs[b.Preds[0].b.ID] {
					if e.v == v {
						// Found a better spot for the spill.
						best = b
						bestArg = e.c
						bestDepth = depth
						break
					}
				}
			} else {
				for _, e := range s.startRegs[b.ID] {
					if e.v == v {
						// Found a better spot for the spill.
						best = b
						bestArg = e.c
						bestDepth = depth
						break
					}
				}
			}
		}

		// Put the spill in the best block we found.
		spill.Block = best
		spill.AddArg(bestArg)
		if best == v.Block && !mustBeFirst(v.Op) {
			// Place immediately after v.
			after[v.ID] = append(after[v.ID], spill)
		} else {
			// Place at the start of best block.
			start[best.ID] = append(start[best.ID], spill)
		}
	}

	// Insert spill instructions into the block schedules.
	var oldSched []*Value
	for _, b := range s.visitOrder {
		nfirst := 0
		for _, v := range b.Values {
			if !mustBeFirst(v.Op) {
				break
			}
			nfirst++
		}
		oldSched = append(oldSched[:0], b.Values[nfirst:]...)
		b.Values = b.Values[:nfirst]
		b.Values = append(b.Values, start[b.ID]...)
		for _, v := range oldSched {
			b.Values = append(b.Values, v)
			b.Values = append(b.Values, after[v.ID]...)
		}
	}
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

	for _, b := range s.visitOrder {
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

	if s.f.pass.debug > regDebug {
		fmt.Printf("post shuffle %s\n", s.f.Name)
		fmt.Println(s.f.String())
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

	usedRegs              regMask // registers currently holding something
	uniqueRegs            regMask // registers holding the only copy of a value
	finalRegs             regMask // registers holding final target
	rematerializeableRegs regMask // registers that hold rematerializeable values
}

type contentRecord struct {
	vid   ID       // pre-regalloc value
	c     *Value   // cached value
	final bool     // this is a satisfied destination
	pos   src.XPos // source position of use of the value
}

type dstRecord struct {
	loc    Location // register or stack slot
	vid    ID       // pre-regalloc value it should contain
	splice **Value  // place to store reference to the generating instruction
	pos    src.XPos // source position of use of this location
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
	e.rematerializeableRegs = 0

	// Live registers can be sources.
	for _, x := range srcReg {
		e.set(&e.s.registers[x.r], x.v.ID, x.c, false, src.NoXPos) // don't care the position of the source
	}
	// So can all of the spill locations.
	for _, spillID := range stacklive {
		v := e.s.orig[spillID]
		spill := e.s.values[v.ID].spill
		if !e.s.sdom.IsAncestorEq(spill.Block, e.p) {
			// Spills were placed that only dominate the uses found
			// during the first regalloc pass. The edge fixup code
			// can't use a spill location if the spill doesn't dominate
			// the edge.
			// We are guaranteed that if the spill doesn't dominate this edge,
			// then the value is available in a register (because we called
			// makeSpill for every value not in a register at the start
			// of an edge).
			continue
		}
		e.set(e.s.f.getHome(spillID), v.ID, spill, false, src.NoXPos) // don't care the position of the source
	}

	// Figure out all the destinations we need.
	dsts := e.destinations[:0]
	for _, x := range dstReg {
		dsts = append(dsts, dstRecord{&e.s.registers[x.r], x.v.ID, nil, x.pos})
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
		dsts = append(dsts, dstRecord{loc, v.Args[idx].ID, &v.Args[idx], v.Pos})
	}
	e.destinations = dsts

	if e.s.f.pass.debug > regDebug {
		for _, vid := range e.cachedVals {
			a := e.cache[vid]
			for _, c := range a {
				fmt.Printf("src %s: v%d cache=%s\n", e.s.f.getHome(c.ID), vid, c)
			}
		}
		for _, d := range e.destinations {
			fmt.Printf("dst %s: v%d\n", d.loc, d.vid)
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
			if !e.processDest(d.loc, d.vid, d.splice, d.pos) {
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
		d := dsts[0]
		loc := d.loc
		vid := e.contents[loc].vid
		c := e.contents[loc].c
		r := e.findRegFor(c.Type)
		if e.s.f.pass.debug > regDebug {
			fmt.Printf("breaking cycle with v%d in %s:%s\n", vid, loc, c)
		}
		e.erase(r)
		pos := d.pos.WithNotStmt()
		if _, isReg := loc.(*Register); isReg {
			c = e.p.NewValue1(pos, OpCopy, c.Type, c)
		} else {
			c = e.p.NewValue1(pos, OpLoadReg, c.Type, c)
		}
		e.set(r, vid, c, false, pos)
		if c.Op == OpLoadReg && e.s.isGReg(register(r.(*Register).num)) {
			e.s.f.Fatalf("process.OpLoadReg targeting g: " + c.LongString())
		}
	}
}

// processDest generates code to put value vid into location loc. Returns true
// if progress was made.
func (e *edgeState) processDest(loc Location, vid ID, splice **Value, pos src.XPos) bool {
	pos = pos.WithNotStmt()
	occupant := e.contents[loc]
	if occupant.vid == vid {
		// Value is already in the correct place.
		e.contents[loc] = contentRecord{vid, occupant.c, true, pos}
		if splice != nil {
			(*splice).Uses--
			*splice = occupant.c
			occupant.c.Uses++
		}
		// Note: if splice==nil then c will appear dead. This is
		// non-SSA formed code, so be careful after this pass not to run
		// deadcode elimination.
		if _, ok := e.s.copies[occupant.c]; ok {
			// The copy at occupant.c was used to avoid spill.
			e.s.copies[occupant.c] = true
		}
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
		fmt.Printf("moving v%d to %s\n", vid, loc)
		fmt.Printf("sources of v%d:", vid)
	}
	for _, w := range e.cache[vid] {
		h := e.s.f.getHome(w.ID)
		if e.s.f.pass.debug > regDebug {
			fmt.Printf(" %s:%s", h, w)
		}
		_, isreg := h.(*Register)
		if src == nil || isreg {
			c = w
			src = h
		}
	}
	if e.s.f.pass.debug > regDebug {
		if src != nil {
			fmt.Printf(" [use %s]\n", src)
		} else {
			fmt.Printf(" [no source]\n")
		}
	}
	_, dstReg := loc.(*Register)

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
	var x *Value
	if c == nil || e.s.values[vid].rematerializeable {
		if !e.s.values[vid].rematerializeable {
			e.s.f.Fatalf("can't find source for %s->%s: %s\n", e.p, e.b, v.LongString())
		}
		if dstReg {
			x = v.copyInto(e.p)
		} else {
			// Rematerialize into stack slot. Need a free
			// register to accomplish this.
			r := e.findRegFor(v.Type)
			e.erase(r)
			x = v.copyIntoWithXPos(e.p, pos)
			e.set(r, vid, x, false, pos)
			// Make sure we spill with the size of the slot, not the
			// size of x (which might be wider due to our dropping
			// of narrowing conversions).
			x = e.p.NewValue1(pos, OpStoreReg, loc.(LocalSlot).Type, x)
		}
	} else {
		// Emit move from src to dst.
		_, srcReg := src.(*Register)
		if srcReg {
			if dstReg {
				x = e.p.NewValue1(pos, OpCopy, c.Type, c)
			} else {
				x = e.p.NewValue1(pos, OpStoreReg, loc.(LocalSlot).Type, c)
			}
		} else {
			if dstReg {
				x = e.p.NewValue1(pos, OpLoadReg, c.Type, c)
			} else {
				// mem->mem. Use temp register.
				r := e.findRegFor(c.Type)
				e.erase(r)
				t := e.p.NewValue1(pos, OpLoadReg, c.Type, c)
				e.set(r, vid, t, false, pos)
				x = e.p.NewValue1(pos, OpStoreReg, loc.(LocalSlot).Type, t)
			}
		}
	}
	e.set(loc, vid, x, true, pos)
	if x.Op == OpLoadReg && e.s.isGReg(register(loc.(*Register).num)) {
		e.s.f.Fatalf("processDest.OpLoadReg targeting g: " + x.LongString())
	}
	if splice != nil {
		(*splice).Uses--
		*splice = x
		x.Uses++
	}
	return true
}

// set changes the contents of location loc to hold the given value and its cached representative.
func (e *edgeState) set(loc Location, vid ID, c *Value, final bool, pos src.XPos) {
	e.s.f.setHome(c, loc)
	e.contents[loc] = contentRecord{vid, c, final, pos}
	a := e.cache[vid]
	if len(a) == 0 {
		e.cachedVals = append(e.cachedVals, vid)
	}
	a = append(a, c)
	e.cache[vid] = a
	if r, ok := loc.(*Register); ok {
		if e.usedRegs&(regMask(1)<<uint(r.num)) != 0 {
			e.s.f.Fatalf("%v is already set (v%d/%v)", r, vid, c)
		}
		e.usedRegs |= regMask(1) << uint(r.num)
		if final {
			e.finalRegs |= regMask(1) << uint(r.num)
		}
		if len(a) == 1 {
			e.uniqueRegs |= regMask(1) << uint(r.num)
		}
		if len(a) == 2 {
			if t, ok := e.s.f.getHome(a[0].ID).(*Register); ok {
				e.uniqueRegs &^= regMask(1) << uint(t.num)
			}
		}
		if e.s.values[vid].rematerializeable {
			e.rematerializeableRegs |= regMask(1) << uint(r.num)
		}
	}
	if e.s.f.pass.debug > regDebug {
		fmt.Printf("%s\n", c.LongString())
		fmt.Printf("v%d now available in %s:%s\n", vid, loc, c)
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
		e.extra = append(e.extra, dstRecord{loc, cr.vid, nil, cr.pos})
	}

	// Remove c from the list of cached values.
	a := e.cache[vid]
	for i, c := range a {
		if e.s.f.getHome(c.ID) == loc {
			if e.s.f.pass.debug > regDebug {
				fmt.Printf("v%d no longer available in %s:%s\n", vid, loc, c)
			}
			a[i], a = a[len(a)-1], a[:len(a)-1]
			break
		}
	}
	e.cache[vid] = a

	// Update register masks.
	if r, ok := loc.(*Register); ok {
		e.usedRegs &^= regMask(1) << uint(r.num)
		if cr.final {
			e.finalRegs &^= regMask(1) << uint(r.num)
		}
		e.rematerializeableRegs &^= regMask(1) << uint(r.num)
	}
	if len(a) == 1 {
		if r, ok := e.s.f.getHome(a[0].ID).(*Register); ok {
			e.uniqueRegs |= regMask(1) << uint(r.num)
		}
	}
}

// findRegFor finds a register we can use to make a temp copy of type typ.
func (e *edgeState) findRegFor(typ *types.Type) Location {
	// Which registers are possibilities.
	types := &e.s.f.Config.Types
	m := e.s.compatRegs(typ)

	// Pick a register. In priority order:
	// 1) an unused register
	// 2) a non-unique register not holding a final value
	// 3) a non-unique register
	// 4) a register holding a rematerializeable value
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
	x = m & e.rematerializeableRegs
	if x != 0 {
		return &e.s.registers[pickReg(x)]
	}

	// No register is available.
	// Pick a register to spill.
	for _, vid := range e.cachedVals {
		a := e.cache[vid]
		for _, c := range a {
			if r, ok := e.s.f.getHome(c.ID).(*Register); ok && m>>uint(r.num)&1 != 0 {
				if !c.rematerializeable() {
					x := e.p.NewValue1(c.Pos, OpStoreReg, c.Type, c)
					// Allocate a temp location to spill a register to.
					// The type of the slot is immaterial - it will not be live across
					// any safepoint. Just use a type big enough to hold any register.
					t := LocalSlot{N: e.s.f.NewLocal(c.Pos, types.Int64), Type: types.Int64}
					// TODO: reuse these slots. They'll need to be erased first.
					e.set(t, vid, x, false, c.Pos)
					if e.s.f.pass.debug > regDebug {
						fmt.Printf("  SPILL %s->%s %s\n", r, t, x.LongString())
					}
				}
				// r will now be overwritten by the caller. At some point
				// later, the newly saved value will be moved back to its
				// final destination in processDest.
				return r
			}
		}
	}

	fmt.Printf("m:%d unique:%d final:%d rematerializable:%d\n", m, e.uniqueRegs, e.finalRegs, e.rematerializeableRegs)
	for _, vid := range e.cachedVals {
		a := e.cache[vid]
		for _, c := range a {
			fmt.Printf("v%d: %s %s\n", vid, c, e.s.f.getHome(c.ID))
		}
	}
	e.s.f.Fatalf("can't find empty register on edge %s->%s", e.p, e.b)
	return nil
}

// rematerializeable reports whether the register allocator should recompute
// a value instead of spilling/restoring it.
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
	ID   ID       // ID of value
	dist int32    // # of instructions before next use
	pos  src.XPos // source position of next use
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

	live := f.newSparseMapPos(f.NumValues())
	defer f.retSparseMapPos(live)
	t := f.newSparseMapPos(f.NumValues())
	defer f.retSparseMapPos(t)

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
	po := f.postorder()
	s.loopnest = f.loopnest()
	s.loopnest.calculateDepths()
	for {
		changed := false

		for _, b := range po {
			// Start with known live values at the end of the block.
			// Add len(b.Values) to adjust from end-of-block distance
			// to beginning-of-block distance.
			live.clear()
			for _, e := range s.live[b.ID] {
				live.set(e.ID, e.dist+int32(len(b.Values)), e.pos)
			}

			// Mark control values as live
			for _, c := range b.ControlValues() {
				if s.values[c.ID].needReg {
					live.set(c.ID, int32(len(b.Values)), b.Pos)
				}
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
				if opcodeTable[v.Op].call {
					c := live.contents()
					for i := range c {
						c[i].val += unlikelyDistance
					}
				}
				for _, a := range v.Args {
					if s.values[a.ID].needReg {
						live.set(a.ID, int32(i), v.Pos)
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
				regspec := s.regspec(v)
				// Cancel desired registers if they get clobbered.
				desired.clobber(regspec.clobbers)
				// Update desired registers if there are any fixed register inputs.
				for _, j := range regspec.inputs {
					if countRegs(j.regs) != 1 {
						continue
					}
					desired.clobber(j.regs)
					desired.add(v.Args[j.idx].ID, pickReg(j.regs))
				}
				// Set desired register of input 0 if this is a 2-operand instruction.
				if opcodeTable[v.Op].resultInArg0 || v.Op == OpAMD64ADDQconst || v.Op == OpAMD64ADDLconst || v.Op == OpSelect0 {
					// ADDQconst is added here because we want to treat it as resultInArg0 for
					// the purposes of desired registers, even though it is not an absolute requirement.
					// This is because we'd rather implement it as ADDQ instead of LEAQ.
					// Same for ADDLconst
					// Select0 is added here to propagate the desired register to the tuple-generating instruction.
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
					t.set(e.ID, e.dist, e.pos)
				}
				update := false

				// Add new live values from scanning this block.
				for _, e := range live.contents() {
					d := e.val + delta
					if !t.contains(e.key) || d < t.get(e.key) {
						update = true
						t.set(e.key, d, e.pos)
					}
				}
				// Also add the correct arg from the saved phi values.
				// All phis are at distance delta (we consider them
				// simultaneously happening at the start of the block).
				for _, v := range phis {
					id := v.Args[i].ID
					if s.values[id].needReg && (!t.contains(id) || delta < t.get(id)) {
						update = true
						t.set(id, delta, v.Pos)
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
					l = append(l, liveInfo{e.key, e.val, e.pos})
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
				fmt.Printf(" v%d(%d)", x.ID, x.dist)
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
						fmt.Print(&s.registers[r])
						first = false
					}
					fmt.Printf("]")
				}
			}
			if avoid := s.desired[b.ID].avoid; avoid != 0 {
				fmt.Printf(" avoid=%v", s.RegMaskString(avoid))
			}
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
	// For opcodes that return tuples, we track desired registers only
	// for the first element of the tuple.
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

func min32(x, y int32) int32 {
	if x < y {
		return x
	}
	return y
}
func max32(x, y int32) int32 {
	if x > y {
		return x
	}
	return y
}
