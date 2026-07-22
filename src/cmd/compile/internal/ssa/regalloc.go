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
	"cmd/compile/internal/ssa/ssabase"
	"cmd/compile/internal/ssa/ssacore"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"cmd/internal/sys"
	"cmp"
	"fmt"
	"internal/buildcfg"
	"math"
	"math/bits"
	"slices"
	"unsafe"
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
func regalloc(f *ssacore.Func) {
	var s regAllocState
	s.init(f)
	s.regalloc(f)
	s.close()
}

const noRegister ssaop.Register = 255

// For bulk initializing
var noRegisters [32]ssaop.Register = [32]ssaop.Register{
	noRegister, noRegister, noRegister, noRegister, noRegister, noRegister, noRegister, noRegister,
	noRegister, noRegister, noRegister, noRegister, noRegister, noRegister, noRegister, noRegister,
	noRegister, noRegister, noRegister, noRegister, noRegister, noRegister, noRegister, noRegister,
	noRegister, noRegister, noRegister, noRegister, noRegister, noRegister, noRegister, noRegister,
}

func (s *regAllocState) RegMaskString(m ssaop.RegMask) string {
	str := ""
	for r := ssaop.Register(0); !m.Empty(); r++ {
		if !m.HasReg(r) {
			continue
		}
		m = m.RemoveReg(r)
		if str != "" {
			str += " "
		}
		str += s.registers[r].String()
	}
	return str
}

// countRegs returns the number of set bits in the register mask.
func countRegs(r ssaop.RegMask) int {
	return bits.OnesCount64(r.V1) + bits.OnesCount64(r.V2)
}

// pickReg picks a register from the register mask.
func (s *regAllocState) pickReg(rm ssaop.RegMask) ssaop.Register {
	if s.f.Config.Ctxt.Arch.Arch == sys.ArchRISCV64 {
		// Prefer x8-x15 and f8-f15 to enable increased use of compressed instructions.
		riscv64CompressedMask := rm.Intersect(ssaop.RegMask{V1: 0x0000ff000000ff00})
		if !riscv64CompressedMask.Empty() {
			rm = riscv64CompressedMask
		}
	}
	return rm.PickReg()
}

type regState struct {
	v *ssacore.Value // Original (preregalloc) Value stored in this register.
	c *ssacore.Value // A Value equal to v which is currently in a register.  Might be v or a copy of it.
	// If a register is unused, v==c==nil
}

type regAllocState struct {
	f *ssacore.Func

	sdom        ssacore.SparseTree
	registers   []ssabase.Register
	numRegs     ssaop.Register
	SPReg       ssaop.Register
	SBReg       ssaop.Register
	GReg        ssaop.Register
	ZeroIntReg  ssaop.Register
	allocatable ssaop.RegMask

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
	values []ssacore.ValState

	// ID of SP, SB values
	sp, sb ssacore.ID

	// For each Value, map from its value ID back to the
	// preregalloc Value it was derived from.
	orig []*ssacore.Value

	// current state of each register.
	// Includes only registers in allocatable.
	regs []regState

	// registers that contain values which can't be kicked out
	nospill ssaop.RegMask

	// mask of registers currently in use
	used ssaop.RegMask

	// mask of registers used since the start of the current block
	usedSinceBlockStart ssaop.RegMask

	// mask of registers used in the current instruction
	tmpused ssaop.RegMask

	// current block we're working on
	curBlock *ssacore.Block

	// cache of use records
	freeUseRecords *ssacore.Use

	// endRegs[blockid] is the register state at the end of each block.
	// encoded as a set of endReg records.
	endRegs [][]endReg

	// startRegs[blockid] is the register state at the start of merge blocks.
	// saved state does not include the state of phi ops in the block.
	startRegs [][]startReg

	// startRegsMask is a mask of the registers in startRegs[curBlock.ID].
	// Registers dropped from startRegsMask are later synchronoized back to
	// startRegs by dropping from there as well.
	startRegsMask ssaop.RegMask

	// spillLive[blockid] is the set of live spills at the end of each block
	spillLive [][]ssacore.ID

	// a set of copies we generated to move things around, and
	// whether it is used in shuffle. Unused copies will be deleted.
	copies map[*ssacore.Value]bool

	loopnest *ssacore.LoopNest

	// choose a good order in which to visit blocks for allocation purposes.
	visitOrder []*ssacore.Block

	// blockOrder[b.ID] corresponds to the index of block b in visitOrder.
	blockOrder []int32

	// whether to insert instructions that clobber dead registers at call sites
	doClobber bool

	// For each instruction index in a basic block, the index of the next call
	// at or after that instruction index.
	// If there is no next call, returns maxInt32.
	// nextCall for a call instruction points to itself.
	// (Indexes and results are pre-regalloc.)
	nextCall []int32

	// Index of the instruction we're currently working on.
	// Index is expressed in terms of the pre-regalloc b.Values list.
	curIdx int
}

type endReg struct {
	r ssaop.Register
	v *ssacore.Value // pre-regalloc value held in this register (TODO: can we use ID here?)
	c *ssacore.Value // cached version of the value
}

type startReg struct {
	r   ssaop.Register
	v   *ssacore.Value // pre-regalloc value needed in this register
	c   *ssacore.Value // cached version of the value
	pos src.XPos       // source position of use of this register
}

// freeReg frees up register r. Any current user of r is kicked out.
func (s *regAllocState) freeReg(r ssaop.Register) {
	if !s.allocatable.HasReg(r) && !s.isGReg(r) {
		return
	}
	v := s.regs[r].v
	if v == nil {
		s.f.Fatalf("tried to free an already free register %d\n", r)
	}

	// Mark r as unused.
	if s.f.Pass.Debug > ssacore.RegDebug {
		fmt.Printf("freeReg %s (dump %s/%s)\n", &s.registers[r], v, s.regs[r].c)
	}
	s.regs[r] = regState{}
	s.values[v.ID].Regs = s.values[v.ID].Regs.RemoveReg(r)
	s.used = s.used.RemoveReg(r)
}

// freeRegs frees up all registers listed in m.
func (s *regAllocState) freeRegs(m ssaop.RegMask) {
	for !m.Intersect(s.used).Empty() {
		s.freeReg(s.pickReg(m.Intersect(s.used)))
	}
}

// clobberRegs inserts instructions that clobber registers listed in m.
func (s *regAllocState) clobberRegs(m ssaop.RegMask) {
	m = m.Intersect(s.allocatable.Intersect(s.f.Config.GpRegMask)) // only integer register can contain pointers, only clobber them
	for !m.Empty() {
		r := s.pickReg(m)
		m = m.RemoveReg(r)
		x := s.curBlock.NewValue0(src.NoXPos, ssaop.OpClobberReg, types.TypeVoid)
		s.f.SetHome(x, &s.registers[r])
	}
}

// setOrig records that c's original value is the same as
// v's original value.
func (s *regAllocState) setOrig(c *ssacore.Value, v *ssacore.Value) {
	if int(c.ID) >= cap(s.orig) {
		x := s.f.Cache.AllocValueSlice(int(c.ID) + 1)
		copy(x, s.orig)
		s.f.Cache.FreeValueSlice(s.orig)
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
func (s *regAllocState) assignReg(r ssaop.Register, v *ssacore.Value, c *ssacore.Value) {
	if s.f.Pass.Debug > ssacore.RegDebug {
		fmt.Printf("assignReg %s %s/%s\n", &s.registers[r], v, c)
	}
	// Allocate v to r.
	s.values[v.ID].Regs = s.values[v.ID].Regs.AddReg(r)
	s.f.SetHome(c, &s.registers[r])

	// Allocate r to v.
	if !s.allocatable.HasReg(r) && !s.isGReg(r) {
		return
	}
	if s.regs[r].v != nil {
		s.f.Fatalf("tried to assign register %d to %s/%s but it is already used by %s", r, v, c, s.regs[r].v)
	}
	s.regs[r] = regState{v, c}
	s.used = s.used.AddReg(r)
}

// allocReg chooses a register from the set of registers in mask.
// If there is no unused register, a Value will be kicked out of
// a register to make room.
func (s *regAllocState) allocReg(mask ssaop.RegMask, v *ssacore.Value) ssaop.Register {
	if v.OnWasmStack {
		return noRegister
	}

	mask = mask.Intersect(s.allocatable)
	mask = mask.Minus(s.nospill)
	if mask.Empty() {
		s.f.Fatalf("no register available for %s", v.LongString())
	}

	// Pick an unused register if one is available.
	if !mask.Minus(s.used).Empty() {
		r := s.pickReg(mask.Minus(s.used))
		s.usedSinceBlockStart = s.usedSinceBlockStart.AddReg(r)
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
	var r ssaop.Register
	maxuse := int32(-1)
	for t := ssaop.Register(0); t < s.numRegs; t++ {
		if !mask.HasReg(t) {
			continue
		}
		v := s.regs[t].v
		if n := s.values[v.ID].Uses.Dist; n > maxuse {
			// v's next use is farther in the future than any value
			// we've seen so far. A new best spill candidate.
			r = t
			maxuse = n
		}
	}
	if maxuse == -1 {
		s.f.Fatalf("couldn't find register to spill")
	}

	if s.f.Config.Ctxt.Arch.Arch == sys.ArchWasm {
		// TODO(neelance): In theory this should never happen, because all wasm registers are equal.
		// So if there is still a free register, the allocation should have picked that one in the first place instead of
		// trying to kick some other value out. In practice, this case does happen and it breaks the stack optimization.
		s.freeReg(r)
		return r
	}

	// Try to move it around before kicking out, if there is a free register.
	// We generate a Copy and record it. It will be deleted if never used.
	v2 := s.regs[r].v
	m := s.compatRegs(v2.Type).Minus(s.used).Minus(s.tmpused).RemoveReg(r)
	if !m.Empty() && !s.values[v2.ID].Rematerializeable && countRegs(s.values[v2.ID].Regs) == 1 {
		s.usedSinceBlockStart = s.usedSinceBlockStart.AddReg(r)
		r2 := s.pickReg(m)
		c := s.curBlock.NewValue1(v2.Pos, ssaop.OpCopy, v2.Type, s.regs[r].c)
		s.copies[c] = false
		if s.f.Pass.Debug > ssacore.RegDebug {
			fmt.Printf("copy %s to %s : %s\n", v2, c, &s.registers[r2])
		}
		s.setOrig(c, v2)
		s.assignReg(r2, v2, c)
	}

	// If the evicted register isn't used between the start of the block
	// and now then there is no reason to even request it on entry. We can
	// drop from startRegs in that case.
	if !s.usedSinceBlockStart.HasReg(r) {
		if s.startRegsMask.HasReg(r) {
			if s.f.Pass.Debug > ssacore.RegDebug {
				fmt.Printf("dropped from startRegs: %s\n", &s.registers[r])
			}
			s.startRegsMask = s.startRegsMask.RemoveReg(r)
		}
	}

	s.freeReg(r)
	s.usedSinceBlockStart = s.usedSinceBlockStart.AddReg(r)
	return r
}

// makeSpill returns a Value which represents the spilled value of v.
// b is the block in which the spill is used.
func (s *regAllocState) makeSpill(v *ssacore.Value, b *ssacore.Block) *ssacore.Value {
	vi := &s.values[v.ID]
	if vi.Spill != nil {
		// Final block not known - keep track of subtree where restores reside.
		vi.RestoreMin = min(vi.RestoreMin, s.sdom[b.ID].Entry)
		vi.RestoreMax = max(vi.RestoreMax, s.sdom[b.ID].Exit)
		return vi.Spill
	}
	// Make a spill for v. We don't know where we want
	// to put it yet, so we leave it blockless for now.
	spill := s.f.NewValueNoBlock(ssaop.OpStoreReg, v.Type, v.Pos)
	// We also don't know what the spill's arg will be.
	// Leave it argless for now.
	s.setOrig(spill, v)
	vi.Spill = spill
	vi.RestoreMin = s.sdom[b.ID].Entry
	vi.RestoreMax = s.sdom[b.ID].Exit
	return spill
}

// allocValToReg allocates v to a register selected from regMask and
// returns the register copy of v. Any previous user is kicked out and spilled
// (if necessary). Load code is added at the current pc. If nospill is set the
// allocated register is marked nospill so the assignment cannot be
// undone until the caller allows it by clearing nospill. Returns a
// *Value which is either v or a copy of v allocated to the chosen register.
func (s *regAllocState) allocValToReg(v *ssacore.Value, mask ssaop.RegMask, nospill bool, pos src.XPos) *ssacore.Value {
	if s.f.Config.Ctxt.Arch.Arch == sys.ArchWasm && v.Rematerializeable() {
		c := v.CopyIntoWithXPos(s.curBlock, pos)
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
	if !mask.Intersect(vi.Regs).Empty() {
		mask = mask.Intersect(vi.Regs)
		r := s.pickReg(mask)
		if mask.HasReg(s.SPReg) {
			// Prefer the stack pointer if it is allowed.
			// (Needed because the op might have an Aux symbol
			// that needs SP as its base.)
			r = s.SPReg
		}
		if !s.allocatable.HasReg(r) {
			return v // v is in a fixed register
		}
		if s.regs[r].v != v || s.regs[r].c == nil {
			panic("bad register state")
		}
		if nospill {
			s.nospill = s.nospill.AddReg(r)
		}
		s.usedSinceBlockStart = s.usedSinceBlockStart.AddReg(r)
		return s.regs[r].c
	}

	var r ssaop.Register
	// If nospill is set, the value is used immediately, so it can live on the WebAssembly stack.
	onWasmStack := nospill && s.f.Config.Ctxt.Arch.Arch == sys.ArchWasm
	if !onWasmStack {
		// Allocate a register.
		r = s.allocReg(mask, v)
	}

	// Allocate v to the new register.
	var c *ssacore.Value
	if !vi.Regs.Empty() {
		// Copy from a register that v is already in.
		var current *ssacore.Value
		if !vi.Regs.Minus(s.allocatable).Empty() {
			// v is in a fixed register, prefer that
			current = v
		} else {
			r2 := s.pickReg(vi.Regs)
			if s.regs[r2].v != v {
				panic("bad register state")
			}
			current = s.regs[r2].c
			s.usedSinceBlockStart = s.usedSinceBlockStart.AddReg(r2)
		}
		c = s.curBlock.NewValue1(pos, ssaop.OpCopy, v.Type, current)
	} else if v.Rematerializeable() {
		// Rematerialize instead of loading from the spill location.
		c = v.CopyIntoWithXPos(s.curBlock, pos)
		// We need to consider its output mask and potentially issue a Copy
		// if there are register mask conflicts.
		// This currently happens for the SIMD package only between GP and FP
		// register. Because Intel's vector extension can put integer value into
		// FP, which is seen as a vector. Example instruction: VPSLL[BWDQ]
		// Because GP and FP masks do not overlap, mask & outputMask == 0
		// detects this situation thoroughly.
		sourceMask := s.regspec(c).Outputs[0].Regs
		if mask.Intersect(sourceMask).Empty() && !onWasmStack {
			s.setOrig(c, v)
			s.assignReg(s.allocReg(sourceMask, v), v, c)
			// v.Type for the new OpCopy is likely wrong and it might delay the problem
			// until ssa to asm lowering, which might need the types to generate the right
			// assembly for OpCopy. For Intel's GP to FP move, it happens to be that
			// MOV instruction has such a variant so it happens to be right.
			// But it's unclear for other architectures or situations, and the problem
			// might be exposed when the assembler sees illegal instructions.
			// Right now make we still pick v.Type, because at least its size should be correct
			// for the rematerialization case the amd64 SIMD package exposed.
			// TODO: We might need to figure out a way to find the correct type or make
			// the asm lowering use reg info only for OpCopy.
			c = s.curBlock.NewValue1(pos, ssaop.OpCopy, v.Type, c)
		}
	} else {
		// Load v from its spill location.
		spill := s.makeSpill(v, s.curBlock)
		if s.f.Pass.Debug > ssacore.LogSpills {
			s.f.Warnl(vi.Spill.Pos, "load spill for %v from %v", v, spill)
		}
		c = s.curBlock.NewValue1(pos, ssaop.OpLoadReg, v.Type, spill)
		sourceMask := s.compatRegs(v.Type)
		if !sourceMask.HasReg(r) && !onWasmStack {
			// Assign a temporary register that can be copied to the desired destination;
			// this at least works where it is currently a problem (x86).
			// This happens processing e.g. ASAN/TSAN with SIMD *simdtype methods.
			s.setOrig(c, v)
			s.assignReg(s.allocReg(sourceMask, v), v, c)
			c = s.curBlock.NewValue1(pos, ssaop.OpCopy, v.Type, c)
		}
	}

	s.setOrig(c, v)

	if onWasmStack {
		c.OnWasmStack = true
		return c
	}

	s.assignReg(r, v, c)
	if c.Op == ssaop.OpLoadReg && s.isGReg(r) {
		s.f.Fatalf("allocValToReg.OpLoadReg targeting g: " + c.LongString())
	}
	if nospill {
		s.nospill = s.nospill.AddReg(r)
	}
	return c
}

// isLeaf reports whether f performs any calls.
func isLeaf(f *ssacore.Func) bool {
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

func (s *regAllocState) init(f *ssacore.Func) {
	s.f = f
	s.f.RegAlloc = s.f.Cache.Locs[:0]
	s.registers = f.Config.Registers
	if nr := len(s.registers); nr == 0 || nr > int(noRegister) || nr > int(unsafe.Sizeof(ssaop.RegMask{})*8) {
		s.f.Fatalf("bad number of registers: %d", nr)
	} else {
		s.numRegs = ssaop.Register(nr)
	}
	// Locate SP, SB, and g registers.
	s.SPReg = noRegister
	s.SBReg = noRegister
	s.GReg = noRegister
	s.ZeroIntReg = noRegister
	for r := ssaop.Register(0); r < s.numRegs; r++ {
		switch s.registers[r].String() {
		case "SP":
			s.SPReg = r
		case "SB":
			s.SBReg = r
		case "g":
			s.GReg = r
		case "ZERO": // TODO: arch-specific?
			s.ZeroIntReg = r
		}
	}
	// Make sure we found all required registers.
	switch noRegister {
	case s.SPReg:
		s.f.Fatalf("no SP register found")
	case s.SBReg:
		s.f.Fatalf("no SB register found")
	case s.GReg:
		if f.Config.HasGReg {
			s.f.Fatalf("no g register found")
		}
	}

	// Figure out which registers we're allowed to use.
	s.allocatable = s.f.Config.GpRegMask.Union(s.f.Config.FpRegMask).Union(s.f.Config.SpecialRegMask).Union(s.f.Config.SimdRegMask)
	s.allocatable = s.allocatable.RemoveReg(s.SPReg)
	s.allocatable = s.allocatable.RemoveReg(s.SBReg)
	if s.f.Config.HasGReg {
		s.allocatable = s.allocatable.RemoveReg(s.GReg)
	}
	if s.ZeroIntReg != noRegister {
		s.allocatable = s.allocatable.RemoveReg(s.ZeroIntReg)
	}
	if buildcfg.FramePointerEnabled && s.f.Config.FPReg >= 0 {
		s.allocatable = s.allocatable.RemoveReg(ssaop.Register(s.f.Config.FPReg))
	}
	if s.f.Config.LinkReg != -1 {
		if isLeaf(f) {
			// Leaf functions don't save/restore the link register.
			s.allocatable = s.allocatable.RemoveReg(ssaop.Register(s.f.Config.LinkReg))
		}
	}
	if s.f.Config.Ctxt.Flag_dynlink {
		switch s.f.Config.Arch {
		case "386":
			// nothing to do.
			// Note that for Flag_shared (position independent code)
			// we do need to be careful, but that carefulness is hidden
			// in the rewrite rules so we always have a free register
			// available for global load/stores. See _gen/386.rules (search for Flag_shared).
		case "amd64":
			s.allocatable = s.allocatable.RemoveReg(15) // R15
		case "arm":
			s.allocatable = s.allocatable.RemoveReg(9) // R9
		case "arm64":
			// nothing to do
		case "loong64": // R2 (aka TP) already reserved.
			// nothing to do
		case "ppc64", "ppc64le": // R2 already reserved.
			// nothing to do
		case "riscv64": // X3 (aka GP) and X4 (aka TP) already reserved.
			// nothing to do
		case "s390x":
			s.allocatable = s.allocatable.RemoveReg(11) // R11
		default:
			s.f.Fe.Fatalf(src.NoXPos, "arch %s not implemented", s.f.Config.Arch)
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
	if cap(s.f.Cache.RegallocValues) >= nv {
		s.f.Cache.RegallocValues = s.f.Cache.RegallocValues[:nv]
	} else {
		s.f.Cache.RegallocValues = make([]ssacore.ValState, nv)
	}
	s.values = s.f.Cache.RegallocValues
	s.orig = s.f.Cache.AllocValueSlice(nv)
	s.copies = make(map[*ssacore.Value]bool)
	for _, b := range s.visitOrder {
		for _, v := range b.Values {
			if v.NeedRegister() {
				s.values[v.ID].NeedReg = true
				s.values[v.ID].Rematerializeable = v.Rematerializeable()
				s.orig[v.ID] = v
			}
			// Note: needReg is false for values returning Tuple types.
			// Instead, we mark the corresponding Selects as needReg.
		}
	}
	s.computeLive()

	s.endRegs = make([][]endReg, f.NumBlocks())
	s.startRegs = make([][]startReg, f.NumBlocks())
	s.spillLive = make([][]ssacore.ID, f.NumBlocks())
	s.sdom = f.Sdom()

	// wasm: Mark instructions that can be optimized to have their values only on the WebAssembly stack.
	if f.Config.Ctxt.Arch.Arch == sys.ArchWasm {
		canLiveOnStack := f.NewSparseSet(f.NumValues())
		defer f.RetSparseSet(canLiveOnStack)
		for _, b := range f.Blocks {
			// New block. Clear candidate set.
			canLiveOnStack.Clear()
			for _, c := range b.ControlValues() {
				if c.Uses == 1 && !ssaop.OpcodeTable[c.Op].Generic {
					canLiveOnStack.Add(c.ID)
				}
			}
			// Walking backwards.
			for i := len(b.Values) - 1; i >= 0; i-- {
				v := b.Values[i]
				if canLiveOnStack.Contains(v.ID) {
					v.OnWasmStack = true
				} else {
					// Value can not live on stack. Values are not allowed to be reordered, so clear candidate set.
					canLiveOnStack.Clear()
				}
				for _, arg := range v.Args {
					// Value can live on the stack if:
					// - it is only used once
					// - it is used in the same basic block
					// - it is not a "mem" value
					// - it is a WebAssembly op
					if arg.Uses == 1 && arg.Block == v.Block && !arg.Type.IsMemory() && !ssaop.OpcodeTable[arg.Op].Generic {
						canLiveOnStack.Add(arg.ID)
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
	s.f.Cache.FreeValueSlice(s.orig)
}

// Adds a use record for id at distance dist from the start of the block.
// All calls to addUse must happen with nonincreasing dist.
func (s *regAllocState) addUse(id ssacore.ID, dist int32, pos src.XPos) {
	r := s.freeUseRecords
	if r != nil {
		s.freeUseRecords = r.Next
	} else {
		r = &ssacore.Use{}
	}
	r.Dist = dist
	r.Pos = pos
	r.Next = s.values[id].Uses
	s.values[id].Uses = r
	if r.Next != nil && dist > r.Next.Dist {
		s.f.Fatalf("uses added in wrong order")
	}
}

// advanceUses advances the uses of v's args from the state before v to the state after v.
// Any values which have no more uses are deallocated from registers.
func (s *regAllocState) advanceUses(v *ssacore.Value) {
	for _, a := range v.Args {
		if !s.values[a.ID].NeedReg {
			continue
		}
		ai := &s.values[a.ID]
		r := ai.Uses
		ai.Uses = r.Next
		if r.Next == nil || (!ssaop.OpcodeTable[a.Op].FixedReg && r.Next.Dist > s.nextCall[s.curIdx]) {
			// Value is dead (or is not used again until after a call), free all registers that hold it.
			s.freeRegs(ai.Regs)
		}
		r.Next = s.freeUseRecords
		s.freeUseRecords = r
	}
	s.dropIfUnused(v)
}

// Drop v from registers if it isn't used again, or its only uses are after
// a call instruction.
func (s *regAllocState) dropIfUnused(v *ssacore.Value) {
	if !s.values[v.ID].NeedReg {
		return
	}
	vi := &s.values[v.ID]
	r := vi.Uses
	nextCall := s.nextCall[s.curIdx]
	if ssaop.OpcodeTable[v.Op].Call {
		if s.curIdx == len(s.nextCall)-1 {
			nextCall = math.MaxInt32
		} else {
			nextCall = s.nextCall[s.curIdx+1]
		}
	}
	if r == nil || (!ssaop.OpcodeTable[v.Op].FixedReg && r.Dist > nextCall) {
		s.freeRegs(vi.Regs)
	}
}

// liveAfterCurrentInstruction reports whether v is live after
// the current instruction is completed.  v must be used by the
// current instruction.
func (s *regAllocState) liveAfterCurrentInstruction(v *ssacore.Value) bool {
	u := s.values[v.ID].Uses
	if u == nil {
		panic(fmt.Errorf("u is nil, v = %s, s.values[v.ID] = %v", v.LongString(), s.values[v.ID]))
	}
	d := u.Dist
	for u != nil && u.Dist == d {
		u = u.Next
	}
	return u != nil && u.Dist > d
}

// Sets the state of the registers to that encoded in regs.
func (s *regAllocState) setState(regs []endReg) {
	s.freeRegs(s.used)
	for _, x := range regs {
		s.assignReg(x.r, x.v, x.c)
	}
}

// compatRegs returns the set of registers which can store a type t.
func (s *regAllocState) compatRegs(t *types.Type) ssaop.RegMask {
	var m ssaop.RegMask
	if t.IsTuple() || t.IsFlags() {
		return ssaop.RegMask{}
	}
	if t.IsSIMD() {
		if t.Size() > 8 {
			return s.f.Config.SimdRegMask.Intersect(s.allocatable)
		} else {
			if !s.f.Config.SpecialRegMask.Empty() {
				// P predicates
				// No instructions can move P <-> GP.
				return s.f.Config.SpecialRegMask.Intersect(s.allocatable)
			}
			// K mask
			// We can move GP <-> K.
			return s.f.Config.GpRegMask.Intersect(s.allocatable)
		}
	}
	if t.IsFloat() || t == types.TypeInt128 {
		if t.Kind() == types.TFLOAT32 && !s.f.Config.Fp32RegMask.Empty() {
			m = s.f.Config.Fp32RegMask
		} else if t.Kind() == types.TFLOAT64 && !s.f.Config.Fp64RegMask.Empty() {
			m = s.f.Config.Fp64RegMask
		} else {
			m = s.f.Config.FpRegMask
		}
	} else {
		m = s.f.Config.GpRegMask
	}
	return m.Intersect(s.allocatable)
}

// regspec returns the regInfo for operation op.
func (s *regAllocState) regspec(v *ssacore.Value) ssaop.RegInfo {
	op := v.Op
	if op == ssaop.OpConvert {
		// OpConvert is a generic op, so it doesn't have a
		// register set in the static table. It can use any
		// allocatable integer register.
		m := s.allocatable.Intersect(s.f.Config.GpRegMask)
		return ssaop.RegInfo{Inputs: []ssaop.InputInfo{{Regs: m}}, Outputs: []ssaop.OutputInfo{{Regs: m}}}
	}
	if op == ssaop.OpArgIntReg {
		reg := v.Block.Func.Config.IntParamRegs[v.AuxInt8()]
		return ssaop.RegInfo{Outputs: []ssaop.OutputInfo{{Regs: ssacore.RegMaskAt(ssaop.Register(reg))}}}
	}
	if op == ssaop.OpArgFloatReg {
		reg := v.Block.Func.Config.FloatParamRegs[v.AuxInt8()]
		return ssaop.RegInfo{Outputs: []ssaop.OutputInfo{{Regs: ssacore.RegMaskAt(ssaop.Register(reg))}}}
	}
	if op.IsCall() {
		if ac, ok := v.Aux.(*ssacore.AuxCall); ok && ac.RegCache != nil {
			return *ac.Reg(&ssaop.OpcodeTable[op].Reg, s.f.Config)
		}
	}
	if op == ssaop.OpMakeResult && s.f.OwnAux.RegCache != nil {
		return *s.f.OwnAux.ResultReg(s.f.Config)
	}
	return ssaop.OpcodeTable[op].Reg
}

func (s *regAllocState) isGReg(r ssaop.Register) bool {
	return s.f.Config.HasGReg && s.GReg == r
}

// Dummy value used to represent the value being held in a temporary register.
var tmpVal ssacore.Value

func (s *regAllocState) regalloc(f *ssacore.Func) {
	regValLiveSet := f.NewSparseSet(f.NumValues()) // set of values that may be live in register
	defer f.RetSparseSet(regValLiveSet)
	var oldSched []*ssacore.Value
	var phis []*ssacore.Value
	var phiRegs []ssaop.Register
	var args []*ssacore.Value

	// Data structure used for computing desired registers.
	var desired desiredState
	desiredSecondReg := map[ssacore.ID][4]ssaop.Register{} // desired register allocation for 2nd part of a tuple

	// Desired registers for inputs & outputs for each instruction in the block.
	type dentry struct {
		out [4]ssaop.Register    // desired output registers
		in  [3][4]ssaop.Register // desired input registers (for inputs 0,1, and 2)
	}
	var dinfo []dentry

	if f.Entry != f.Blocks[0] {
		f.Fatalf("entry block must be first")
	}

	for _, b := range s.visitOrder {
		if s.f.Pass.Debug > ssacore.RegDebug {
			fmt.Printf("Begin processing block %v\n", b)
		}
		s.curBlock = b
		s.startRegsMask = ssaop.RegMask{}
		s.usedSinceBlockStart = ssaop.RegMask{}
		clear(desiredSecondReg)

		// Initialize regValLiveSet and uses fields for this block.
		// Walk backwards through the block doing liveness analysis.
		regValLiveSet.Clear()
		if s.live != nil {
			for _, e := range s.live[b.ID] {
				s.addUse(e.ID, int32(len(b.Values))+e.dist, e.pos) // pseudo-uses from beyond end of block
				regValLiveSet.Add(e.ID)
			}
		}
		for _, v := range b.ControlValues() {
			if s.values[v.ID].NeedReg {
				s.addUse(v.ID, int32(len(b.Values)), b.Pos) // pseudo-use by control values
				regValLiveSet.Add(v.ID)
			}
		}
		if cap(s.nextCall) < len(b.Values) {
			c := cap(s.nextCall)
			s.nextCall = append(s.nextCall[:c], make([]int32, len(b.Values)-c)...)
		} else {
			s.nextCall = s.nextCall[:len(b.Values)]
		}
		var nextCall int32 = math.MaxInt32
		for i := len(b.Values) - 1; i >= 0; i-- {
			v := b.Values[i]
			regValLiveSet.Remove(v.ID)
			if v.Op == ssaop.OpPhi {
				// Remove v from the live set, but don't add
				// any inputs. This is the state the len(b.Preds)>1
				// case below desires; it wants to process phis specially.
				s.nextCall[i] = nextCall
				continue
			}
			if ssaop.OpcodeTable[v.Op].Call {
				// Function call clobbers all the registers but SP and SB.
				regValLiveSet.Clear()
				if s.sp != 0 && s.values[s.sp].Uses != nil {
					regValLiveSet.Add(s.sp)
				}
				if s.sb != 0 && s.values[s.sb].Uses != nil {
					regValLiveSet.Add(s.sb)
				}
				nextCall = int32(i)
			}
			for _, a := range v.Args {
				if !s.values[a.ID].NeedReg {
					continue
				}
				s.addUse(a.ID, int32(i), v.Pos)
				regValLiveSet.Add(a.ID)
			}
			s.nextCall[i] = nextCall
		}
		if s.f.Pass.Debug > ssacore.RegDebug {
			fmt.Printf("use distances for %s\n", b)
			for i := range s.values {
				vi := &s.values[i]
				u := vi.Uses
				if u == nil {
					continue
				}
				fmt.Printf("  v%d:", i)
				for u != nil {
					fmt.Printf(" %d", u.Dist)
					u = u.Next
				}
				fmt.Println()
			}
		}

		// Make a copy of the block schedule so we can generate a new one in place.
		// We make a separate copy for phis and regular values.
		nphi := 0
		for _, v := range b.Values {
			if v.Op != ssaop.OpPhi {
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
			s.setState(s.endRegs[b.Preds[0].B.ID])
			if nphi > 0 {
				f.Fatalf("phis in single-predecessor block")
			}
			// Drop any values which are no longer live.
			// This may happen because at the end of p, a value may be
			// live but only used by some other successor of p.
			for r := ssaop.Register(0); r < s.numRegs; r++ {
				v := s.regs[r].v
				if v != nil && !regValLiveSet.Contains(v.ID) {
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
				pb := p.B
				if s.blockOrder[pb.ID] >= s.blockOrder[b.ID] {
					continue
				}
				if idx == -1 {
					idx = i
					continue
				}
				pSel := b.Preds[idx].B
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
					if pb.LikelyBranch() && !pSel.LikelyBranch() || s.blockOrder[pb.ID] < s.blockOrder[pSel.ID] {
						idx = i
					}
				}
			}
			if idx < 0 {
				f.Fatalf("bad visitOrder, no predecessor of %s has been visited before it", b)
			}
			p := b.Preds[idx].B
			s.setState(s.endRegs[p.ID])

			if s.f.Pass.Debug > ssacore.RegDebug {
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
			var phiUsed ssaop.RegMask

			for _, v := range phis {
				if !s.values[v.ID].NeedReg {
					phiRegs = append(phiRegs, noRegister)
					continue
				}
				a := v.Args[idx]
				// Some instructions target not-allocatable registers.
				// They're not suitable for further (phi-function) allocation.
				m := s.values[a.ID].Regs.Minus(phiUsed).Intersect(s.allocatable)
				if !m.Empty() {
					r := s.pickReg(m)
					phiUsed = phiUsed.AddReg(r)
					phiRegs = append(phiRegs, r)
				} else {
					phiRegs = append(phiRegs, noRegister)
				}
			}

			// Second pass - deallocate all in-register phi inputs.
			for i, v := range phis {
				if !s.values[v.ID].NeedReg {
					continue
				}
				a := v.Args[idx]
				r := phiRegs[i]
				if r == noRegister {
					continue
				}
				if regValLiveSet.Contains(a.ID) {
					// Input value is still live (it is used by something other than Phi).
					// Try to move it around before kicking out, if there is a free register.
					// We generate a Copy in the predecessor block and record it. It will be
					// deleted later if never used.
					//
					// Pick a free register. At this point some registers used in the predecessor
					// block may have been deallocated. Those are the ones used for Phis. Exclude
					// them (and they are not going to be helpful anyway).
					m := s.compatRegs(a.Type).Minus(s.used).Minus(phiUsed)
					if !m.Empty() && !s.values[a.ID].Rematerializeable && countRegs(s.values[a.ID].Regs) == 1 {
						r2 := s.pickReg(m)
						c := p.NewValue1(a.Pos, ssaop.OpCopy, a.Type, s.regs[r].c)
						s.copies[c] = false
						if s.f.Pass.Debug > ssacore.RegDebug {
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
				if !s.values[v.ID].NeedReg {
					continue
				}
				if phiRegs[i] != noRegister {
					continue
				}
				m := s.compatRegs(v.Type).Minus(phiUsed).Minus(s.used)
				// If one of the other inputs of v is in a register, and the register is available,
				// select this register, which can save some unnecessary copies.
				for i, pe := range b.Preds {
					if i == idx {
						continue
					}
					ri := noRegister
					for _, er := range s.endRegs[pe.B.ID] {
						if er.v == s.orig[v.Args[i].ID] {
							ri = er.r
							break
						}
					}
					if ri != noRegister && m.HasReg(ri) {
						m = ssacore.RegMaskAt(ri)
						break
					}
				}
				if !m.Empty() {
					r := s.pickReg(m)
					phiRegs[i] = r
					phiUsed = phiUsed.AddReg(r)
				}
			}

			// Set registers for phis. Add phi spill code.
			for i, v := range phis {
				if !s.values[v.ID].NeedReg {
					continue
				}
				r := phiRegs[i]
				if r == noRegister {
					// stack-based phi
					// Spills will be inserted in all the predecessors below.
					s.values[v.ID].Spill = v // v starts life spilled
					continue
				}
				// register-based phi
				s.assignReg(r, v, v)
			}

			// Deallocate any values which are no longer live. Phis are excluded.
			for r := ssaop.Register(0); r < s.numRegs; r++ {
				if phiUsed.HasReg(r) {
					continue
				}
				v := s.regs[r].v
				if v != nil && !regValLiveSet.Contains(v.ID) {
					s.freeReg(r)
				}
			}

			// Look for loop headers of loops that contain unavoidable calls.
			// That call will clobber all registers.
			// Any value that's unused before the first such call is doomed.
			// To avoid pointless backedge reloads, free such doomed values instead,
			// and reload them lazily at their first use, after the call.
			//
			//	v := ...      // in a register
			//	for ... {
			//		...       // no use of v
			//		f()       // clobbers registers
			//		... = v   // reload v here, not on the backedge
			//	}
			doomDist := int32(math.MaxInt32)
			if l := s.loopnest.B2L[b.ID]; l != nil && l.Header == b && l.ContainsUnavoidableCall {
				// The first call, if any, is at s.nextCall[0].
				// A call in a later block is at least unlikelyDistance away.
				doomDist = unlikelyDistance
				if len(s.nextCall) > 0 {
					doomDist = min(doomDist, s.nextCall[0])
				}
			}

			// Save the starting state for use by merge edges.
			// We append to a stack allocated variable that we'll
			// later copy into s.startRegs in one fell swoop, to save
			// on allocations.
			regList := make([]startReg, 0, 32)
			for r := ssaop.Register(0); r < s.numRegs; r++ {
				v := s.regs[r].v
				if v == nil {
					continue
				}
				if phiUsed.HasReg(r) {
					// Skip registers that phis used, we'll handle those
					// specially during merge edge processing.
					continue
				}
				// Drop values doomed by an intervening unavoidable call.
				if s.values[v.ID].Uses.Dist >= doomDist && s.allocatable.HasReg(r) && !ssaop.OpcodeTable[v.Op].FixedReg {
					s.freeReg(r)
					continue
				}
				regList = append(regList, startReg{r, v, s.regs[r].c, s.values[v.ID].Uses.Pos})
				s.startRegsMask = s.startRegsMask.AddReg(r)
			}
			s.startRegs[b.ID] = make([]startReg, len(regList))
			copy(s.startRegs[b.ID], regList)

			if s.f.Pass.Debug > ssacore.RegDebug {
				fmt.Printf("after phis\n")
				for _, x := range s.startRegs[b.ID] {
					fmt.Printf("  %s: v%d\n", &s.registers[x.r], x.v.ID)
				}
			}
		}

		// Drop phis from registers if they immediately go dead.
		for i, v := range phis {
			s.curIdx = i
			s.dropIfUnused(v)
		}

		// Allocate space to record the desired registers for each value.
		if l := len(oldSched); cap(dinfo) < l {
			dinfo = make([]dentry, l)
		} else {
			dinfo = dinfo[:l]
			clear(dinfo)
		}

		// Load static desired register info at the end of the block.
		if s.desired != nil {
			desired.copy(&s.desired[b.ID])
		}

		// Check actual assigned registers at the start of the next block(s).
		// Dynamically assigned registers will trump the static
		// desired registers computed during liveness analysis.
		// Note that we do this phase after startRegs is set above, so that
		// we get the right behavior for a block which branches to itself.
		for _, e := range b.Succs {
			succ := e.B
			// TODO: prioritize likely successor?
			for _, x := range s.startRegs[succ.ID] {
				desired.add(x.v.ID, x.r)
			}
			// Process phi ops in succ.
			pidx := e.I
			for _, v := range succ.Values {
				if v.Op != ssaop.OpPhi {
					break
				}
				if !s.values[v.ID].NeedReg {
					continue
				}
				rp, ok := s.f.GetHome(v.ID).(*ssabase.Register)
				if !ok {
					// If v is not assigned a register, pick a register assigned to one of v's inputs.
					// Hopefully v will get assigned that register later.
					// If the inputs have allocated register information, add it to desired,
					// which may reduce spill or copy operations when the register is available.
					for _, a := range v.Args {
						rp, ok = s.f.GetHome(a.ID).(*ssabase.Register)
						if ok {
							break
						}
					}
					if !ok {
						continue
					}
				}
				desired.add(v.Args[pidx].ID, ssaop.Register(rp.Num))
			}
		}
		// Walk values backwards computing desired register info.
		// See computeDesired for more comments.
		for i := len(oldSched) - 1; i >= 0; i-- {
			v := oldSched[i]
			prefs := desired.remove(v.ID)
			regspec := s.regspec(v)
			desired.clobber(regspec.Clobbers)
			for _, j := range regspec.Inputs {
				if countRegs(j.Regs) != 1 {
					continue
				}
				desired.clobber(j.Regs)
				desired.add(v.Args[j.Idx].ID, s.pickReg(j.Regs))
			}
			if ssaop.OpcodeTable[v.Op].ResultInArg0 || v.Op == ssaop.OpAMD64ADDQconst || v.Op == ssaop.OpAMD64ADDLconst || v.Op == ssaop.OpSelect0 {
				if ssaop.OpcodeTable[v.Op].Commutative {
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
			if v.Op == ssaop.OpSelect1 && prefs[0] != noRegister {
				// Save desired registers of select1 for
				// use by the tuple generating instruction.
				desiredSecondReg[v.Args[0].ID] = prefs
			}
		}

		// Process all the non-phi values.
		for idx, v := range oldSched {
			s.curIdx = nphi + idx
			tmpReg := noRegister
			if s.f.Pass.Debug > ssacore.RegDebug {
				fmt.Printf("  processing %s\n", v.LongString())
			}
			regspec := s.regspec(v)
			if v.Op == ssaop.OpPhi {
				f.Fatalf("phi %s not at start of block", v)
			}
			if ssaop.OpcodeTable[v.Op].FixedReg {
				switch v.Op {
				case ssaop.OpSP:
					s.assignReg(s.SPReg, v, v)
					s.sp = v.ID
				case ssaop.OpSB:
					s.assignReg(s.SBReg, v, v)
					s.sb = v.ID
				case ssaop.OpARM64ZERO, ssaop.OpLOONG64ZERO, ssaop.OpMIPS64ZERO:
					s.assignReg(s.ZeroIntReg, v, v)
				case ssaop.OpAMD64Zero128, ssaop.OpAMD64Zero256, ssaop.OpAMD64Zero512:
					regspec := s.regspec(v)
					m := regspec.Outputs[0].Regs
					if countRegs(m) != 1 {
						f.Fatalf("bad fixed-register op %s", v)
					}
					s.assignReg(s.pickReg(m), v, v)
				default:
					f.Fatalf("unknown fixed-register op %s", v)
				}
				b.Values = append(b.Values, v)
				s.advanceUses(v)
				continue
			}
			if v.Op == ssaop.OpSelect0 || v.Op == ssaop.OpSelect1 || v.Op == ssaop.OpSelectN {
				if s.values[v.ID].NeedReg {
					if v.Op == ssaop.OpSelectN {
						s.assignReg(ssaop.Register(s.f.GetHome(v.Args[0].ID).(ssacore.LocResults)[int(v.AuxInt)].(*ssabase.Register).Num), v, v)
					} else {
						var i = 0
						if v.Op == ssaop.OpSelect1 {
							i = 1
						}
						s.assignReg(ssaop.Register(s.f.GetHome(v.Args[0].ID).(ssacore.LocPair)[i].(*ssabase.Register).Num), v, v)
					}
				}
				b.Values = append(b.Values, v)
				s.advanceUses(v)
				continue
			}
			if v.Op == ssaop.OpGetG && s.f.Config.HasGReg {
				// use hardware g register
				if s.regs[s.GReg].v != nil {
					s.freeReg(s.GReg) // kick out the old value
				}
				s.assignReg(s.GReg, v, v)
				b.Values = append(b.Values, v)
				s.advanceUses(v)
				continue
			}
			if v.Op == ssaop.OpArg {
				// Args are "pre-spilled" values. We don't allocate
				// any register here. We just set up the spill pointer to
				// point at itself and any later user will restore it to use it.
				s.values[v.ID].Spill = v
				b.Values = append(b.Values, v)
				s.advanceUses(v)
				continue
			}
			if v.Op == ssaop.OpKeepAlive {
				// Make sure the argument to v is still live here.
				s.advanceUses(v)
				a := v.Args[0]
				vi := &s.values[a.ID]
				if vi.Regs.Empty() && !vi.Rematerializeable {
					// Use the spill location.
					// This forces later liveness analysis to make the
					// value live at this point.
					v.SetArg(0, s.makeSpill(a, b))
				} else if _, ok := a.Aux.(*ir.Name); ok && vi.Rematerializeable {
					// Rematerializeable value with a *ir.Name. This is the address of
					// a stack object (e.g. an LEAQ). Keep the object live.
					// Change it to VarLive, which is what plive expects for locals.
					v.Op = ssaop.OpVarLive
					v.SetArgs1(v.Args[1])
					v.Aux = a.Aux
				} else {
					// In-register and rematerializeable values are already live.
					// These are typically rematerializeable constants like nil,
					// or values of a variable that were modified since the last call.
					v.Op = ssaop.OpCopy
					v.SetArgs1(v.Args[1])
				}
				b.Values = append(b.Values, v)
				continue
			}
			if len(regspec.Inputs) == 0 && len(regspec.Outputs) == 0 {
				// No register allocation required (or none specified yet)
				if s.doClobber && v.Op.IsCall() {
					s.clobberRegs(regspec.Clobbers)
				}
				s.freeRegs(regspec.Clobbers)
				b.Values = append(b.Values, v)
				s.advanceUses(v)
				continue
			}

			if s.values[v.ID].Rematerializeable {
				// Value is rematerializeable, don't issue it here.
				// It will get issued just before each use (see
				// allocValueToReg).
				for _, a := range v.Args {
					a.Uses--
				}
				s.advanceUses(v)
				continue
			}

			if s.f.Pass.Debug > ssacore.RegDebug {
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
			args = append(args[:0], make([]*ssacore.Value, len(v.Args))...)
			for i, a := range v.Args {
				if !s.values[a.ID].NeedReg {
					args[i] = a
				}
			}
			for _, i := range regspec.Inputs {
				mask := i.Regs
				if countRegs(mask) == 1 && !mask.Intersect(s.values[v.Args[i.Idx].ID].Regs).Empty() {
					args[i.Idx] = s.allocValToReg(v.Args[i.Idx], mask, true, v.Pos)
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
				for _, i := range regspec.Inputs {
					if args[i.Idx] != nil {
						continue // already allocated
					}
					mask := i.Regs
					if countRegs(mask) == 1 && !mask.Minus(s.used).Empty() {
						args[i.Idx] = s.allocValToReg(v.Args[i.Idx], mask, true, v.Pos)
						// If the input is in other registers that will be clobbered by v,
						// or the input is dead, free the registers. This may make room
						// for other inputs.
						oldregs := s.values[v.Args[i.Idx].ID].Regs
						if oldregs.Minus(regspec.Clobbers).Empty() || !s.liveAfterCurrentInstruction(v.Args[i.Idx]) {
							s.freeRegs(oldregs.Minus(mask).Minus(s.nospill))
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
			for _, i := range regspec.Inputs {
				if args[i.Idx] != nil {
					continue // already allocated
				}
				mask := i.Regs
				if mask.Intersect(s.values[v.Args[i.Idx].ID].Regs).Empty() {
					// Need a new register for the input.
					mask = mask.Intersect(s.allocatable)
					mask = mask.Minus(s.nospill)
					// Used desired register if available.
					if i.Idx < 3 {
						for _, r := range dinfo[idx].in[i.Idx] {
							if r != noRegister && mask.Minus(s.used).HasReg(r) {
								// Desired register is allowed and unused.
								mask = ssacore.RegMaskAt(r)
								break
							}
						}
					}
					// Avoid registers we're saving for other values.
					if !mask.Minus(desired.avoid).Empty() {
						mask = mask.Minus(desired.avoid)
					}
				}
				if mask.Intersect(s.values[v.Args[i.Idx].ID].Regs).HasReg(s.SPReg) {
					// Prefer SP register. This ensures that local variables
					// use SP as their base register (instead of a copy of the
					// stack pointer living in another register). See issue 74836.
					mask = ssacore.RegMaskAt(s.SPReg)
				}
				args[i.Idx] = s.allocValToReg(v.Args[i.Idx], mask, true, v.Pos)
			}

			// If the output clobbers the input register, make sure we have
			// at least two copies of the input register so we don't
			// have to reload the value from the spill location.
			if ssaop.OpcodeTable[v.Op].ResultInArg0 {
				var m ssaop.RegMask
				if !s.liveAfterCurrentInstruction(v.Args[0]) {
					// arg0 is dead.  We can clobber its register.
					goto ok
				}
				if ssaop.OpcodeTable[v.Op].Commutative && !s.liveAfterCurrentInstruction(v.Args[1]) {
					args[0], args[1] = args[1], args[0]
					goto ok
				}
				if s.values[v.Args[0].ID].Rematerializeable {
					// We can rematerialize the input, don't worry about clobbering it.
					goto ok
				}
				if ssaop.OpcodeTable[v.Op].Commutative && s.values[v.Args[1].ID].Rematerializeable {
					args[0], args[1] = args[1], args[0]
					goto ok
				}
				if countRegs(s.values[v.Args[0].ID].Regs) >= 2 {
					// we have at least 2 copies of arg0.  We can afford to clobber one.
					goto ok
				}
				if ssaop.OpcodeTable[v.Op].Commutative && countRegs(s.values[v.Args[1].ID].Regs) >= 2 {
					args[0], args[1] = args[1], args[0]
					goto ok
				}

				// We can't overwrite arg0 (or arg1, if commutative).  So we
				// need to make a copy of an input so we have a register we can modify.

				// Possible new registers to copy into.
				m = s.compatRegs(v.Args[0].Type).Minus(s.used)
				if m.Empty() {
					// No free registers.  In this case we'll just clobber
					// an input and future uses of that input must use a restore.
					// TODO(khr): We should really do this like allocReg does it,
					// spilling the value with the most distant next use.
					goto ok
				}

				// Try to move an input to the desired output, if allowed.
				for _, r := range dinfo[idx].out {
					if r != noRegister && m.Intersect(regspec.Outputs[0].Regs).HasReg(r) {
						m = ssacore.RegMaskAt(r)
						args[0] = s.allocValToReg(v.Args[0], m, true, v.Pos)
						// Note: we update args[0] so the instruction will
						// use the register copy we just made.
						goto ok
					}
				}
				// Try to copy input to its desired location & use its old
				// location as the result register.
				for _, r := range dinfo[idx].in[0] {
					if r != noRegister && m.HasReg(r) {
						m = ssacore.RegMaskAt(r)
						c := s.allocValToReg(v.Args[0], m, true, v.Pos)
						s.copies[c] = false
						// Note: no update to args[0] so the instruction will
						// use the original copy.
						goto ok
					}
				}
				if ssaop.OpcodeTable[v.Op].Commutative {
					for _, r := range dinfo[idx].in[1] {
						if r != noRegister && m.HasReg(r) {
							m = ssacore.RegMaskAt(r)
							c := s.allocValToReg(v.Args[1], m, true, v.Pos)
							s.copies[c] = false
							args[0], args[1] = args[1], args[0]
							goto ok
						}
					}
				}

				// Avoid future fixed uses if we can.
				if !m.Minus(desired.avoid).Empty() {
					m = m.Minus(desired.avoid)
				}
				// Save input 0 to a new register so we can clobber it.
				c := s.allocValToReg(v.Args[0], m, true, v.Pos)
				s.copies[c] = false

				// Normally we use the register of the old copy of input 0 as the target.
				// However, if input 0 is already in its desired register then we use
				// the register of the new copy instead.
				if regspec.Outputs[0].Regs.HasReg(ssaop.Register(s.f.GetHome(c.ID).(*ssabase.Register).Num)) {
					if rp, ok := s.f.GetHome(args[0].ID).(*ssabase.Register); ok {
						r := ssaop.Register(rp.Num)
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
			for i := 0; i < 2; i++ {
				if !(i == 0 && regspec.ClobbersArg0 || i == 1 && regspec.ClobbersArg1) {
					continue
				}
				if !s.liveAfterCurrentInstruction(v.Args[i]) {
					// arg is dead.  We can clobber its register.
					continue
				}
				if s.values[v.Args[i].ID].Rematerializeable {
					// We can rematerialize the input, don't worry about clobbering it.
					continue
				}
				if countRegs(s.values[v.Args[i].ID].Regs) >= 2 {
					// We have at least 2 copies of arg.  We can afford to clobber one.
					continue
				}
				// Possible new registers to copy into.
				m := s.compatRegs(v.Args[i].Type).Minus(s.used)
				if m.Empty() {
					// No free registers.  In this case we'll just clobber the
					// input and future uses of that input must use a restore.
					// TODO(khr): We should really do this like allocReg does it,
					// spilling the value with the most distant next use.
					continue
				}
				// Copy input to a different register that won't be clobbered.
				c := s.allocValToReg(v.Args[i], m, true, v.Pos)
				s.copies[c] = false
			}

			// Pick a temporary register if needed.
			// It should be distinct from all the input registers, so we
			// allocate it after all the input registers, but before
			// the input registers are freed via advanceUses below.
			// (Not all instructions need that distinct part, but it is conservative.)
			// We also ensure it is not any of the single-choice output registers.
			if ssaop.OpcodeTable[v.Op].NeedIntTemp {
				m := s.allocatable.Intersect(s.f.Config.GpRegMask)
				for _, out := range regspec.Outputs {
					if countRegs(out.Regs) == 1 {
						m = m.Minus(out.Regs)
					}
				}
				if !m.Minus(desired.avoid).Minus(s.nospill).Empty() {
					m = m.Minus(desired.avoid)
				}
				tmpReg = s.allocReg(m, &tmpVal)
				s.nospill = s.nospill.AddReg(tmpReg)
				s.tmpused = s.tmpused.AddReg(tmpReg)
			}

			if regspec.ClobbersArg0 {
				s.freeReg(ssaop.Register(s.f.GetHome(args[0].ID).(*ssabase.Register).Num))
			}
			if regspec.ClobbersArg1 && !(regspec.ClobbersArg0 && s.f.GetHome(args[0].ID) == s.f.GetHome(args[1].ID)) {
				s.freeReg(ssaop.Register(s.f.GetHome(args[1].ID).(*ssabase.Register).Num))
			}

			// Now that all args are in regs, we're ready to issue the value itself.
			// Before we pick a register for the output value, allow input registers
			// to be deallocated. We do this here so that the output can use the
			// same register as a dying input.
			if !ssaop.OpcodeTable[v.Op].ResultNotInArgs {
				s.tmpused = s.nospill
				s.nospill = ssaop.RegMask{}
				s.advanceUses(v) // frees any registers holding args that are no longer live
			}

			// Dump any registers which will be clobbered
			if s.doClobber && v.Op.IsCall() {
				// clobber registers that are marked as clobber in regmask, but
				// don't clobber inputs.
				s.clobberRegs(regspec.Clobbers.Minus(s.tmpused).Minus(s.nospill))
			}
			s.freeRegs(regspec.Clobbers)
			s.tmpused = s.tmpused.Union(regspec.Clobbers)

			// Pick registers for outputs.
			{
				outRegs := noRegisters // TODO if this is costly, hoist and clear incrementally below.
				maxOutIdx := -1
				var used ssaop.RegMask
				if tmpReg != noRegister {
					// Ensure output registers are distinct from the temporary register.
					// (Not all instructions need that distinct part, but it is conservative.)
					used = used.AddReg(tmpReg)
				}
				for _, out := range regspec.Outputs {
					if out.Regs.Empty() {
						continue
					}
					mask := out.Regs.Intersect(s.allocatable).Minus(used)
					if mask.Empty() {
						s.f.Fatalf("can't find any output register %s", v.LongString())
					}
					if ssaop.OpcodeTable[v.Op].ResultInArg0 && out.Idx == 0 {
						if !ssaop.OpcodeTable[v.Op].Commutative {
							// Output must use the same register as input 0.
							r := ssaop.Register(s.f.GetHome(args[0].ID).(*ssabase.Register).Num)
							if !mask.HasReg(r) {
								s.f.Fatalf("resultInArg0 value's input %v cannot be an output of %s", s.f.GetHome(args[0].ID).(*ssabase.Register), v.LongString())
							}
							mask = ssacore.RegMaskAt(r)
						} else {
							// Output must use the same register as input 0 or 1.
							r0 := ssaop.Register(s.f.GetHome(args[0].ID).(*ssabase.Register).Num)
							r1 := ssaop.Register(s.f.GetHome(args[1].ID).(*ssabase.Register).Num)
							// Check r0 and r1 for desired output register.
							found := false
							for _, r := range dinfo[idx].out {
								if (r == r0 || r == r1) && mask.Minus(s.used).HasReg(r) {
									mask = ssacore.RegMaskAt(r)
									found = true
									if r == r1 {
										args[0], args[1] = args[1], args[0]
									}
									break
								}
							}
							if !found {
								// Neither are desired, pick r0.
								mask = ssacore.RegMaskAt(r0)
							}
						}
					}
					if out.Idx == 0 { // desired registers only apply to the first element of a tuple result
						for _, r := range dinfo[idx].out {
							if r != noRegister && mask.Minus(s.used).HasReg(r) {
								// Desired register is allowed and unused.
								mask = ssacore.RegMaskAt(r)
								break
							}
						}
					}
					if out.Idx == 1 {
						if prefs, ok := desiredSecondReg[v.ID]; ok {
							for _, r := range prefs {
								if r != noRegister && mask.Minus(s.used).HasReg(r) {
									// Desired register is allowed and unused.
									mask = ssacore.RegMaskAt(r)
									break
								}
							}
						}
					}
					// Avoid registers we're saving for other values.
					if !mask.Minus(desired.avoid).Minus(s.nospill).Minus(s.used).Empty() {
						mask = mask.Minus(desired.avoid)
					}
					r := s.allocReg(mask, v)
					if out.Idx > maxOutIdx {
						maxOutIdx = out.Idx
					}
					outRegs[out.Idx] = r
					used = used.AddReg(r)
					s.tmpused = s.tmpused.AddReg(r)
				}
				// Record register choices
				if v.Type.IsTuple() {
					var outLocs ssacore.LocPair
					if r := outRegs[0]; r != noRegister {
						outLocs[0] = &s.registers[r]
					}
					if r := outRegs[1]; r != noRegister {
						outLocs[1] = &s.registers[r]
					}
					s.f.SetHome(v, outLocs)
					// Note that subsequent SelectX instructions will do the assignReg calls.
				} else if v.Type.IsResults() {
					// preallocate outLocs to the right size, which is maxOutIdx+1
					outLocs := make(ssacore.LocResults, maxOutIdx+1, maxOutIdx+1)
					for i := 0; i <= maxOutIdx; i++ {
						if r := outRegs[i]; r != noRegister {
							outLocs[i] = &s.registers[r]
						}
					}
					s.f.SetHome(v, outLocs)
				} else {
					if r := outRegs[0]; r != noRegister {
						s.assignReg(r, v, v)
					}
				}
				if tmpReg != noRegister {
					// Remember the temp register allocation, if any.
					if s.f.TempRegs == nil {
						s.f.TempRegs = map[ssacore.ID]*ssabase.Register{}
					}
					s.f.TempRegs[v.ID] = &s.registers[tmpReg]
				}
			}

			// deallocate dead args, if we have not done so
			if ssaop.OpcodeTable[v.Op].ResultNotInArgs {
				s.nospill = ssaop.RegMask{}
				s.advanceUses(v) // frees any registers holding args that are no longer live
			}
			s.tmpused = ssaop.RegMask{}

			// Issue the Value itself.
			for i, a := range args {
				v.SetArg(i, a) // use register version of arguments
			}
			b.Values = append(b.Values, v)
			s.dropIfUnused(v)
		}

		// Copy the control values - we need this so we can reduce the
		// uses property of these values later.
		controls := append(make([]*ssacore.Value, 0, 2), b.ControlValues()...)

		// Load control values into registers.
		for i, v := range b.ControlValues() {
			if !s.values[v.ID].NeedReg {
				continue
			}
			if s.f.Pass.Debug > ssacore.RegDebug {
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
			if !vi.NeedReg {
				continue
			}
			// Remove this use from the uses list.
			u := vi.Uses
			vi.Uses = u.Next
			if u.Next == nil {
				s.freeRegs(vi.Regs) // value is dead
			}
			u.Next = s.freeUseRecords
			s.freeUseRecords = u
		}

		// If we are approaching a merge point and we are the primary
		// predecessor of it, find live values that we use soon after
		// the merge point and promote them to registers now.
		if len(b.Succs) == 1 {
			if s.f.Config.HasGReg && s.regs[s.GReg].v != nil {
				s.freeReg(s.GReg) // Spill value in G register before any merge.
			}
			if s.blockOrder[b.ID] > s.blockOrder[b.Succs[0].B.ID] {
				// No point if we've already regalloc'd the destination.
				goto badloop
			}
			// For this to be worthwhile, the loop must have no calls in it.
			top := b.Succs[0].B
			loop := s.loopnest.B2L[top.ID]
			if loop == nil || loop.Header != top || loop.ContainsUnavoidableCall {
				goto badloop
			}

			// Look into target block, find Phi arguments that come from b.
			phiArgs := regValLiveSet // reuse this space
			phiArgs.Clear()
			for _, v := range b.Succs[0].B.Values {
				if v.Op == ssaop.OpPhi {
					phiArgs.Add(v.Args[b.Succs[0].I].ID)
				}
			}

			// Get mask of all registers that might be used soon in the destination.
			// We don't want to kick values out of these registers, but we will
			// kick out an unlikely-to-be-used value for a likely-to-be-used one.
			var likelyUsedRegs ssaop.RegMask
			for _, live := range s.live[b.ID] {
				if live.dist < unlikelyDistance {
					likelyUsedRegs = likelyUsedRegs.Union(s.values[live.ID].Regs)
				}
			}
			// Promote values we're going to use soon in the destination to registers.
			// Note that this iterates nearest-use first, as we sorted
			// live lists by distance in computeLive.
			for _, live := range s.live[b.ID] {
				if live.dist >= unlikelyDistance {
					// Don't preload anything live after the loop.
					continue
				}
				vid := live.ID
				vi := &s.values[vid]
				v := s.orig[vid]
				if phiArgs.Contains(vid) {
					// A phi argument needs its value in a regular register,
					// as returned by compatRegs. Being in a fixed register
					// (e.g. the zero register) or being easily
					// rematerializeable isn't enough.
					if !vi.Regs.Intersect(s.compatRegs(v.Type)).Empty() {
						continue
					}
				} else {
					if !vi.Regs.Empty() {
						continue
					}
					if vi.Rematerializeable {
						// TODO: maybe we should not skip rematerializeable
						// values here. One rematerialization outside the loop
						// is better than N in the loop. But rematerializations
						// are cheap, and spilling another value may not be.
						// And we don't want to materialize the zero register
						// into a different register when it is just the
						// argument to a store.
						continue
					}
				}
				if vi.Rematerializeable && s.f.Config.Ctxt.Arch.Arch == sys.ArchWasm {
					continue
				}
				// Registers we could load v into.
				// Don't kick out other likely-used values.
				m := s.compatRegs(v.Type).Minus(likelyUsedRegs)
				if m.Empty() {
					// To many likely-used values to give them all a register.
					continue
				}

				// Used desired register if available.
			outerloop:
				for _, e := range desired.entries {
					if e.ID != v.ID {
						continue
					}
					for _, r := range e.regs {
						if r != noRegister && m.HasReg(r) {
							m = ssacore.RegMaskAt(r)
							break outerloop
						}
					}
				}
				if !m.Minus(desired.avoid).Empty() {
					m = m.Minus(desired.avoid)
				}
				s.allocValToReg(v, m, false, b.Pos)
				likelyUsedRegs = likelyUsedRegs.Union(s.values[v.ID].Regs)
			}
		}
	badloop:
		;

		// Save end-of-block register state.
		// First count how many, this cuts allocations in half.
		k := 0
		for r := ssaop.Register(0); r < s.numRegs; r++ {
			v := s.regs[r].v
			if v == nil {
				continue
			}
			k++
		}
		regList := make([]endReg, 0, k)
		for r := ssaop.Register(0); r < s.numRegs; r++ {
			v := s.regs[r].v
			if v == nil {
				continue
			}
			regList = append(regList, endReg{r, v, s.regs[r].c})
		}
		s.endRegs[b.ID] = regList

		if checkEnabled {
			regValLiveSet.Clear()
			if s.live != nil {
				for _, x := range s.live[b.ID] {
					regValLiveSet.Add(x.ID)
				}
			}
			for r := ssaop.Register(0); r < s.numRegs; r++ {
				v := s.regs[r].v
				if v == nil {
					continue
				}
				if !regValLiveSet.Contains(v.ID) {
					s.f.Fatalf("val %s is in reg but not live at end of %s", v, b)
				}
			}
		}

		// If a value is live at the end of the block and
		// isn't in a register, generate a use for the spill location.
		// We need to remember this information so that
		// the liveness analysis in stackalloc is correct.
		if s.live != nil {
			for _, e := range s.live[b.ID] {
				vi := &s.values[e.ID]
				if !vi.Regs.Empty() {
					// in a register, we'll use that source for the merge.
					continue
				}
				if vi.Rematerializeable {
					// we'll rematerialize during the merge.
					continue
				}
				if s.f.Pass.Debug > ssacore.RegDebug {
					fmt.Printf("live-at-end spill for %s at %s\n", s.orig[e.ID], b)
				}
				spill := s.makeSpill(s.orig[e.ID], b)
				s.spillLive[b.ID] = append(s.spillLive[b.ID], spill.ID)
			}

			// Clear any final uses.
			// All that is left should be the pseudo-uses added for values which
			// are live at the end of b.
			for _, e := range s.live[b.ID] {
				u := s.values[e.ID].Uses
				if u == nil {
					f.Fatalf("live at end, no uses v%d", e.ID)
				}
				if u.Next != nil {
					f.Fatalf("live at end, too many uses v%d", e.ID)
				}
				s.values[e.ID].Uses = nil
				u.Next = s.freeUseRecords
				s.freeUseRecords = u
			}
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
				if !s.startRegsMask.HasReg(sr.r) {
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
				if s.f.Pass.Debug > ssacore.RegDebug {
					fmt.Printf("delete copied value %s\n", c.LongString())
				}
				c.ResetArgs()
				f.FreeValue(c)
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
			if v.Op == ssaop.OpInvalid {
				continue
			}
			b.Values[i] = v
			i++
		}
		b.Values = b.Values[:i]
	}
}

func (s *regAllocState) placeSpills() {
	mustBeFirst := func(op ssaop.Op) bool {
		return op.IsLoweredGetClosurePtr() || op == ssaop.OpPhi || op == ssaop.OpArgIntReg || op == ssaop.OpArgFloatReg
	}

	// Start maps block IDs to the list of spills
	// that go at the start of the block (but after any phis).
	start := map[ssacore.ID][]*ssacore.Value{}
	// After maps value IDs to the list of spills
	// that go immediately after that value ID.
	after := map[ssacore.ID][]*ssacore.Value{}

	for i := range s.values {
		vi := s.values[i]
		spill := vi.Spill
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
		if s.loopnest != nil && s.loopnest.B2L[best.ID] != nil {
			bestDepth = s.loopnest.B2L[best.ID].Depth
		}
		b := best
		const maxSpillSearch = 100
		for i := 0; i < maxSpillSearch; i++ {
			// Find the child of b in the dominator tree which
			// dominates all restores.
			p := b
			b = nil
			for c := s.sdom.Child(p); c != nil && i < maxSpillSearch; c, i = s.sdom.Sibling(c), i+1 {
				if s.sdom[c.ID].Entry <= vi.RestoreMin && s.sdom[c.ID].Exit >= vi.RestoreMax {
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
			if s.loopnest != nil && s.loopnest.B2L[b.ID] != nil {
				depth = s.loopnest.B2L[b.ID].Depth
			}
			if depth > bestDepth {
				// Don't push the spill into a deeper loop.
				continue
			}

			// If v is in a register at the start of b, we can
			// place the spill here (after the phis).
			if len(b.Preds) == 1 {
				for _, e := range s.endRegs[b.Preds[0].B.ID] {
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
	var oldSched []*ssacore.Value
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
func (s *regAllocState) shuffle(stacklive [][]ssacore.ID) {
	var e edgeState
	e.s = s
	e.cache = map[ssacore.ID][]*ssacore.Value{}
	e.contents = map[ssacore.Location]contentRecord{}
	if s.f.Pass.Debug > ssacore.RegDebug {
		fmt.Printf("shuffle %s\n", s.f.Name)
		fmt.Println(s.f.String())
	}

	for _, b := range s.visitOrder {
		if len(b.Preds) <= 1 {
			continue
		}
		e.b = b
		for i, edge := range b.Preds {
			p := edge.B
			e.p = p
			e.setup(i, s.endRegs[p.ID], s.startRegs[b.ID], stacklive[p.ID])
			e.process()
		}
	}

	if s.f.Pass.Debug > ssacore.RegDebug {
		fmt.Printf("post shuffle %s\n", s.f.Name)
		fmt.Println(s.f.String())
	}
}

type edgeState struct {
	s    *regAllocState
	p, b *ssacore.Block // edge goes from p->b.

	// for each pre-regalloc value, a list of equivalent cached values
	cache      map[ssacore.ID][]*ssacore.Value
	cachedVals []ssacore.ID // (superset of) keys of the above map, for deterministic iteration

	// map from location to the value it contains
	contents map[ssacore.Location]contentRecord

	// desired destination locations
	destinations []dstRecord
	extra        []dstRecord

	usedRegs              ssaop.RegMask // registers currently holding something
	uniqueRegs            ssaop.RegMask // registers holding the only copy of a value
	finalRegs             ssaop.RegMask // registers holding final target
	rematerializeableRegs ssaop.RegMask // registers that hold rematerializeable values
}

type contentRecord struct {
	vid   ssacore.ID     // pre-regalloc value
	c     *ssacore.Value // cached value
	final bool           // this is a satisfied destination
	pos   src.XPos       // source position of use of the value
}

type dstRecord struct {
	loc    ssacore.Location // register or stack slot
	vid    ssacore.ID       // pre-regalloc value it should contain
	splice **ssacore.Value  // place to store reference to the generating instruction
	pos    src.XPos         // source position of use of this location
}

// setup initializes the edge state for shuffling.
func (e *edgeState) setup(idx int, srcReg []endReg, dstReg []startReg, stacklive []ssacore.ID) {
	if e.s.f.Pass.Debug > ssacore.RegDebug {
		fmt.Printf("edge %s->%s\n", e.p, e.b)
	}

	// Clear state.
	clear(e.cache)
	e.cachedVals = e.cachedVals[:0]
	clear(e.contents)
	e.usedRegs = ssaop.RegMask{}
	e.uniqueRegs = ssaop.RegMask{}
	e.finalRegs = ssaop.RegMask{}
	e.rematerializeableRegs = ssaop.RegMask{}

	// Live registers can be sources.
	for _, x := range srcReg {
		e.set(&e.s.registers[x.r], x.v.ID, x.c, false, src.NoXPos) // don't care the position of the source
	}
	// So can all of the spill locations.
	for _, spillID := range stacklive {
		v := e.s.orig[spillID]
		spill := e.s.values[v.ID].Spill
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
		e.set(e.s.f.GetHome(spillID), v.ID, spill, false, src.NoXPos) // don't care the position of the source
	}

	// Figure out all the destinations we need.
	dsts := e.destinations[:0]
	for _, x := range dstReg {
		dsts = append(dsts, dstRecord{&e.s.registers[x.r], x.v.ID, nil, x.pos})
	}
	// Phis need their args to end up in a specific location.
	for _, v := range e.b.Values {
		if v.Op != ssaop.OpPhi {
			break
		}
		loc := e.s.f.GetHome(v.ID)
		if loc == nil {
			continue
		}
		dsts = append(dsts, dstRecord{loc, v.Args[idx].ID, &v.Args[idx], v.Pos})
	}
	e.destinations = dsts

	if e.s.f.Pass.Debug > ssacore.RegDebug {
		for _, vid := range e.cachedVals {
			a := e.cache[vid]
			for _, c := range a {
				fmt.Printf("src %s: v%d cache=%s\n", e.s.f.GetHome(c.ID), vid, c)
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
		if e.s.f.Pass.Debug > ssacore.RegDebug {
			fmt.Printf("breaking cycle with v%d in %s:%s\n", vid, loc, c)
		}
		e.erase(r)
		pos := d.pos.WithNotStmt()
		if _, isReg := loc.(*ssabase.Register); isReg {
			c = e.p.NewValue1(pos, ssaop.OpCopy, c.Type, c)
		} else {
			c = e.p.NewValue1(pos, ssaop.OpLoadReg, c.Type, c)
		}
		e.set(r, vid, c, false, pos)
		if c.Op == ssaop.OpLoadReg && e.s.isGReg(ssaop.Register(r.(*ssabase.Register).Num)) {
			e.s.f.Fatalf("process.OpLoadReg targeting g: " + c.LongString())
		}
	}
}

// processDest generates code to put value vid into location loc. Returns true
// if progress was made.
func (e *edgeState) processDest(loc ssacore.Location, vid ssacore.ID, splice **ssacore.Value, pos src.XPos) bool {
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
	if len(e.cache[occupant.vid]) == 1 && !e.s.values[occupant.vid].Rematerializeable && !ssaop.OpcodeTable[e.s.orig[occupant.vid].Op].FixedReg {
		// We can't overwrite the last copy
		// of a value that needs to survive.
		return false
	}

	// Copy from a source of v, register preferred.
	v := e.s.orig[vid]
	var c *ssacore.Value
	var src ssacore.Location
	if e.s.f.Pass.Debug > ssacore.RegDebug {
		fmt.Printf("moving v%d to %s\n", vid, loc)
		fmt.Printf("sources of v%d:", vid)
	}
	if ssaop.OpcodeTable[v.Op].FixedReg {
		c = v
		src = e.s.f.GetHome(v.ID)
	} else {
		for _, w := range e.cache[vid] {
			h := e.s.f.GetHome(w.ID)
			if e.s.f.Pass.Debug > ssacore.RegDebug {
				fmt.Printf(" %s:%s", h, w)
			}
			_, isreg := h.(*ssabase.Register)
			if src == nil || isreg {
				c = w
				src = h
			}
		}
	}
	if e.s.f.Pass.Debug > ssacore.RegDebug {
		if src != nil {
			fmt.Printf(" [use %s]\n", src)
		} else {
			fmt.Printf(" [no source]\n")
		}
	}
	_, dstReg := loc.(*ssabase.Register)

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
	var x *ssacore.Value
	if c == nil || e.s.values[vid].Rematerializeable {
		if !e.s.values[vid].Rematerializeable {
			e.s.f.Fatalf("can't find source for %s->%s: %s\n", e.p, e.b, v.LongString())
		}
		if dstReg {
			// We want to rematerialize v into a register that is incompatible with v's op's register mask.
			// Instead of setting the wrong register for the rematerialized v, we should find the right register
			// for it and emit an additional copy to move to the desired register.
			// For #70451.
			if !e.s.regspec(v).Outputs[0].Regs.HasReg(ssaop.Register(loc.(*ssabase.Register).Num)) {
				_, srcReg := src.(*ssabase.Register)
				if srcReg {
					// It exists in a valid register already, so just copy it to the desired register
					// If src is a Register, c must have already been set.
					x = e.p.NewValue1(pos, ssaop.OpCopy, c.Type, c)
				} else {
					// We need a tmp register
					x = v.CopyInto(e.p)
					r := e.findRegFor(x.Type)
					e.erase(r)
					// Rematerialize to the tmp register
					e.set(r, vid, x, false, pos)
					// Copy from tmp to the desired register
					x = e.p.NewValue1(pos, ssaop.OpCopy, x.Type, x)
				}
			} else {
				x = v.CopyInto(e.p)
			}
		} else {
			// Rematerialize into stack slot. Need a free
			// register to accomplish this.
			r := e.findRegFor(v.Type)
			e.erase(r)
			x = v.CopyIntoWithXPos(e.p, pos)
			e.set(r, vid, x, false, pos)
			// Make sure we spill with the size of the slot, not the
			// size of x (which might be wider due to our dropping
			// of narrowing conversions).
			x = e.p.NewValue1(pos, ssaop.OpStoreReg, loc.(ssacore.LocalSlot).Type, x)
		}
	} else {
		// Emit move from src to dst.
		_, srcReg := src.(*ssabase.Register)
		if srcReg {
			if dstReg {
				x = e.p.NewValue1(pos, ssaop.OpCopy, c.Type, c)
			} else {
				x = e.p.NewValue1(pos, ssaop.OpStoreReg, loc.(ssacore.LocalSlot).Type, c)
			}
		} else {
			if dstReg {
				x = e.p.NewValue1(pos, ssaop.OpLoadReg, c.Type, c)
			} else {
				// mem->mem. Use temp register.
				r := e.findRegFor(c.Type)
				e.erase(r)
				t := e.p.NewValue1(pos, ssaop.OpLoadReg, c.Type, c)
				e.set(r, vid, t, false, pos)
				x = e.p.NewValue1(pos, ssaop.OpStoreReg, loc.(ssacore.LocalSlot).Type, t)
			}
		}
	}
	e.set(loc, vid, x, true, pos)
	if x.Op == ssaop.OpLoadReg && e.s.isGReg(ssaop.Register(loc.(*ssabase.Register).Num)) {
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
func (e *edgeState) set(loc ssacore.Location, vid ssacore.ID, c *ssacore.Value, final bool, pos src.XPos) {
	e.s.f.SetHome(c, loc)
	e.contents[loc] = contentRecord{vid, c, final, pos}
	a := e.cache[vid]
	if len(a) == 0 {
		e.cachedVals = append(e.cachedVals, vid)
	}
	a = append(a, c)
	e.cache[vid] = a
	if r, ok := loc.(*ssabase.Register); ok {
		if e.usedRegs.HasReg(ssaop.Register(r.Num)) {
			e.s.f.Fatalf("%v is already set (v%d/%v)", r, vid, c)
		}
		e.usedRegs = e.usedRegs.AddReg(ssaop.Register(r.Num))
		if final {
			e.finalRegs = e.finalRegs.AddReg(ssaop.Register(r.Num))
		}
		if len(a) == 1 {
			e.uniqueRegs = e.uniqueRegs.AddReg(ssaop.Register(r.Num))
		}
		if len(a) == 2 {
			if t, ok := e.s.f.GetHome(a[0].ID).(*ssabase.Register); ok {
				e.uniqueRegs = e.uniqueRegs.RemoveReg(ssaop.Register(t.Num))
			}
		}
		if e.s.values[vid].Rematerializeable {
			e.rematerializeableRegs = e.rematerializeableRegs.AddReg(ssaop.Register(r.Num))
		}
	}
	if e.s.f.Pass.Debug > ssacore.RegDebug {
		fmt.Printf("%s\n", c.LongString())
		fmt.Printf("v%d now available in %s:%s\n", vid, loc, c)
	}
}

// erase removes any user of loc.
func (e *edgeState) erase(loc ssacore.Location) {
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
		if e.s.f.GetHome(c.ID) == loc {
			if e.s.f.Pass.Debug > ssacore.RegDebug {
				fmt.Printf("v%d no longer available in %s:%s\n", vid, loc, c)
			}
			a[i], a = a[len(a)-1], a[:len(a)-1]
			break
		}
	}
	e.cache[vid] = a

	// Update register masks.
	if r, ok := loc.(*ssabase.Register); ok {
		e.usedRegs = e.usedRegs.RemoveReg(ssaop.Register(r.Num))
		if cr.final {
			e.finalRegs = e.finalRegs.RemoveReg(ssaop.Register(r.Num))
		}
		e.rematerializeableRegs = e.rematerializeableRegs.RemoveReg(ssaop.Register(r.Num))
	}
	if len(a) == 1 {
		if r, ok := e.s.f.GetHome(a[0].ID).(*ssabase.Register); ok {
			e.uniqueRegs = e.uniqueRegs.AddReg(ssaop.Register(r.Num))
		}
	}
}

// findRegFor finds a register we can use to make a temp copy of type typ.
func (e *edgeState) findRegFor(typ *types.Type) ssacore.Location {
	// Which registers are possibilities.
	m := e.s.compatRegs(typ)

	// Pick a register. In priority order:
	// 1) an unused register
	// 2) a non-unique register not holding a final value
	// 3) a non-unique register
	// 4) a register holding a rematerializeable value
	x := m.Minus(e.usedRegs)
	if !x.Empty() {
		return &e.s.registers[e.s.pickReg(x)]
	}
	x = m.Minus(e.uniqueRegs).Minus(e.finalRegs)
	if !x.Empty() {
		return &e.s.registers[e.s.pickReg(x)]
	}
	x = m.Minus(e.uniqueRegs)
	if !x.Empty() {
		return &e.s.registers[e.s.pickReg(x)]
	}
	x = m.Intersect(e.rematerializeableRegs)
	if !x.Empty() {
		return &e.s.registers[e.s.pickReg(x)]
	}

	// No register is available.
	// Pick a register to spill.
	for _, vid := range e.cachedVals {
		a := e.cache[vid]
		for _, c := range a {
			if r, ok := e.s.f.GetHome(c.ID).(*ssabase.Register); ok && m.HasReg(ssaop.Register(r.Num)) {
				if !c.Rematerializeable() {
					x := e.p.NewValue1(c.Pos, ssaop.OpStoreReg, c.Type, c)
					// Allocate a temp location to spill a register to.
					t := ssacore.LocalSlot{N: e.s.f.NewLocal(c.Pos, c.Type), Type: c.Type}
					// TODO: reuse these slots. They'll need to be erased first.
					e.set(t, vid, x, false, c.Pos)
					if e.s.f.Pass.Debug > ssacore.RegDebug {
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
			fmt.Printf("v%d: %s %s\n", vid, c, e.s.f.GetHome(c.ID))
		}
	}
	e.s.f.Fatalf("can't find empty register on edge %s->%s", e.p, e.b)
	return nil
}

type liveInfo struct {
	ID   ssacore.ID // ID of value
	dist int32      // # of instructions before next use
	pos  src.XPos   // source position of next use
}

// computeLive computes a map from block ID to a list of value IDs live at the end
// of that block. Together with the value ID is a count of how many instructions
// to the next use of that value. The resulting map is stored in s.live.
func (s *regAllocState) computeLive() {
	f := s.f
	// single block functions do not have variables that are live across
	// branches
	if len(f.Blocks) == 1 {
		return
	}
	po := f.Postorder()
	s.live = make([][]liveInfo, f.NumBlocks())
	s.desired = make([]desiredState, f.NumBlocks())
	s.loopnest = f.Loopnest()

	rematIDs := make([]ssacore.ID, 0, 64)

	live := f.NewSparseMapPos(f.NumValues())
	defer f.RetSparseMapPos(live)
	t := f.NewSparseMapPos(f.NumValues())
	defer f.RetSparseMapPos(t)

	s.loopnest.ComputeUnavoidableCalls()

	// Liveness analysis.
	// This is an adapted version of the algorithm described in chapter 2.4.2
	// of Fabrice Rastello's On Sparse Intermediate Representations.
	//   https://web.archive.org/web/20240417212122if_/https://inria.hal.science/hal-00761555/file/habilitation.pdf#section.50
	//
	// For our implementation, we fall back to a traditional iterative algorithm when we encounter
	// Irreducible CFGs. They are very uncommon in Go code because they need to be constructed with
	// gotos and our current loopnest definition does not compute all the information that
	// we'd need to compute the loop ancestors for that step of the algorithm.
	//
	// Additionally, instead of only considering non-loop successors in the initial DFS phase,
	// we compute the liveout as the union of all successors. This larger liveout set is a subset
	// of the final liveout for the block and adding this information in the DFS phase means that
	// we get slightly more accurate distance information.
	var loopLiveIn map[*ssacore.Loop][]liveInfo
	var numCalls []int32
	if len(s.loopnest.Loops) > 0 && !s.loopnest.HasIrreducible {
		loopLiveIn = make(map[*ssacore.Loop][]liveInfo)
		numCalls = f.Cache.AllocInt32Slice(f.NumBlocks())
		defer f.Cache.FreeInt32Slice(numCalls)
	}

	for {
		changed := false

		for _, b := range po {
			// Start with known live values at the end of the block.
			live.Clear()
			for _, e := range s.live[b.ID] {
				live.Set(e.ID, e.dist, e.pos)
			}
			update := false
			// arguments to phi nodes are live at this blocks out
			for _, e := range b.Succs {
				succ := e.B
				delta := branchDistance(b, succ)
				for _, v := range succ.Values {
					if v.Op != ssaop.OpPhi {
						break
					}
					arg := v.Args[e.I]
					if s.values[arg.ID].NeedReg && (!live.Contains(arg.ID) || delta < live.Get(arg.ID)) {
						live.Set(arg.ID, delta, v.Pos)
						update = true
					}
				}
			}
			if update {
				s.live[b.ID] = updateLive(live, s.live[b.ID])
			}
			// Add len(b.Values) to adjust from end-of-block distance
			// to beginning-of-block distance.
			c := live.Contents()
			for i := range c {
				c[i].Val += int32(len(b.Values))
			}

			// Mark control values as live
			for _, c := range b.ControlValues() {
				if s.values[c.ID].NeedReg {
					live.Set(c.ID, int32(len(b.Values)), b.Pos)
				}
			}

			for i := len(b.Values) - 1; i >= 0; i-- {
				v := b.Values[i]
				live.Remove(v.ID)
				if v.Op == ssaop.OpPhi {
					continue
				}
				if ssaop.OpcodeTable[v.Op].Call {
					if numCalls != nil {
						numCalls[b.ID]++
					}
					rematIDs = rematIDs[:0]
					c := live.Contents()
					for i := range c {
						c[i].Val += unlikelyDistance
						vid := c[i].Key
						if s.values[vid].Rematerializeable {
							rematIDs = append(rematIDs, vid)
						}
					}
					// We don't spill rematerializeable values, and assuming they
					// are live across a call would only force shuffle to add some
					// (dead) constant rematerialization. Remove them.
					for _, r := range rematIDs {
						live.Remove(r)
					}
				}
				for _, a := range v.Args {
					if s.values[a.ID].NeedReg {
						live.Set(a.ID, int32(i), v.Pos)
					}
				}
			}
			// This is a loop header, save our live-in so that
			// we can use it to fill in the loop bodies later
			if loopLiveIn != nil {
				loop := s.loopnest.B2L[b.ID]
				if loop != nil && loop.Header.ID == b.ID {
					loopLiveIn[loop] = updateLive(live, nil)
				}
			}
			// For each predecessor of b, expand its list of live-at-end values.
			// invariant: live contains the values live at the start of b
			for _, e := range b.Preds {
				p := e.B
				delta := branchDistance(p, b)

				// Start t off with the previously known live values at the end of p.
				t.Clear()
				for _, e := range s.live[p.ID] {
					t.Set(e.ID, e.dist, e.pos)
				}
				update := false

				// Add new live values from scanning this block.
				for _, e := range live.Contents() {
					d := e.Val + delta
					if !t.Contains(e.Key) || d < t.Get(e.Key) {
						update = true
						t.Set(e.Key, d, e.Pos)
					}
				}

				if !update {
					continue
				}
				s.live[p.ID] = updateLive(t, s.live[p.ID])
				changed = true
			}
		}

		// Doing a traditional iterative algorithm and have run
		// out of changes
		if !changed {
			break
		}

		// Doing a pre-pass and will fill in the liveness information
		// later
		if loopLiveIn != nil {
			break
		}
		// For loopless code, we have full liveness info after a single
		// iteration
		if len(s.loopnest.Loops) == 0 {
			break
		}
	}
	if f.Pass.Debug > ssacore.RegDebug {
		s.debugPrintLive("after dfs walk", f, s.live, s.desired)
	}

	// irreducible CFGs and functions without loops are already
	// done, compute their desired registers and return
	if loopLiveIn == nil {
		s.computeDesired()
		return
	}

	// Walk the loopnest from outer to inner, adding
	// all live-in values from their parent. Instead of
	// a recursive algorithm, iterate in depth order.
	// TODO(dmo): can we permute the loopnest? can we avoid this copy?
	loops := slices.Clone(s.loopnest.Loops)
	slices.SortFunc(loops, func(a, b *ssacore.Loop) int {
		return cmp.Compare(a.Depth, b.Depth)
	})

	loopset := f.NewSparseMapPos(f.NumValues())
	defer f.RetSparseMapPos(loopset)
	for _, loop := range loops {
		if loop.Outer == nil {
			continue
		}
		livein := loopLiveIn[loop]
		loopset.Clear()
		for _, l := range livein {
			loopset.Set(l.ID, l.dist, l.pos)
		}
		update := false
		for _, l := range loopLiveIn[loop.Outer] {
			if !loopset.Contains(l.ID) {
				loopset.Set(l.ID, l.dist, l.pos)
				update = true
			}
		}
		if update {
			loopLiveIn[loop] = updateLive(loopset, livein)
		}
	}
	// unknownDistance is a sentinel value for when we know a variable
	// is live at any given block, but we do not yet know how far until it's next
	// use. The distance will be computed later.
	const unknownDistance = -1

	// add live-in values of the loop headers to their children.
	// This includes the loop headers themselves, since they can have values
	// that die in the middle of the block and aren't live-out
	for _, b := range po {
		loop := s.loopnest.B2L[b.ID]
		if loop == nil {
			continue
		}
		headerLive := loopLiveIn[loop]
		loopset.Clear()
		for _, l := range s.live[b.ID] {
			loopset.Set(l.ID, l.dist, l.pos)
		}
		update := false
		for _, l := range headerLive {
			if !loopset.Contains(l.ID) {
				loopset.Set(l.ID, unknownDistance, src.NoXPos)
				update = true
			}
		}
		if update {
			s.live[b.ID] = updateLive(loopset, s.live[b.ID])
		}
	}
	if f.Pass.Debug > ssacore.RegDebug {
		s.debugPrintLive("after live loop prop", f, s.live, s.desired)
	}
	// Filling in liveness from loops leaves some blocks with no distance information
	// Run over them and fill in the information from their successors.
	// To stabilize faster, we quit when no block has missing values and we only
	// look at blocks that still have missing values in subsequent iterations
	unfinishedBlocks := f.Cache.AllocBlockSlice(len(po))
	defer f.Cache.FreeBlockSlice(unfinishedBlocks)
	copy(unfinishedBlocks, po)

	for len(unfinishedBlocks) > 0 {
		n := 0
		for _, b := range unfinishedBlocks {
			live.Clear()
			unfinishedValues := 0
			for _, l := range s.live[b.ID] {
				if l.dist == unknownDistance {
					unfinishedValues++
				}
				live.Set(l.ID, l.dist, l.pos)
			}
			update := false
			for _, e := range b.Succs {
				succ := e.B
				for _, l := range s.live[succ.ID] {
					if !live.Contains(l.ID) || l.dist == unknownDistance {
						continue
					}
					dist := int32(len(succ.Values)) + l.dist + branchDistance(b, succ)
					dist += numCalls[succ.ID] * unlikelyDistance
					val := live.Get(l.ID)
					switch {
					case val == unknownDistance:
						unfinishedValues--
						fallthrough
					case dist < val:
						update = true
						live.Set(l.ID, dist, l.pos)
					}
				}
			}
			if update {
				s.live[b.ID] = updateLive(live, s.live[b.ID])
			}
			if unfinishedValues > 0 {
				unfinishedBlocks[n] = b
				n++
			}
		}
		unfinishedBlocks = unfinishedBlocks[:n]
	}

	// Sort live values in order of their nearest next use.
	// Useful for promoting values to registers, nearest use first.
	for _, b := range f.Blocks {
		slices.SortFunc(s.live[b.ID], func(a, b liveInfo) int {
			if a.dist != b.dist {
				return cmp.Compare(a.dist, b.dist)
			}
			return cmp.Compare(a.ID, b.ID) // for deterministic sorting
		})
	}

	s.computeDesired()

	if f.Pass.Debug > ssacore.RegDebug {
		s.debugPrintLive("final", f, s.live, s.desired)
	}
}

// computeDesired computes the desired register information at the end of each block.
// It is essentially a liveness analysis on machine registers instead of SSA values
// The desired register information is stored in s.desired.
func (s *regAllocState) computeDesired() {

	// TODO: Can we speed this up using the liveness information we have already
	// from computeLive?
	var desired desiredState
	f := s.f
	po := f.Postorder()
	maxPreds := 0
	for _, b := range f.Blocks {
		maxPreds = max(maxPreds, len(b.Preds))
	}
	// phiPrefs[i] collects desired registers for phi inputs coming from b.Preds[i].
	phiPrefs := make([]desiredState, maxPreds)
	for {
		changed := false
		for _, b := range po {
			desired.copy(&s.desired[b.ID])
			for i := range b.Preds {
				phiPrefs[i].reset()
			}
			var headerLoop *ssacore.Loop // loop whose header is b, if any
			if l := s.loopnest.B2L[b.ID]; l != nil && l.Header == b {
				headerLoop = l
			}
			// Process non-phis, then phis.
			i := len(b.Values) - 1
			for ; i >= 0; i-- {
				v := b.Values[i]
				if v.Op == ssaop.OpPhi {
					break
				}
				prefs := desired.remove(v.ID)
				regspec := s.regspec(v)
				// Cancel desired registers if they get clobbered.
				desired.clobber(regspec.Clobbers)
				// Update desired registers if there are any fixed register inputs.
				for _, j := range regspec.Inputs {
					if countRegs(j.Regs) != 1 {
						continue
					}
					desired.clobber(j.Regs)
					desired.add(v.Args[j.Idx].ID, s.pickReg(j.Regs))
				}
				// Set desired register of input 0 if this is a 2-operand instruction.
				if ssaop.OpcodeTable[v.Op].ResultInArg0 || v.Op == ssaop.OpAMD64ADDQconst || v.Op == ssaop.OpAMD64ADDLconst || v.Op == ssaop.OpSelect0 {
					// ADDQconst is added here because we want to treat it as resultInArg0 for
					// the purposes of desired registers, even though it is not an absolute requirement.
					// This is because we'd rather implement it as ADDQ instead of LEAQ.
					// Same for ADDLconst
					// Select0 is added here to propagate the desired register to the tuple-generating instruction.
					if ssaop.OpcodeTable[v.Op].Commutative {
						desired.addList(v.Args[1].ID, prefs)
					}
					desired.addList(v.Args[0].ID, prefs)
				}
			}
			for ; i >= 0; i-- {
				v := b.Values[i]
				prefs := desired.remove(v.ID)
				if prefs[0] == noRegister {
					continue
				}
				// Phi desires go to phiPrefs (per-pred), so drop them from desired.avoid.
				// The merge below re-adds any bits other entries still need.
				for _, r := range prefs {
					if r != noRegister {
						desired.avoid = desired.avoid.Minus(ssacore.RegMaskAt(r))
					}
				}
				// Propagate v's desired registers back to its args.
				for pidx, a := range v.Args {
					if headerLoop != nil && s.loopnest.B2L[b.Preds[pidx].B.ID] == headerLoop {
						// Skip direct back-edges to avoid pessimizing the loop body to skip a single reg-reg move.
						// We check only the immediate loop; it is simple and empirically sufficient.
						continue
					}
					phiPrefs[pidx].addList(a.ID, prefs)
				}
			}
			for pidx, e := range b.Preds {
				p := e.B
				changed = s.desired[p.ID].merge(&desired) || changed
				changed = s.desired[p.ID].merge(&phiPrefs[pidx]) || changed
			}
		}
		if !changed || (!s.loopnest.HasIrreducible && len(s.loopnest.Loops) == 0) {
			break
		}
	}
}

// updateLive updates a given liveInfo slice with the contents of t
func updateLive(t *ssacore.SparseMapPos, live []liveInfo) []liveInfo {
	live = live[:0]
	if cap(live) < t.Size() {
		live = make([]liveInfo, 0, t.Size())
	}
	for _, e := range t.Contents() {
		live = append(live, liveInfo{e.Key, e.Val, e.Pos})
	}
	return live
}

// branchDistance calculates the distance between a block and a
// successor in pseudo-instructions. This is used to indicate
// likeliness
func branchDistance(b *ssacore.Block, s *ssacore.Block) int32 {
	if len(b.Succs) == 2 {
		if b.Succs[0].B == s && b.Likely == ssacore.BranchLikely ||
			b.Succs[1].B == s && b.Likely == ssacore.BranchUnlikely {
			return likelyDistance
		}
		if b.Succs[0].B == s && b.Likely == ssacore.BranchUnlikely ||
			b.Succs[1].B == s && b.Likely == ssacore.BranchLikely {
			return unlikelyDistance
		}
	}
	// Note: the branch distance must be at least 1 to distinguish the control
	// value use from the first user in a successor block.
	return normalDistance
}

func (s *regAllocState) debugPrintLive(stage string, f *ssacore.Func, live [][]liveInfo, desired []desiredState) {
	fmt.Printf("%s: live values at end of each block: %s\n", stage, f.Name)
	for _, b := range f.Blocks {
		s.debugPrintLiveBlock(b, live[b.ID], &desired[b.ID])
	}
}

func (s *regAllocState) debugPrintLiveBlock(b *ssacore.Block, live []liveInfo, desired *desiredState) {
	fmt.Printf("  %s:", b)
	slices.SortFunc(live, func(a, b liveInfo) int {
		return cmp.Compare(a.ID, b.ID)
	})
	for _, x := range live {
		fmt.Printf(" v%d(%d)", x.ID, x.dist)
		for _, e := range desired.entries {
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
	if avoid := desired.avoid; !avoid.Empty() {
		fmt.Printf(" avoid=%v", s.RegMaskString(avoid))
	}
	fmt.Println()
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
	avoid ssaop.RegMask
}
type desiredStateEntry struct {
	// (pre-regalloc) value
	ID ssacore.ID
	// Registers it would like to be in, in priority order.
	// Unused slots are filled with noRegister.
	// For opcodes that return tuples, we track desired registers only
	// for the first element of the tuple (see desiredSecondReg for
	// tracking the desired register for second part of a tuple).
	regs [4]ssaop.Register
}

// get returns a list of desired registers for value vid.
func (d *desiredState) get(vid ssacore.ID) [4]ssaop.Register {
	for _, e := range d.entries {
		if e.ID == vid {
			return e.regs
		}
	}
	return [4]ssaop.Register{noRegister, noRegister, noRegister, noRegister}
}

// add records that we'd like value vid to be in register r.
func (d *desiredState) add(vid ssacore.ID, r ssaop.Register) {
	d.avoid = d.avoid.AddReg(r)
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
	d.entries = append(d.entries, desiredStateEntry{vid, [4]ssaop.Register{r, noRegister, noRegister, noRegister}})
}

func (d *desiredState) addList(vid ssacore.ID, regs [4]ssaop.Register) {
	// regs is in priority order, so iterate in reverse order.
	for i := len(regs) - 1; i >= 0; i-- {
		r := regs[i]
		if r != noRegister {
			d.add(vid, r)
		}
	}
}

// clobber erases any desired registers in the set m.
func (d *desiredState) clobber(m ssaop.RegMask) {
	for i := 0; i < len(d.entries); {
		e := &d.entries[i]
		j := 0
		for _, r := range e.regs {
			if r != noRegister && !m.HasReg(r) {
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
	d.avoid = d.avoid.Minus(m)
}

// reset prepares d for re-use.
func (d *desiredState) reset() {
	d.entries = d.entries[:0]
	d.avoid = ssaop.RegMask{}
}

// copy copies a desired state from another desiredState x.
func (d *desiredState) copy(x *desiredState) {
	d.entries = append(d.entries[:0], x.entries...)
	d.avoid = x.avoid
}

// remove removes the desired registers for vid and returns them.
func (d *desiredState) remove(vid ssacore.ID) [4]ssaop.Register {
	for i := range d.entries {
		if d.entries[i].ID == vid {
			regs := d.entries[i].regs
			d.entries[i] = d.entries[len(d.entries)-1]
			d.entries = d.entries[:len(d.entries)-1]
			return regs
		}
	}
	return [4]ssaop.Register{noRegister, noRegister, noRegister, noRegister}
}

// merge merges another desired state x into d. Returns whether the set has
// changed
func (d *desiredState) merge(x *desiredState) bool {
	oldAvoid := d.avoid
	d.avoid = d.avoid.Union(x.avoid)
	// There should only be a few desired registers, so
	// linear insert is ok.
	for _, e := range x.entries {
		d.addList(e.ID, e.regs)
	}
	return oldAvoid != d.avoid
}
