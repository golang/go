// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package ssa

import (
	"cmd/internal/obj"
	"fmt"
	"strings"
)

type SlotID int32

// A FuncDebug contains all the debug information for the variables in a
// function. Variables are identified by their LocalSlot, which may be the
// result of decomposing a larger variable.
type FuncDebug struct {
	// Slots are all the slots in the function, indexed by their SlotID as
	// used in various functions and parallel to BlockDebug.Variables.
	Slots []*LocalSlot
	// The blocks in the function, in program text order.
	Blocks []*BlockDebug
	// The registers of the current architecture, indexed by Register.num.
	Registers []Register
}

func (f *FuncDebug) BlockString(b *BlockDebug) string {
	var vars []string

	for slot, list := range b.Variables {
		if len(list.Locations) == 0 {
			continue
		}
		vars = append(vars, fmt.Sprintf("%v = %v", f.Slots[slot], list))
	}
	return fmt.Sprintf("{%v}", strings.Join(vars, ", "))
}

func (f *FuncDebug) SlotLocsString(id SlotID) string {
	var locs []string
	for _, block := range f.Blocks {
		for _, loc := range block.Variables[id].Locations {
			locs = append(locs, block.LocString(loc))
		}
	}
	return strings.Join(locs, " ")
}

type BlockDebug struct {
	// The SSA block that this tracks. For debug logging only.
	Block *Block
	// The variables in this block, indexed by their SlotID.
	Variables []VarLocList
}

func (b *BlockDebug) LocString(loc *VarLoc) string {
	registers := b.Block.Func.Config.registers

	var storage []string
	if loc.OnStack {
		storage = append(storage, "stack")
	}

	for reg := 0; reg < 64; reg++ {
		if loc.Registers&(1<<uint8(reg)) == 0 {
			continue
		}
		if registers != nil {
			storage = append(storage, registers[reg].String())
		} else {
			storage = append(storage, fmt.Sprintf("reg%d", reg))
		}
	}
	if len(storage) == 0 {
		storage = append(storage, "!!!no storage!!!")
	}
	pos := func(v *Value, p *obj.Prog, pc int64) string {
		if v == nil {
			return "?"
		}
		vStr := fmt.Sprintf("v%d", v.ID)
		if v == BlockStart {
			vStr = fmt.Sprintf("b%dStart", b.Block.ID)
		}
		if v == BlockEnd {
			vStr = fmt.Sprintf("b%dEnd", b.Block.ID)
		}
		if p == nil {
			return vStr
		}
		return fmt.Sprintf("%s/%x", vStr, pc)
	}
	start := pos(loc.Start, loc.StartProg, loc.StartPC)
	end := pos(loc.End, loc.EndProg, loc.EndPC)
	return fmt.Sprintf("%v-%v@%s", start, end, strings.Join(storage, ","))

}

// append adds a location to the location list for slot.
func (b *BlockDebug) append(slot SlotID, loc *VarLoc) {
	b.Variables[slot].append(loc)
}

// lastLoc returns the last VarLoc for slot, or nil if it has none.
func (b *BlockDebug) lastLoc(slot SlotID) *VarLoc {
	return b.Variables[slot].last()
}

// A VarLocList contains the locations for a variable, in program text order.
// It will often have gaps.
type VarLocList struct {
	Locations []*VarLoc
}

func (l *VarLocList) append(loc *VarLoc) {
	l.Locations = append(l.Locations, loc)
}

// last returns the last location in the list.
func (l *VarLocList) last() *VarLoc {
	if l == nil || len(l.Locations) == 0 {
		return nil
	}
	return l.Locations[len(l.Locations)-1]
}

// A VarLoc describes a variable's location in a single contiguous range
// of program text. It is generated from the SSA representation, but it
// refers to the generated machine code, so the Values referenced are better
// understood as PCs than actual Values, and the ranges can cross blocks.
// The range is defined first by Values, which are then mapped to Progs
// during genssa and finally to function PCs after assembly.
// A variable can be on the stack and in any number of registers.
type VarLoc struct {
	// Inclusive -- the first SSA value that the range covers. The value
	// doesn't necessarily have anything to do with the variable; it just
	// identifies a point in the program text.
	// The special sentinel value BlockStart indicates that the range begins
	// at the beginning of the containing block, even if the block doesn't
	// actually have a Value to use to indicate that.
	Start *Value
	// Exclusive -- the first SSA value after start that the range doesn't
	// cover. A location with start == end is empty.
	// The special sentinel value BlockEnd indicates that the variable survives
	// to the end of the of the containing block, after all its Values and any
	// control flow instructions added later.
	End *Value

	// The prog/PCs corresponding to Start and End above. These are for the
	// convenience of later passes, since code generation isn't done when
	// BuildFuncDebug runs.
	// Control flow instructions don't correspond to a Value, so EndProg
	// may point to a Prog in the next block if SurvivedBlock is true. For
	// the last block, where there's no later Prog, it will be nil to indicate
	// the end of the function.
	StartProg, EndProg *obj.Prog
	StartPC, EndPC     int64

	// The registers this variable is available in. There can be more than
	// one in various situations, e.g. it's being moved between registers.
	Registers RegisterSet
	// Indicates whether the variable is on the stack. The stack position is
	// stored in the associated gc.Node.
	OnStack bool
}

var BlockStart = &Value{
	ID:  -10000,
	Op:  OpInvalid,
	Aux: "BlockStart",
}

var BlockEnd = &Value{
	ID:  -20000,
	Op:  OpInvalid,
	Aux: "BlockEnd",
}

// RegisterSet is a bitmap of registers, indexed by Register.num.
type RegisterSet uint64

// unexpected is used to indicate an inconsistency or bug in the debug info
// generation process. These are not fixable by users. At time of writing,
// changing this to a Fprintf(os.Stderr) and running make.bash generates
// thousands of warnings.
func (s *debugState) unexpected(v *Value, msg string, args ...interface{}) {
	s.f.Logf("unexpected at "+fmt.Sprint(v.ID)+":"+msg, args...)
}

func (s *debugState) logf(msg string, args ...interface{}) {
	s.f.Logf(msg, args...)
}

type debugState struct {
	loggingEnabled bool
	slots          []*LocalSlot
	f              *Func
	cache          *Cache
	numRegisters   int

	// working storage for BuildFuncDebug, reused between blocks.
	registerContents [][]SlotID
}

func (s *debugState) BlockString(b *BlockDebug) string {
	f := &FuncDebug{
		Slots:     s.slots,
		Registers: s.f.Config.registers,
	}
	return f.BlockString(b)
}

// BuildFuncDebug returns debug information for f.
// f must be fully processed, so that each Value is where it will be when
// machine code is emitted.
func BuildFuncDebug(f *Func, loggingEnabled bool) *FuncDebug {
	if f.RegAlloc == nil {
		f.Fatalf("BuildFuncDebug on func %v that has not been fully processed", f)
	}
	state := &debugState{
		loggingEnabled:   loggingEnabled,
		slots:            make([]*LocalSlot, len(f.Names)),
		cache:            f.Cache,
		f:                f,
		numRegisters:     len(f.Config.registers),
		registerContents: make([][]SlotID, len(f.Config.registers)),
	}
	// TODO: consider storing this in Cache and reusing across functions.
	valueNames := make([][]SlotID, f.NumValues())

	for i, slot := range f.Names {
		slot := slot
		state.slots[i] = &slot

		if isSynthetic(&slot) {
			continue
		}
		for _, value := range f.NamedValues[slot] {
			valueNames[value.ID] = append(valueNames[value.ID], SlotID(i))
		}
	}

	if state.loggingEnabled {
		var names []string
		for i, name := range f.Names {
			names = append(names, fmt.Sprintf("%d = %s", i, name))
		}
		state.logf("Name table: %v\n", strings.Join(names, ", "))
	}

	// Build up block states, starting with the first block, then
	// processing blocks once their predecessors have been processed.

	// TODO: use a reverse post-order traversal instead of the work queue.

	// Location list entries for each block.
	blockLocs := make([]*BlockDebug, f.NumBlocks())

	// Work queue of blocks to visit. Some of them may already be processed.
	work := []*Block{f.Entry}

	for len(work) > 0 {
		b := work[0]
		work = work[1:]
		if blockLocs[b.ID] != nil {
			continue // already processed
		}
		if !state.predecessorsDone(b, blockLocs) {
			continue // not ready yet
		}

		for _, edge := range b.Succs {
			if blockLocs[edge.Block().ID] != nil {
				continue
			}
			work = append(work, edge.Block())
		}

		// Build the starting state for the block from the final
		// state of its predecessors.
		locs := state.mergePredecessors(b, blockLocs)
		if state.loggingEnabled {
			state.logf("Processing %v, initial locs %v, regs %v\n", b, state.BlockString(locs), state.registerContents)
		}
		// Update locs/registers with the effects of each Value.
		for _, v := range b.Values {
			slots := valueNames[v.ID]

			// Loads and stores inherit the names of their sources.
			var source *Value
			switch v.Op {
			case OpStoreReg:
				source = v.Args[0]
			case OpLoadReg:
				switch a := v.Args[0]; a.Op {
				case OpArg:
					source = a
				case OpStoreReg:
					source = a.Args[0]
				default:
					state.unexpected(v, "load with unexpected source op %v", a)
				}
			}
			if source != nil {
				slots = append(slots, valueNames[source.ID]...)
				// As of writing, the compiler never uses a load/store as a
				// source of another load/store, so there's no reason this should
				// ever be consulted. Update just in case, and so that when
				// valueNames is cached, we can reuse the memory.
				valueNames[v.ID] = slots
			}

			if len(slots) == 0 {
				continue
			}

			reg, _ := f.getHome(v.ID).(*Register)
			state.processValue(locs, v, slots, reg)

		}

		// The block is done; mark any live locations as ending with the block.
		for _, locList := range locs.Variables {
			last := locList.last()
			if last == nil || last.End != nil {
				continue
			}
			last.End = BlockEnd
		}
		if state.loggingEnabled {
			f.Logf("Block done: locs %v, regs %v. work = %+v\n", state.BlockString(locs), state.registerContents, work)
		}
		blockLocs[b.ID] = locs
	}

	info := &FuncDebug{
		Slots:     state.slots,
		Registers: f.Config.registers,
	}
	// Consumers want the information in textual order, not by block ID.
	for _, b := range f.Blocks {
		info.Blocks = append(info.Blocks, blockLocs[b.ID])
	}

	if state.loggingEnabled {
		f.Logf("Final result:\n")
		for slot := range info.Slots {
			f.Logf("\t%v => %v\n", info.Slots[slot], info.SlotLocsString(SlotID(slot)))
		}
	}
	return info
}

// isSynthetic reports whether if slot represents a compiler-inserted variable,
// e.g. an autotmp or an anonymous return value that needed a stack slot.
func isSynthetic(slot *LocalSlot) bool {
	c := slot.String()[0]
	return c == '.' || c == '~'
}

// predecessorsDone reports whether block is ready to be processed.
func (state *debugState) predecessorsDone(b *Block, blockLocs []*BlockDebug) bool {
	f := b.Func
	for _, edge := range b.Preds {
		// Ignore back branches, e.g. the continuation of a for loop.
		// This may not work for functions with mutual gotos, which are not
		// reducible, in which case debug information will be missing for any
		// code after that point in the control flow.
		if f.sdom().isAncestorEq(b, edge.b) {
			if state.loggingEnabled {
				f.Logf("ignoring back branch from %v to %v\n", edge.b, b)
			}
			continue // back branch
		}
		if blockLocs[edge.b.ID] == nil {
			if state.loggingEnabled {
				f.Logf("%v is not ready because %v isn't done\n", b, edge.b)
			}
			return false
		}
	}
	return true
}

// mergePredecessors takes the end state of each of b's predecessors and
// intersects them to form the starting state for b.
// The registers slice (the second return value) will be reused for each call to mergePredecessors.
func (state *debugState) mergePredecessors(b *Block, blockLocs []*BlockDebug) *BlockDebug {
	live := make([]VarLocList, len(state.slots))

	// Filter out back branches.
	var preds []*Block
	for _, pred := range b.Preds {
		if blockLocs[pred.b.ID] != nil {
			preds = append(preds, pred.b)
		}
	}

	if len(preds) > 0 {
		p := preds[0]
		for slot, locList := range blockLocs[p.ID].Variables {
			last := locList.last()
			if last == nil || last.End != BlockEnd {
				continue
			}
			loc := state.cache.NewVarLoc()
			loc.Start = BlockStart
			loc.OnStack = last.OnStack
			loc.Registers = last.Registers
			live[slot].append(loc)
		}
	}
	if state.loggingEnabled && len(b.Preds) > 1 {
		state.logf("Starting merge with state from %v: %v\n", b.Preds[0].b, state.BlockString(blockLocs[b.Preds[0].b.ID]))
	}
	for i := 1; i < len(preds); i++ {
		p := preds[i]
		if state.loggingEnabled {
			state.logf("Merging in state from %v: %v &= %v\n", p, live, state.BlockString(blockLocs[p.ID]))
		}

		for slot, liveVar := range live {
			liveLoc := liveVar.last()
			if liveLoc == nil {
				continue
			}

			predLoc := blockLocs[p.ID].Variables[SlotID(slot)].last()
			// Clear out slots missing/dead in p.
			if predLoc == nil || predLoc.End != BlockEnd {
				live[slot].Locations = nil
				continue
			}

			// Unify storage locations.
			liveLoc.OnStack = liveLoc.OnStack && predLoc.OnStack
			liveLoc.Registers &= predLoc.Registers
		}
	}

	// Create final result.
	locs := &BlockDebug{Variables: live}
	if state.loggingEnabled {
		locs.Block = b
	}
	for reg := range state.registerContents {
		state.registerContents[reg] = state.registerContents[reg][:0]
	}
	for slot, locList := range live {
		loc := locList.last()
		if loc == nil {
			continue
		}
		for reg := 0; reg < state.numRegisters; reg++ {
			if loc.Registers&(1<<uint8(reg)) != 0 {
				state.registerContents[reg] = append(state.registerContents[reg], SlotID(slot))
			}
		}
	}
	return locs
}

// processValue updates locs and state.registerContents to reflect v, a value with
// the names in vSlots and homed in vReg.
func (state *debugState) processValue(locs *BlockDebug, v *Value, vSlots []SlotID, vReg *Register) {
	switch {
	case v.Op == OpRegKill:
		if state.loggingEnabled {
			existingSlots := make([]bool, len(state.slots))
			for _, slot := range state.registerContents[vReg.num] {
				existingSlots[slot] = true
			}
			for _, slot := range vSlots {
				if existingSlots[slot] {
					existingSlots[slot] = false
				} else {
					state.unexpected(v, "regkill of unassociated name %v\n", state.slots[slot])
				}
			}
			for slot, live := range existingSlots {
				if live {
					state.unexpected(v, "leftover register name: %v\n", state.slots[slot])
				}
			}
		}
		state.registerContents[vReg.num] = nil

		for _, slot := range vSlots {
			last := locs.lastLoc(slot)
			if last == nil {
				state.unexpected(v, "regkill of already dead %s, %+v\n", vReg, state.slots[slot])
				continue
			}
			if state.loggingEnabled {
				state.logf("at %v: %v regkilled out of %s\n", v.ID, state.slots[slot], vReg)
			}
			if last.End != nil {
				state.unexpected(v, "regkill of dead slot, died at %v\n", last.End)
			}
			last.End = v

			regs := last.Registers &^ (1 << uint8(vReg.num))
			if !last.OnStack && regs == 0 {
				continue
			}
			loc := state.cache.NewVarLoc()
			loc.Start = v
			loc.OnStack = last.OnStack
			loc.Registers = regs
			locs.append(slot, loc)
		}
	case v.Op == OpArg:
		for _, slot := range vSlots {
			if last := locs.lastLoc(slot); last != nil {
				state.unexpected(v, "Arg op on already-live slot %v", state.slots[slot])
				last.End = v
			}
			if state.loggingEnabled {
				state.logf("at %v: %v now on stack from arg\n", v.ID, state.slots[slot])
			}
			loc := state.cache.NewVarLoc()
			loc.Start = v
			loc.OnStack = true
			locs.append(slot, loc)
		}

	case v.Op == OpStoreReg:
		for _, slot := range vSlots {
			if state.loggingEnabled {
				state.logf("at %v: %v spilled to stack\n", v.ID, state.slots[slot])
			}
			last := locs.lastLoc(slot)
			if last == nil {
				state.unexpected(v, "spill of unnamed register %s\n", vReg)
				break
			}
			last.End = v
			loc := state.cache.NewVarLoc()
			loc.Start = v
			loc.OnStack = true
			loc.Registers = last.Registers
			locs.append(slot, loc)
		}

	case vReg != nil:
		if state.loggingEnabled {
			newSlots := make([]bool, len(state.slots))
			for _, slot := range vSlots {
				newSlots[slot] = true
			}

			for _, slot := range state.registerContents[vReg.num] {
				if !newSlots[slot] {
					state.unexpected(v, "%v clobbered\n", state.slots[slot])
				}
			}
		}

		for _, slot := range vSlots {
			if state.loggingEnabled {
				state.logf("at %v: %v now in %s\n", v.ID, state.slots[slot], vReg)
			}
			last := locs.lastLoc(slot)
			if last != nil && last.End == nil {
				last.End = v
			}
			state.registerContents[vReg.num] = append(state.registerContents[vReg.num], slot)
			loc := state.cache.NewVarLoc()
			loc.Start = v
			if last != nil {
				loc.OnStack = last.OnStack
				loc.Registers = last.Registers
			}
			loc.Registers |= 1 << uint8(vReg.num)
			locs.append(slot, loc)
		}
	default:
		state.unexpected(v, "named value with no reg\n")
	}
}
