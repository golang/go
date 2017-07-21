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
	Slots     []*LocalSlot
	Variables []VarLocList
	Registers []Register
}

// append adds a location to the location list for slot.
func (f *FuncDebug) append(slot SlotID, loc *VarLoc) {
	f.Variables[slot].append(loc)
}

// lastLoc returns the last VarLoc for slot, or nil if it has none.
func (f *FuncDebug) lastLoc(slot SlotID) *VarLoc {
	return f.Variables[slot].last()
}

func (f *FuncDebug) String() string {
	var vars []string
	for slot, list := range f.Variables {
		if len(list.Locations) == 0 {
			continue
		}
		vars = append(vars, fmt.Sprintf("%v = %v", f.Slots[slot], list))
	}
	return fmt.Sprintf("{%v}", strings.Join(vars, ", "))
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
	Start *Value
	// Exclusive -- the first SSA value after start that the range doesn't
	// cover. A location with start == end is empty.
	End *Value
	// The prog/PCs corresponding to Start and End above. These are for the
	// convenience of later passes, since code generation isn't done when
	// BuildFuncDebug runs.
	StartProg, EndProg *obj.Prog
	StartPC, EndPC     int64

	// The registers this variable is available in. There can be more than
	// one in various situations, e.g. it's being moved between registers.
	Registers RegisterSet
	// Indicates whether the variable is on the stack. The stack position is
	// stored in the associated gc.Node.
	OnStack bool

	// Used only during generation. Indicates whether this location lasts
	// past the block's end. Without this, there would be no way to distinguish
	// between a range that ended on the last Value of a block and one that
	// didn't end at all.
	survivedBlock bool
}

// RegisterSet is a bitmap of registers, indexed by Register.num.
type RegisterSet uint64

func (v *VarLoc) String() string {
	var registers []Register
	if v.Start != nil {
		registers = v.Start.Block.Func.Config.registers
	}
	loc := ""
	if !v.OnStack && v.Registers == 0 {
		loc = "!!!no location!!!"
	}
	if v.OnStack {
		loc += "stack,"
	}
	var regnames []string
	for reg := 0; reg < 64; reg++ {
		if v.Registers&(1<<uint8(reg)) == 0 {
			continue
		}
		if registers != nil {
			regnames = append(regnames, registers[reg].Name())
		} else {
			regnames = append(regnames, fmt.Sprintf("reg%v", reg))
		}
	}
	loc += strings.Join(regnames, ",")
	pos := func(v *Value, p *obj.Prog, pc int64) string {
		if v == nil {
			return "?"
		}
		if p == nil {
			return fmt.Sprintf("v%v", v.ID)
		}
		return fmt.Sprintf("v%v/%x", v.ID, pc)
	}
	surv := ""
	if v.survivedBlock {
		surv = "+"
	}
	return fmt.Sprintf("%v-%v%s@%s", pos(v.Start, v.StartProg, v.StartPC), pos(v.End, v.EndProg, v.EndPC), surv, loc)
}

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
			names = append(names, fmt.Sprintf("%v = %v", i, name))
		}
		state.logf("Name table: %v\n", strings.Join(names, ", "))
	}

	// Build up block states, starting with the first block, then
	// processing blocks once their predecessors have been processed.

	// TODO: use a reverse post-order traversal instead of the work queue.

	// Location list entries for each block.
	blockLocs := make([]*FuncDebug, f.NumBlocks())

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
			state.logf("Processing %v, initial locs %v, regs %v\n", b, locs, state.registerContents)
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

		// The block is done; end the locations for all its slots.
		for _, locList := range locs.Variables {
			last := locList.last()
			if last == nil || last.End != nil {
				continue
			}
			if len(b.Values) != 0 {
				last.End = b.Values[len(b.Values)-1]
			} else {
				// This happens when a value survives into an empty block from its predecessor.
				// Just carry it forward for liveness's sake.
				last.End = last.Start
			}
			last.survivedBlock = true
		}
		if state.loggingEnabled {
			f.Logf("Block done: locs %v, regs %v. work = %+v\n", locs, state.registerContents, work)
		}
		blockLocs[b.ID] = locs
	}

	// Build the complete debug info by concatenating each of the blocks'
	// locations together.
	info := &FuncDebug{
		Variables: make([]VarLocList, len(state.slots)),
		Slots:     state.slots,
		Registers: f.Config.registers,
	}
	for _, b := range f.Blocks {
		// Ignore empty blocks; there will be some records for liveness
		// but they're all useless.
		if len(b.Values) == 0 {
			continue
		}
		if blockLocs[b.ID] == nil {
			state.unexpected(b.Values[0], "Never processed block %v\n", b)
			continue
		}
		for slot, blockLocList := range blockLocs[b.ID].Variables {
			for _, loc := range blockLocList.Locations {
				if !loc.OnStack && loc.Registers == 0 {
					state.unexpected(loc.Start, "Location for %v with no storage: %+v\n", state.slots[slot], loc)
					continue // don't confuse downstream with our bugs
				}
				if loc.Start == nil || loc.End == nil {
					state.unexpected(b.Values[0], "Location for %v missing start or end: %v\n", state.slots[slot], loc)
					continue
				}
				info.append(SlotID(slot), loc)
			}
		}
	}
	if state.loggingEnabled {
		f.Logf("Final result:\n")
		for slot, locList := range info.Variables {
			f.Logf("\t%v => %v\n", state.slots[slot], locList)
		}
	}
	return info
}

// isSynthetic reports whether if slot represents a compiler-inserted variable,
// e.g. an autotmp or an anonymous return value that needed a stack slot.
func isSynthetic(slot *LocalSlot) bool {
	c := slot.Name()[0]
	return c == '.' || c == '~'
}

// predecessorsDone reports whether block is ready to be processed.
func (state *debugState) predecessorsDone(b *Block, blockLocs []*FuncDebug) bool {
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
func (state *debugState) mergePredecessors(b *Block, blockLocs []*FuncDebug) *FuncDebug {
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
			if last == nil || !last.survivedBlock {
				continue
			}
			// If this block is empty, carry forward the end value for liveness.
			// It'll be ignored later.
			start := last.End
			if len(b.Values) != 0 {
				start = b.Values[0]
			}
			loc := state.cache.NewVarLoc()
			loc.Start = start
			loc.OnStack = last.OnStack
			loc.Registers = last.Registers
			live[slot].append(loc)
		}
	}
	if state.loggingEnabled && len(b.Preds) > 1 {
		state.logf("Starting merge with state from %v: %v\n", b.Preds[0].b, blockLocs[b.Preds[0].b.ID])
	}
	for i := 1; i < len(preds); i++ {
		p := preds[i]
		if state.loggingEnabled {
			state.logf("Merging in state from %v: %v &= %v\n", p, live, blockLocs[p.ID])
		}

		for slot, liveVar := range live {
			liveLoc := liveVar.last()
			if liveLoc == nil {
				continue
			}

			predLoc := blockLocs[p.ID].lastLoc(SlotID(slot))
			// Clear out slots missing/dead in p.
			if predLoc == nil || !predLoc.survivedBlock {
				live[slot].Locations = nil
				continue
			}

			// Unify storage locations.
			liveLoc.OnStack = liveLoc.OnStack && predLoc.OnStack
			liveLoc.Registers &= predLoc.Registers
		}
	}

	// Create final result.
	locs := &FuncDebug{Variables: live, Slots: state.slots}
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
func (state *debugState) processValue(locs *FuncDebug, v *Value, vSlots []SlotID, vReg *Register) {
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
				state.unexpected(v, "regkill of already dead %v, %+v\n", vReg, state.slots[slot])
				continue
			}
			if state.loggingEnabled {
				state.logf("at %v: %v regkilled out of %v\n", v.ID, state.slots[slot], vReg.Name())
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
				state.unexpected(v, "spill of unnamed register %v\n", vReg)
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
				state.logf("at %v: %v now in %v\n", v.ID, state.slots[slot], vReg.Name())
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
