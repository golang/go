// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package ssa

import (
	"cmd/internal/dwarf"
	"cmd/internal/obj"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
)

type SlotID int32
type VarID int32

// A FuncDebug contains all the debug information for the variables in a
// function. Variables are identified by their LocalSlot, which may be the
// result of decomposing a larger variable.
type FuncDebug struct {
	// Slots is all the slots used in the debug info, indexed by their SlotID.
	Slots []*LocalSlot
	// The user variables, indexed by VarID.
	Vars []GCNode
	// The slots that make up each variable, indexed by VarID.
	VarSlots [][]SlotID
	// The location list data, indexed by VarID. Must be processed by PutLocationList.
	LocationLists [][]byte

	// Filled in by the user. Translates Block and Value ID to PC.
	GetPC func(ID, ID) int64
}

type BlockDebug struct {
	// The SSA block that this tracks. For debug logging only.
	Block *Block
	// State at entry to the block. Both this and endState are immutable
	// once initialized.
	startState []liveSlot
	// State at the end of the block if it's fully processed.
	endState []liveSlot
}

// A liveSlot is a slot that's live in loc at entry/exit of a block.
type liveSlot struct {
	slot SlotID
	loc  VarLoc
}

// stateAtPC is the current state of all variables at some point.
type stateAtPC struct {
	// The location of each known slot, indexed by SlotID.
	slots []VarLoc
	// The slots present in each register, indexed by register number.
	registers [][]SlotID
}

// reset fills state with the live variables from live.
func (state *stateAtPC) reset(live []liveSlot) {
	for i := range state.slots {
		state.slots[i] = VarLoc{}
	}
	for i := range state.registers {
		state.registers[i] = state.registers[i][:0]
	}
	for _, live := range live {
		state.slots[live.slot] = live.loc
		for reg, regMask := 0, 1; reg < len(state.registers); reg, regMask = reg+1, regMask<<1 {
			if live.loc.Registers&RegisterSet(regMask) != 0 {
				state.registers[reg] = append(state.registers[reg], SlotID(live.slot))
			}
		}
	}
}

func (b *BlockDebug) LocString(loc VarLoc) string {
	if loc.absent() {
		return "<nil>"
	}
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
	return strings.Join(storage, ",")
}

// A VarLoc describes the storage for part of a user variable.
type VarLoc struct {
	// The registers this variable is available in. There can be more than
	// one in various situations, e.g. it's being moved between registers.
	Registers RegisterSet
	// OnStack indicates that the variable is on the stack at StackOffset.
	OnStack     bool
	StackOffset int32
}

func (loc *VarLoc) absent() bool {
	return loc.Registers == 0 && !loc.OnStack
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
	// See FuncDebug.
	slots    []*LocalSlot
	vars     []GCNode
	varSlots [][]SlotID

	// The user variable that each slot rolls up to, indexed by SlotID.
	slotVars []VarID

	f              *Func
	loggingEnabled bool
	cache          *Cache
	registers      []Register
	stackOffset    func(*LocalSlot) int32

	// The names (slots) associated with each value, indexed by Value ID.
	valueNames [][]SlotID

	// The current state of whatever analysis is running.
	currentState stateAtPC
	changedVars  []bool
}

func (s *debugState) blockEndStateString(b *BlockDebug) string {
	endState := stateAtPC{slots: make([]VarLoc, len(s.slots)), registers: make([][]SlotID, len(s.slots))}
	endState.reset(b.endState)
	return s.stateString(b, endState)
}

func (s *debugState) stateString(b *BlockDebug, state stateAtPC) string {
	var strs []string
	for slotID, loc := range state.slots {
		if !loc.absent() {
			strs = append(strs, fmt.Sprintf("\t%v = %v\n", s.slots[slotID], b.LocString(loc)))
		}
	}

	strs = append(strs, "\n")
	for reg, slots := range state.registers {
		if len(slots) != 0 {
			var slotStrs []string
			for _, slot := range slots {
				slotStrs = append(slotStrs, s.slots[slot].String())
			}
			strs = append(strs, fmt.Sprintf("\t%v = %v\n", &s.registers[reg], slotStrs))
		}
	}

	if len(strs) == 1 {
		return "(no vars)\n"
	}
	return strings.Join(strs, "")
}

// BuildFuncDebug returns debug information for f.
// f must be fully processed, so that each Value is where it will be when
// machine code is emitted.
func BuildFuncDebug(ctxt *obj.Link, f *Func, loggingEnabled bool, stackOffset func(*LocalSlot) int32) *FuncDebug {
	if f.RegAlloc == nil {
		f.Fatalf("BuildFuncDebug on func %v that has not been fully processed", f)
	}
	state := &debugState{
		loggingEnabled: loggingEnabled,
		slots:          make([]*LocalSlot, len(f.Names)),

		f:            f,
		cache:        f.Cache,
		registers:    f.Config.registers,
		stackOffset:  stackOffset,
		currentState: stateAtPC{make([]VarLoc, len(f.Names)), make([][]SlotID, len(f.Config.registers))},
	}
	// TODO: consider storing this in Cache and reusing across functions.
	state.valueNames = make([][]SlotID, f.NumValues())

	// Recompose any decomposed variables, and record the names associated with each value.
	varParts := map[GCNode][]SlotID{}
	for i, slot := range f.Names {
		slot := slot
		state.slots[i] = &slot
		if isSynthetic(&slot) {
			continue
		}
		for _, value := range f.NamedValues[slot] {
			state.valueNames[value.ID] = append(state.valueNames[value.ID], SlotID(i))
		}

		topSlot := &slot
		for topSlot.SplitOf != nil {
			topSlot = topSlot.SplitOf
		}
		if _, ok := varParts[topSlot.N]; !ok {
			state.vars = append(state.vars, topSlot.N)
		}
		varParts[topSlot.N] = append(varParts[topSlot.N], SlotID(i))
	}

	// Fill in the var<->slot mappings.
	state.varSlots = make([][]SlotID, len(state.vars))
	state.slotVars = make([]VarID, len(state.slots))
	for varID, n := range state.vars {
		parts := varParts[n]
		state.varSlots[varID] = parts
		for _, slotID := range parts {
			state.slotVars[slotID] = VarID(varID)
		}
	}
	state.changedVars = make([]bool, len(state.vars))

	blockLocs := state.liveness()
	lists := state.buildLocationLists(ctxt, stackOffset, blockLocs)

	return &FuncDebug{
		Slots:         state.slots,
		VarSlots:      state.varSlots,
		Vars:          state.vars,
		LocationLists: lists,
	}
}

// isSynthetic reports whether if slot represents a compiler-inserted variable,
// e.g. an autotmp or an anonymous return value that needed a stack slot.
func isSynthetic(slot *LocalSlot) bool {
	c := slot.N.String()[0]
	return c == '.' || c == '~'
}

// liveness walks the function in control flow order, calculating the start
// and end state of each block.
func (state *debugState) liveness() []*BlockDebug {
	blockLocs := make([]*BlockDebug, state.f.NumBlocks())

	// Reverse postorder: visit a block after as many as possible of its
	// predecessors have been visited.
	po := state.f.Postorder()
	for i := len(po) - 1; i >= 0; i-- {
		b := po[i]

		// Build the starting state for the block from the final
		// state of its predecessors.
		locs := state.mergePredecessors(b, blockLocs)

		if state.loggingEnabled {
			state.logf("Processing %v, initial state:\n%v", b, state.stateString(locs, state.currentState))
		}

		// Update locs/registers with the effects of each Value.
		for _, v := range b.Values {
			slots := state.valueNames[v.ID]

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
			// Update valueNames with the source so that later steps
			// don't need special handling.
			if source != nil {
				slots = append(slots, state.valueNames[source.ID]...)
				state.valueNames[v.ID] = slots
			}

			reg, _ := state.f.getHome(v.ID).(*Register)
			state.processValue(v, slots, reg)
		}

		if state.loggingEnabled {
			state.f.Logf("Block %v done, locs:\n%v", b, state.stateString(locs, state.currentState))
		}

		for slotID, slotLoc := range state.currentState.slots {
			if slotLoc.absent() {
				continue
			}
			locs.endState = append(locs.endState, liveSlot{SlotID(slotID), slotLoc})
		}

		blockLocs[b.ID] = locs
	}
	return blockLocs
}

// mergePredecessors takes the end state of each of b's predecessors and
// intersects them to form the starting state for b. It returns that state in
// the BlockDebug, and fills state.currentState with it.
func (state *debugState) mergePredecessors(b *Block, blockLocs []*BlockDebug) *BlockDebug {
	result := &BlockDebug{}
	if state.loggingEnabled {
		result.Block = b
	}

	// Filter out back branches.
	var preds []*Block
	for _, pred := range b.Preds {
		if blockLocs[pred.b.ID] != nil {
			preds = append(preds, pred.b)
		}
	}

	if state.loggingEnabled {
		state.logf("Merging %v into %v\n", preds, b)
	}

	if len(preds) == 0 {
		if state.loggingEnabled {
		}
		state.currentState.reset(nil)
		return result
	}

	if len(preds) == 1 {
		p := blockLocs[preds[0].ID]
		result.startState = p.endState
		state.currentState.reset(p.endState)
		return result
	}

	if state.loggingEnabled {
		state.logf("Starting %v with state from %v:\n%v", b, preds[0], state.blockEndStateString(blockLocs[preds[0].ID]))
	}

	count := make([]int, len(state.slots))
	slotLocs := state.currentState.slots
	for _, predSlot := range blockLocs[preds[0].ID].endState {
		slotLocs[predSlot.slot] = predSlot.loc
		count[predSlot.slot] = 1
	}
	for i := 1; i < len(preds); i++ {
		if state.loggingEnabled {
			state.logf("Merging in state from %v:\n%v", preds[i], state.blockEndStateString(blockLocs[preds[i].ID]))
		}
		for _, predSlot := range blockLocs[preds[i].ID].endState {
			count[predSlot.slot]++
			liveLoc := slotLocs[predSlot.slot]
			if !liveLoc.OnStack || !predSlot.loc.OnStack || liveLoc.StackOffset != predSlot.loc.StackOffset {
				liveLoc.OnStack = false
				liveLoc.StackOffset = 0
			}
			liveLoc.Registers &= predSlot.loc.Registers
			slotLocs[predSlot.slot] = liveLoc
		}
	}

	for reg := range state.currentState.registers {
		state.currentState.registers[reg] = state.currentState.registers[reg][:0]
	}

	// A slot is live if it was seen in all predecessors, and they all had
	// some storage in common.
	for slotID, slotLoc := range slotLocs {
		// Not seen in any predecessor.
		if slotLoc.absent() {
			continue
		}
		// Seen in only some predecessors. Clear it out.
		if count[slotID] != len(preds) {
			slotLocs[slotID] = VarLoc{}
			continue
		}
		// Present in all predecessors.
		result.startState = append(result.startState, liveSlot{SlotID(slotID), slotLoc})
		if slotLoc.Registers == 0 {
			continue
		}
		for reg, regMask := 0, 1; reg < len(state.registers); reg, regMask = reg+1, regMask<<1 {
			if slotLoc.Registers&RegisterSet(regMask) != 0 {
				state.currentState.registers[reg] = append(state.currentState.registers[reg], SlotID(slotID))
			}
		}
	}
	return result
}

// processValue updates locs and state.registerContents to reflect v, a value with
// the names in vSlots and homed in vReg.  "v" becomes visible after execution of
// the instructions evaluating it. It returns which VarIDs were modified by the
// Value's execution.
func (state *debugState) processValue(v *Value, vSlots []SlotID, vReg *Register) {
	locs := state.currentState
	setSlot := func(slot SlotID, loc VarLoc) {
		state.changedVars[state.slotVars[slot]] = true
		state.currentState.slots[slot] = loc
	}

	// Handle any register clobbering. Call operations, for example,
	// clobber all registers even though they don't explicitly write to
	// them.
	if clobbers := opcodeTable[v.Op].reg.clobbers; clobbers != 0 {
		for reg := 0; reg < len(state.registers); reg++ {
			if clobbers&(1<<uint8(reg)) == 0 {
				continue
			}

			for _, slot := range locs.registers[reg] {
				if state.loggingEnabled {
					state.logf("at %v: %v clobbered out of %v\n", v.ID, state.slots[slot], &state.registers[reg])
				}

				last := locs.slots[slot]
				if last.absent() {
					state.f.Fatalf("at %v: slot %v in register %v with no location entry", v, state.slots[slot], &state.registers[reg])
					continue
				}
				regs := last.Registers &^ (1 << uint8(reg))
				setSlot(slot, VarLoc{regs, last.OnStack, last.StackOffset})
			}

			locs.registers[reg] = locs.registers[reg][:0]
		}
	}

	switch {
	case v.Op == OpArg:
		home := state.f.getHome(v.ID).(LocalSlot)
		stackOffset := state.stackOffset(&home)
		for _, slot := range vSlots {
			if state.loggingEnabled {
				state.logf("at %v: arg %v now on stack in location %v\n", v.ID, state.slots[slot], home)
				if last := locs.slots[slot]; !last.absent() {
					state.unexpected(v, "Arg op on already-live slot %v", state.slots[slot])
				}
			}

			setSlot(slot, VarLoc{0, true, stackOffset})
		}

	case v.Op == OpStoreReg:
		home := state.f.getHome(v.ID).(LocalSlot)
		stackOffset := state.stackOffset(&home)
		for _, slot := range vSlots {
			last := locs.slots[slot]
			if last.absent() {
				state.unexpected(v, "spill of unnamed register %s\n", vReg)
				break
			}

			setSlot(slot, VarLoc{last.Registers, true, stackOffset})
			if state.loggingEnabled {
				state.logf("at %v: %v spilled to stack location %v\n", v.ID, state.slots[slot], home)
			}
		}

	case vReg != nil:
		if state.loggingEnabled {
			newSlots := make([]bool, len(state.slots))
			for _, slot := range vSlots {
				newSlots[slot] = true
			}

			for _, slot := range locs.registers[vReg.num] {
				if !newSlots[slot] {
					state.logf("at %v: overwrote %v in register %v\n", v, state.slots[slot], vReg)
				}
			}
		}

		for _, slot := range locs.registers[vReg.num] {
			last := locs.slots[slot]
			setSlot(slot, VarLoc{last.Registers &^ (1 << uint8(vReg.num)), last.OnStack, last.StackOffset})
		}
		locs.registers[vReg.num] = locs.registers[vReg.num][:0]
		locs.registers[vReg.num] = append(locs.registers[vReg.num], vSlots...)
		for _, slot := range vSlots {
			if state.loggingEnabled {
				state.logf("at %v: %v now in %s\n", v.ID, state.slots[slot], vReg)
			}
			var loc VarLoc
			loc.Registers |= 1 << uint8(vReg.num)
			if last := locs.slots[slot]; !last.absent() {
				loc.OnStack = last.OnStack
				loc.StackOffset = last.StackOffset
				loc.Registers |= last.Registers
			}
			setSlot(slot, loc)
		}
	}
}

// varOffset returns the offset of slot within the user variable it was
// decomposed from. This has nothing to do with its stack offset.
func varOffset(slot *LocalSlot) int64 {
	offset := slot.Off
	for ; slot.SplitOf != nil; slot = slot.SplitOf {
		offset += slot.SplitOffset
	}
	return offset
}

// This type is deleted in a subsequent CL.
type varPart struct {
	varOffset int64
	slot      SlotID
}

type partsByVarOffset []varPart

func (a partsByVarOffset) Len() int           { return len(a) }
func (a partsByVarOffset) Less(i, j int) bool { return a[i].varOffset < a[j].varOffset }
func (a partsByVarOffset) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

// A pendingEntry represents the beginning of a location list entry, missing
// only its end coordinate.
type pendingEntry struct {
	present                bool
	startBlock, startValue ID
	// The location of each piece of the variable, indexed by *SlotID*,
	// even though only a few slots are used in each entry. This could be
	// improved by only storing the relevant slots.
	pieces []VarLoc
}

func (e *pendingEntry) clear() {
	e.present = false
	e.startBlock = 0
	e.startValue = 0
	for i := range e.pieces {
		e.pieces[i] = VarLoc{}
	}
}

// canMerge returns true if the location description for new is the same as
// pending.
func canMerge(pending, new VarLoc) bool {
	if pending.absent() && new.absent() {
		return true
	}
	if pending.absent() || new.absent() {
		return false
	}
	if pending.OnStack {
		return new.OnStack && pending.StackOffset == new.StackOffset
	}
	if pending.Registers != 0 && new.Registers != 0 {
		return firstReg(pending.Registers) == firstReg(new.Registers)
	}
	return false
}

// firstReg returns the first register in set that is present.
func firstReg(set RegisterSet) uint8 {
	for reg := 0; reg < 64; reg++ {
		if set&(1<<uint8(reg)) != 0 {
			return uint8(reg)
		}
	}
	// This is wrong, but there seem to be some situations where we
	// produce locations with no storage.
	return 0
}

// buildLocationLists builds location lists for all the user variables in
// state.f, using the information about block state in blockLocs.
// The returned location lists are not fully complete. They are in terms of
// SSA values rather than PCs, and have no base address/end entries. They will
// be finished by PutLocationList.
func (state *debugState) buildLocationLists(Ctxt *obj.Link, stackOffset func(*LocalSlot) int32, blockLocs []*BlockDebug) [][]byte {
	lists := make([][]byte, len(state.vars))
	varParts := make([][]varPart, len(lists))
	pendingEntries := make([]pendingEntry, len(lists))

	for varID, parts := range state.varSlots {
		for _, slotID := range parts {
			varParts[varID] = append(varParts[varID], varPart{varOffset(state.slots[slotID]), slotID})
		}
		// Get the order the parts need to be in to represent the memory
		// of the decomposed user variable.
		sort.Sort(partsByVarOffset(varParts[varID]))

		pendingEntries[varID].pieces = make([]VarLoc, len(state.slots))
	}

	// writePendingEntry writes out the pending entry for varID, if any,
	// terminated at endBlock/Value.
	writePendingEntry := func(varID VarID, endBlock, endValue ID) {
		list := lists[varID]
		pending := pendingEntries[varID]
		if !pending.present {
			return
		}

		// Pack the start/end coordinates into the start/end addresses
		// of the entry, for decoding by PutLocationList.
		start, startOK := encodeValue(Ctxt, pending.startBlock, pending.startValue)
		end, endOK := encodeValue(Ctxt, endBlock, endValue)
		if !startOK || !endOK {
			// If someone writes a function that uses >65K values,
			// they get incomplete debug info on 32-bit platforms.
			return
		}
		list = appendPtr(Ctxt, list, start)
		list = appendPtr(Ctxt, list, end)
		// Where to write the length of the location description once
		// we know how big it is.
		sizeIdx := len(list)
		list = list[:len(list)+2]

		if state.loggingEnabled {
			var partStrs []string
			for _, part := range varParts[varID] {
				partStrs = append(partStrs, fmt.Sprintf("%v@%v", state.slots[part.slot], blockLocs[endBlock].LocString(pending.pieces[part.slot])))
			}
			state.logf("Add entry for %v: \tb%vv%v-b%vv%v = \t%v\n", state.vars[varID], pending.startBlock, pending.startValue, endBlock, endValue, strings.Join(partStrs, " "))
		}

		for _, part := range varParts[varID] {
			loc := pending.pieces[part.slot]
			slot := state.slots[part.slot]

			if !loc.absent() {
				if loc.OnStack {
					if loc.StackOffset == 0 {
						list = append(list, dwarf.DW_OP_call_frame_cfa)
					} else {
						list = append(list, dwarf.DW_OP_fbreg)
						list = dwarf.AppendSleb128(list, int64(loc.StackOffset))
					}
				} else {
					regnum := Ctxt.Arch.DWARFRegisters[state.registers[firstReg(loc.Registers)].ObjNum()]
					if regnum < 32 {
						list = append(list, dwarf.DW_OP_reg0+byte(regnum))
					} else {
						list = append(list, dwarf.DW_OP_regx)
						list = dwarf.AppendUleb128(list, uint64(regnum))
					}
				}
			}

			if len(varParts[varID]) > 1 {
				list = append(list, dwarf.DW_OP_piece)
				list = dwarf.AppendUleb128(list, uint64(slot.Type.Size()))
			}
		}
		Ctxt.Arch.ByteOrder.PutUint16(list[sizeIdx:], uint16(len(list)-sizeIdx-2))
		lists[varID] = list
	}

	// updateVar updates the pending location list entry for varID to
	// reflect the new locations in curLoc, caused by v.
	updateVar := func(varID VarID, v *Value, curLoc []VarLoc) {
		// Assemble the location list entry with whatever's live.
		empty := true
		for _, part := range varParts[varID] {
			if !curLoc[part.slot].absent() {
				empty = false
				break
			}
		}
		pending := &pendingEntries[varID]
		if empty {
			writePendingEntry(varID, v.Block.ID, v.ID)
			pending.clear()
			return
		}

		// Extend the previous entry if possible.
		if pending.present {
			merge := true
			for _, part := range varParts[varID] {
				if !canMerge(pending.pieces[part.slot], curLoc[part.slot]) {
					merge = false
					break
				}
			}
			if merge {
				return
			}
		}

		writePendingEntry(varID, v.Block.ID, v.ID)
		pending.present = true
		pending.startBlock = v.Block.ID
		pending.startValue = v.ID
		copy(pending.pieces, curLoc)
		return

	}

	// Run through the function in program text order, building up location
	// lists as we go. The heavy lifting has mostly already been done.
	for _, b := range state.f.Blocks {
		state.currentState.reset(blockLocs[b.ID].startState)

		for _, v := range b.Values {
			slots := state.valueNames[v.ID]
			reg, _ := state.f.getHome(v.ID).(*Register)
			state.processValue(v, slots, reg)

			if v.Op == OpPhi {
				continue
			}

			for varID := range state.changedVars {
				if !state.changedVars[varID] {
					continue
				}
				state.changedVars[varID] = false
				updateVar(VarID(varID), v, state.currentState.slots)
			}
		}

	}

	if state.loggingEnabled {
		state.logf("location lists:\n")
	}

	// Flush any leftover entries live at the end of the last block.
	for varID := range lists {
		writePendingEntry(VarID(varID), state.f.Blocks[len(state.f.Blocks)-1].ID, BlockEnd.ID)
		list := lists[varID]
		if len(list) == 0 {
			continue
		}

		if state.loggingEnabled {
			state.logf("\t%v : %q\n", state.vars[varID], hex.EncodeToString(lists[varID]))
		}
	}
	return lists
}

// PutLocationList adds list (a location list in its intermediate representation) to listSym.
func (debugInfo *FuncDebug) PutLocationList(list []byte, ctxt *obj.Link, listSym, startPC *obj.LSym) {
	getPC := debugInfo.GetPC
	// Re-read list, translating its address from block/value ID to PC.
	for i := 0; i < len(list); {
		translate := func() {
			bv := readPtr(ctxt, list[i:])
			pc := getPC(decodeValue(ctxt, bv))
			writePtr(ctxt, list[i:], uint64(pc))
			i += ctxt.Arch.PtrSize
		}
		translate()
		translate()
		i += 2 + int(ctxt.Arch.ByteOrder.Uint16(list[i:]))
	}

	// Base address entry.
	listSym.WriteInt(ctxt, listSym.Size, ctxt.Arch.PtrSize, ^0)
	listSym.WriteAddr(ctxt, listSym.Size, ctxt.Arch.PtrSize, startPC, 0)
	// Location list contents, now with real PCs.
	listSym.WriteBytes(ctxt, listSym.Size, list)
	// End entry.
	listSym.WriteInt(ctxt, listSym.Size, ctxt.Arch.PtrSize, 0)
	listSym.WriteInt(ctxt, listSym.Size, ctxt.Arch.PtrSize, 0)
}

// Pack a value and block ID into an address-sized uint, returning ~0 if they
// don't fit.
func encodeValue(ctxt *obj.Link, b, v ID) (uint64, bool) {
	if ctxt.Arch.PtrSize == 8 {
		result := uint64(b)<<32 | uint64(uint32(v))
		//ctxt.Logf("b %#x (%d) v %#x (%d) -> %#x\n", b, b, v, v, result)
		return result, true
	}
	if ctxt.Arch.PtrSize != 4 {
		panic("unexpected pointer size")
	}
	if ID(int16(b)) != b || ID(int16(v)) != v {
		return 0, false
	}
	return uint64(b)<<16 | uint64(uint16(v)), true
}

// Unpack a value and block ID encoded by encodeValue.
func decodeValue(ctxt *obj.Link, word uint64) (ID, ID) {
	if ctxt.Arch.PtrSize == 8 {
		b, v := ID(word>>32), ID(word)
		//ctxt.Logf("%#x -> b %#x (%d) v %#x (%d)\n", word, b, b, v, v)
		return b, v
	}
	if ctxt.Arch.PtrSize != 4 {
		panic("unexpected pointer size")
	}
	return ID(word >> 16), ID(word)
}

// Append a pointer-sized uint to buf.
func appendPtr(ctxt *obj.Link, buf []byte, word uint64) []byte {
	if cap(buf) < len(buf)+100 {
		b := make([]byte, len(buf), 100+cap(buf)*2)
		copy(b, buf)
		buf = b
	}
	writeAt := len(buf)
	buf = buf[0 : len(buf)+ctxt.Arch.PtrSize]
	writePtr(ctxt, buf[writeAt:], word)
	return buf
}

// Write a pointer-sized uint to the beginning of buf.
func writePtr(ctxt *obj.Link, buf []byte, word uint64) {
	switch ctxt.Arch.PtrSize {
	case 4:
		ctxt.Arch.ByteOrder.PutUint32(buf, uint32(word))
	case 8:
		ctxt.Arch.ByteOrder.PutUint64(buf, word)
	default:
		panic("unexpected pointer size")
	}

}

// Read a pointer-sized uint from the beginning of buf.
func readPtr(ctxt *obj.Link, buf []byte) uint64 {
	switch ctxt.Arch.PtrSize {
	case 4:
		return uint64(ctxt.Arch.ByteOrder.Uint32(buf))
	case 8:
		return ctxt.Arch.ByteOrder.Uint64(buf)
	default:
		panic("unexpected pointer size")
	}

}
