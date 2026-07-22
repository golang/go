// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacore

import (
	"fmt"
	"math/bits"
	"strings"

	"cmd/compile/internal/abt"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/ssa/ssabase"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/internal/dwarf"
	"cmd/internal/obj"
)

type BlockDebug struct {
	// State at the start and end of the block. These are initialized,
	// and updated from new information that flows on back edges.
	startState, endState abt.T
	// Use these to avoid excess work in the merge. If none of the
	// predecessors has changed since the last check, the old answer is
	// still good.
	lastCheckedTime, lastChangedTime int32
	// Whether the block had any changes to user variables at all.
	relevant bool
	// false until the block has been processed at least once. This
	// affects how the merge is done; the goal is to maximize sharing
	// and avoid allocation.
	everProcessed bool
}

var BlockEnd = &Value{
	ID:  -20000,
	Op:  ssaop.OpInvalid,
	Aux: StringToAux("BlockEnd"),
}

var BlockStart = &Value{
	ID:  -10000,
	Op:  ssaop.OpInvalid,
	Aux: StringToAux("BlockStart"),
}

type DebugState struct {
	// See FuncDebug.
	Slots    []LocalSlot
	Vars     []*ir.Name
	VarSlots [][]SlotID
	Lists    [][]LocListEntry

	// The user variable that each slot rolls up to, indexed by SlotID.
	SlotVars []VarID

	F             *Func
	LoggingLevel  int
	ConvergeCount int // testing; iterate over block debug state this many times
	Registers     []ssabase.Register
	StackOffset   func(LocalSlot) int32
	Ctxt          *obj.Link

	// The names (slots) associated with each value, indexed by Value ID.
	ValueNames [][]SlotID

	// The current state of whatever analysis is running.
	currentState StateAtPC
	changedVars  *SparseSet
	changedSlots *SparseSet

	// The pending location list entry for each user variable, indexed by VarID.
	pendingEntries []pendingEntry

	VarParts        map[*ir.Name][]SlotID
	blockDebug      []BlockDebug
	pendingSlotLocs []VarLoc
}

var FuncEnd = &Value{
	ID:  -30000,
	Op:  ssaop.OpInvalid,
	Aux: StringToAux("FuncEnd"),
}

// IsVarWantedForDebug returns true if the debug info for the node should
// be generated.
// For example, internal variables for range-over-func loops have little
// value to users, so we don't generate debug info for them.
func IsVarWantedForDebug(n ir.Node) bool {
	name := n.Sym().Name
	if len(name) > 0 && name[0] == '&' {
		name = name[1:]
	}
	if len(name) > 0 && name[0] == '#' {
		// #yield is used by delve.
		return strings.HasPrefix(name, "#yield")
	}
	return true
}

// LocListEntry represents a single entry in a location list.
// StartBlock/StartValue and EndBlock/EndValue are SSA coordinates
// that get resolved to PCs during final encoding.
type LocListEntry struct {
	StartBlock, StartValue ID
	EndBlock, EndValue     ID
	Expr                   []byte // DWARF location expression (DW_OP_*)
}

// RegisterSet is a bitmap of registers, indexed by Register.num.
type RegisterSet uint64

type SlotID int32

// StackOffset encodes whether a value is on the stack and if so, where.
// It is a 31-bit integer followed by a presence flag at the low-order
// bit.
type StackOffset int32

// StateAtPC is the current state of all variables at some point.
type StateAtPC struct {
	// The location of each known slot, indexed by SlotID.
	slots []VarLoc
	// The slots present in each register, indexed by register number.
	registers [][]SlotID
}

type VarID int32

// A VarLoc describes the storage for part of a user variable.
type VarLoc struct {
	// The registers this variable is available in. There can be more than
	// one in various situations, e.g. it's being moved between registers.
	Registers RegisterSet

	StackOffset
}

// canMerge reports whether a new location description is a superset
// of the (non-empty) pending location description, if so, the two
// can be merged (i.e., pending is still a valid and useful location
// description).
func canMerge(pending, new VarLoc) bool {
	if pending.absent() && new.absent() {
		return true
	}
	if pending.absent() || new.absent() {
		return false
	}
	// pending is not absent, therefore it has either a stack mapping,
	// or registers, or both.
	if pending.onStack() && pending.StackOffset != new.StackOffset {
		// if pending has a stack offset, then new must also, and it
		// must be the same (StackOffset encodes onStack).
		return false
	}
	if pending.Registers&new.Registers != pending.Registers {
		// There is at least one register in pending not mentioned in new.
		return false
	}
	return true
}

// firstReg returns the first register in set that is present.
func firstReg(set RegisterSet) uint8 {
	if set == 0 {
		// This is wrong, but there seem to be some situations where we
		// produce locations with no storage.
		return 0
	}
	return uint8(bits.TrailingZeros64(uint64(set)))
}

// A liveSlot is a slot that's live in loc at entry/exit of a block.
type liveSlot struct {
	VarLoc
}

// A pendingEntry represents the beginning of a location list entry, missing
// only its end coordinate.
type pendingEntry struct {
	present                bool
	startBlock, startValue ID
	// The location of each piece of the variable, in the same order as the
	// SlotIDs in varParts.
	pieces []VarLoc
}

func (ls *liveSlot) String() string {
	return fmt.Sprintf("0x%x.%d.%d", ls.Registers, ls.stackOffsetValue(), int32(ls.StackOffset)&1)
}

func (s StackOffset) onStack() bool {
	return s != 0
}

func (s StackOffset) stackOffsetValue() int32 {
	return int32(s) >> 1
}

// reset fills state with the live variables from live.
func (state *StateAtPC) reset(live abt.T) {
	slots, registers := state.slots, state.registers
	clear(slots)
	for i := range registers {
		registers[i] = registers[i][:0]
	}
	for it := live.Iterator(); !it.Done(); {
		k, d := it.Next()
		live := d.(*liveSlot)
		slots[k] = live.VarLoc
		if live.VarLoc.Registers == 0 {
			continue
		}

		mask := uint64(live.VarLoc.Registers)
		for {
			if mask == 0 {
				break
			}
			reg := uint8(bits.TrailingZeros64(mask))
			mask &^= 1 << reg

			registers[reg] = append(registers[reg], SlotID(k))
		}
	}
	state.slots, state.registers = slots, registers
}

func (s *DebugState) LocString(loc VarLoc) string {
	if loc.absent() {
		return "<nil>"
	}

	var storage []string
	if loc.onStack() {
		storage = append(storage, fmt.Sprintf("@%+d", loc.stackOffsetValue()))
	}

	mask := uint64(loc.Registers)
	for {
		if mask == 0 {
			break
		}
		reg := uint8(bits.TrailingZeros64(mask))
		mask &^= 1 << reg

		storage = append(storage, s.Registers[reg].String())
	}
	return strings.Join(storage, ",")
}

func (loc VarLoc) absent() bool {
	return loc.Registers == 0 && !loc.onStack()
}

func (loc VarLoc) intersect(other VarLoc) VarLoc {
	if !loc.onStack() || !other.onStack() || loc.StackOffset != other.StackOffset {
		loc.StackOffset = 0
	}
	loc.Registers &= other.Registers
	return loc
}

// Logf prints debug-specific logging to stdout (always stdout) if the
// current function is tagged by GOSSAFUNC (for ssa output directed
// either to stdout or html).
func (s *DebugState) Logf(msg string, args ...any) {
	if s.F.PrintOrHtmlSSA {
		fmt.Printf(msg, args...)
	}
}

func (state *DebugState) InitializeCache(f *Func, numVars, numSlots int) {
	// One blockDebug per block. Initialized in allocBlock.
	if cap(state.blockDebug) < f.NumBlocks() {
		state.blockDebug = make([]BlockDebug, f.NumBlocks())
	} else {
		clear(state.blockDebug[:f.NumBlocks()])
	}

	// A list of slots per Value. Reuse the previous child slices.
	if cap(state.ValueNames) < f.NumValues() {
		old := state.ValueNames
		state.ValueNames = make([][]SlotID, f.NumValues())
		copy(state.ValueNames, old)
	}
	vn := state.ValueNames[:f.NumValues()]
	for i := range vn {
		vn[i] = vn[i][:0]
	}

	// Slot and register contents for currentState. Cleared by reset().
	if cap(state.currentState.slots) < numSlots {
		state.currentState.slots = make([]VarLoc, numSlots)
	} else {
		state.currentState.slots = state.currentState.slots[:numSlots]
	}
	if cap(state.currentState.registers) < len(state.Registers) {
		state.currentState.registers = make([][]SlotID, len(state.Registers))
	} else {
		state.currentState.registers = state.currentState.registers[:len(state.Registers)]
	}

	// A relatively small slice, but used many times as the return from processValue.
	state.changedVars = NewSparseSet(numVars)
	state.changedSlots = NewSparseSet(numSlots)

	// A pending entry per user variable, with space to track each of its pieces.
	numPieces := 0
	for i := range state.VarSlots {
		numPieces += len(state.VarSlots[i])
	}
	if cap(state.pendingSlotLocs) < numPieces {
		state.pendingSlotLocs = make([]VarLoc, numPieces)
	} else {
		clear(state.pendingSlotLocs[:numPieces])
	}
	if cap(state.pendingEntries) < numVars {
		state.pendingEntries = make([]pendingEntry, numVars)
	}
	pe := state.pendingEntries[:numVars]
	freePieceIdx := 0
	for varID, slots := range state.VarSlots {
		pe[varID] = pendingEntry{
			pieces: state.pendingSlotLocs[freePieceIdx : freePieceIdx+len(slots)],
		}
		freePieceIdx += len(slots)
	}
	state.pendingEntries = pe

	if cap(state.Lists) < numVars {
		state.Lists = make([][]LocListEntry, numVars)
	} else {
		state.Lists = state.Lists[:numVars]
		clear(state.Lists)
	}
}

func (state *DebugState) allocBlock(b *Block) *BlockDebug {
	return &state.blockDebug[b.ID]
}

func (s *DebugState) blockEndStateString(b *BlockDebug) string {
	endState := StateAtPC{slots: make([]VarLoc, len(s.Slots)), registers: make([][]SlotID, len(s.Registers))}
	endState.reset(b.endState)
	return s.stateString(endState)
}

func (s *DebugState) stateString(state StateAtPC) string {
	var strs []string
	for slotID, loc := range state.slots {
		if !loc.absent() {
			strs = append(strs, fmt.Sprintf("\t%v = %v\n", s.Slots[slotID], s.LocString(loc)))
		}
	}

	strs = append(strs, "\n")
	for reg, slots := range state.registers {
		if len(slots) != 0 {
			var slotStrs []string
			for _, slot := range slots {
				slotStrs = append(slotStrs, s.Slots[slot].String())
			}
			strs = append(strs, fmt.Sprintf("\t%v = %v\n", &s.Registers[reg], slotStrs))
		}
	}

	if len(strs) == 1 {
		return "(no vars)\n"
	}
	return strings.Join(strs, "")
}

// Liveness walks the function in control flow order, calculating the start
// and end state of each block.
func (state *DebugState) Liveness() []*BlockDebug {
	blockLocs := make([]*BlockDebug, state.F.NumBlocks())
	counterTime := int32(1)

	// Reverse postorder: visit a block after as many as possible of its
	// predecessors have been visited.
	po := state.F.Postorder()
	converged := false

	// The iteration rule is that by default, run until converged, but
	// if a particular iteration count is specified, run that many
	// iterations, no more, no less.  A count is specified as the
	// thousands digit of the location lists debug flag,
	// e.g. -d=locationlists=4000
	keepGoing := func(k int) bool {
		if state.ConvergeCount == 0 {
			return !converged
		}
		return k < state.ConvergeCount
	}
	for k := 0; keepGoing(k); k++ {
		if state.LoggingLevel > 0 {
			state.Logf("Liveness pass %d\n", k)
		}
		converged = true
		for i := len(po) - 1; i >= 0; i-- {
			b := po[i]
			locs := blockLocs[b.ID]
			if locs == nil {
				locs = state.allocBlock(b)
				blockLocs[b.ID] = locs
			}

			// Build the starting state for the block from the final
			// state of its predecessors.
			startState, blockChanged := state.mergePredecessors(b, blockLocs, nil, false)
			locs.lastCheckedTime = counterTime
			counterTime++
			if state.LoggingLevel > 1 {
				state.Logf("Processing %v, block changed %v, initial state:\n%v", b, blockChanged, state.stateString(state.currentState))
			}

			if blockChanged {
				// If the start did not change, then the old endState is good
				converged = false
				changed := false
				state.changedSlots.Clear()

				// Update locs/registers with the effects of each Value.
				for _, v := range b.Values {
					slots := state.ValueNames[v.ID]

					// Loads and stores inherit the names of their sources.
					var source *Value
					switch v.Op {
					case ssaop.OpStoreReg:
						source = v.Args[0]
					case ssaop.OpLoadReg:
						switch a := v.Args[0]; a.Op {
						case ssaop.OpArg, ssaop.OpPhi:
							source = a
						case ssaop.OpStoreReg:
							source = a.Args[0]
						default:
							if state.LoggingLevel > 1 {
								state.Logf("at %v: load with unexpected source op: %v (%v)\n", v, a.Op, a)
							}
						}
					}
					// Update valueNames with the source so that later steps
					// don't need special handling.
					if source != nil && k == 0 {
						// limit to k == 0 otherwise there are duplicates.
						slots = append(slots, state.ValueNames[source.ID]...)
						state.ValueNames[v.ID] = slots
					}

					reg, _ := state.F.GetHome(v.ID).(*ssabase.Register)
					c := state.processValue(v, slots, reg)
					changed = changed || c
				}

				if state.LoggingLevel > 1 {
					state.Logf("Block %v done, locs:\n%v", b, state.stateString(state.currentState))
				}

				locs.relevant = locs.relevant || changed
				if !changed {
					locs.endState = startState
				} else {
					for _, id := range state.changedSlots.Contents() {
						slotID := SlotID(id)
						slotLoc := state.currentState.slots[slotID]
						if slotLoc.absent() {
							startState.Delete(int32(slotID))
							continue
						}
						old := startState.Find(int32(slotID)) // do NOT replace existing values
						if oldLS, ok := old.(*liveSlot); !ok || oldLS.VarLoc != slotLoc {
							startState.Insert(int32(slotID),
								&liveSlot{VarLoc: slotLoc})
						}
					}
					locs.endState = startState
				}
				locs.lastChangedTime = counterTime
			}
			counterTime++
		}
	}
	return blockLocs
}

// mergePredecessors takes the end state of each of b's predecessors and
// intersects them to form the starting state for b. It puts that state
// in blockLocs[b.ID].startState, and fills state.currentState with it.
// It returns the start state and whether this is changed from the
// previously approximated value of startState for this block.  After
// the first call, subsequent calls can only shrink startState.
//
// Passing forLocationLists=true enables additional side-effects that
// are necessary for building location lists but superfluous while still
// iterating to an answer.
//
// If previousBlock is non-nil, it registers changes vs. that block's
// end state in state.changedVars. Note that previousBlock will often
// not be a predecessor.
//
// Note that mergePredecessors behaves slightly differently between
// first and subsequent calls for a block.  For the first call, the
// starting state is approximated by taking the state from the
// predecessor whose state is smallest, and removing any elements not
// in all the other predecessors; this makes the smallest number of
// changes and shares the most state.  On subsequent calls the old
// value of startState is adjusted with new information; this is judged
// to do the least amount of extra work.
//
// To improve performance, each block's state information is marked with
// lastChanged and lastChecked "times" so unchanged predecessors can be
// skipped on after-the-first iterations.  Doing this allows extra
// iterations by the caller to be almost free.
//
// It is important to know that the set representation used for
// startState, endState, and merges can share data for two sets where
// one is a small delta from the other.  Doing this does require a
// little care in how sets are updated, both in mergePredecessors, and
// using its result.
func (state *DebugState) mergePredecessors(b *Block, blockLocs []*BlockDebug, previousBlock *Block, forLocationLists bool) (abt.T, bool) {
	// Filter out back branches.
	var predsBuf [10]*Block

	preds := predsBuf[:0]
	locs := blockLocs[b.ID]

	blockChanged := !locs.everProcessed // the first time it always changes.
	updating := locs.everProcessed

	// For the first merge, exclude predecessors that have not been seen yet.
	// I.e., backedges.
	for _, pred := range b.Preds {
		if bl := blockLocs[pred.B.ID]; bl != nil && bl.everProcessed {
			// crucially, a self-edge has bl != nil, but bl.everProcessed is false the first time.
			preds = append(preds, pred.B)
		}
	}

	locs.everProcessed = true

	if state.LoggingLevel > 1 {
		// The logf below would cause preds to be heap-allocated if
		// it were passed directly.
		preds2 := make([]*Block, len(preds))
		copy(preds2, preds)
		state.Logf("Merging %v into %v (changed=%d, checked=%d)\n", preds2, b, locs.lastChangedTime, locs.lastCheckedTime)
	}

	state.changedVars.Clear()

	markChangedVars := func(slots, merged abt.T) {
		if !forLocationLists {
			return
		}
		// Fill changedVars with those that differ between the previous
		// block (in the emit order, not necessarily a flow predecessor)
		// and the start state for this block.
		for it := slots.Iterator(); !it.Done(); {
			k, v := it.Next()
			m := merged.Find(k)
			if m == nil || v.(*liveSlot).VarLoc != m.(*liveSlot).VarLoc {
				state.changedVars.Add(ID(state.SlotVars[k]))
			}
		}
	}

	reset := func(ourStartState abt.T) {
		if !(forLocationLists || blockChanged) {
			// there is no change and this is not for location lists, do
			// not bother to reset currentState because it will not be
			// examined.
			return
		}
		state.currentState.reset(ourStartState)
	}

	// Zero predecessors
	if len(preds) == 0 {
		if previousBlock != nil {
			state.F.Fatalf("Function %v, block %s with no predecessors is not first block, has previous %s", state.F, b.String(), previousBlock.String())
		}
		// startState is empty
		reset(abt.T{})
		return abt.T{}, blockChanged
	}

	// One predecessor
	l0 := blockLocs[preds[0].ID]
	p0 := l0.endState
	if len(preds) == 1 {
		if previousBlock != nil && preds[0].ID != previousBlock.ID {
			// Change from previous block is its endState minus the predecessor's endState
			markChangedVars(blockLocs[previousBlock.ID].endState, p0)
		}
		locs.startState = p0
		blockChanged = blockChanged || l0.lastChangedTime > locs.lastCheckedTime
		reset(p0)
		return p0, blockChanged
	}

	// More than one predecessor

	if updating {
		// After the first approximation, i.e., when updating, results
		// can only get smaller, because initially backedge
		// predecessors do not participate in the intersection.  This
		// means that for the update, given the prior approximation of
		// startState, there is no need to re-intersect with unchanged
		// blocks.  Therefore remove unchanged blocks from the
		// predecessor list.
		for i := len(preds) - 1; i >= 0; i-- {
			pred := preds[i]
			if blockLocs[pred.ID].lastChangedTime > locs.lastCheckedTime {
				continue // keep this predecessor
			}
			preds[i] = preds[len(preds)-1]
			preds = preds[:len(preds)-1]
			if state.LoggingLevel > 2 {
				state.Logf("Pruned b%d, lastChanged was %d but b%d lastChecked is %d\n", pred.ID, blockLocs[pred.ID].lastChangedTime, b.ID, locs.lastCheckedTime)
			}
		}
		// Check for an early out; this should always hit for the update
		// if there are no cycles.
		if len(preds) == 0 {
			blockChanged = false

			reset(locs.startState)
			if state.LoggingLevel > 2 {
				state.Logf("Early out, no predecessors changed since last check\n")
			}
			if previousBlock != nil {
				markChangedVars(blockLocs[previousBlock.ID].endState, locs.startState)
			}
			return locs.startState, blockChanged
		}
	}

	baseID := preds[0].ID
	baseState := p0

	// Choose the predecessor with the smallest endState for intersection work
	for _, pred := range preds[1:] {
		if blockLocs[pred.ID].endState.Size() < baseState.Size() {
			baseState = blockLocs[pred.ID].endState
			baseID = pred.ID
		}
	}

	if state.LoggingLevel > 2 {
		state.Logf("Starting %v with state from b%v:\n%v", b, baseID, state.blockEndStateString(blockLocs[baseID]))
		for _, pred := range preds {
			if pred.ID == baseID {
				continue
			}
			state.Logf("Merging in state from %v:\n%v", pred, state.blockEndStateString(blockLocs[pred.ID]))
		}
	}

	state.currentState.reset(abt.T{})
	// The normal logic of "reset" is included in the intersection loop below.

	slotLocs := state.currentState.slots

	// If this is the first call, do updates on the "baseState"; if this
	// is a subsequent call, tweak the startState instead. Note that
	// these "set" values are values; there are no side effects to
	// other values as these are modified.
	newState := baseState
	if updating {
		newState = blockLocs[b.ID].startState
	}

	for it := newState.Iterator(); !it.Done(); {
		k, d := it.Next()
		thisSlot := d.(*liveSlot)
		x := thisSlot.VarLoc
		x0 := x // initial value in newState

		// Intersect this slot with the slot in all the predecessors
		for _, other := range preds {
			if !updating && other.ID == baseID {
				continue
			}
			otherSlot := blockLocs[other.ID].endState.Find(k)
			if otherSlot == nil {
				x = VarLoc{}
				break
			}
			y := otherSlot.(*liveSlot).VarLoc
			x = x.intersect(y)
			if x.absent() {
				x = VarLoc{}
				break
			}
		}

		// Delete if necessary, but not otherwise (in order to maximize sharing).
		if x.absent() {
			if !x0.absent() {
				blockChanged = true
				newState.Delete(k)
			}
			slotLocs[k] = VarLoc{}
			continue
		}
		if x != x0 {
			blockChanged = true
			newState.Insert(k, &liveSlot{VarLoc: x})
		}

		slotLocs[k] = x
		mask := uint64(x.Registers)
		for {
			if mask == 0 {
				break
			}
			reg := uint8(bits.TrailingZeros64(mask))
			mask &^= 1 << reg
			state.currentState.registers[reg] = append(state.currentState.registers[reg], SlotID(k))
		}
	}

	if previousBlock != nil {
		markChangedVars(blockLocs[previousBlock.ID].endState, newState)
	}
	locs.startState = newState
	return newState, blockChanged
}

// processValue updates locs and state.registerContents to reflect v, a
// value with the names in vSlots and homed in vReg.  "v" becomes
// visible after execution of the instructions evaluating it. It
// returns which VarIDs were modified by the Value's execution.
func (state *DebugState) processValue(v *Value, vSlots []SlotID, vReg *ssabase.Register) bool {
	locs := state.currentState
	changed := false
	setSlot := func(slot SlotID, loc VarLoc) {
		changed = true
		state.changedVars.Add(ID(state.SlotVars[slot]))
		state.changedSlots.Add(ID(slot))
		state.currentState.slots[slot] = loc
	}

	// Handle any register clobbering. Call operations, for example,
	// clobber all registers even though they don't explicitly write to
	// them.
	clobbers := ssaop.OpcodeTable[v.Op].Reg.Clobbers
	for {
		if clobbers.Empty() {
			break
		}
		reg := clobbers.PickReg()
		clobbers = clobbers.RemoveReg(reg)

		for _, slot := range locs.registers[reg] {
			if state.LoggingLevel > 1 {
				state.Logf("at %v: %v clobbered out of %v\n", v, state.Slots[slot], &state.Registers[reg])
			}

			last := locs.slots[slot]
			if last.absent() {
				state.F.Fatalf("at %v: slot %v in register %v with no location entry", v, state.Slots[slot], &state.Registers[reg])
				continue
			}
			regs := last.Registers &^ (1 << reg)
			setSlot(slot, VarLoc{regs, last.StackOffset})
		}

		locs.registers[reg] = locs.registers[reg][:0]
	}

	switch {
	case v.Op == ssaop.OpVarDef:
		n := v.Aux.(*ir.Name)
		if ir.IsSynthetic(n) || !IsVarWantedForDebug(n) {
			break
		}

		slotID := state.VarParts[n][0]
		var stackOffset StackOffset
		if v.Op == ssaop.OpVarDef {
			stackOffset = StackOffset(state.StackOffset(state.Slots[slotID])<<1 | 1)
		}
		setSlot(slotID, VarLoc{0, stackOffset})
		if state.LoggingLevel > 1 {
			if v.Op == ssaop.OpVarDef {
				state.Logf("at %v: stack-only var %v now live\n", v, state.Slots[slotID])
			} else {
				state.Logf("at %v: stack-only var %v now dead\n", v, state.Slots[slotID])
			}
		}

	case v.Op == ssaop.OpArg:
		home := state.F.GetHome(v.ID).(LocalSlot)
		stackOffset := state.StackOffset(home)<<1 | 1
		for _, slot := range vSlots {
			if state.LoggingLevel > 1 {
				state.Logf("at %v: arg %v now on stack in location %v\n", v, state.Slots[slot], home)
				if last := locs.slots[slot]; !last.absent() {
					state.Logf("at %v: unexpected arg op on already-live slot %v\n", v, state.Slots[slot])
				}
			}

			setSlot(slot, VarLoc{0, StackOffset(stackOffset)})
		}

	case v.Op == ssaop.OpStoreReg:
		home := state.F.GetHome(v.ID).(LocalSlot)
		stackOffset := state.StackOffset(home)<<1 | 1
		for _, slot := range vSlots {
			last := locs.slots[slot]
			if last.absent() {
				if state.LoggingLevel > 1 {
					state.Logf("at %v: unexpected spill of unnamed register %s\n", v, vReg)
				}
				break
			}

			setSlot(slot, VarLoc{last.Registers, StackOffset(stackOffset)})
			if state.LoggingLevel > 1 {
				state.Logf("at %v: %v spilled to stack location %v@%d\n", v, state.Slots[slot], home, state.StackOffset(home))
			}
		}

	case vReg != nil:
		if state.LoggingLevel > 1 {
			newSlots := make([]bool, len(state.Slots))
			for _, slot := range vSlots {
				newSlots[slot] = true
			}

			for _, slot := range locs.registers[vReg.Num] {
				if !newSlots[slot] {
					state.Logf("at %v: overwrote %v in register %v\n", v, state.Slots[slot], vReg)
				}
			}
		}

		for _, slot := range locs.registers[vReg.Num] {
			last := locs.slots[slot]
			setSlot(slot, VarLoc{last.Registers &^ (1 << uint8(vReg.Num)), last.StackOffset})
		}
		locs.registers[vReg.Num] = locs.registers[vReg.Num][:0]
		locs.registers[vReg.Num] = append(locs.registers[vReg.Num], vSlots...)
		for _, slot := range vSlots {
			if state.LoggingLevel > 1 {
				state.Logf("at %v: %v now in %s\n", v, state.Slots[slot], vReg)
			}

			last := locs.slots[slot]
			setSlot(slot, VarLoc{1<<uint8(vReg.Num) | last.Registers, last.StackOffset})
		}
	}
	return changed
}

func (e *pendingEntry) clear() {
	e.present = false
	e.startBlock = 0
	e.startValue = 0
	clear(e.pieces)
}

// BuildLocationLists builds location lists for all the user variables
// in state.f, using the information about block state in blockLocs.
// The returned location lists are not fully complete. They are in
// terms of SSA values rather than PCs, and have no base address/end
// entries. They will be finished by PutLocationList.
func (state *DebugState) BuildLocationLists(blockLocs []*BlockDebug) {
	// Run through the function in program text order, building up location
	// lists as we go. The heavy lifting has mostly already been done.

	var prevBlock *Block
	for _, b := range state.F.Blocks {
		state.mergePredecessors(b, blockLocs, prevBlock, true)

		// Handle any differences among predecessor blocks and previous block (perhaps not a predecessor)
		for _, varID := range state.changedVars.Contents() {
			state.updateVar(VarID(varID), b, BlockStart)
		}
		state.changedVars.Clear()

		if !blockLocs[b.ID].relevant {
			continue
		}

		mustBeFirst := func(v *Value) bool {
			return v.Op == ssaop.OpPhi || v.Op.IsLoweredGetClosurePtr() ||
				v.Op == ssaop.OpArgIntReg || v.Op == ssaop.OpArgFloatReg
		}

		blockPrologComplete := func(v *Value) bool {
			if b.ID != state.F.Entry.ID {
				return !ssaop.OpcodeTable[v.Op].ZeroWidth
			} else {
				return v.Op == ssaop.OpInitMem
			}
		}

		// Examine the prolog portion of the block to process special
		// zero-width ops such as Arg, Phi, LoweredGetClosurePtr (etc)
		// whose lifetimes begin at the block starting point. In an
		// entry block, allow for the possibility that we may see Arg
		// ops that appear _after_ other non-zero-width operations.
		// Example:
		//
		//   v33 = ArgIntReg <uintptr> {foo+0} [0] : AX (foo)
		//   v34 = ArgIntReg <uintptr> {bar+0} [0] : BX (bar)
		//   ...
		//   v77 = StoreReg <unsafe.Pointer> v67 : ctx+8[unsafe.Pointer]
		//   v78 = StoreReg <unsafe.Pointer> v68 : ctx[unsafe.Pointer]
		//   v79 = Arg <*uint8> {args} : args[*uint8] (args[*uint8])
		//   v80 = Arg <int> {args} [8] : args+8[int] (args+8[int])
		//   ...
		//   v1 = InitMem <mem>
		//
		// We can stop scanning the initial portion of the block when
		// we either see the InitMem op (for entry blocks) or the
		// first non-zero-width op (for other blocks).
		for idx := 0; idx < len(b.Values); idx++ {
			v := b.Values[idx]
			if blockPrologComplete(v) {
				break
			}
			// Consider only "lifetime begins at block start" ops.
			if !mustBeFirst(v) && v.Op != ssaop.OpArg {
				continue
			}
			slots := state.ValueNames[v.ID]
			reg, _ := state.F.GetHome(v.ID).(*ssabase.Register)
			changed := state.processValue(v, slots, reg) // changed == added to state.changedVars
			if changed {
				for _, varID := range state.changedVars.Contents() {
					state.updateVar(VarID(varID), v.Block, BlockStart)
				}
				state.changedVars.Clear()
			}
		}

		// Now examine the block again, handling things other than the
		// "begins at block start" lifetimes.
		zeroWidthPending := false
		prologComplete := false
		// expect to see values in pattern (apc)* (zerowidth|real)*
		for _, v := range b.Values {
			if blockPrologComplete(v) {
				prologComplete = true
			}
			slots := state.ValueNames[v.ID]
			reg, _ := state.F.GetHome(v.ID).(*ssabase.Register)
			changed := state.processValue(v, slots, reg) // changed == added to state.changedVars

			if ssaop.OpcodeTable[v.Op].ZeroWidth {
				if prologComplete && mustBeFirst(v) {
					panic(fmt.Errorf("Unexpected placement of op '%s' appearing after non-pseudo-op at beginning of block %s in %s\n%s", v.LongString(), b, b.Func.Name, b.Func))
				}
				if changed {
					if mustBeFirst(v) || v.Op == ssaop.OpArg {
						// already taken care of above
						continue
					}
					zeroWidthPending = true
				}
				continue
			}
			if !changed && !zeroWidthPending {
				continue
			}

			// Not zero-width; i.e., a "real" instruction.
			zeroWidthPending = false
			for _, varID := range state.changedVars.Contents() {
				state.updateVar(VarID(varID), v.Block, v)
			}
			state.changedVars.Clear()
		}
		for _, varID := range state.changedVars.Contents() {
			state.updateVar(VarID(varID), b, BlockEnd)
		}

		prevBlock = b
	}

	if state.LoggingLevel > 0 {
		state.Logf("location lists:\n")
	}

	// Flush any leftover entries live at the end of the last block.
	for varID := range state.Lists {
		state.writePendingEntry(VarID(varID), -1, FuncEnd.ID)
		list := state.Lists[varID]
		if state.LoggingLevel > 0 {
			if len(list) == 0 {
				state.Logf("\t%v : empty list\n", state.Vars[varID])
			} else {
				state.Logf("\t%v : %d entries\n", state.Vars[varID], len(list))
			}
		}
	}
}

// updateVar updates the pending location list entry for varID to
// reflect the new locations in curLoc, beginning at v in block b.
// v may be one of the special values indicating block start or end.
func (state *DebugState) updateVar(varID VarID, b *Block, v *Value) {
	curLoc := state.currentState.slots
	// Assemble the location list entry with whatever's live.
	empty := true
	for _, slotID := range state.VarSlots[varID] {
		if !curLoc[slotID].absent() {
			empty = false
			break
		}
	}
	pending := &state.pendingEntries[varID]
	if empty {
		state.writePendingEntry(varID, b.ID, v.ID)
		pending.clear()
		return
	}

	// Extend the previous entry if possible.
	if pending.present {
		merge := true
		for i, slotID := range state.VarSlots[varID] {
			if !canMerge(pending.pieces[i], curLoc[slotID]) {
				merge = false
				break
			}
		}
		if merge {
			return
		}
	}

	state.writePendingEntry(varID, b.ID, v.ID)
	pending.present = true
	pending.startBlock = b.ID
	pending.startValue = v.ID
	for i, slot := range state.VarSlots[varID] {
		pending.pieces[i] = curLoc[slot]
	}
}

// writePendingEntry writes out the pending entry for varID, if any,
// terminated at endBlock/Value.
func (state *DebugState) writePendingEntry(varID VarID, endBlock, endValue ID) {
	pending := state.pendingEntries[varID]
	if !pending.present {
		return
	}

	// Skip zero-width entries where start and end coordinates are identical.
	if pending.startBlock == endBlock && pending.startValue == endValue {
		if state.LoggingLevel > 1 {
			state.Logf("Skipping empty location list for %v in %s\n", state.Vars[varID], state.F.Name)
		}
		return
	}

	if state.LoggingLevel > 1 {
		var partStrs []string
		for i, slot := range state.VarSlots[varID] {
			partStrs = append(partStrs, fmt.Sprintf("%v@%v", state.Slots[slot], state.LocString(pending.pieces[i])))
		}
		state.Logf("Add entry for %v: \tb%vv%v-b%vv%v = \t%v\n", state.Vars[varID], pending.startBlock, pending.startValue, endBlock, endValue, strings.Join(partStrs, " "))
	}

	// Build the DWARF location expression.
	var expr []byte
	for i, slotID := range state.VarSlots[varID] {
		loc := pending.pieces[i]
		slot := state.Slots[slotID]

		if !loc.absent() {
			if loc.onStack() {
				if loc.stackOffsetValue() == 0 {
					expr = append(expr, dwarf.DW_OP_call_frame_cfa)
				} else {
					expr = append(expr, dwarf.DW_OP_fbreg)
					expr = dwarf.AppendSleb128(expr, int64(loc.stackOffsetValue()))
				}
			} else {
				regnum := state.Ctxt.Arch.DWARFRegisters[state.Registers[firstReg(loc.Registers)].ObjNum]
				if regnum < 32 {
					expr = append(expr, dwarf.DW_OP_reg0+byte(regnum))
				} else {
					expr = append(expr, dwarf.DW_OP_regx)
					expr = dwarf.AppendUleb128(expr, uint64(regnum))
				}
			}
		}

		if len(state.VarSlots[varID]) > 1 {
			expr = append(expr, dwarf.DW_OP_piece)
			expr = dwarf.AppendUleb128(expr, uint64(slot.Type.Size()))
		}
	}

	entry := LocListEntry{
		StartBlock: pending.startBlock,
		StartValue: pending.startValue,
		EndBlock:   endBlock,
		EndValue:   endValue,
		Expr:       expr,
	}
	state.Lists[varID] = append(state.Lists[varID], entry)
}
