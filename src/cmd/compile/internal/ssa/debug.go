// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/abi"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/ssa/ssabase"
	"cmd/compile/internal/ssa/ssacore"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
	"cmd/internal/dwarf"
	"cmd/internal/obj"
	"cmd/internal/src"
	"cmp"
	"internal/buildcfg"
	"slices"
)

// A FuncDebug contains all the debug information for the variables in a
// function. Variables are identified by their LocalSlot, which may be
// the result of decomposing a larger variable.
type FuncDebug struct {
	// Slots is all the slots used in the debug info, indexed by their SlotID.
	Slots []ssacore.LocalSlot
	// The user variables, indexed by VarID.
	Vars []*ir.Name
	// The slots that make up each variable, indexed by VarID.
	VarSlots [][]ssacore.SlotID
	// The location list data, indexed by VarID. Must be processed by PutLocationList.
	LocationLists [][]ssacore.LocListEntry
	// Register-resident output parameters for the function. This is filled in at
	// SSA generation time.
	RegOutputParams []*ir.Name
	// Variable declarations that were removed during optimization
	OptDcl []*ir.Name
	// The ssa.Func.EntryID value, used to build location lists for
	// return values promoted to heap in later DWARF generation.
	EntryID ssacore.ID

	// Filled in by the user. Translates Block and Value ID to PC.
	//
	// NOTE: block is only used if value is BlockStart.ID or BlockEnd.ID.
	// Otherwise, it is ignored.
	GetPC func(block, value ssacore.ID) int64
}

// slotCanonicalizer is a table used to lookup and canonicalize
// LocalSlot's in a type insensitive way (e.g. taking into account the
// base name, offset, and width of the slot, but ignoring the slot
// type).
type slotCanonicalizer struct {
	slmap  map[slotKey]SlKeyIdx
	slkeys []ssacore.LocalSlot
}

func newSlotCanonicalizer() *slotCanonicalizer {
	return &slotCanonicalizer{
		slmap:  make(map[slotKey]SlKeyIdx),
		slkeys: []ssacore.LocalSlot{ssacore.LocalSlot{N: nil}},
	}
}

type SlKeyIdx uint32

const noSlot = SlKeyIdx(0)

// slotKey is a type-insensitive encapsulation of a LocalSlot; it
// is used to key a map within slotCanonicalizer.
type slotKey struct {
	name        *ir.Name
	offset      int64
	width       int64
	splitOf     SlKeyIdx // idx in slkeys slice in slotCanonicalizer
	splitOffset int64
}

// lookup looks up a LocalSlot in the slot canonicalizer "sc", returning
// a canonical index for the slot, and adding it to the table if need
// be. Return value is the canonical slot index, and a boolean indicating
// whether the slot was found in the table already (TRUE => found).
func (sc *slotCanonicalizer) lookup(ls ssacore.LocalSlot) (SlKeyIdx, bool) {
	split := noSlot
	if ls.SplitOf != nil {
		split, _ = sc.lookup(*ls.SplitOf)
	}
	k := slotKey{
		name: ls.N, offset: ls.Off, width: ls.Type.Size(),
		splitOf: split, splitOffset: ls.SplitOffset,
	}
	if idx, ok := sc.slmap[k]; ok {
		return idx, true
	}
	rv := SlKeyIdx(len(sc.slkeys))
	sc.slkeys = append(sc.slkeys, ls)
	sc.slmap[k] = rv
	return rv, false
}

func (sc *slotCanonicalizer) canonSlot(idx SlKeyIdx) ssacore.LocalSlot {
	return sc.slkeys[idx]
}

// PopulateABIInRegArgOps examines the entry block of the function
// and looks for incoming parameters that have missing or partial
// OpArg{Int,Float}Reg values, inserting additional values in
// cases where they are missing. Example:
//
//	func foo(s string, used int, notused int) int {
//	  return len(s) + used
//	}
//
// In the function above, the incoming parameter "used" is fully live,
// "notused" is not live, and "s" is partially live (only the length
// field of the string is used). At the point where debug value
// analysis runs, we might expect to see an entry block with:
//
//	b1:
//	  v4 = ArgIntReg <uintptr> {s+8} [0] : BX
//	  v5 = ArgIntReg <int> {used} [0] : CX
//
// While this is an accurate picture of the live incoming params,
// we also want to have debug locations for non-live params (or
// their non-live pieces), e.g. something like
//
//	b1:
//	  v9 = ArgIntReg <*uint8> {s+0} [0] : AX
//	  v4 = ArgIntReg <uintptr> {s+8} [0] : BX
//	  v5 = ArgIntReg <int> {used} [0] : CX
//	  v10 = ArgIntReg <int> {unused} [0] : DI
//
// This function examines the live OpArg{Int,Float}Reg values and
// synthesizes new (dead) values for the non-live params or the
// non-live pieces of partially live params.
func PopulateABIInRegArgOps(f *ssacore.Func) {
	pri := f.ABISelf.ABIAnalyzeFuncType(f.Type)

	// When manufacturing new slots that correspond to splits of
	// composite parameters, we want to avoid creating a new sub-slot
	// that differs from some existing sub-slot only by type, since
	// the debug location analysis will treat that slot as a separate
	// entity. To achieve this, create a lookup table of existing
	// slots that is type-insenstitive.
	sc := newSlotCanonicalizer()
	for _, sl := range f.Names {
		sc.lookup(sl)
	}

	// Add slot -> value entry to f.NamedValues if not already present.
	addToNV := func(v *ssacore.Value, sl ssacore.LocalSlot) {
		values, ok := f.NamedValues[sl]
		if !ok {
			// Haven't seen this slot yet.
			f.Names = append(f.Names, sl)
		} else {
			for _, ev := range values {
				if v == ev {
					return
				}
			}
		}
		values = append(values, v)
		f.NamedValues[sl] = values
	}

	newValues := []*ssacore.Value{}

	abiRegIndexToRegister := func(reg abi.RegIndex) int8 {
		i := f.ABISelf.FloatIndexFor(reg)
		if i >= 0 { // float PR
			return f.Config.FloatParamRegs[i]
		} else {
			return f.Config.IntParamRegs[reg]
		}
	}

	// Helper to construct a new OpArg{Float,Int}Reg op value.
	var pos src.XPos
	if len(f.Entry.Values) != 0 {
		pos = f.Entry.Values[0].Pos
	}
	synthesizeOpIntFloatArg := func(n *ir.Name, t *types.Type, reg abi.RegIndex, sl ssacore.LocalSlot) *ssacore.Value {
		aux := &ssacore.AuxNameOffset{Name: n, Offset: sl.Off}
		op, auxInt := ssacore.ArgOpAndRegisterFor(reg, f.ABISelf)
		v := f.NewValueNoBlock(op, t, pos)
		v.AuxInt = auxInt
		v.Aux = aux
		v.Args = nil
		v.Block = f.Entry
		newValues = append(newValues, v)
		addToNV(v, sl)
		f.SetHome(v, &f.Config.Registers[abiRegIndexToRegister(reg)])
		return v
	}

	// Make a pass through the entry block looking for
	// OpArg{Int,Float}Reg ops. Record the slots they use in a table
	// ("sc"). We use a type-insensitive lookup for the slot table,
	// since the type we get from the ABI analyzer won't always match
	// what the compiler uses when creating OpArg{Int,Float}Reg ops.
	for _, v := range f.Entry.Values {
		if v.Op == ssaop.OpArgIntReg || v.Op == ssaop.OpArgFloatReg {
			aux := v.Aux.(*ssacore.AuxNameOffset)
			sl := ssacore.LocalSlot{N: aux.Name, Type: v.Type, Off: aux.Offset}
			// install slot in lookup table
			idx, _ := sc.lookup(sl)
			// add to f.NamedValues if not already present
			addToNV(v, sc.canonSlot(idx))
		} else if v.Op.IsCall() {
			// if we hit a call, we've gone too far.
			break
		}
	}

	// Now make a pass through the ABI in-params, looking for params
	// or pieces of params that we didn't encounter in the loop above.
	for _, inp := range pri.InParams() {
		if !isNamedRegParam(inp) {
			continue
		}
		n := inp.Name

		// Param is spread across one or more registers. Walk through
		// each piece to see whether we've seen an arg reg op for it.
		types, offsets := inp.RegisterTypesAndOffsets()
		for k, t := range types {
			// Note: this recipe for creating a LocalSlot is designed
			// to be compatible with the one used in expand_calls.go
			// as opposed to decompose.go. The expand calls code just
			// takes the base name and creates an offset into it,
			// without using the SplitOf/SplitOffset fields. The code
			// in decompose.go does the opposite -- it creates a
			// LocalSlot object with "Off" set to zero, but with
			// SplitOf pointing to a parent slot, and SplitOffset
			// holding the offset into the parent object.
			pieceSlot := ssacore.LocalSlot{N: n, Type: t, Off: offsets[k]}

			// Look up this piece to see if we've seen a reg op
			// for it. If not, create one.
			_, found := sc.lookup(pieceSlot)
			if !found {
				// This slot doesn't appear in the map, meaning it
				// corresponds to an in-param that is not live, or
				// a portion of an in-param that is not live/used.
				// Add a new dummy OpArg{Int,Float}Reg for it.
				synthesizeOpIntFloatArg(n, t, inp.Registers[k],
					pieceSlot)
			}
		}
	}

	// Insert the new values into the head of the block.
	f.Entry.Values = append(newValues, f.Entry.Values...)
}

// BuildFuncDebug builds debug information for f, placing the results
// in "rval". f must be fully processed, so that each Value is where it
// will be when machine code is emitted.
func BuildFuncDebug(ctxt *obj.Link, f *ssacore.Func, loggingLevel int, stackOffset func(ssacore.LocalSlot) int32, rval *FuncDebug) {
	if f.RegAlloc == nil {
		f.Fatalf("BuildFuncDebug on func %v that has not been fully processed", f)
	}
	state := &f.Cache.DebugState
	state.LoggingLevel = loggingLevel % 1000

	// A specific number demands exactly that many iterations. Under
	// particular circumstances it make require more than the total of
	// 2 passes implied by a single run through liveness and a single
	// run through location list generation.
	state.ConvergeCount = loggingLevel / 1000
	state.F = f
	state.Registers = f.Config.Registers
	state.StackOffset = stackOffset
	state.Ctxt = ctxt

	if buildcfg.Experiment.RegabiArgs {
		PopulateABIInRegArgOps(f)
	}

	if state.LoggingLevel > 0 {
		state.Logf("Generating location lists for function %q\n", f.Name)
	}

	if state.VarParts == nil {
		state.VarParts = make(map[*ir.Name][]ssacore.SlotID)
	} else {
		clear(state.VarParts)
	}

	// Recompose any decomposed variables, and establish the canonical
	// IDs for each var and slot by filling out state.vars and state.slots.

	state.Slots = state.Slots[:0]
	state.Vars = state.Vars[:0]
	for i, slot := range f.Names {
		state.Slots = append(state.Slots, slot)
		if ir.IsSynthetic(slot.N) || !ssacore.IsVarWantedForDebug(slot.N) {
			continue
		}

		topSlot := slot
		for topSlot.SplitOf != nil {
			topSlot = *topSlot.SplitOf
		}
		if _, ok := state.VarParts[topSlot.N]; !ok {
			state.Vars = append(state.Vars, topSlot.N)
		}
		state.VarParts[topSlot.N] = append(state.VarParts[topSlot.N], ssacore.SlotID(i))
	}

	// Recreate the LocalSlot for each stack-only variable.
	// This would probably be better as an output from stackframe.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op == ssaop.OpVarDef {
				n := v.Aux.(*ir.Name)
				if ir.IsSynthetic(n) || !ssacore.IsVarWantedForDebug(n) {
					continue
				}

				if _, ok := state.VarParts[n]; !ok {
					slot := ssacore.LocalSlot{N: n, Type: v.Type, Off: 0}
					state.Slots = append(state.Slots, slot)
					state.VarParts[n] = []ssacore.SlotID{ssacore.SlotID(len(state.Slots) - 1)}
					state.Vars = append(state.Vars, n)
				}
			}
		}
	}

	// Fill in the var<->slot mappings.
	if cap(state.VarSlots) < len(state.Vars) {
		state.VarSlots = make([][]ssacore.SlotID, len(state.Vars))
	} else {
		state.VarSlots = state.VarSlots[:len(state.Vars)]
		for i := range state.VarSlots {
			state.VarSlots[i] = state.VarSlots[i][:0]
		}
	}
	if cap(state.SlotVars) < len(state.Slots) {
		state.SlotVars = make([]ssacore.VarID, len(state.Slots))
	} else {
		state.SlotVars = state.SlotVars[:len(state.Slots)]
	}

	for varID, n := range state.Vars {
		parts := state.VarParts[n]
		slices.SortFunc(parts, func(a, b ssacore.SlotID) int {
			return cmp.Compare(varOffset(state.Slots[a]), varOffset(state.Slots[b]))
		})

		state.VarSlots[varID] = parts
		for _, slotID := range parts {
			state.SlotVars[slotID] = ssacore.VarID(varID)
		}
	}

	state.InitializeCache(f, len(state.VarParts), len(state.Slots))

	for i, slot := range f.Names {
		if ir.IsSynthetic(slot.N) || !ssacore.IsVarWantedForDebug(slot.N) {
			continue
		}
		for _, value := range f.NamedValues[slot] {
			state.ValueNames[value.ID] = append(state.ValueNames[value.ID], ssacore.SlotID(i))
		}
	}

	blockLocs := state.Liveness()
	state.BuildLocationLists(blockLocs)

	// Populate "rval" with what we've computed.
	rval.Slots = state.Slots
	rval.VarSlots = state.VarSlots
	rval.Vars = state.Vars
	rval.LocationLists = state.Lists
}

// varOffset returns the offset of slot within the user variable it was
// decomposed from. This has nothing to do with its stack offset.
func varOffset(slot ssacore.LocalSlot) int64 {
	offset := slot.Off
	s := &slot
	for ; s.SplitOf != nil; s = s.SplitOf {
		offset += s.SplitOffset
	}
	return offset
}

// PutLocationList adds entries (a location list in structured form)
// to listSym, encoding it in the appropriate DWARF format.
func (debugInfo *FuncDebug) PutLocationList(entries []ssacore.LocListEntry, ctxt *obj.Link, listSym, startPC *obj.LSym) {
	if buildcfg.Experiment.Dwarf5 {
		debugInfo.PutLocationListDwarf5(entries, ctxt, listSym, startPC)
	} else {
		debugInfo.PutLocationListDwarf4(entries, ctxt, listSym, startPC)
	}
}

// PutLocationListDwarf5 adds entries (a location list in structured form)
// to listSym in DWARF 5 format.
func (debugInfo *FuncDebug) PutLocationListDwarf5(entries []ssacore.LocListEntry, ctxt *obj.Link, listSym, startPC *obj.LSym) {
	getPC := debugInfo.GetPC

	// base address entry
	listSym.WriteInt(ctxt, listSym.Size, 1, dwarf.DW_LLE_base_addressx)
	listSym.WriteDwTxtAddrx(ctxt, listSym.Size, startPC, ctxt.DwTextCount*2)

	var stbuf, enbuf [10]byte
	for _, entry := range entries {
		begin := getPC(entry.StartBlock, entry.StartValue)
		end := getPC(entry.EndBlock, entry.EndValue)

		// Write LLE_offset_pair tag followed by payload (ULEB for start
		// and then end).
		listSym.WriteInt(ctxt, listSym.Size, 1, dwarf.DW_LLE_offset_pair)
		stb := stbuf[:0]
		enb := enbuf[:0]
		stb = dwarf.AppendUleb128(stb, uint64(begin))
		enb = dwarf.AppendUleb128(enb, uint64(end))
		listSym.WriteBytes(ctxt, listSym.Size, stb)
		listSym.WriteBytes(ctxt, listSym.Size, enb)

		// DWARF5 uses ULEB128-encoded length for the location expression.
		stb = stbuf[:0]
		stb = dwarf.AppendUleb128(stb, uint64(len(entry.Expr)))
		listSym.WriteBytes(ctxt, listSym.Size, stb)
		listSym.WriteBytes(ctxt, listSym.Size, entry.Expr)
	}

	// Terminator
	listSym.WriteInt(ctxt, listSym.Size, 1, dwarf.DW_LLE_end_of_list)
}

// PutLocationListDwarf4 adds entries (a location list in structured form)
// to listSym in DWARF 4 format.
func (debugInfo *FuncDebug) PutLocationListDwarf4(entries []ssacore.LocListEntry, ctxt *obj.Link, listSym, startPC *obj.LSym) {
	getPC := debugInfo.GetPC

	if ctxt.UseBASEntries {
		listSym.WriteInt(ctxt, listSym.Size, ctxt.Arch.PtrSize, ^0)
		listSym.WriteAddr(ctxt, listSym.Size, ctxt.Arch.PtrSize, startPC, 0)
	}

	for _, entry := range entries {
		begin := getPC(entry.StartBlock, entry.StartValue)
		end := getPC(entry.EndBlock, entry.EndValue)

		// Horrible hack. If a range contains only zero-width
		// instructions, e.g. an Arg, and it's at the beginning of the
		// function, this would be indistinguishable from an
		// end entry. Fudge it.
		if begin == 0 && end == 0 {
			end = 1
		}

		if ctxt.UseBASEntries {
			listSym.WriteInt(ctxt, listSym.Size, ctxt.Arch.PtrSize, begin)
			listSym.WriteInt(ctxt, listSym.Size, ctxt.Arch.PtrSize, end)
		} else {
			listSym.WriteCURelativeAddr(ctxt, listSym.Size, startPC, begin)
			listSym.WriteCURelativeAddr(ctxt, listSym.Size, startPC, end)
		}

		// Write 2-byte length prefix followed by the location expression.
		listSym.WriteInt(ctxt, listSym.Size, 2, int64(len(entry.Expr)))
		listSym.WriteBytes(ctxt, listSym.Size, entry.Expr)
	}

	// End entry.
	listSym.WriteInt(ctxt, listSym.Size, ctxt.Arch.PtrSize, 0)
	listSym.WriteInt(ctxt, listSym.Size, ctxt.Arch.PtrSize, 0)
}

// locatePrologEnd walks the entry block of a function with incoming
// register arguments and locates the last instruction in the prolog
// that spills a register arg. It returns the ID of that instruction,
// and (where appropriate) the prolog's lowered closure ptr store inst.
//
// Example:
//
//	b1:
//	    v3 = ArgIntReg <int> {p1+0} [0] : AX
//	    ... more arg regs ..
//	    v4 = ArgFloatReg <float32> {f1+0} [0] : X0
//	    v52 = MOVQstore <mem> {p1} v2 v3 v1
//	    ... more stores ...
//	    v68 = MOVSSstore <mem> {f4} v2 v67 v66
//	    v38 = MOVQstoreconst <mem> {blob} [val=0,off=0] v2 v32
//
// Important: locatePrologEnd is expected to work properly only with
// optimization turned off (e.g. "-N"). If optimization is enabled
// we can't be assured of finding all input arguments spilled in the
// entry block prolog.
func locatePrologEnd(f *ssacore.Func, needCloCtx bool) (ssacore.ID, *ssacore.Value) {

	// returns true if this instruction looks like it moves an ABI
	// register (or context register for rangefunc bodies) to the
	// stack, along with the value being stored.
	isRegMoveLike := func(v *ssacore.Value) (bool, ssacore.ID) {
		n, ok := v.Aux.(*ir.Name)
		var r ssacore.ID
		if (!ok || n.Class != ir.PPARAM) && !needCloCtx {
			return false, r
		}
		regInputs, memInputs, spInputs := 0, 0, 0
		for _, a := range v.Args {
			if a.Op == ssaop.OpArgIntReg || a.Op == ssaop.OpArgFloatReg ||
				(needCloCtx && a.Op.IsLoweredGetClosurePtr()) {
				regInputs++
				r = a.ID
			} else if a.Type.IsMemory() {
				memInputs++
			} else if a.Op == ssaop.OpSP {
				spInputs++
			} else {
				return false, r
			}
		}
		return v.Type.IsMemory() && memInputs == 1 &&
			regInputs == 1 && spInputs == 1, r
	}

	// OpArg*Reg values we've seen so far on our forward walk,
	// for which we have not yet seen a corresponding spill.
	regArgs := make([]ssacore.ID, 0, 32)

	// removeReg tries to remove a value from regArgs, returning true
	// if found and removed, or false otherwise.
	removeReg := func(r ssacore.ID) bool {
		for i := 0; i < len(regArgs); i++ {
			if regArgs[i] == r {
				regArgs = slices.Delete(regArgs, i, i+1)
				return true
			}
		}
		return false
	}

	// Walk forwards through the block. When we see OpArg*Reg, record
	// the value it produces in the regArgs list. When see a store that uses
	// the value, remove the entry. When we hit the last store (use)
	// then we've arrived at the end of the prolog.
	var cloRegStore *ssacore.Value
	for k, v := range f.Entry.Values {
		if v.Op == ssaop.OpArgIntReg || v.Op == ssaop.OpArgFloatReg {
			regArgs = append(regArgs, v.ID)
			continue
		}
		if needCloCtx && v.Op.IsLoweredGetClosurePtr() {
			regArgs = append(regArgs, v.ID)
			cloRegStore = v
			continue
		}
		if ok, r := isRegMoveLike(v); ok {
			if removed := removeReg(r); removed {
				if len(regArgs) == 0 {
					// Found our last spill; return the value after
					// it. Note that it is possible that this spill is
					// the last instruction in the block. If so, then
					// return the "end of block" sentinel.
					if k < len(f.Entry.Values)-1 {
						return f.Entry.Values[k+1].ID, cloRegStore
					}
					return ssacore.BlockEnd.ID, cloRegStore
				}
			}
		}
		if v.Op.IsCall() {
			// if we hit a call, we've gone too far.
			return v.ID, cloRegStore
		}
	}
	// nothing found
	return ssacore.ID(-1), cloRegStore
}

// isNamedRegParam returns true if the param corresponding to "p"
// is a named, non-blank input parameter assigned to one or more
// registers.
func isNamedRegParam(p abi.ABIParamAssignment) bool {
	if p.Name == nil {
		return false
	}
	n := p.Name
	if n.Sym() == nil || n.Sym().IsBlank() {
		return false
	}
	if len(p.Registers) == 0 {
		return false
	}
	return true
}

// BuildFuncDebugNoOptimized populates a FuncDebug object "rval" with
// entries corresponding to the register-resident input parameters for
// the function "f"; it is used when we are compiling without
// optimization but the register ABI is enabled. For each reg param,
// it constructs a 2-element location list: the first element holds
// the input register, and the second element holds the stack location
// of the param (the assumption being that when optimization is off,
// each input param reg will be spilled in the prolog). In addition
// to the register params, here we also build location lists (where
// appropriate for the ".closureptr" compiler-synthesized variable
// needed by the debugger for range func bodies.
func BuildFuncDebugNoOptimized(ctxt *obj.Link, f *ssacore.Func, loggingEnabled bool, stackOffset func(ssacore.LocalSlot) int32, rval *FuncDebug) {
	needCloCtx := f.CloSlot != nil
	pri := f.ABISelf.ABIAnalyzeFuncType(f.Type)

	// Look to see if we have any named register-promoted parameters,
	// and/or whether we need location info for the ".closureptr"
	// synthetic variable; if not bail early and let the caller sort
	// things out for the remainder of the params/locals.
	numRegParams := 0
	for _, inp := range pri.InParams() {
		if isNamedRegParam(inp) {
			numRegParams++
		}
	}
	if numRegParams == 0 && !needCloCtx {
		return
	}

	state := ssacore.DebugState{F: f}

	if loggingEnabled {
		state.Logf("generating -N reg param loc lists for func %q\n", f.Name)
	}

	// cloReg stores the obj register num that the context register
	// appears in within the function prolog, where appropriate.
	var cloReg int16

	extraForCloCtx := 0
	if needCloCtx {
		extraForCloCtx = 1
	}

	// Allocate location lists.
	rval.LocationLists = make([][]ssacore.LocListEntry, numRegParams+extraForCloCtx)

	// Locate the value corresponding to the last spill of
	// an input register.
	afterPrologVal, cloRegStore := locatePrologEnd(f, needCloCtx)

	if needCloCtx {
		reg, _ := state.F.GetHome(cloRegStore.ID).(*ssabase.Register)
		cloReg = reg.ObjNum
		if loggingEnabled {
			state.Logf("needCloCtx is true for func %q, cloreg=%v\n",
				f.Name, reg)
		}
	}

	addVarSlot := func(name *ir.Name, typ *types.Type) {
		sl := ssacore.LocalSlot{N: name, Type: typ, Off: 0}
		rval.Vars = append(rval.Vars, name)
		rval.Slots = append(rval.Slots, sl)
		slid := len(rval.VarSlots)
		rval.VarSlots = append(rval.VarSlots, []ssacore.SlotID{ssacore.SlotID(slid)})
	}

	// Make an initial pass to populate the vars/slots for our return
	// value, covering first the input parameters and then (if needed)
	// the special ".closureptr" var for rangefunc bodies.
	params := []abi.ABIParamAssignment{}
	for _, inp := range pri.InParams() {
		if !isNamedRegParam(inp) {
			// will be sorted out elsewhere
			continue
		}
		if !ssacore.IsVarWantedForDebug(inp.Name) {
			continue
		}
		addVarSlot(inp.Name, inp.Type)
		params = append(params, inp)
	}
	if needCloCtx {
		addVarSlot(f.CloSlot, f.CloSlot.Type())
		cloAssign := abi.ABIParamAssignment{
			Type:      f.CloSlot.Type(),
			Name:      f.CloSlot,
			Registers: []abi.RegIndex{0}, // dummy
		}
		params = append(params, cloAssign)
	}

	// Walk the input params again and process the register-resident elements.
	pidx := 0
	for _, inp := range params {
		if !isNamedRegParam(inp) {
			// will be sorted out elsewhere
			continue
		}
		if !ssacore.IsVarWantedForDebug(inp.Name) {
			continue
		}

		sl := rval.Slots[pidx]
		n := rval.Vars[pidx]

		if afterPrologVal == ssacore.ID(-1) {
			// This can happen for degenerate functions with infinite
			// loops such as that in issue 45948. In such cases, leave
			// the var/slot set up for the param, but don't try to
			// emit a location list.
			if loggingEnabled {
				state.Logf("locatePrologEnd failed, skipping %v\n", n)
			}
			pidx++
			continue
		}

		// Param is arriving in one or more registers. We need a 2-element
		// location expression for it. First entry in location list
		// will correspond to lifetime in input registers.
		if loggingEnabled {
			state.Logf("param %v:\n  [<entry>, %d]:\n", n, afterPrologVal)
		}
		var regExpr []byte
		rtypes, _ := inp.RegisterTypesAndOffsets()
		padding := make([]uint64, 0, 32)
		padding = inp.ComputePadding(padding)
		for k, r := range inp.Registers {
			var reg int16
			if n == f.CloSlot {
				reg = cloReg
			} else {
				reg = ssacore.ObjRegForAbiReg(r, f.Config)
			}
			dwreg := ctxt.Arch.DWARFRegisters[reg]
			if dwreg < 32 {
				regExpr = append(regExpr, dwarf.DW_OP_reg0+byte(dwreg))
			} else {
				regExpr = append(regExpr, dwarf.DW_OP_regx)
				regExpr = dwarf.AppendUleb128(regExpr, uint64(dwreg))
			}
			if loggingEnabled {
				state.Logf("    piece %d -> dwreg %d", k, dwreg)
			}
			if len(inp.Registers) > 1 {
				regExpr = append(regExpr, dwarf.DW_OP_piece)
				ts := rtypes[k].Size()
				regExpr = dwarf.AppendUleb128(regExpr, uint64(ts))
				if padding[k] > 0 {
					if loggingEnabled {
						state.Logf(" [pad %d bytes]", padding[k])
					}
					regExpr = append(regExpr, dwarf.DW_OP_piece)
					regExpr = dwarf.AppendUleb128(regExpr, padding[k])
				}
			}
			if loggingEnabled {
				state.Logf("\n")
			}
		}
		rval.LocationLists[pidx] = append(rval.LocationLists[pidx], ssacore.LocListEntry{
			StartBlock: f.Entry.ID,
			StartValue: ssacore.BlockStart.ID,
			EndBlock:   f.Entry.ID,
			EndValue:   afterPrologVal,
			Expr:       regExpr,
		})

		// Second entry in the location list will be the stack home
		// of the param, once it has been spilled.  Emit that now.
		var stackExpr []byte
		soff := stackOffset(sl)
		if soff == 0 {
			stackExpr = append(stackExpr, dwarf.DW_OP_call_frame_cfa)
		} else {
			stackExpr = append(stackExpr, dwarf.DW_OP_fbreg)
			stackExpr = dwarf.AppendSleb128(stackExpr, int64(soff))
		}
		if loggingEnabled {
			state.Logf("  [%d, <end>): stackOffset=%d\n", afterPrologVal, soff)
		}

		rval.LocationLists[pidx] = append(rval.LocationLists[pidx], ssacore.LocListEntry{
			StartBlock: f.Entry.ID,
			StartValue: afterPrologVal,
			EndBlock:   f.Entry.ID,
			EndValue:   ssacore.FuncEnd.ID,
			Expr:       stackExpr,
		})

		pidx++
	}
}
