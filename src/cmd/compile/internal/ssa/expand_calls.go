// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/abi"
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"fmt"
)

func postExpandCallsDecompose(f *Func) {
	decomposeUser(f)    // redo user decompose to cleanup after expand calls
	decomposeBuiltIn(f) // handles both regular decomposition and cleanup.
}

func expandCalls(f *Func) {
	// Convert each aggregate arg to a call into "dismantle aggregate, store/pass parts"
	// Convert each aggregate result from a call into "assemble aggregate from parts"
	// Convert each multivalue exit into "dismantle aggregate, store/return parts"
	// Convert incoming aggregate arg into assembly of parts.
	// Feed modified AST to decompose.

	sp, _ := f.spSb()

	x := &expandState{
		f:               f,
		debug:           f.pass.debug,
		regSize:         f.Config.RegSize,
		sp:              sp,
		typs:            &f.Config.Types,
		wideSelects:     make(map[*Value]*Value),
		commonArgs:      make(map[selKey]*Value),
		commonSelectors: make(map[selKey]*Value),
		memForCall:      make(map[ID]*Value),
	}

	// For 32-bit, need to deal with decomposition of 64-bit integers, which depends on endianness.
	if f.Config.BigEndian {
		x.firstOp = OpInt64Hi
		x.secondOp = OpInt64Lo
		x.firstType = x.typs.Int32
		x.secondType = x.typs.UInt32
	} else {
		x.firstOp = OpInt64Lo
		x.secondOp = OpInt64Hi
		x.firstType = x.typs.UInt32
		x.secondType = x.typs.Int32
	}

	// Defer select processing until after all calls and selects are seen.
	var selects []*Value
	var calls []*Value
	var args []*Value
	var exitBlocks []*Block

	var m0 *Value

	// Accumulate lists of calls, args, selects, and exit blocks to process,
	// note "wide" selects consumed by stores,
	// rewrite mem for each call,
	// rewrite each OpSelectNAddr.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			switch v.Op {
			case OpInitMem:
				m0 = v

			case OpClosureLECall, OpInterLECall, OpStaticLECall, OpTailLECall:
				calls = append(calls, v)

			case OpArg:
				args = append(args, v)

			case OpStore:
				if a := v.Args[1]; a.Op == OpSelectN && !CanSSA(a.Type) {
					if a.Uses > 1 {
						panic(fmt.Errorf("Saw double use of wide SelectN %s operand of Store %s",
							a.LongString(), v.LongString()))
					}
					x.wideSelects[a] = v
				}

			case OpSelectN:
				if v.Type == types.TypeMem {
					// rewrite the mem selector in place
					call := v.Args[0]
					aux := call.Aux.(*AuxCall)
					mem := x.memForCall[call.ID]
					if mem == nil {
						v.AuxInt = int64(aux.abiInfo.OutRegistersUsed())
						x.memForCall[call.ID] = v
					} else {
						panic(fmt.Errorf("Saw two memories for call %v, %v and %v", call, mem, v))
					}
				} else {
					selects = append(selects, v)
				}

			case OpSelectNAddr:
				call := v.Args[0]
				which := v.AuxInt
				aux := call.Aux.(*AuxCall)
				pt := v.Type
				off := x.offsetFrom(x.f.Entry, x.sp, aux.OffsetOfResult(which), pt)
				v.copyOf(off)
			}
		}

		// rewrite function results from an exit block
		// values returned by function need to be split out into registers.
		if isBlockMultiValueExit(b) {
			exitBlocks = append(exitBlocks, b)
		}
	}

	// Convert each aggregate arg into Make of its parts (and so on, to primitive types)
	for _, v := range args {
		var rc registerCursor
		a := x.prAssignForArg(v)
		aux := x.f.OwnAux
		regs := a.Registers
		var offset int64
		if len(regs) == 0 {
			offset = a.FrameOffset(aux.abiInfo)
		}
		auxBase := x.offsetFrom(x.f.Entry, x.sp, offset, types.NewPtr(v.Type))
		rc.init(regs, aux.abiInfo, nil, auxBase, 0)
		x.rewriteSelectOrArg(f.Entry.Pos, f.Entry, v, v, m0, v.Type, rc)
	}

	// Rewrite selects of results (which may be aggregates) into make-aggregates of register/memory-targeted selects
	for _, v := range selects {
		if v.Op == OpInvalid {
			continue
		}

		call := v.Args[0]
		aux := call.Aux.(*AuxCall)
		mem := x.memForCall[call.ID]
		if mem == nil {
			mem = call.Block.NewValue1I(call.Pos, OpSelectN, types.TypeMem, int64(aux.abiInfo.OutRegistersUsed()), call)
			x.memForCall[call.ID] = mem
		}

		i := v.AuxInt
		regs := aux.RegsOfResult(i)

		// If this select cannot fit into SSA and is stored, either disaggregate to register stores, or mem-mem move.
		if store := x.wideSelects[v]; store != nil {
			// Use the mem that comes from the store operation.
			storeAddr := store.Args[0]
			mem := store.Args[2]
			if len(regs) > 0 {
				// Cannot do a rewrite that builds up a result from pieces; instead, copy pieces to the store operation.
				var rc registerCursor
				rc.init(regs, aux.abiInfo, nil, storeAddr, 0)
				mem = x.rewriteWideSelectToStores(call.Pos, call.Block, v, mem, v.Type, rc)
				store.copyOf(mem)
			} else {
				// Move directly from AuxBase to store target; rewrite the store instruction.
				offset := aux.OffsetOfResult(i)
				auxBase := x.offsetFrom(x.f.Entry, x.sp, offset, types.NewPtr(v.Type))
				// was Store dst, v, mem
				// now Move dst, auxBase, mem
				move := store.Block.NewValue3A(store.Pos, OpMove, types.TypeMem, v.Type, storeAddr, auxBase, mem)
				move.AuxInt = v.Type.Size()
				store.copyOf(move)
			}
			continue
		}

		var auxBase *Value
		if len(regs) == 0 {
			offset := aux.OffsetOfResult(i)
			auxBase = x.offsetFrom(x.f.Entry, x.sp, offset, types.NewPtr(v.Type))
		}
		var rc registerCursor
		rc.init(regs, aux.abiInfo, nil, auxBase, 0)
		x.rewriteSelectOrArg(call.Pos, call.Block, v, v, mem, v.Type, rc)
	}

	rewriteCall := func(v *Value, newOp Op, argStart int) {
		// Break aggregate args passed to call into smaller pieces.
		x.rewriteCallArgs(v, argStart)
		v.Op = newOp
		rts := abi.RegisterTypes(v.Aux.(*AuxCall).abiInfo.OutParams())
		v.Type = types.NewResults(append(rts, types.TypeMem))
	}

	// Rewrite calls
	for _, v := range calls {
		switch v.Op {
		case OpStaticLECall:
			rewriteCall(v, OpStaticCall, 0)
		case OpTailLECall:
			rewriteCall(v, OpTailCall, 0)
		case OpClosureLECall:
			rewriteCall(v, OpClosureCall, 2)
		case OpInterLECall:
			rewriteCall(v, OpInterCall, 1)
		}
	}

	// Rewrite results from exit blocks
	for _, b := range exitBlocks {
		v := b.Controls[0]
		x.rewriteFuncResults(v, b, f.OwnAux)
		b.SetControl(v)
	}

}

func (x *expandState) rewriteFuncResults(v *Value, b *Block, aux *AuxCall) {
	// This is very similar to rewriteCallArgs
	// differences:
	// firstArg + preArgs
	// sp vs auxBase

	m0 := v.MemoryArg()
	mem := m0

	allResults := []*Value{}
	var oldArgs []*Value
	argsWithoutMem := v.Args[:len(v.Args)-1]

	for j, a := range argsWithoutMem {
		oldArgs = append(oldArgs, a)
		i := int64(j)
		auxType := aux.TypeOfResult(i)
		auxBase := b.NewValue2A(v.Pos, OpLocalAddr, types.NewPtr(auxType), aux.NameOfResult(i), x.sp, mem)
		auxOffset := int64(0)
		aRegs := aux.RegsOfResult(int64(j))
		if a.Op == OpDereference {
			a.Op = OpLoad
		}
		var rc registerCursor
		var result *[]*Value
		if len(aRegs) > 0 {
			result = &allResults
		} else {
			if a.Op == OpLoad && a.Args[0].Op == OpLocalAddr {
				addr := a.Args[0]
				if addr.MemoryArg() == a.MemoryArg() && addr.Aux == aux.NameOfResult(i) {
					continue // Self move to output parameter
				}
			}
		}
		rc.init(aRegs, aux.abiInfo, result, auxBase, auxOffset)
		mem = x.decomposeAsNecessary(v.Pos, b, a, mem, rc)
	}
	v.resetArgs()
	v.AddArgs(allResults...)
	v.AddArg(mem)
	for _, a := range oldArgs {
		if a.Uses == 0 {
			if x.debug > 1 {
				x.Printf("...marking %v unused\n", a.LongString())
			}
			x.invalidateRecursively(a)
		}
	}
	v.Type = types.NewResults(append(abi.RegisterTypes(aux.abiInfo.OutParams()), types.TypeMem))
	return
}

func (x *expandState) rewriteCallArgs(v *Value, firstArg int) {
	if x.debug > 1 {
		x.indent(3)
		defer x.indent(-3)
		x.Printf("rewriteCallArgs(%s; %d)\n", v.LongString(), firstArg)
	}
	// Thread the stores on the memory arg
	aux := v.Aux.(*AuxCall)
	m0 := v.MemoryArg()
	mem := m0
	allResults := []*Value{}
	oldArgs := []*Value{}
	argsWithoutMem := v.Args[firstArg : len(v.Args)-1] // Also strip closure/interface Op-specific args

	sp := x.sp
	if v.Op == OpTailLECall {
		// For tail call, we unwind the frame before the call so we'll use the caller's
		// SP.
		sp = x.f.Entry.NewValue1(src.NoXPos, OpGetCallerSP, x.typs.Uintptr, mem)
	}

	for i, a := range argsWithoutMem { // skip leading non-parameter SSA Args and trailing mem SSA Arg.
		oldArgs = append(oldArgs, a)
		auxI := int64(i)
		aRegs := aux.RegsOfArg(auxI)
		aType := aux.TypeOfArg(auxI)

		if a.Op == OpDereference {
			a.Op = OpLoad
		}
		var rc registerCursor
		var result *[]*Value
		var aOffset int64
		if len(aRegs) > 0 {
			result = &allResults
		} else {
			aOffset = aux.OffsetOfArg(auxI)
		}
		if v.Op == OpTailLECall && a.Op == OpArg && a.AuxInt == 0 {
			// It's common for a tail call passing the same arguments (e.g. method wrapper),
			// so this would be a self copy. Detect this and optimize it out.
			n := a.Aux.(*ir.Name)
			if n.Class == ir.PPARAM && n.FrameOffset()+x.f.Config.ctxt.Arch.FixedFrameSize == aOffset {
				continue
			}
		}
		if x.debug > 1 {
			x.Printf("...storeArg %s, %v, %d\n", a.LongString(), aType, aOffset)
		}

		rc.init(aRegs, aux.abiInfo, result, sp, aOffset)
		mem = x.decomposeAsNecessary(v.Pos, v.Block, a, mem, rc)
	}
	var preArgStore [2]*Value
	preArgs := append(preArgStore[:0], v.Args[0:firstArg]...)
	v.resetArgs()
	v.AddArgs(preArgs...)
	v.AddArgs(allResults...)
	v.AddArg(mem)
	for _, a := range oldArgs {
		if a.Uses == 0 {
			x.invalidateRecursively(a)
		}
	}

	return
}

func (x *expandState) decomposePair(pos src.XPos, b *Block, a, mem *Value, t0, t1 *types.Type, o0, o1 Op, rc *registerCursor) *Value {
	e := b.NewValue1(pos, o0, t0, a)
	pos = pos.WithNotStmt()
	mem = x.decomposeAsNecessary(pos, b, e, mem, rc.next(t0))
	e = b.NewValue1(pos, o1, t1, a)
	mem = x.decomposeAsNecessary(pos, b, e, mem, rc.next(t1))
	return mem
}

func (x *expandState) decomposeOne(pos src.XPos, b *Block, a, mem *Value, t0 *types.Type, o0 Op, rc *registerCursor) *Value {
	e := b.NewValue1(pos, o0, t0, a)
	pos = pos.WithNotStmt()
	mem = x.decomposeAsNecessary(pos, b, e, mem, rc.next(t0))
	return mem
}

// decomposeAsNecessary converts a value (perhaps an aggregate) passed to a call or returned by a function,
// into the appropriate sequence of stores and register assignments to transmit that value in a given ABI, and
// returns the current memory after this convert/rewrite (it may be the input memory, perhaps stores were needed.)
// 'pos' is the source position all this is tied to
// 'b' is the enclosing block
// 'a' is the value to decompose
// 'm0' is the input memory arg used for the first store (or returned if there are no stores)
// 'rc' is a registerCursor which identifies the register/memory destination for the value
func (x *expandState) decomposeAsNecessary(pos src.XPos, b *Block, a, m0 *Value, rc registerCursor) *Value {
	if x.debug > 1 {
		x.indent(3)
		defer x.indent(-3)
	}
	at := a.Type
	if at.Size() == 0 {
		return m0
	}
	if a.Op == OpDereference {
		a.Op = OpLoad // For purposes of parameter passing expansion, a Dereference is a Load.
	}

	if !rc.hasRegs() && !CanSSA(at) {
		dst := x.offsetFrom(b, rc.storeDest, rc.storeOffset, types.NewPtr(at))
		if x.debug > 1 {
			x.Printf("...recur store %s at %s\n", a.LongString(), dst.LongString())
		}
		if a.Op == OpLoad {
			m0 = b.NewValue3A(pos, OpMove, types.TypeMem, at, dst, a.Args[0], m0)
			m0.AuxInt = at.Size()
			return m0
		} else {
			panic(fmt.Errorf("Store of not a load"))
		}
	}

	mem := m0
	switch at.Kind() {
	case types.TARRAY:
		et := at.Elem()
		for i := int64(0); i < at.NumElem(); i++ {
			e := b.NewValue1I(pos, OpArraySelect, et, i, a)
			pos = pos.WithNotStmt()
			mem = x.decomposeAsNecessary(pos, b, e, mem, rc.next(et))
		}
		return mem

	case types.TSTRUCT:
		for i := 0; i < at.NumFields(); i++ {
			et := at.Field(i).Type // might need to read offsets from the fields
			e := b.NewValue1I(pos, OpStructSelect, et, int64(i), a)
			pos = pos.WithNotStmt()
			if x.debug > 1 {
				x.Printf("...recur decompose %s, %v\n", e.LongString(), et)
			}
			mem = x.decomposeAsNecessary(pos, b, e, mem, rc.next(et))
		}
		return mem

	case types.TSLICE:
		mem = x.decomposeOne(pos, b, a, mem, at.Elem().PtrTo(), OpSlicePtr, &rc)
		pos = pos.WithNotStmt()
		mem = x.decomposeOne(pos, b, a, mem, x.typs.Int, OpSliceLen, &rc)
		return x.decomposeOne(pos, b, a, mem, x.typs.Int, OpSliceCap, &rc)

	case types.TSTRING:
		return x.decomposePair(pos, b, a, mem, x.typs.BytePtr, x.typs.Int, OpStringPtr, OpStringLen, &rc)

	case types.TINTER:
		mem = x.decomposeOne(pos, b, a, mem, x.typs.Uintptr, OpITab, &rc)
		pos = pos.WithNotStmt()
		// Immediate interfaces cause so many headaches.
		if a.Op == OpIMake {
			data := a.Args[1]
			for data.Op == OpStructMake1 || data.Op == OpArrayMake1 {
				data = data.Args[0]
			}
			return x.decomposeAsNecessary(pos, b, data, mem, rc.next(data.Type))
		}
		return x.decomposeOne(pos, b, a, mem, x.typs.BytePtr, OpIData, &rc)

	case types.TCOMPLEX64:
		return x.decomposePair(pos, b, a, mem, x.typs.Float32, x.typs.Float32, OpComplexReal, OpComplexImag, &rc)

	case types.TCOMPLEX128:
		return x.decomposePair(pos, b, a, mem, x.typs.Float64, x.typs.Float64, OpComplexReal, OpComplexImag, &rc)

	case types.TINT64:
		if at.Size() > x.regSize {
			return x.decomposePair(pos, b, a, mem, x.firstType, x.secondType, x.firstOp, x.secondOp, &rc)
		}
	case types.TUINT64:
		if at.Size() > x.regSize {
			return x.decomposePair(pos, b, a, mem, x.typs.UInt32, x.typs.UInt32, x.firstOp, x.secondOp, &rc)
		}
	}

	// An atomic type, either record the register or store it and update the memory.

	if rc.hasRegs() {
		if x.debug > 1 {
			x.Printf("...recur addArg %s\n", a.LongString())
		}
		rc.addArg(a)
	} else {
		dst := x.offsetFrom(b, rc.storeDest, rc.storeOffset, types.NewPtr(at))
		if x.debug > 1 {
			x.Printf("...recur store %s at %s\n", a.LongString(), dst.LongString())
		}
		mem = b.NewValue3A(pos, OpStore, types.TypeMem, at, dst, a, mem)
	}

	return mem
}

// Convert scalar OpArg into the proper OpWhateverArg instruction
// Convert scalar OpSelectN into perhaps-differently-indexed OpSelectN
// Convert aggregate OpArg into Make of its parts (which are eventually scalars)
// Convert aggregate OpSelectN into Make of its parts (which are eventually scalars)
// Returns the converted value.
//
//   - "pos" the position for any generated instructions
//   - "b" the block for any generated instructions
//   - "container" the outermost OpArg/OpSelectN
//   - "a" the instruction to overwrite, if any (only the outermost caller)
//   - "m0" the memory arg for any loads that are necessary
//   - "at" the type of the Arg/part
//   - "rc" the register/memory cursor locating the various parts of the Arg.
func (x *expandState) rewriteSelectOrArg(pos src.XPos, b *Block, container, a, m0 *Value, at *types.Type, rc registerCursor) *Value {

	if at == types.TypeMem {
		a.copyOf(m0)
		return a
	}

	makeOf := func(a *Value, op Op, args []*Value) *Value {
		if a == nil {
			a = b.NewValue0(pos, op, at)
			a.AddArgs(args...)
		} else {
			a.resetArgs()
			a.Aux, a.AuxInt = nil, 0
			a.Pos, a.Op, a.Type = pos, op, at
			a.AddArgs(args...)
		}
		return a
	}

	if at.Size() == 0 {
		// For consistency, create these values even though they'll ultimately be unused
		if at.IsArray() {
			return makeOf(a, OpArrayMake0, nil)
		}
		if at.IsStruct() {
			return makeOf(a, OpStructMake0, nil)
		}
		return a
	}

	sk := selKey{from: container, size: 0, offsetOrIndex: rc.storeOffset, typ: at}
	dupe := x.commonSelectors[sk]
	if dupe != nil {
		if a == nil {
			return dupe
		}
		a.copyOf(dupe)
		return a
	}

	var argStore [10]*Value
	args := argStore[:0]

	addArg := func(a0 *Value) {
		if a0 == nil {
			as := "<nil>"
			if a != nil {
				as = a.LongString()
			}
			panic(fmt.Errorf("a0 should not be nil, a=%v, container=%v, at=%v", as, container.LongString(), at))
		}
		args = append(args, a0)
	}

	switch at.Kind() {
	case types.TARRAY:
		et := at.Elem()
		for i := int64(0); i < at.NumElem(); i++ {
			e := x.rewriteSelectOrArg(pos, b, container, nil, m0, et, rc.next(et))
			addArg(e)
		}
		a = makeOf(a, OpArrayMake1, args)
		x.commonSelectors[sk] = a
		return a

	case types.TSTRUCT:
		// Assume ssagen/ssa.go (in buildssa) spills large aggregates so they won't appear here.
		for i := 0; i < at.NumFields(); i++ {
			et := at.Field(i).Type
			e := x.rewriteSelectOrArg(pos, b, container, nil, m0, et, rc.next(et))
			if e == nil {
				panic(fmt.Errorf("nil e, et=%v, et.Size()=%d, i=%d", et, et.Size(), i))
			}
			addArg(e)
			pos = pos.WithNotStmt()
		}
		if at.NumFields() > 4 {
			panic(fmt.Errorf("Too many fields (%d, %d bytes), container=%s", at.NumFields(), at.Size(), container.LongString()))
		}
		a = makeOf(a, StructMakeOp(at.NumFields()), args)
		x.commonSelectors[sk] = a
		return a

	case types.TSLICE:
		addArg(x.rewriteSelectOrArg(pos, b, container, nil, m0, at.Elem().PtrTo(), rc.next(x.typs.BytePtr)))
		pos = pos.WithNotStmt()
		addArg(x.rewriteSelectOrArg(pos, b, container, nil, m0, x.typs.Int, rc.next(x.typs.Int)))
		addArg(x.rewriteSelectOrArg(pos, b, container, nil, m0, x.typs.Int, rc.next(x.typs.Int)))
		a = makeOf(a, OpSliceMake, args)
		x.commonSelectors[sk] = a
		return a

	case types.TSTRING:
		addArg(x.rewriteSelectOrArg(pos, b, container, nil, m0, x.typs.BytePtr, rc.next(x.typs.BytePtr)))
		pos = pos.WithNotStmt()
		addArg(x.rewriteSelectOrArg(pos, b, container, nil, m0, x.typs.Int, rc.next(x.typs.Int)))
		a = makeOf(a, OpStringMake, args)
		x.commonSelectors[sk] = a
		return a

	case types.TINTER:
		addArg(x.rewriteSelectOrArg(pos, b, container, nil, m0, x.typs.Uintptr, rc.next(x.typs.Uintptr)))
		pos = pos.WithNotStmt()
		addArg(x.rewriteSelectOrArg(pos, b, container, nil, m0, x.typs.BytePtr, rc.next(x.typs.BytePtr)))
		a = makeOf(a, OpIMake, args)
		x.commonSelectors[sk] = a
		return a

	case types.TCOMPLEX64:
		addArg(x.rewriteSelectOrArg(pos, b, container, nil, m0, x.typs.Float32, rc.next(x.typs.Float32)))
		pos = pos.WithNotStmt()
		addArg(x.rewriteSelectOrArg(pos, b, container, nil, m0, x.typs.Float32, rc.next(x.typs.Float32)))
		a = makeOf(a, OpComplexMake, args)
		x.commonSelectors[sk] = a
		return a

	case types.TCOMPLEX128:
		addArg(x.rewriteSelectOrArg(pos, b, container, nil, m0, x.typs.Float64, rc.next(x.typs.Float64)))
		pos = pos.WithNotStmt()
		addArg(x.rewriteSelectOrArg(pos, b, container, nil, m0, x.typs.Float64, rc.next(x.typs.Float64)))
		a = makeOf(a, OpComplexMake, args)
		x.commonSelectors[sk] = a
		return a

	case types.TINT64:
		if at.Size() > x.regSize {
			addArg(x.rewriteSelectOrArg(pos, b, container, nil, m0, x.firstType, rc.next(x.firstType)))
			pos = pos.WithNotStmt()
			addArg(x.rewriteSelectOrArg(pos, b, container, nil, m0, x.secondType, rc.next(x.secondType)))
			if !x.f.Config.BigEndian {
				// Int64Make args are big, little
				args[0], args[1] = args[1], args[0]
			}
			a = makeOf(a, OpInt64Make, args)
			x.commonSelectors[sk] = a
			return a
		}
	case types.TUINT64:
		if at.Size() > x.regSize {
			addArg(x.rewriteSelectOrArg(pos, b, container, nil, m0, x.typs.UInt32, rc.next(x.typs.UInt32)))
			pos = pos.WithNotStmt()
			addArg(x.rewriteSelectOrArg(pos, b, container, nil, m0, x.typs.UInt32, rc.next(x.typs.UInt32)))
			if !x.f.Config.BigEndian {
				// Int64Make args are big, little
				args[0], args[1] = args[1], args[0]
			}
			a = makeOf(a, OpInt64Make, args)
			x.commonSelectors[sk] = a
			return a
		}
	}

	// An atomic type, either record the register or store it and update the memory.

	// Depending on the container Op, the leaves are either OpSelectN or OpArg{Int,Float}Reg

	if container.Op == OpArg {
		if rc.hasRegs() {
			op, i := rc.ArgOpAndRegisterFor()
			name := container.Aux.(*ir.Name)
			a = makeOf(a, op, nil)
			a.AuxInt = i
			a.Aux = &AuxNameOffset{name, rc.storeOffset}
		} else {
			key := selKey{container, rc.storeOffset, at.Size(), at}
			w := x.commonArgs[key]
			if w != nil && w.Uses != 0 {
				if a == nil {
					a = w
				} else {
					a.copyOf(w)
				}
			} else {
				if a == nil {
					aux := container.Aux
					auxInt := container.AuxInt + rc.storeOffset
					a = container.Block.NewValue0IA(container.Pos, OpArg, at, auxInt, aux)
				} else {
					// do nothing, the original should be okay.
				}
				x.commonArgs[key] = a
			}
		}
	} else if container.Op == OpSelectN {
		call := container.Args[0]
		aux := call.Aux.(*AuxCall)
		which := container.AuxInt

		if at == types.TypeMem {
			if a != m0 || a != x.memForCall[call.ID] {
				panic(fmt.Errorf("Memories %s, %s, and %s should all be equal after %s", a.LongString(), m0.LongString(), x.memForCall[call.ID], call.LongString()))
			}
		} else if rc.hasRegs() {
			firstReg := uint32(0)
			for i := 0; i < int(which); i++ {
				firstReg += uint32(len(aux.abiInfo.OutParam(i).Registers))
			}
			reg := int64(rc.nextSlice + Abi1RO(firstReg))
			a = makeOf(a, OpSelectN, []*Value{call})
			a.AuxInt = reg
		} else {
			off := x.offsetFrom(x.f.Entry, x.sp, rc.storeOffset+aux.OffsetOfResult(which), types.NewPtr(at))
			a = makeOf(a, OpLoad, []*Value{off, m0})
		}

	} else {
		panic(fmt.Errorf("Expected container OpArg or OpSelectN, saw %v instead", container.LongString()))
	}

	x.commonSelectors[sk] = a
	return a
}

// rewriteWideSelectToStores handles the case of a SelectN'd result from a function call that is too large for SSA,
// but is transferred in registers.  In this case the register cursor tracks both operands; the register sources and
// the memory destinations.
// This returns the memory flowing out of the last store
func (x *expandState) rewriteWideSelectToStores(pos src.XPos, b *Block, container, m0 *Value, at *types.Type, rc registerCursor) *Value {

	if at.Size() == 0 {
		return m0
	}

	switch at.Kind() {
	case types.TARRAY:
		et := at.Elem()
		for i := int64(0); i < at.NumElem(); i++ {
			m0 = x.rewriteWideSelectToStores(pos, b, container, m0, et, rc.next(et))
		}
		return m0

	case types.TSTRUCT:
		// Assume ssagen/ssa.go (in buildssa) spills large aggregates so they won't appear here.
		for i := 0; i < at.NumFields(); i++ {
			et := at.Field(i).Type
			m0 = x.rewriteWideSelectToStores(pos, b, container, m0, et, rc.next(et))
			pos = pos.WithNotStmt()
		}
		return m0

	case types.TSLICE:
		m0 = x.rewriteWideSelectToStores(pos, b, container, m0, at.Elem().PtrTo(), rc.next(x.typs.BytePtr))
		pos = pos.WithNotStmt()
		m0 = x.rewriteWideSelectToStores(pos, b, container, m0, x.typs.Int, rc.next(x.typs.Int))
		m0 = x.rewriteWideSelectToStores(pos, b, container, m0, x.typs.Int, rc.next(x.typs.Int))
		return m0

	case types.TSTRING:
		m0 = x.rewriteWideSelectToStores(pos, b, container, m0, x.typs.BytePtr, rc.next(x.typs.BytePtr))
		pos = pos.WithNotStmt()
		m0 = x.rewriteWideSelectToStores(pos, b, container, m0, x.typs.Int, rc.next(x.typs.Int))
		return m0

	case types.TINTER:
		m0 = x.rewriteWideSelectToStores(pos, b, container, m0, x.typs.Uintptr, rc.next(x.typs.Uintptr))
		pos = pos.WithNotStmt()
		m0 = x.rewriteWideSelectToStores(pos, b, container, m0, x.typs.BytePtr, rc.next(x.typs.BytePtr))
		return m0

	case types.TCOMPLEX64:
		m0 = x.rewriteWideSelectToStores(pos, b, container, m0, x.typs.Float32, rc.next(x.typs.Float32))
		pos = pos.WithNotStmt()
		m0 = x.rewriteWideSelectToStores(pos, b, container, m0, x.typs.Float32, rc.next(x.typs.Float32))
		return m0

	case types.TCOMPLEX128:
		m0 = x.rewriteWideSelectToStores(pos, b, container, m0, x.typs.Float64, rc.next(x.typs.Float64))
		pos = pos.WithNotStmt()
		m0 = x.rewriteWideSelectToStores(pos, b, container, m0, x.typs.Float64, rc.next(x.typs.Float64))
		return m0

	case types.TINT64:
		if at.Size() > x.regSize {
			m0 = x.rewriteWideSelectToStores(pos, b, container, m0, x.firstType, rc.next(x.firstType))
			pos = pos.WithNotStmt()
			m0 = x.rewriteWideSelectToStores(pos, b, container, m0, x.secondType, rc.next(x.secondType))
			return m0
		}
	case types.TUINT64:
		if at.Size() > x.regSize {
			m0 = x.rewriteWideSelectToStores(pos, b, container, m0, x.typs.UInt32, rc.next(x.typs.UInt32))
			pos = pos.WithNotStmt()
			m0 = x.rewriteWideSelectToStores(pos, b, container, m0, x.typs.UInt32, rc.next(x.typs.UInt32))
			return m0
		}
	}

	// TODO could change treatment of too-large OpArg, would deal with it here.
	if container.Op == OpSelectN {
		call := container.Args[0]
		aux := call.Aux.(*AuxCall)
		which := container.AuxInt

		if rc.hasRegs() {
			firstReg := uint32(0)
			for i := 0; i < int(which); i++ {
				firstReg += uint32(len(aux.abiInfo.OutParam(i).Registers))
			}
			reg := int64(rc.nextSlice + Abi1RO(firstReg))
			a := b.NewValue1I(pos, OpSelectN, at, reg, call)
			dst := x.offsetFrom(b, rc.storeDest, rc.storeOffset, types.NewPtr(at))
			m0 = b.NewValue3A(pos, OpStore, types.TypeMem, at, dst, a, m0)
		} else {
			panic(fmt.Errorf("Expected rc to have registers"))
		}
	} else {
		panic(fmt.Errorf("Expected container OpSelectN, saw %v instead", container.LongString()))
	}
	return m0
}

func isBlockMultiValueExit(b *Block) bool {
	return (b.Kind == BlockRet || b.Kind == BlockRetJmp) && b.Controls[0] != nil && b.Controls[0].Op == OpMakeResult
}

type Abi1RO uint8 // An offset within a parameter's slice of register indices, for abi1.

// A registerCursor tracks which register is used for an Arg or regValues, or a piece of such.
type registerCursor struct {
	storeDest   *Value // if there are no register targets, then this is the base of the store.
	storeOffset int64
	regs        []abi.RegIndex // the registers available for this Arg/result (which is all in registers or not at all)
	nextSlice   Abi1RO         // the next register/register-slice offset
	config      *abi.ABIConfig
	regValues   *[]*Value // values assigned to registers accumulate here
}

func (c *registerCursor) String() string {
	dest := "<none>"
	if c.storeDest != nil {
		dest = fmt.Sprintf("%s+%d", c.storeDest.String(), c.storeOffset)
	}
	regs := "<none>"
	if c.regValues != nil {
		regs = ""
		for i, x := range *c.regValues {
			if i > 0 {
				regs = regs + "; "
			}
			regs = regs + x.LongString()
		}
	}

	// not printing the config because that has not been useful
	return fmt.Sprintf("RCSR{storeDest=%v, regsLen=%d, nextSlice=%d, regValues=[%s]}", dest, len(c.regs), c.nextSlice, regs)
}

// next effectively post-increments the register cursor; the receiver is advanced,
// the (aligned) old value is returned.
func (c *registerCursor) next(t *types.Type) registerCursor {
	c.storeOffset = types.RoundUp(c.storeOffset, t.Alignment())
	rc := *c
	c.storeOffset = types.RoundUp(c.storeOffset+t.Size(), t.Alignment())
	if int(c.nextSlice) < len(c.regs) {
		w := c.config.NumParamRegs(t)
		c.nextSlice += Abi1RO(w)
	}
	return rc
}

// plus returns a register cursor offset from the original, without modifying the original.
func (c *registerCursor) plus(regWidth Abi1RO) registerCursor {
	rc := *c
	rc.nextSlice += regWidth
	return rc
}

// at returns the register cursor for component i of t, where the first
// component is numbered 0.
func (c *registerCursor) at(t *types.Type, i int) registerCursor {
	rc := *c
	if i == 0 || len(c.regs) == 0 {
		return rc
	}
	if t.IsArray() {
		w := c.config.NumParamRegs(t.Elem())
		rc.nextSlice += Abi1RO(i * w)
		return rc
	}
	if t.IsStruct() {
		for j := 0; j < i; j++ {
			rc.next(t.FieldType(j))
		}
		return rc
	}
	panic("Haven't implemented this case yet, do I need to?")
}

func (c *registerCursor) init(regs []abi.RegIndex, info *abi.ABIParamResultInfo, result *[]*Value, storeDest *Value, storeOffset int64) {
	c.regs = regs
	c.nextSlice = 0
	c.storeOffset = storeOffset
	c.storeDest = storeDest
	c.config = info.Config()
	c.regValues = result
}

func (c *registerCursor) addArg(v *Value) {
	*c.regValues = append(*c.regValues, v)
}

func (c *registerCursor) hasRegs() bool {
	return len(c.regs) > 0
}

func (c *registerCursor) ArgOpAndRegisterFor() (Op, int64) {
	r := c.regs[c.nextSlice]
	return ArgOpAndRegisterFor(r, c.config)
}

// ArgOpAndRegisterFor converts an abi register index into an ssa Op and corresponding
// arg register index.
func ArgOpAndRegisterFor(r abi.RegIndex, abiConfig *abi.ABIConfig) (Op, int64) {
	i := abiConfig.FloatIndexFor(r)
	if i >= 0 { // float PR
		return OpArgFloatReg, i
	}
	return OpArgIntReg, int64(r)
}

type selKey struct {
	from          *Value // what is selected from
	offsetOrIndex int64  // whatever is appropriate for the selector
	size          int64
	typ           *types.Type
}

type expandState struct {
	f       *Func
	debug   int // odd values log lost statement markers, so likely settings are 1 (stmts), 2 (expansion), and 3 (both)
	regSize int64
	sp      *Value
	typs    *Types

	firstOp    Op          // for 64-bit integers on 32-bit machines, first word in memory
	secondOp   Op          // for 64-bit integers on 32-bit machines, second word in memory
	firstType  *types.Type // first half type, for Int64
	secondType *types.Type // second half type, for Int64

	wideSelects     map[*Value]*Value // Selects that are not SSA-able, mapped to consuming stores.
	commonSelectors map[selKey]*Value // used to de-dupe selectors
	commonArgs      map[selKey]*Value // used to de-dupe OpArg/OpArgIntReg/OpArgFloatReg
	memForCall      map[ID]*Value     // For a call, need to know the unique selector that gets the mem.
	indentLevel     int               // Indentation for debugging recursion
}

// intPairTypes returns the pair of 32-bit int types needed to encode a 64-bit integer type on a target
// that has no 64-bit integer registers.
func (x *expandState) intPairTypes(et types.Kind) (tHi, tLo *types.Type) {
	tHi = x.typs.UInt32
	if et == types.TINT64 {
		tHi = x.typs.Int32
	}
	tLo = x.typs.UInt32
	return
}

// offsetFrom creates an offset from a pointer, simplifying chained offsets and offsets from SP
func (x *expandState) offsetFrom(b *Block, from *Value, offset int64, pt *types.Type) *Value {
	ft := from.Type
	if offset == 0 {
		if ft == pt {
			return from
		}
		// This captures common, (apparently) safe cases.  The unsafe cases involve ft == uintptr
		if (ft.IsPtr() || ft.IsUnsafePtr()) && pt.IsPtr() {
			return from
		}
	}
	// Simplify, canonicalize
	for from.Op == OpOffPtr {
		offset += from.AuxInt
		from = from.Args[0]
	}
	if from == x.sp {
		return x.f.ConstOffPtrSP(pt, offset, x.sp)
	}
	return b.NewValue1I(from.Pos.WithNotStmt(), OpOffPtr, pt, offset, from)
}

func (x *expandState) regWidth(t *types.Type) Abi1RO {
	return Abi1RO(x.f.ABI1.NumParamRegs(t))
}

// regOffset returns the register offset of the i'th element of type t
func (x *expandState) regOffset(t *types.Type, i int) Abi1RO {
	// TODO maybe cache this in a map if profiling recommends.
	if i == 0 {
		return 0
	}
	if t.IsArray() {
		return Abi1RO(i) * x.regWidth(t.Elem())
	}
	if t.IsStruct() {
		k := Abi1RO(0)
		for j := 0; j < i; j++ {
			k += x.regWidth(t.FieldType(j))
		}
		return k
	}
	panic("Haven't implemented this case yet, do I need to?")
}

// prAssignForArg returns the ABIParamAssignment for v, assumed to be an OpArg.
func (x *expandState) prAssignForArg(v *Value) *abi.ABIParamAssignment {
	if v.Op != OpArg {
		panic(fmt.Errorf("Wanted OpArg, instead saw %s", v.LongString()))
	}
	return ParamAssignmentForArgName(x.f, v.Aux.(*ir.Name))
}

// ParamAssignmentForArgName returns the ABIParamAssignment for f's arg with matching name.
func ParamAssignmentForArgName(f *Func, name *ir.Name) *abi.ABIParamAssignment {
	abiInfo := f.OwnAux.abiInfo
	ip := abiInfo.InParams()
	for i, a := range ip {
		if a.Name == name {
			return &ip[i]
		}
	}
	panic(fmt.Errorf("Did not match param %v in prInfo %+v", name, abiInfo.InParams()))
}

// indent increments (or decrements) the indentation.
func (x *expandState) indent(n int) {
	x.indentLevel += n
}

// Printf does an indented fmt.Printf on the format and args.
func (x *expandState) Printf(format string, a ...interface{}) (n int, err error) {
	if x.indentLevel > 0 {
		fmt.Printf("%[1]*s", x.indentLevel, "")
	}
	return fmt.Printf(format, a...)
}

func (x *expandState) invalidateRecursively(a *Value) {
	var s string
	if x.debug > 0 {
		plus := " "
		if a.Pos.IsStmt() == src.PosIsStmt {
			plus = " +"
		}
		s = a.String() + plus + a.Pos.LineNumber() + " " + a.LongString()
		if x.debug > 1 {
			x.Printf("...marking %v unused\n", s)
		}
	}
	lost := a.invalidateRecursively()
	if x.debug&1 != 0 && lost { // For odd values of x.debug, do this.
		x.Printf("Lost statement marker in %s on former %s\n", base.Ctxt.Pkgpath+"."+x.f.Name, s)
	}
}
