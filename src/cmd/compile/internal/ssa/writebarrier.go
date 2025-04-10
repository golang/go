// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"fmt"
	"internal/buildcfg"
)

// A ZeroRegion records parts of an object which are known to be zero.
// A ZeroRegion only applies to a single memory state.
// Each bit in mask is set if the corresponding pointer-sized word of
// the base object is known to be zero.
// In other words, if mask & (1<<i) != 0, then [base+i*ptrSize, base+(i+1)*ptrSize)
// is known to be zero.
type ZeroRegion struct {
	base *Value
	mask uint64
}

// mightBeHeapPointer reports whether v might point to the heap.
// v must have pointer type.
func mightBeHeapPointer(v *Value) bool {
	if IsGlobalAddr(v) {
		return false
	}
	return true
}

// mightContainHeapPointer reports whether the data currently at addresses
// [ptr,ptr+size) might contain heap pointers. "currently" means at memory state mem.
// zeroes contains ZeroRegion data to help make that decision (see computeZeroMap).
func mightContainHeapPointer(ptr *Value, size int64, mem *Value, zeroes map[ID]ZeroRegion) bool {
	if IsReadOnlyGlobalAddr(ptr) {
		// The read-only globals section cannot contain any heap pointers.
		return false
	}

	// See if we can prove that the queried memory is all zero.

	// Find base pointer and offset. Hopefully, the base is the result of a new(T).
	var off int64
	for ptr.Op == OpOffPtr {
		off += ptr.AuxInt
		ptr = ptr.Args[0]
	}

	ptrSize := ptr.Block.Func.Config.PtrSize
	if off%ptrSize != 0 {
		return true // see issue 61187
	}
	if size%ptrSize != 0 {
		ptr.Fatalf("unaligned pointer write")
	}
	if off < 0 || off+size > 64*ptrSize {
		// memory range goes off end of tracked offsets
		return true
	}
	z := zeroes[mem.ID]
	if ptr != z.base {
		// This isn't the object we know about at this memory state.
		return true
	}
	// Mask of bits we're asking about
	m := (uint64(1)<<(size/ptrSize) - 1) << (off / ptrSize)

	if z.mask&m == m {
		// All locations are known to be zero, so no heap pointers.
		return false
	}
	return true
}

// needwb reports whether we need write barrier for store op v.
// v must be Store/Move/Zero.
// zeroes provides known zero information (keyed by ID of memory-type values).
func needwb(v *Value, zeroes map[ID]ZeroRegion) bool {
	t, ok := v.Aux.(*types.Type)
	if !ok {
		v.Fatalf("store aux is not a type: %s", v.LongString())
	}
	if !t.HasPointers() {
		return false
	}
	dst := v.Args[0]
	if IsStackAddr(dst) {
		return false // writes into the stack don't need write barrier
	}
	// If we're writing to a place that might have heap pointers, we need
	// the write barrier.
	if mightContainHeapPointer(dst, t.Size(), v.MemoryArg(), zeroes) {
		return true
	}
	// Lastly, check if the values we're writing might be heap pointers.
	// If they aren't, we don't need a write barrier.
	switch v.Op {
	case OpStore:
		if !mightBeHeapPointer(v.Args[1]) {
			return false
		}
	case OpZero:
		return false // nil is not a heap pointer
	case OpMove:
		if !mightContainHeapPointer(v.Args[1], t.Size(), v.Args[2], zeroes) {
			return false
		}
	default:
		v.Fatalf("store op unknown: %s", v.LongString())
	}
	return true
}

// needWBsrc reports whether GC needs to see v when it is the source of a store.
func needWBsrc(v *Value) bool {
	return !IsGlobalAddr(v)
}

// needWBdst reports whether GC needs to see what used to be in *ptr when ptr is
// the target of a pointer store.
func needWBdst(ptr, mem *Value, zeroes map[ID]ZeroRegion) bool {
	// Detect storing to zeroed memory.
	var off int64
	for ptr.Op == OpOffPtr {
		off += ptr.AuxInt
		ptr = ptr.Args[0]
	}
	ptrSize := ptr.Block.Func.Config.PtrSize
	if off%ptrSize != 0 {
		return true // see issue 61187
	}
	if off < 0 || off >= 64*ptrSize {
		// write goes off end of tracked offsets
		return true
	}
	z := zeroes[mem.ID]
	if ptr != z.base {
		return true
	}
	// If destination is known to be zeroed, we don't need the write barrier
	// to record the old value in *ptr.
	return z.mask>>uint(off/ptrSize)&1 == 0
}

// writebarrier pass inserts write barriers for store ops (Store, Move, Zero)
// when necessary (the condition above). It rewrites store ops to branches
// and runtime calls, like
//
//	if writeBarrier.enabled {
//		buf := gcWriteBarrier2()	// Not a regular Go call
//		buf[0] = val
//		buf[1] = *ptr
//	}
//	*ptr = val
//
// A sequence of WB stores for many pointer fields of a single type will
// be emitted together, with a single branch.
func writebarrier(f *Func) {
	if !f.fe.UseWriteBarrier() {
		return
	}

	// Number of write buffer entries we can request at once.
	// Must match runtime/mwbbuf.go:wbMaxEntriesPerCall.
	// It must also match the number of instances of runtime.gcWriteBarrier{X}.
	const maxEntries = 8

	var sb, sp, wbaddr, const0 *Value
	var cgoCheckPtrWrite, cgoCheckMemmove *obj.LSym
	var wbZero, wbMove *obj.LSym
	var stores, after []*Value
	var sset, sset2 *sparseSet
	var storeNumber []int32

	// Compute map from a value to the SelectN [1] value that uses it.
	select1 := f.Cache.allocValueSlice(f.NumValues())
	defer func() { f.Cache.freeValueSlice(select1) }()
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op != OpSelectN {
				continue
			}
			if v.AuxInt != 1 {
				continue
			}
			select1[v.Args[0].ID] = v
		}
	}

	zeroes := f.computeZeroMap(select1)
	for _, b := range f.Blocks { // range loop is safe since the blocks we added contain no stores to expand
		// first, identify all the stores that need to insert a write barrier.
		// mark them with WB ops temporarily. record presence of WB ops.
		nWBops := 0 // count of temporarily created WB ops remaining to be rewritten in the current block
		for _, v := range b.Values {
			switch v.Op {
			case OpStore, OpMove, OpZero:
				if needwb(v, zeroes) {
					switch v.Op {
					case OpStore:
						v.Op = OpStoreWB
					case OpMove:
						v.Op = OpMoveWB
					case OpZero:
						v.Op = OpZeroWB
					}
					nWBops++
				}
			}
		}
		if nWBops == 0 {
			continue
		}

		if wbaddr == nil {
			// lazily initialize global values for write barrier test and calls
			// find SB and SP values in entry block
			initpos := f.Entry.Pos
			sp, sb = f.spSb()
			wbsym := f.fe.Syslook("writeBarrier")
			wbaddr = f.Entry.NewValue1A(initpos, OpAddr, f.Config.Types.UInt32Ptr, wbsym, sb)
			wbZero = f.fe.Syslook("wbZero")
			wbMove = f.fe.Syslook("wbMove")
			if buildcfg.Experiment.CgoCheck2 {
				cgoCheckPtrWrite = f.fe.Syslook("cgoCheckPtrWrite")
				cgoCheckMemmove = f.fe.Syslook("cgoCheckMemmove")
			}
			const0 = f.ConstInt32(f.Config.Types.UInt32, 0)

			// allocate auxiliary data structures for computing store order
			sset = f.newSparseSet(f.NumValues())
			defer f.retSparseSet(sset)
			sset2 = f.newSparseSet(f.NumValues())
			defer f.retSparseSet(sset2)
			storeNumber = f.Cache.allocInt32Slice(f.NumValues())
			defer f.Cache.freeInt32Slice(storeNumber)
		}

		// order values in store order
		b.Values = storeOrder(b.Values, sset, storeNumber)
	again:
		// find the start and end of the last contiguous WB store sequence.
		// a branch will be inserted there. values after it will be moved
		// to a new block.
		var last *Value
		var start, end int
		var nonPtrStores int
		values := b.Values
		hasMove := false
	FindSeq:
		for i := len(values) - 1; i >= 0; i-- {
			w := values[i]
			switch w.Op {
			case OpStoreWB, OpMoveWB, OpZeroWB:
				start = i
				if last == nil {
					last = w
					end = i + 1
				}
				nonPtrStores = 0
				if w.Op == OpMoveWB {
					hasMove = true
				}
			case OpVarDef, OpVarLive:
				continue
			case OpStore:
				if last == nil {
					continue
				}
				nonPtrStores++
				if nonPtrStores > 2 {
					break FindSeq
				}
				if hasMove {
					// We need to ensure that this store happens
					// before we issue a wbMove, as the wbMove might
					// use the result of this store as its source.
					// Even though this store is not write-barrier
					// eligible, it might nevertheless be the store
					// of a pointer to the stack, which is then the
					// source of the move.
					// See issue 71228.
					break FindSeq
				}
			default:
				if last == nil {
					continue
				}
				break FindSeq
			}
		}
		stores = append(stores[:0], b.Values[start:end]...) // copy to avoid aliasing
		after = append(after[:0], b.Values[end:]...)
		b.Values = b.Values[:start]

		// find the memory before the WB stores
		mem := stores[0].MemoryArg()
		pos := stores[0].Pos

		// If the source of a MoveWB is volatile (will be clobbered by a
		// function call), we need to copy it to a temporary location, as
		// marshaling the args of wbMove might clobber the value we're
		// trying to move.
		// Look for volatile source, copy it to temporary before we check
		// the write barrier flag.
		// It is unlikely to have more than one of them. Just do a linear
		// search instead of using a map.
		// See issue 15854.
		type volatileCopy struct {
			src *Value // address of original volatile value
			tmp *Value // address of temporary we've copied the volatile value into
		}
		var volatiles []volatileCopy

		if !(f.ABIDefault == f.ABI1 && len(f.Config.intParamRegs) >= 3) {
			// We don't need to do this if the calls we're going to do take
			// all their arguments in registers.
			// 3 is the magic number because it covers wbZero, wbMove, cgoCheckMemmove.
		copyLoop:
			for _, w := range stores {
				if w.Op == OpMoveWB {
					val := w.Args[1]
					if isVolatile(val) {
						for _, c := range volatiles {
							if val == c.src {
								continue copyLoop // already copied
							}
						}

						t := val.Type.Elem()
						tmp := f.NewLocal(w.Pos, t)
						mem = b.NewValue1A(w.Pos, OpVarDef, types.TypeMem, tmp, mem)
						tmpaddr := b.NewValue2A(w.Pos, OpLocalAddr, t.PtrTo(), tmp, sp, mem)
						siz := t.Size()
						mem = b.NewValue3I(w.Pos, OpMove, types.TypeMem, siz, tmpaddr, val, mem)
						mem.Aux = t
						volatiles = append(volatiles, volatileCopy{val, tmpaddr})
					}
				}
			}
		}

		// Build branch point.
		bThen := f.NewBlock(BlockPlain)
		bEnd := f.NewBlock(b.Kind)
		bThen.Pos = pos
		bEnd.Pos = b.Pos
		b.Pos = pos

		// Set up control flow for end block.
		bEnd.CopyControls(b)
		bEnd.Likely = b.Likely
		for _, e := range b.Succs {
			bEnd.Succs = append(bEnd.Succs, e)
			e.b.Preds[e.i].b = bEnd
		}

		// set up control flow for write barrier test
		// load word, test word, avoiding partial register write from load byte.
		cfgtypes := &f.Config.Types
		flag := b.NewValue2(pos, OpLoad, cfgtypes.UInt32, wbaddr, mem)
		flag = b.NewValue2(pos, OpNeq32, cfgtypes.Bool, flag, const0)
		b.Kind = BlockIf
		b.SetControl(flag)
		b.Likely = BranchUnlikely
		b.Succs = b.Succs[:0]
		b.AddEdgeTo(bThen)
		b.AddEdgeTo(bEnd)
		bThen.AddEdgeTo(bEnd)

		// For each write barrier store, append write barrier code to bThen.
		memThen := mem

		// Note: we can issue the write barrier code in any order. In particular,
		// it doesn't matter if they are in a different order *even if* they end
		// up referring to overlapping memory regions. For instance if an OpStore
		// stores to a location that is later read by an OpMove. In all cases
		// any pointers we must get into the write barrier buffer still make it,
		// possibly in a different order and possibly a different (but definitely
		// more than 0) number of times.
		// In light of that, we process all the OpStoreWBs first. This minimizes
		// the amount of spill/restore code we need around the Zero/Move calls.

		// srcs contains the value IDs of pointer values we've put in the write barrier buffer.
		srcs := sset
		srcs.clear()
		// dsts contains the value IDs of locations which we've read a pointer out of
		// and put the result in the write barrier buffer.
		dsts := sset2
		dsts.clear()

		// Buffer up entries that we need to put in the write barrier buffer.
		type write struct {
			ptr *Value   // value to put in write barrier buffer
			pos src.XPos // location to use for the write
		}
		var writeStore [maxEntries]write
		writes := writeStore[:0]

		flush := func() {
			if len(writes) == 0 {
				return
			}
			// Issue a call to get a write barrier buffer.
			t := types.NewTuple(types.Types[types.TUINTPTR].PtrTo(), types.TypeMem)
			call := bThen.NewValue1I(pos, OpWB, t, int64(len(writes)), memThen)
			curPtr := bThen.NewValue1(pos, OpSelect0, types.Types[types.TUINTPTR].PtrTo(), call)
			memThen = bThen.NewValue1(pos, OpSelect1, types.TypeMem, call)
			// Write each pending pointer to a slot in the buffer.
			for i, write := range writes {
				wbuf := bThen.NewValue1I(write.pos, OpOffPtr, types.Types[types.TUINTPTR].PtrTo(), int64(i)*f.Config.PtrSize, curPtr)
				memThen = bThen.NewValue3A(write.pos, OpStore, types.TypeMem, types.Types[types.TUINTPTR], wbuf, write.ptr, memThen)
			}
			writes = writes[:0]
		}
		addEntry := func(pos src.XPos, ptr *Value) {
			writes = append(writes, write{ptr: ptr, pos: pos})
			if len(writes) == maxEntries {
				flush()
			}
		}

		// Find all the pointers we need to write to the buffer.
		for _, w := range stores {
			if w.Op != OpStoreWB {
				continue
			}
			pos := w.Pos
			ptr := w.Args[0]
			val := w.Args[1]
			if !srcs.contains(val.ID) && needWBsrc(val) {
				srcs.add(val.ID)
				addEntry(pos, val)
			}
			if !dsts.contains(ptr.ID) && needWBdst(ptr, w.Args[2], zeroes) {
				dsts.add(ptr.ID)
				// Load old value from store target.
				// Note: This turns bad pointer writes into bad
				// pointer reads, which could be confusing. We could avoid
				// reading from obviously bad pointers, which would
				// take care of the vast majority of these. We could
				// patch this up in the signal handler, or use XCHG to
				// combine the read and the write.
				oldVal := bThen.NewValue2(pos, OpLoad, types.Types[types.TUINTPTR], ptr, memThen)
				// Save old value to write buffer.
				addEntry(pos, oldVal)
			}
			f.fe.Func().SetWBPos(pos)
			nWBops--
		}
		flush()

		// Now do the rare cases, Zeros and Moves.
		for _, w := range stores {
			pos := w.Pos
			switch w.Op {
			case OpZeroWB:
				dst := w.Args[0]
				typ := reflectdata.TypeLinksym(w.Aux.(*types.Type))
				// zeroWB(&typ, dst)
				taddr := b.NewValue1A(pos, OpAddr, b.Func.Config.Types.Uintptr, typ, sb)
				memThen = wbcall(pos, bThen, wbZero, sp, memThen, taddr, dst)
				f.fe.Func().SetWBPos(pos)
				nWBops--
			case OpMoveWB:
				dst := w.Args[0]
				src := w.Args[1]
				if isVolatile(src) {
					for _, c := range volatiles {
						if src == c.src {
							src = c.tmp
							break
						}
					}
				}
				typ := reflectdata.TypeLinksym(w.Aux.(*types.Type))
				// moveWB(&typ, dst, src)
				taddr := b.NewValue1A(pos, OpAddr, b.Func.Config.Types.Uintptr, typ, sb)
				memThen = wbcall(pos, bThen, wbMove, sp, memThen, taddr, dst, src)
				f.fe.Func().SetWBPos(pos)
				nWBops--
			}
		}

		// merge memory
		mem = bEnd.NewValue2(pos, OpPhi, types.TypeMem, mem, memThen)

		// Do raw stores after merge point.
		for _, w := range stores {
			pos := w.Pos
			switch w.Op {
			case OpStoreWB:
				ptr := w.Args[0]
				val := w.Args[1]
				if buildcfg.Experiment.CgoCheck2 {
					// Issue cgo checking code.
					mem = wbcall(pos, bEnd, cgoCheckPtrWrite, sp, mem, ptr, val)
				}
				mem = bEnd.NewValue3A(pos, OpStore, types.TypeMem, w.Aux, ptr, val, mem)
			case OpZeroWB:
				dst := w.Args[0]
				mem = bEnd.NewValue2I(pos, OpZero, types.TypeMem, w.AuxInt, dst, mem)
				mem.Aux = w.Aux
			case OpMoveWB:
				dst := w.Args[0]
				src := w.Args[1]
				if isVolatile(src) {
					for _, c := range volatiles {
						if src == c.src {
							src = c.tmp
							break
						}
					}
				}
				if buildcfg.Experiment.CgoCheck2 {
					// Issue cgo checking code.
					typ := reflectdata.TypeLinksym(w.Aux.(*types.Type))
					taddr := b.NewValue1A(pos, OpAddr, b.Func.Config.Types.Uintptr, typ, sb)
					mem = wbcall(pos, bEnd, cgoCheckMemmove, sp, mem, taddr, dst, src)
				}
				mem = bEnd.NewValue3I(pos, OpMove, types.TypeMem, w.AuxInt, dst, src, mem)
				mem.Aux = w.Aux
			case OpVarDef, OpVarLive:
				mem = bEnd.NewValue1A(pos, w.Op, types.TypeMem, w.Aux, mem)
			case OpStore:
				ptr := w.Args[0]
				val := w.Args[1]
				mem = bEnd.NewValue3A(pos, OpStore, types.TypeMem, w.Aux, ptr, val, mem)
			}
		}

		// The last store becomes the WBend marker. This marker is used by the liveness
		// pass to determine what parts of the code are preemption-unsafe.
		// All subsequent memory operations use this memory, so we have to sacrifice the
		// previous last memory op to become this new value.
		bEnd.Values = append(bEnd.Values, last)
		last.Block = bEnd
		last.reset(OpWBend)
		last.Pos = last.Pos.WithNotStmt()
		last.Type = types.TypeMem
		last.AddArg(mem)

		// Free all the old stores, except last which became the WBend marker.
		for _, w := range stores {
			if w != last {
				w.resetArgs()
			}
		}
		for _, w := range stores {
			if w != last {
				f.freeValue(w)
			}
		}

		// put values after the store sequence into the end block
		bEnd.Values = append(bEnd.Values, after...)
		for _, w := range after {
			w.Block = bEnd
		}

		// if we have more stores in this block, do this block again
		if nWBops > 0 {
			goto again
		}
	}
}

// computeZeroMap returns a map from an ID of a memory value to
// a set of locations that are known to be zeroed at that memory value.
func (f *Func) computeZeroMap(select1 []*Value) map[ID]ZeroRegion {

	ptrSize := f.Config.PtrSize
	// Keep track of which parts of memory are known to be zero.
	// This helps with removing write barriers for various initialization patterns.
	// This analysis is conservative. We only keep track, for each memory state, of
	// which of the first 64 words of a single object are known to be zero.
	zeroes := map[ID]ZeroRegion{}
	// Find new objects.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if mem, ok := IsNewObject(v, select1); ok {
				// While compiling package runtime itself, we might see user
				// calls to newobject, which will have result type
				// unsafe.Pointer instead. We can't easily infer how large the
				// allocated memory is, so just skip it.
				if types.LocalPkg.Path == "runtime" && v.Type.IsUnsafePtr() {
					continue
				}

				nptr := min(64, v.Type.Elem().Size()/ptrSize)
				zeroes[mem.ID] = ZeroRegion{base: v, mask: 1<<uint(nptr) - 1}
			}
		}
	}
	// Find stores to those new objects.
	for {
		changed := false
		for _, b := range f.Blocks {
			// Note: iterating forwards helps convergence, as values are
			// typically (but not always!) in store order.
			for _, v := range b.Values {
				if v.Op != OpStore {
					continue
				}
				z, ok := zeroes[v.MemoryArg().ID]
				if !ok {
					continue
				}
				ptr := v.Args[0]
				var off int64
				size := v.Aux.(*types.Type).Size()
				for ptr.Op == OpOffPtr {
					off += ptr.AuxInt
					ptr = ptr.Args[0]
				}
				if ptr != z.base {
					// Different base object - we don't know anything.
					// We could even be writing to the base object we know
					// about, but through an aliased but offset pointer.
					// So we have to throw all the zero information we have away.
					continue
				}
				// Round to cover any partially written pointer slots.
				// Pointer writes should never be unaligned like this, but non-pointer
				// writes to pointer-containing types will do this.
				if d := off % ptrSize; d != 0 {
					off -= d
					size += d
				}
				if d := size % ptrSize; d != 0 {
					size += ptrSize - d
				}
				// Clip to the 64 words that we track.
				min := off
				max := off + size
				if min < 0 {
					min = 0
				}
				if max > 64*ptrSize {
					max = 64 * ptrSize
				}
				// Clear bits for parts that we are writing (and hence
				// will no longer necessarily be zero).
				for i := min; i < max; i += ptrSize {
					bit := i / ptrSize
					z.mask &^= 1 << uint(bit)
				}
				if z.mask == 0 {
					// No more known zeros - don't bother keeping.
					continue
				}
				// Save updated known zero contents for new store.
				if zeroes[v.ID] != z {
					zeroes[v.ID] = z
					changed = true
				}
			}
		}
		if !changed {
			break
		}
	}
	if f.pass.debug > 0 {
		fmt.Printf("func %s\n", f.Name)
		for mem, z := range zeroes {
			fmt.Printf("  memory=v%d ptr=%v zeromask=%b\n", mem, z.base, z.mask)
		}
	}
	return zeroes
}

// wbcall emits write barrier runtime call in b, returns memory.
func wbcall(pos src.XPos, b *Block, fn *obj.LSym, sp, mem *Value, args ...*Value) *Value {
	config := b.Func.Config
	typ := config.Types.Uintptr // type of all argument values
	nargs := len(args)

	// TODO (register args) this is a bit of a hack.
	inRegs := b.Func.ABIDefault == b.Func.ABI1 && len(config.intParamRegs) >= 3

	if !inRegs {
		// Store arguments to the appropriate stack slot.
		off := config.ctxt.Arch.FixedFrameSize
		for _, arg := range args {
			stkaddr := b.NewValue1I(pos, OpOffPtr, typ.PtrTo(), off, sp)
			mem = b.NewValue3A(pos, OpStore, types.TypeMem, typ, stkaddr, arg, mem)
			off += typ.Size()
		}
		args = args[:0]
	}

	args = append(args, mem)

	// issue call
	argTypes := make([]*types.Type, nargs, 3) // at most 3 args; allows stack allocation
	for i := 0; i < nargs; i++ {
		argTypes[i] = typ
	}
	call := b.NewValue0A(pos, OpStaticCall, types.TypeResultMem, StaticAuxCall(fn, b.Func.ABIDefault.ABIAnalyzeTypes(argTypes, nil)))
	call.AddArgs(args...)
	call.AuxInt = int64(nargs) * typ.Size()
	return b.NewValue1I(pos, OpSelectN, types.TypeMem, 0, call)
}

// IsStackAddr reports whether v is known to be an address of a stack slot.
func IsStackAddr(v *Value) bool {
	for v.Op == OpOffPtr || v.Op == OpAddPtr || v.Op == OpPtrIndex || v.Op == OpCopy {
		v = v.Args[0]
	}
	switch v.Op {
	case OpSP, OpLocalAddr, OpSelectNAddr, OpGetCallerSP:
		return true
	}
	return false
}

// IsGlobalAddr reports whether v is known to be an address of a global (or nil).
func IsGlobalAddr(v *Value) bool {
	for v.Op == OpOffPtr || v.Op == OpAddPtr || v.Op == OpPtrIndex || v.Op == OpCopy {
		v = v.Args[0]
	}
	if v.Op == OpAddr && v.Args[0].Op == OpSB {
		return true // address of a global
	}
	if v.Op == OpConstNil {
		return true
	}
	if v.Op == OpLoad && IsReadOnlyGlobalAddr(v.Args[0]) {
		return true // loading from a read-only global - the resulting address can't be a heap address.
	}
	return false
}

// IsReadOnlyGlobalAddr reports whether v is known to be an address of a read-only global.
func IsReadOnlyGlobalAddr(v *Value) bool {
	if v.Op == OpConstNil {
		// Nil pointers are read only. See issue 33438.
		return true
	}
	if v.Op == OpAddr && v.Aux != nil && v.Aux.(*obj.LSym).Type == objabi.SRODATA {
		return true
	}
	return false
}

// IsNewObject reports whether v is a pointer to a freshly allocated & zeroed object,
// if so, also returns the memory state mem at which v is zero.
func IsNewObject(v *Value, select1 []*Value) (mem *Value, ok bool) {
	f := v.Block.Func
	c := f.Config
	if f.ABIDefault == f.ABI1 && len(c.intParamRegs) >= 1 {
		if v.Op != OpSelectN || v.AuxInt != 0 {
			return nil, false
		}
		mem = select1[v.Args[0].ID]
		if mem == nil {
			return nil, false
		}
	} else {
		if v.Op != OpLoad {
			return nil, false
		}
		mem = v.MemoryArg()
		if mem.Op != OpSelectN {
			return nil, false
		}
		if mem.Type != types.TypeMem {
			return nil, false
		} // assume it is the right selection if true
	}
	call := mem.Args[0]
	if call.Op != OpStaticCall {
		return nil, false
	}
	if !isSameCall(call.Aux, "runtime.newobject") {
		return nil, false
	}
	if f.ABIDefault == f.ABI1 && len(c.intParamRegs) >= 1 {
		if v.Args[0] == call {
			return mem, true
		}
		return nil, false
	}
	if v.Args[0].Op != OpOffPtr {
		return nil, false
	}
	if v.Args[0].Args[0].Op != OpSP {
		return nil, false
	}
	if v.Args[0].AuxInt != c.ctxt.Arch.FixedFrameSize+c.RegSize { // offset of return value
		return nil, false
	}
	return mem, true
}

// IsSanitizerSafeAddr reports whether v is known to be an address
// that doesn't need instrumentation.
func IsSanitizerSafeAddr(v *Value) bool {
	for v.Op == OpOffPtr || v.Op == OpAddPtr || v.Op == OpPtrIndex || v.Op == OpCopy {
		v = v.Args[0]
	}
	switch v.Op {
	case OpSP, OpLocalAddr, OpSelectNAddr:
		// Stack addresses are always safe.
		return true
	case OpITab, OpStringPtr, OpGetClosurePtr:
		// Itabs, string data, and closure fields are
		// read-only once initialized.
		return true
	case OpAddr:
		vt := v.Aux.(*obj.LSym).Type
		return vt == objabi.SRODATA || vt == objabi.SLIBFUZZER_8BIT_COUNTER || vt == objabi.SCOVERAGE_COUNTER || vt == objabi.SCOVERAGE_AUXVAR
	}
	return false
}

// isVolatile reports whether v is a pointer to argument region on stack which
// will be clobbered by a function call.
func isVolatile(v *Value) bool {
	for v.Op == OpOffPtr || v.Op == OpAddPtr || v.Op == OpPtrIndex || v.Op == OpCopy || v.Op == OpSelectNAddr {
		v = v.Args[0]
	}
	return v.Op == OpSP
}
