// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
	"strings"
)

// needwb reports whether we need write barrier for store op v.
// v must be Store/Move/Zero.
func needwb(v *Value) bool {
	t, ok := v.Aux.(*types.Type)
	if !ok {
		v.Fatalf("store aux is not a type: %s", v.LongString())
	}
	if !t.HasHeapPointer() {
		return false
	}
	if IsStackAddr(v.Args[0]) {
		return false // write on stack doesn't need write barrier
	}
	if v.Op == OpStore && IsGlobalAddr(v.Args[1]) && IsNewObject(v.Args[0], v.MemoryArg()) {
		// Storing pointers to non-heap locations into a fresh object doesn't need a write barrier.
		return false
	}
	if v.Op == OpMove && IsReadOnlyGlobalAddr(v.Args[1]) && IsNewObject(v.Args[0], v.MemoryArg()) {
		// Copying data from readonly memory into a fresh object doesn't need a write barrier.
		return false
	}
	return true
}

// writebarrier pass inserts write barriers for store ops (Store, Move, Zero)
// when necessary (the condition above). It rewrites store ops to branches
// and runtime calls, like
//
// if writeBarrier.enabled {
//   gcWriteBarrier(ptr, val)	// Not a regular Go call
// } else {
//   *ptr = val
// }
//
// A sequence of WB stores for many pointer fields of a single type will
// be emitted together, with a single branch.
func writebarrier(f *Func) {
	if !f.fe.UseWriteBarrier() {
		return
	}

	var sb, sp, wbaddr, const0 *Value
	var typedmemmove, typedmemclr, gcWriteBarrier *obj.LSym
	var stores, after []*Value
	var sset *sparseSet
	var storeNumber []int32

	for _, b := range f.Blocks { // range loop is safe since the blocks we added contain no stores to expand
		// first, identify all the stores that need to insert a write barrier.
		// mark them with WB ops temporarily. record presence of WB ops.
		nWBops := 0 // count of temporarily created WB ops remaining to be rewritten in the current block
		for _, v := range b.Values {
			switch v.Op {
			case OpStore, OpMove, OpZero:
				if needwb(v) {
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
			for _, v := range f.Entry.Values {
				if v.Op == OpSB {
					sb = v
				}
				if v.Op == OpSP {
					sp = v
				}
				if sb != nil && sp != nil {
					break
				}
			}
			if sb == nil {
				sb = f.Entry.NewValue0(initpos, OpSB, f.Config.Types.Uintptr)
			}
			if sp == nil {
				sp = f.Entry.NewValue0(initpos, OpSP, f.Config.Types.Uintptr)
			}
			wbsym := f.fe.Syslook("writeBarrier")
			wbaddr = f.Entry.NewValue1A(initpos, OpAddr, f.Config.Types.UInt32Ptr, wbsym, sb)
			gcWriteBarrier = f.fe.Syslook("gcWriteBarrier")
			typedmemmove = f.fe.Syslook("typedmemmove")
			typedmemclr = f.fe.Syslook("typedmemclr")
			const0 = f.ConstInt32(f.Config.Types.UInt32, 0)

			// allocate auxiliary data structures for computing store order
			sset = f.newSparseSet(f.NumValues())
			defer f.retSparseSet(sset)
			storeNumber = make([]int32, f.NumValues())
		}

		// order values in store order
		b.Values = storeOrder(b.Values, sset, storeNumber)

		firstSplit := true
	again:
		// find the start and end of the last contiguous WB store sequence.
		// a branch will be inserted there. values after it will be moved
		// to a new block.
		var last *Value
		var start, end int
		values := b.Values
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
			case OpVarDef, OpVarLive, OpVarKill:
				continue
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
		bThen := f.NewBlock(BlockPlain)
		bElse := f.NewBlock(BlockPlain)
		bEnd := f.NewBlock(b.Kind)
		bThen.Pos = pos
		bElse.Pos = pos
		bEnd.Pos = b.Pos
		b.Pos = pos

		// set up control flow for end block
		bEnd.SetControl(b.Control)
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
		b.AddEdgeTo(bElse)
		// TODO: For OpStoreWB and the buffered write barrier,
		// we could move the write out of the write barrier,
		// which would lead to fewer branches. We could do
		// something similar to OpZeroWB, since the runtime
		// could provide just the barrier half and then we
		// could unconditionally do an OpZero (which could
		// also generate better zeroing code). OpMoveWB is
		// trickier and would require changing how
		// cgoCheckMemmove works.
		bThen.AddEdgeTo(bEnd)
		bElse.AddEdgeTo(bEnd)

		// for each write barrier store, append write barrier version to bThen
		// and simple store version to bElse
		memThen := mem
		memElse := mem

		// If the source of a MoveWB is volatile (will be clobbered by a
		// function call), we need to copy it to a temporary location, as
		// marshaling the args of typedmemmove might clobber the value we're
		// trying to move.
		// Look for volatile source, copy it to temporary before we emit any
		// call.
		// It is unlikely to have more than one of them. Just do a linear
		// search instead of using a map.
		type volatileCopy struct {
			src *Value // address of original volatile value
			tmp *Value // address of temporary we've copied the volatile value into
		}
		var volatiles []volatileCopy
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
					tmp := f.fe.Auto(w.Pos, t)
					memThen = bThen.NewValue1A(w.Pos, OpVarDef, types.TypeMem, tmp, memThen)
					tmpaddr := bThen.NewValue2A(w.Pos, OpLocalAddr, t.PtrTo(), tmp, sp, memThen)
					siz := t.Size()
					memThen = bThen.NewValue3I(w.Pos, OpMove, types.TypeMem, siz, tmpaddr, val, memThen)
					memThen.Aux = t
					volatiles = append(volatiles, volatileCopy{val, tmpaddr})
				}
			}
		}

		for _, w := range stores {
			ptr := w.Args[0]
			pos := w.Pos

			var fn *obj.LSym
			var typ *obj.LSym
			var val *Value
			switch w.Op {
			case OpStoreWB:
				val = w.Args[1]
				nWBops--
			case OpMoveWB:
				fn = typedmemmove
				val = w.Args[1]
				typ = w.Aux.(*types.Type).Symbol()
				nWBops--
			case OpZeroWB:
				fn = typedmemclr
				typ = w.Aux.(*types.Type).Symbol()
				nWBops--
			case OpVarDef, OpVarLive, OpVarKill:
			}

			// then block: emit write barrier call
			switch w.Op {
			case OpStoreWB, OpMoveWB, OpZeroWB:
				if w.Op == OpStoreWB {
					memThen = bThen.NewValue3A(pos, OpWB, types.TypeMem, gcWriteBarrier, ptr, val, memThen)
				} else {
					srcval := val
					if w.Op == OpMoveWB && isVolatile(srcval) {
						for _, c := range volatiles {
							if srcval == c.src {
								srcval = c.tmp
								break
							}
						}
					}
					memThen = wbcall(pos, bThen, fn, typ, ptr, srcval, memThen, sp, sb)
				}
				// Note that we set up a writebarrier function call.
				f.fe.SetWBPos(pos)
			case OpVarDef, OpVarLive, OpVarKill:
				memThen = bThen.NewValue1A(pos, w.Op, types.TypeMem, w.Aux, memThen)
			}

			// else block: normal store
			switch w.Op {
			case OpStoreWB:
				memElse = bElse.NewValue3A(pos, OpStore, types.TypeMem, w.Aux, ptr, val, memElse)
			case OpMoveWB:
				memElse = bElse.NewValue3I(pos, OpMove, types.TypeMem, w.AuxInt, ptr, val, memElse)
				memElse.Aux = w.Aux
			case OpZeroWB:
				memElse = bElse.NewValue2I(pos, OpZero, types.TypeMem, w.AuxInt, ptr, memElse)
				memElse.Aux = w.Aux
			case OpVarDef, OpVarLive, OpVarKill:
				memElse = bElse.NewValue1A(pos, w.Op, types.TypeMem, w.Aux, memElse)
			}
		}

		// mark volatile temps dead
		for _, c := range volatiles {
			tmpNode := c.tmp.Aux
			memThen = bThen.NewValue1A(memThen.Pos, OpVarKill, types.TypeMem, tmpNode, memThen)
		}

		// merge memory
		// Splice memory Phi into the last memory of the original sequence,
		// which may be used in subsequent blocks. Other memories in the
		// sequence must be dead after this block since there can be only
		// one memory live.
		bEnd.Values = append(bEnd.Values, last)
		last.Block = bEnd
		last.reset(OpPhi)
		last.Type = types.TypeMem
		last.AddArg(memThen)
		last.AddArg(memElse)
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

		// Preemption is unsafe between loading the write
		// barrier-enabled flag and performing the write
		// because that would allow a GC phase transition,
		// which would invalidate the flag. Remember the
		// conditional block so liveness analysis can disable
		// safe-points. This is somewhat subtle because we're
		// splitting b bottom-up.
		if firstSplit {
			// Add b itself.
			b.Func.WBLoads = append(b.Func.WBLoads, b)
			firstSplit = false
		} else {
			// We've already split b, so we just pushed a
			// write barrier test into bEnd.
			b.Func.WBLoads = append(b.Func.WBLoads, bEnd)
		}

		// if we have more stores in this block, do this block again
		if nWBops > 0 {
			goto again
		}
	}
}

// wbcall emits write barrier runtime call in b, returns memory.
func wbcall(pos src.XPos, b *Block, fn, typ *obj.LSym, ptr, val, mem, sp, sb *Value) *Value {
	config := b.Func.Config

	// put arguments on stack
	off := config.ctxt.FixedFrameSize()

	if typ != nil { // for typedmemmove
		taddr := b.NewValue1A(pos, OpAddr, b.Func.Config.Types.Uintptr, typ, sb)
		off = round(off, taddr.Type.Alignment())
		arg := b.NewValue1I(pos, OpOffPtr, taddr.Type.PtrTo(), off, sp)
		mem = b.NewValue3A(pos, OpStore, types.TypeMem, ptr.Type, arg, taddr, mem)
		off += taddr.Type.Size()
	}

	off = round(off, ptr.Type.Alignment())
	arg := b.NewValue1I(pos, OpOffPtr, ptr.Type.PtrTo(), off, sp)
	mem = b.NewValue3A(pos, OpStore, types.TypeMem, ptr.Type, arg, ptr, mem)
	off += ptr.Type.Size()

	if val != nil {
		off = round(off, val.Type.Alignment())
		arg = b.NewValue1I(pos, OpOffPtr, val.Type.PtrTo(), off, sp)
		mem = b.NewValue3A(pos, OpStore, types.TypeMem, val.Type, arg, val, mem)
		off += val.Type.Size()
	}
	off = round(off, config.PtrSize)

	// issue call
	mem = b.NewValue1A(pos, OpStaticCall, types.TypeMem, fn, mem)
	mem.AuxInt = off - config.ctxt.FixedFrameSize()
	return mem
}

// round to a multiple of r, r is a power of 2
func round(o int64, r int64) int64 {
	return (o + r - 1) &^ (r - 1)
}

// IsStackAddr reports whether v is known to be an address of a stack slot.
func IsStackAddr(v *Value) bool {
	for v.Op == OpOffPtr || v.Op == OpAddPtr || v.Op == OpPtrIndex || v.Op == OpCopy {
		v = v.Args[0]
	}
	switch v.Op {
	case OpSP, OpLocalAddr:
		return true
	}
	return false
}

// IsGlobalAddr reports whether v is known to be an address of a global.
func IsGlobalAddr(v *Value) bool {
	return v.Op == OpAddr && v.Args[0].Op == OpSB
}

// IsReadOnlyGlobalAddr reports whether v is known to be an address of a read-only global.
func IsReadOnlyGlobalAddr(v *Value) bool {
	if !IsGlobalAddr(v) {
		return false
	}
	// See TODO in OpAddr case in IsSanitizerSafeAddr below.
	return strings.HasPrefix(v.Aux.(*obj.LSym).Name, `"".statictmp_`)
}

// IsNewObject reports whether v is a pointer to a freshly allocated & zeroed object at memory state mem.
// TODO: Be more precise. We really want "IsNilPointer" for the particular field in question.
// Right now, we can only detect a new object before any writes have been done to it.
// We could ignore non-pointer writes, writes to offsets which
// are known not to overlap the write in question, etc.
func IsNewObject(v *Value, mem *Value) bool {
	if v.Op != OpLoad {
		return false
	}
	if v.MemoryArg() != mem {
		return false
	}
	if mem.Op != OpStaticCall {
		return false
	}
	if !isSameSym(mem.Aux, "runtime.newobject") {
		return false
	}
	if v.Args[0].Op != OpOffPtr {
		return false
	}
	if v.Args[0].Args[0].Op != OpSP {
		return false
	}
	c := v.Block.Func.Config
	if v.Args[0].AuxInt != c.ctxt.FixedFrameSize()+c.RegSize { // offset of return value
		return false
	}
	return true
}

// IsSanitizerSafeAddr reports whether v is known to be an address
// that doesn't need instrumentation.
func IsSanitizerSafeAddr(v *Value) bool {
	for v.Op == OpOffPtr || v.Op == OpAddPtr || v.Op == OpPtrIndex || v.Op == OpCopy {
		v = v.Args[0]
	}
	switch v.Op {
	case OpSP, OpLocalAddr:
		// Stack addresses are always safe.
		return true
	case OpITab, OpStringPtr, OpGetClosurePtr:
		// Itabs, string data, and closure fields are
		// read-only once initialized.
		return true
	case OpAddr:
		sym := v.Aux.(*obj.LSym)
		// TODO(mdempsky): Find a cleaner way to
		// detect this. It would be nice if we could
		// test sym.Type==objabi.SRODATA, but we don't
		// initialize sym.Type until after function
		// compilation.
		if strings.HasPrefix(sym.Name, `"".statictmp_`) {
			return true
		}
	}
	return false
}

// isVolatile reports whether v is a pointer to argument region on stack which
// will be clobbered by a function call.
func isVolatile(v *Value) bool {
	for v.Op == OpOffPtr || v.Op == OpAddPtr || v.Op == OpPtrIndex || v.Op == OpCopy {
		v = v.Args[0]
	}
	return v.Op == OpSP
}
