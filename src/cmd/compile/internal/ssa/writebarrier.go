// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
)

// needwb returns whether we need write barrier for store op v.
// v must be Store/Move/Zero.
func needwb(v *Value) bool {
	t, ok := v.Aux.(*types.Type)
	if !ok {
		v.Fatalf("store aux is not a type: %s", v.LongString())
	}
	if !t.HasPointer() {
		return false
	}
	if IsStackAddr(v.Args[0]) {
		return false // write on stack doesn't need write barrier
	}
	return true
}

// writebarrier pass inserts write barriers for store ops (Store, Move, Zero)
// when necessary (the condition above). It rewrites store ops to branches
// and runtime calls, like
//
// if writeBarrier.enabled {
//   writebarrierptr(ptr, val)
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
	var writebarrierptr, typedmemmove, typedmemclr *obj.LSym
	var stores, after []*Value
	var sset *sparseSet
	var storeNumber []int32

	for _, b := range f.Blocks { // range loop is safe since the blocks we added contain no stores to expand
		// first, identify all the stores that need to insert a write barrier.
		// mark them with WB ops temporarily. record presence of WB ops.
		hasStore := false
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
					hasStore = true
				}
			}
		}
		if !hasStore {
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
			wbsym := &ExternSymbol{Sym: f.fe.Syslook("writeBarrier")}
			wbaddr = f.Entry.NewValue1A(initpos, OpAddr, f.Config.Types.UInt32Ptr, wbsym, sb)
			writebarrierptr = f.fe.Syslook("writebarrierptr")
			typedmemmove = f.fe.Syslook("typedmemmove")
			typedmemclr = f.fe.Syslook("typedmemclr")
			const0 = f.ConstInt32(initpos, f.Config.Types.UInt32, 0)

			// allocate auxiliary data structures for computing store order
			sset = f.newSparseSet(f.NumValues())
			defer f.retSparseSet(sset)
			storeNumber = make([]int32, f.NumValues())
		}

		// order values in store order
		b.Values = storeOrder(b.Values, sset, storeNumber)

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
		bThen.AddEdgeTo(bEnd)
		bElse.AddEdgeTo(bEnd)

		// for each write barrier store, append write barrier version to bThen
		// and simple store version to bElse
		memThen := mem
		memElse := mem
		for _, w := range stores {
			ptr := w.Args[0]
			pos := w.Pos

			var fn *obj.LSym
			var typ *ExternSymbol
			var val *Value
			switch w.Op {
			case OpStoreWB:
				fn = writebarrierptr
				val = w.Args[1]
			case OpMoveWB:
				fn = typedmemmove
				val = w.Args[1]
				typ = &ExternSymbol{Sym: w.Aux.(*types.Type).Symbol()}
			case OpZeroWB:
				fn = typedmemclr
				typ = &ExternSymbol{Sym: w.Aux.(*types.Type).Symbol()}
			case OpVarDef, OpVarLive, OpVarKill:
			}

			// then block: emit write barrier call
			switch w.Op {
			case OpStoreWB, OpMoveWB, OpZeroWB:
				volatile := w.Op == OpMoveWB && isVolatile(val)
				memThen = wbcall(pos, bThen, fn, typ, ptr, val, memThen, sp, sb, volatile)
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

			if fn != nil {
				// Note that we set up a writebarrier function call.
				if !f.WBPos.IsKnown() {
					f.WBPos = pos
				}
				if f.fe.Debug_wb() {
					f.Warnl(pos, "write barrier")
				}
			}
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

		// if we have more stores in this block, do this block again
		// check from end to beginning, to avoid quadratic behavior; issue 13554
		// TODO: track the final value to avoid any looping here at all
		for i := len(b.Values) - 1; i >= 0; i-- {
			switch b.Values[i].Op {
			case OpStoreWB, OpMoveWB, OpZeroWB:
				goto again
			}
		}
	}
}

// wbcall emits write barrier runtime call in b, returns memory.
// if valIsVolatile, it moves val into temp space before making the call.
func wbcall(pos src.XPos, b *Block, fn *obj.LSym, typ *ExternSymbol, ptr, val, mem, sp, sb *Value, valIsVolatile bool) *Value {
	config := b.Func.Config

	var tmp GCNode
	if valIsVolatile {
		// Copy to temp location if the source is volatile (will be clobbered by
		// a function call). Marshaling the args to typedmemmove might clobber the
		// value we're trying to move.
		t := val.Type.ElemType()
		tmp = b.Func.fe.Auto(val.Pos, t)
		aux := &AutoSymbol{Node: tmp}
		mem = b.NewValue1A(pos, OpVarDef, types.TypeMem, tmp, mem)
		tmpaddr := b.NewValue1A(pos, OpAddr, t.PtrTo(), aux, sp)
		siz := t.Size()
		mem = b.NewValue3I(pos, OpMove, types.TypeMem, siz, tmpaddr, val, mem)
		mem.Aux = t
		val = tmpaddr
	}

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

	if valIsVolatile {
		mem = b.NewValue1A(pos, OpVarKill, types.TypeMem, tmp, mem) // mark temp dead
	}

	return mem
}

// round to a multiple of r, r is a power of 2
func round(o int64, r int64) int64 {
	return (o + r - 1) &^ (r - 1)
}

// IsStackAddr returns whether v is known to be an address of a stack slot
func IsStackAddr(v *Value) bool {
	for v.Op == OpOffPtr || v.Op == OpAddPtr || v.Op == OpPtrIndex || v.Op == OpCopy {
		v = v.Args[0]
	}
	switch v.Op {
	case OpSP:
		return true
	case OpAddr:
		return v.Args[0].Op == OpSP
	}
	return false
}

// isVolatile returns whether v is a pointer to argument region on stack which
// will be clobbered by a function call.
func isVolatile(v *Value) bool {
	for v.Op == OpOffPtr || v.Op == OpAddPtr || v.Op == OpPtrIndex || v.Op == OpCopy {
		v = v.Args[0]
	}
	return v.Op == OpSP
}
