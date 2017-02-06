// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/internal/obj"
	"cmd/internal/src"
)

// needwb returns whether we need write barrier for store op v.
// v must be Store/Move/Zero.
func needwb(v *Value) bool {
	t, ok := v.Aux.(Type)
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
	if !f.Config.fe.UseWriteBarrier() {
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
				sb = f.Entry.NewValue0(initpos, OpSB, f.Config.fe.TypeUintptr())
			}
			if sp == nil {
				sp = f.Entry.NewValue0(initpos, OpSP, f.Config.fe.TypeUintptr())
			}
			wbsym := &ExternSymbol{Typ: f.Config.fe.TypeBool(), Sym: f.Config.fe.Syslook("writeBarrier")}
			wbaddr = f.Entry.NewValue1A(initpos, OpAddr, f.Config.fe.TypeUInt32().PtrTo(), wbsym, sb)
			writebarrierptr = f.Config.fe.Syslook("writebarrierptr")
			typedmemmove = f.Config.fe.Syslook("typedmemmove")
			typedmemclr = f.Config.fe.Syslook("typedmemclr")
			const0 = f.ConstInt32(initpos, f.Config.fe.TypeUInt32(), 0)

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
		for i := len(values) - 1; i >= 0; i-- {
			w := values[i]
			if w.Op == OpStoreWB || w.Op == OpMoveWB || w.Op == OpZeroWB {
				if last == nil {
					last = w
					end = i + 1
				}
			} else {
				if last != nil {
					start = i + 1
					break
				}
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
		flag := b.NewValue2(pos, OpLoad, f.Config.fe.TypeUInt32(), wbaddr, mem)
		flag = b.NewValue2(pos, OpNeq32, f.Config.fe.TypeBool(), flag, const0)
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
			var val *Value
			ptr := w.Args[0]
			siz := w.AuxInt
			var typ interface{}
			if w.Op != OpStoreWB {
				typ = &ExternSymbol{Typ: f.Config.fe.TypeUintptr(), Sym: w.Aux.(Type).Symbol()}
			}
			pos = w.Pos

			var op Op
			var fn *obj.LSym
			switch w.Op {
			case OpStoreWB:
				op = OpStore
				fn = writebarrierptr
				val = w.Args[1]
			case OpMoveWB:
				op = OpMove
				fn = typedmemmove
				val = w.Args[1]
			case OpZeroWB:
				op = OpZero
				fn = typedmemclr
			}

			// then block: emit write barrier call
			volatile := w.Op == OpMoveWB && isVolatile(val)
			memThen = wbcall(pos, bThen, fn, typ, ptr, val, memThen, sp, sb, volatile)

			// else block: normal store
			if op == OpZero {
				memElse = bElse.NewValue2I(pos, op, TypeMem, siz, ptr, memElse)
			} else {
				memElse = bElse.NewValue3I(pos, op, TypeMem, siz, ptr, val, memElse)
			}

			if f.Config.fe.Debug_wb() {
				f.Config.Warnl(pos, "write barrier")
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
		last.Type = TypeMem
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
		for _, w := range b.Values {
			if w.Op == OpStoreWB || w.Op == OpMoveWB || w.Op == OpZeroWB {
				goto again
			}
		}
	}
}

// wbcall emits write barrier runtime call in b, returns memory.
// if valIsVolatile, it moves val into temp space before making the call.
func wbcall(pos src.XPos, b *Block, fn *obj.LSym, typ interface{}, ptr, val, mem, sp, sb *Value, valIsVolatile bool) *Value {
	config := b.Func.Config

	var tmp GCNode
	if valIsVolatile {
		// Copy to temp location if the source is volatile (will be clobbered by
		// a function call). Marshaling the args to typedmemmove might clobber the
		// value we're trying to move.
		t := val.Type.ElemType()
		tmp = config.fe.Auto(t)
		aux := &AutoSymbol{Typ: t, Node: tmp}
		mem = b.NewValue1A(pos, OpVarDef, TypeMem, tmp, mem)
		tmpaddr := b.NewValue1A(pos, OpAddr, t.PtrTo(), aux, sp)
		siz := MakeSizeAndAlign(t.Size(), t.Alignment()).Int64()
		mem = b.NewValue3I(pos, OpMove, TypeMem, siz, tmpaddr, val, mem)
		val = tmpaddr
	}

	// put arguments on stack
	off := config.ctxt.FixedFrameSize()

	if typ != nil { // for typedmemmove
		taddr := b.NewValue1A(pos, OpAddr, config.fe.TypeUintptr(), typ, sb)
		off = round(off, taddr.Type.Alignment())
		arg := b.NewValue1I(pos, OpOffPtr, taddr.Type.PtrTo(), off, sp)
		mem = b.NewValue3I(pos, OpStore, TypeMem, ptr.Type.Size(), arg, taddr, mem)
		off += taddr.Type.Size()
	}

	off = round(off, ptr.Type.Alignment())
	arg := b.NewValue1I(pos, OpOffPtr, ptr.Type.PtrTo(), off, sp)
	mem = b.NewValue3I(pos, OpStore, TypeMem, ptr.Type.Size(), arg, ptr, mem)
	off += ptr.Type.Size()

	if val != nil {
		off = round(off, val.Type.Alignment())
		arg = b.NewValue1I(pos, OpOffPtr, val.Type.PtrTo(), off, sp)
		mem = b.NewValue3I(pos, OpStore, TypeMem, val.Type.Size(), arg, val, mem)
		off += val.Type.Size()
	}
	off = round(off, config.PtrSize)

	// issue call
	mem = b.NewValue1A(pos, OpStaticCall, TypeMem, fn, mem)
	mem.AuxInt = off - config.ctxt.FixedFrameSize()

	if valIsVolatile {
		mem = b.NewValue1A(pos, OpVarKill, TypeMem, tmp, mem) // mark temp dead
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
