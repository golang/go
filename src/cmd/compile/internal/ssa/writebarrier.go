// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "fmt"

// writebarrier expands write barrier ops (StoreWB, MoveWB, etc.) into
// branches and runtime calls, like
//
// if writeBarrier.enabled {
//   writebarrierptr(ptr, val)
// } else {
//   *ptr = val
// }
//
// If ptr is an address of a stack slot, write barrier will be removed
// and a normal store will be used.
// A sequence of WB stores for many pointer fields of a single type will
// be emitted together, with a single branch.
//
// Expanding WB ops introduces new control flows, and we would need to
// split a block into two if there were values after WB ops, which would
// require scheduling the values. To avoid this complexity, when building
// SSA, we make sure that WB ops are always at the end of a block. We do
// this before fuse as it may merge blocks. It also helps to reduce
// number of blocks as fuse merges blocks introduced in this phase.
func writebarrier(f *Func) {
	var sb, sp, wbaddr *Value
	var writebarrierptr, typedmemmove, typedmemclr interface{} // *gc.Sym
	var storeWBs, others []*Value
	var wbs *sparseSet
	for _, b := range f.Blocks { // range loop is safe since the blocks we added contain no WB stores
	valueLoop:
		for i, v := range b.Values {
			switch v.Op {
			case OpStoreWB, OpMoveWB, OpMoveWBVolatile, OpZeroWB:
				if IsStackAddr(v.Args[0]) {
					switch v.Op {
					case OpStoreWB:
						v.Op = OpStore
					case OpMoveWB, OpMoveWBVolatile:
						v.Op = OpMove
						v.Aux = nil
					case OpZeroWB:
						v.Op = OpZero
						v.Aux = nil
					}
					continue
				}

				if wbaddr == nil {
					// initalize global values for write barrier test and calls
					// find SB and SP values in entry block
					initln := f.Entry.Line
					for _, v := range f.Entry.Values {
						if v.Op == OpSB {
							sb = v
						}
						if v.Op == OpSP {
							sp = v
						}
					}
					if sb == nil {
						sb = f.Entry.NewValue0(initln, OpSB, f.Config.fe.TypeUintptr())
					}
					if sp == nil {
						sp = f.Entry.NewValue0(initln, OpSP, f.Config.fe.TypeUintptr())
					}
					wbsym := &ExternSymbol{Typ: f.Config.fe.TypeBool(), Sym: f.Config.fe.Syslook("writeBarrier").(fmt.Stringer)}
					wbaddr = f.Entry.NewValue1A(initln, OpAddr, f.Config.fe.TypeUInt32().PtrTo(), wbsym, sb)
					writebarrierptr = f.Config.fe.Syslook("writebarrierptr")
					typedmemmove = f.Config.fe.Syslook("typedmemmove")
					typedmemclr = f.Config.fe.Syslook("typedmemclr")

					wbs = f.newSparseSet(f.NumValues())
					defer f.retSparseSet(wbs)
				}

				line := v.Line

				// there may be a sequence of WB stores in the current block. find them.
				storeWBs = storeWBs[:0]
				others = others[:0]
				wbs.clear()
				for _, w := range b.Values[i:] {
					if w.Op == OpStoreWB || w.Op == OpMoveWB || w.Op == OpMoveWBVolatile || w.Op == OpZeroWB {
						storeWBs = append(storeWBs, w)
						wbs.add(w.ID)
					} else {
						others = append(others, w)
					}
				}

				// make sure that no value in this block depends on WB stores
				for _, w := range b.Values {
					if w.Op == OpStoreWB || w.Op == OpMoveWB || w.Op == OpMoveWBVolatile || w.Op == OpZeroWB {
						continue
					}
					for _, a := range w.Args {
						if wbs.contains(a.ID) {
							f.Fatalf("value %v depends on WB store %v in the same block %v", w, a, b)
						}
					}
				}

				// find the memory before the WB stores
				// this memory is not a WB store but it is used in a WB store.
				var mem *Value
				for _, w := range storeWBs {
					a := w.Args[len(w.Args)-1]
					if wbs.contains(a.ID) {
						continue
					}
					if mem != nil {
						b.Fatalf("two stores live simultaneously: %s, %s", mem, a)
					}
					mem = a
				}

				b.Values = append(b.Values[:i], others...) // move WB ops out of this block

				bThen := f.NewBlock(BlockPlain)
				bElse := f.NewBlock(BlockPlain)
				bEnd := f.NewBlock(b.Kind)
				bThen.Line = line
				bElse.Line = line
				bEnd.Line = line

				// set up control flow for end block
				bEnd.SetControl(b.Control)
				bEnd.Likely = b.Likely
				for _, e := range b.Succs {
					bEnd.Succs = append(bEnd.Succs, e)
					e.b.Preds[e.i].b = bEnd
				}

				// set up control flow for write barrier test
				// load word, test word, avoiding partial register write from load byte.
				flag := b.NewValue2(line, OpLoad, f.Config.fe.TypeUInt32(), wbaddr, mem)
				const0 := f.ConstInt32(line, f.Config.fe.TypeUInt32(), 0)
				flag = b.NewValue2(line, OpNeq32, f.Config.fe.TypeBool(), flag, const0)
				b.Kind = BlockIf
				b.SetControl(flag)
				b.Likely = BranchUnlikely
				b.Succs = b.Succs[:0]
				b.AddEdgeTo(bThen)
				b.AddEdgeTo(bElse)
				bThen.AddEdgeTo(bEnd)
				bElse.AddEdgeTo(bEnd)

				memThen := mem
				memElse := mem
				for _, w := range storeWBs {
					var val *Value
					ptr := w.Args[0]
					siz := w.AuxInt
					typ := w.Aux // only non-nil for MoveWB, MoveWBVolatile, ZeroWB

					var op Op
					var fn interface{} // *gc.Sym
					switch w.Op {
					case OpStoreWB:
						op = OpStore
						fn = writebarrierptr
						val = w.Args[1]
					case OpMoveWB, OpMoveWBVolatile:
						op = OpMove
						fn = typedmemmove
						val = w.Args[1]
					case OpZeroWB:
						op = OpZero
						fn = typedmemclr
					}

					// then block: emit write barrier call
					memThen = wbcall(line, bThen, fn, typ, ptr, val, memThen, sp, sb, w.Op == OpMoveWBVolatile)

					// else block: normal store
					if op == OpZero {
						memElse = bElse.NewValue2I(line, op, TypeMem, siz, ptr, memElse)
					} else {
						memElse = bElse.NewValue3I(line, op, TypeMem, siz, ptr, val, memElse)
					}
				}

				// merge memory
				// Splice memory Phi into the last memory of the original sequence,
				// which may be used in subsequent blocks. Other memories in the
				// sequence must be dead after this block since there can be only
				// one memory live.
				last := storeWBs[0]
				if len(storeWBs) > 1 {
					// find the last store
					last = nil
					wbs.clear() // we reuse wbs to record WB stores that is used in another WB store
					for _, w := range storeWBs {
						wbs.add(w.Args[len(w.Args)-1].ID)
					}
					for _, w := range storeWBs {
						if wbs.contains(w.ID) {
							continue
						}
						if last != nil {
							b.Fatalf("two stores live simultaneously: %s, %s", last, w)
						}
						last = w
					}
				}
				bEnd.Values = append(bEnd.Values, last)
				last.Block = bEnd
				last.reset(OpPhi)
				last.Type = TypeMem
				last.AddArg(memThen)
				last.AddArg(memElse)
				for _, w := range storeWBs {
					if w != last {
						w.resetArgs()
					}
				}
				for _, w := range storeWBs {
					if w != last {
						f.freeValue(w)
					}
				}

				if f.Config.fe.Debug_wb() {
					f.Config.Warnl(line, "write barrier")
				}

				break valueLoop
			}
		}
	}
}

// wbcall emits write barrier runtime call in b, returns memory.
// if valIsVolatile, it moves val into temp space before making the call.
func wbcall(line int32, b *Block, fn interface{}, typ interface{}, ptr, val, mem, sp, sb *Value, valIsVolatile bool) *Value {
	config := b.Func.Config

	var tmp GCNode
	if valIsVolatile {
		// Copy to temp location if the source is volatile (will be clobbered by
		// a function call). Marshaling the args to typedmemmove might clobber the
		// value we're trying to move.
		t := val.Type.ElemType()
		tmp = config.fe.Auto(t)
		aux := &AutoSymbol{Typ: t, Node: tmp}
		mem = b.NewValue1A(line, OpVarDef, TypeMem, tmp, mem)
		tmpaddr := b.NewValue1A(line, OpAddr, t.PtrTo(), aux, sp)
		siz := MakeSizeAndAlign(t.Size(), t.Alignment()).Int64()
		mem = b.NewValue3I(line, OpMove, TypeMem, siz, tmpaddr, val, mem)
		val = tmpaddr
	}

	// put arguments on stack
	off := config.ctxt.FixedFrameSize()

	if typ != nil { // for typedmemmove
		taddr := b.NewValue1A(line, OpAddr, config.fe.TypeUintptr(), typ, sb)
		off = round(off, taddr.Type.Alignment())
		arg := b.NewValue1I(line, OpOffPtr, taddr.Type.PtrTo(), off, sp)
		mem = b.NewValue3I(line, OpStore, TypeMem, ptr.Type.Size(), arg, taddr, mem)
		off += taddr.Type.Size()
	}

	off = round(off, ptr.Type.Alignment())
	arg := b.NewValue1I(line, OpOffPtr, ptr.Type.PtrTo(), off, sp)
	mem = b.NewValue3I(line, OpStore, TypeMem, ptr.Type.Size(), arg, ptr, mem)
	off += ptr.Type.Size()

	if val != nil {
		off = round(off, val.Type.Alignment())
		arg = b.NewValue1I(line, OpOffPtr, val.Type.PtrTo(), off, sp)
		mem = b.NewValue3I(line, OpStore, TypeMem, val.Type.Size(), arg, val, mem)
		off += val.Type.Size()
	}
	off = round(off, config.PtrSize)

	// issue call
	mem = b.NewValue1A(line, OpStaticCall, TypeMem, fn, mem)
	mem.AuxInt = off - config.ctxt.FixedFrameSize()

	if valIsVolatile {
		mem = b.NewValue1A(line, OpVarKill, TypeMem, tmp, mem) // mark temp dead
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
