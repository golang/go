// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package liveness

import (
	"fmt"
	"internal/abi"

	"cmd/compile/internal/base"
	"cmd/compile/internal/bitvec"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/ssa"
	"cmd/internal/obj"
)

// Argument liveness tracking.
//
// For arguments passed in registers, this file tracks if their spill slots
// are live for runtime traceback. An argument spill slot is live at a PC
// if we know that an actual value has stored into it at or before this point.
//
// Stack args are always live and not tracked in this code. Stack args are
// laid out before register spill slots, so we emit the smallest offset that
// needs tracking. Slots before that offset are always live. That offset is
// usually the offset of the first spill slot. But if the first spill slot is
// always live (e.g. if it is address-taken), it will be the offset of a later
// one.
//
// The liveness information is emitted as a FUNCDATA and a PCDATA.
//
// FUNCDATA format:
// - start (smallest) offset that needs tracking (1 byte)
// - a list of bitmaps.
//   In a bitmap bit i is set if the i-th spill slot is live.
//
// At a PC where the liveness info changes, a PCDATA indicates the
// byte offset of the liveness map in the FUNCDATA. PCDATA -1 is a
// special case indicating all slots are live (for binary size
// saving).

const allLiveIdx = -1

// name and offset
type nameOff struct {
	n   *ir.Name
	off int64
}

func (a nameOff) FrameOffset() int64 { return a.n.FrameOffset() + a.off }
func (a nameOff) String() string     { return fmt.Sprintf("%v+%d", a.n, a.off) }

type blockArgEffects struct {
	livein  bitvec.BitVec // variables live at block entry
	liveout bitvec.BitVec // variables live at block exit
}

type argLiveness struct {
	fn   *ir.Func
	f    *ssa.Func
	args []nameOff         // name and offset of spill slots
	idx  map[nameOff]int32 // index in args

	be []blockArgEffects // indexed by block ID

	bvset bvecSet // Set of liveness bitmaps, used for uniquifying.

	// Liveness map indices at each Value (where it changes) and Block entry.
	// During the computation the indices are temporarily index to bvset.
	// At the end they will be index (offset) to the output funcdata (changed
	// in (*argLiveness).emit).
	blockIdx map[ssa.ID]int
	valueIdx map[ssa.ID]int
}

// ArgLiveness computes the liveness information of register argument spill slots.
// An argument's spill slot is "live" if we know it contains a meaningful value,
// that is, we have stored the register value to it.
// Returns the liveness map indices at each Block entry and at each Value (where
// it changes).
func ArgLiveness(fn *ir.Func, f *ssa.Func, pp *objw.Progs) (blockIdx, valueIdx map[ssa.ID]int) {
	if f.OwnAux.ABIInfo().InRegistersUsed() == 0 || base.Flag.N != 0 {
		// No register args. Nothing to emit.
		// Or if -N is used we spill everything upfront so it is always live.
		return nil, nil
	}

	lv := &argLiveness{
		fn:       fn,
		f:        f,
		idx:      make(map[nameOff]int32),
		be:       make([]blockArgEffects, f.NumBlocks()),
		blockIdx: make(map[ssa.ID]int),
		valueIdx: make(map[ssa.ID]int),
	}
	// Gather all register arg spill slots.
	for _, a := range f.OwnAux.ABIInfo().InParams() {
		n := a.Name
		if n == nil || len(a.Registers) == 0 {
			continue
		}
		_, offs := a.RegisterTypesAndOffsets()
		for _, off := range offs {
			if n.FrameOffset()+off > 0xff {
				// We only print a limited number of args, with stack
				// offsets no larger than 255.
				continue
			}
			lv.args = append(lv.args, nameOff{n, off})
		}
	}
	if len(lv.args) > 10 {
		lv.args = lv.args[:10] // We print no more than 10 args.
	}

	// We spill address-taken or non-SSA-able value upfront, so they are always live.
	alwaysLive := func(n *ir.Name) bool { return n.Addrtaken() || !ssa.CanSSA(n.Type()) }

	// We'll emit the smallest offset for the slots that need liveness info.
	// No need to include a slot with a lower offset if it is always live.
	for len(lv.args) > 0 && alwaysLive(lv.args[0].n) {
		lv.args = lv.args[1:]
	}
	if len(lv.args) == 0 {
		return // everything is always live
	}

	for i, a := range lv.args {
		lv.idx[a] = int32(i)
	}

	nargs := int32(len(lv.args))
	bulk := bitvec.NewBulk(nargs, int32(len(f.Blocks)*2))
	for _, b := range f.Blocks {
		be := &lv.be[b.ID]
		be.livein = bulk.Next()
		be.liveout = bulk.Next()

		// initialize to all 1s, so we can AND them
		be.livein.Not()
		be.liveout.Not()
	}

	entrybe := &lv.be[f.Entry.ID]
	entrybe.livein.Clear()
	for i, a := range lv.args {
		if alwaysLive(a.n) {
			entrybe.livein.Set(int32(i))
		}
	}

	// Visit blocks in reverse-postorder, compute block effects.
	po := f.Postorder()
	for i := len(po) - 1; i >= 0; i-- {
		b := po[i]
		be := &lv.be[b.ID]

		// A slot is live at block entry if it is live in all predecessors.
		for _, pred := range b.Preds {
			pb := pred.Block()
			be.livein.And(be.livein, lv.be[pb.ID].liveout)
		}

		be.liveout.Copy(be.livein)
		for _, v := range b.Values {
			lv.valueEffect(v, be.liveout)
		}
	}

	// Coalesce identical live vectors. Compute liveness indices at each PC
	// where it changes.
	live := bitvec.New(nargs)
	addToSet := func(bv bitvec.BitVec) (int, bool) {
		if bv.Count() == int(nargs) { // special case for all live
			return allLiveIdx, false
		}
		return lv.bvset.add(bv)
	}
	for _, b := range lv.f.Blocks {
		be := &lv.be[b.ID]
		lv.blockIdx[b.ID], _ = addToSet(be.livein)

		live.Copy(be.livein)
		var lastv *ssa.Value
		for i, v := range b.Values {
			if lv.valueEffect(v, live) {
				// Record that liveness changes but not emit a map now.
				// For a sequence of StoreRegs we only need to emit one
				// at last.
				lastv = v
			}
			if lastv != nil && (mayFault(v) || i == len(b.Values)-1) {
				// Emit the liveness map if it may fault or at the end of
				// the block. We may need a traceback if the instruction
				// may cause a panic.
				var added bool
				lv.valueIdx[lastv.ID], added = addToSet(live)
				if added {
					// live is added to bvset and we cannot modify it now.
					// Make a copy.
					t := live
					live = bitvec.New(nargs)
					live.Copy(t)
				}
				lastv = nil
			}
		}

		// Sanity check.
		if !live.Eq(be.liveout) {
			panic("wrong arg liveness map at block end")
		}
	}

	// Emit funcdata symbol, update indices to offsets in the symbol data.
	lsym := lv.emit()
	fn.LSym.Func().ArgLiveInfo = lsym

	//lv.print()

	p := pp.Prog(obj.AFUNCDATA)
	p.From.SetConst(abi.FUNCDATA_ArgLiveInfo)
	p.To.Type = obj.TYPE_MEM
	p.To.Name = obj.NAME_EXTERN
	p.To.Sym = lsym

	return lv.blockIdx, lv.valueIdx
}

// valueEffect applies the effect of v to live, return whether it is changed.
func (lv *argLiveness) valueEffect(v *ssa.Value, live bitvec.BitVec) bool {
	if v.Op != ssa.OpStoreReg { // TODO: include other store instructions?
		return false
	}
	n, off := ssa.AutoVar(v)
	if n.Class != ir.PPARAM {
		return false
	}
	i, ok := lv.idx[nameOff{n, off}]
	if !ok || live.Get(i) {
		return false
	}
	live.Set(i)
	return true
}

func mayFault(v *ssa.Value) bool {
	switch v.Op {
	case ssa.OpLoadReg, ssa.OpStoreReg, ssa.OpCopy, ssa.OpPhi,
		ssa.OpVarDef, ssa.OpVarLive, ssa.OpKeepAlive,
		ssa.OpSelect0, ssa.OpSelect1, ssa.OpSelectN, ssa.OpMakeResult,
		ssa.OpConvert, ssa.OpInlMark, ssa.OpGetG:
		return false
	}
	if len(v.Args) == 0 {
		return false // assume constant op cannot fault
	}
	return true // conservatively assume all other ops could fault
}

func (lv *argLiveness) print() {
	fmt.Println("argument liveness:", lv.f.Name)
	live := bitvec.New(int32(len(lv.args)))
	for _, b := range lv.f.Blocks {
		be := &lv.be[b.ID]

		fmt.Printf("%v: live in: ", b)
		lv.printLivenessVec(be.livein)
		if idx, ok := lv.blockIdx[b.ID]; ok {
			fmt.Printf("   #%d", idx)
		}
		fmt.Println()

		for _, v := range b.Values {
			if lv.valueEffect(v, live) {
				fmt.Printf("  %v: ", v)
				lv.printLivenessVec(live)
				if idx, ok := lv.valueIdx[v.ID]; ok {
					fmt.Printf("   #%d", idx)
				}
				fmt.Println()
			}
		}

		fmt.Printf("%v: live out: ", b)
		lv.printLivenessVec(be.liveout)
		fmt.Println()
	}
	fmt.Println("liveness maps data:", lv.fn.LSym.Func().ArgLiveInfo.P)
}

func (lv *argLiveness) printLivenessVec(bv bitvec.BitVec) {
	for i, a := range lv.args {
		if bv.Get(int32(i)) {
			fmt.Printf("%v ", a)
		}
	}
}

func (lv *argLiveness) emit() *obj.LSym {
	livenessMaps := lv.bvset.extractUnique()

	// stack offsets of register arg spill slots
	argOffsets := make([]uint8, len(lv.args))
	for i, a := range lv.args {
		off := a.FrameOffset()
		if off > 0xff {
			panic("offset too large")
		}
		argOffsets[i] = uint8(off)
	}

	idx2off := make([]int, len(livenessMaps))

	lsym := base.Ctxt.Lookup(lv.fn.LSym.Name + ".argliveinfo")
	lsym.Set(obj.AttrContentAddressable, true)

	off := objw.Uint8(lsym, 0, argOffsets[0]) // smallest offset that needs liveness info.
	for idx, live := range livenessMaps {
		idx2off[idx] = off
		off = objw.BitVec(lsym, off, live)
	}

	// Update liveness indices to offsets.
	for i, x := range lv.blockIdx {
		if x != allLiveIdx {
			lv.blockIdx[i] = idx2off[x]
		}
	}
	for i, x := range lv.valueIdx {
		if x != allLiveIdx {
			lv.valueIdx[i] = idx2off[x]
		}
	}

	return lsym
}
