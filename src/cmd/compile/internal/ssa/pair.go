// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"slices"
)

// The pair pass finds memory operations that can be paired up
// into single 2-register memory instructions.
func pair(f *Func) {
	// Only arm64 for now. This pass is fairly arch-specific.
	switch f.Config.arch {
	case "arm64":
	default:
		return
	}
	pairLoads(f)
	pairStores(f)
}

type pairableLoadInfo struct {
	width int64 // width of one element in the pair, in bytes
	pair  Op
}

// All pairableLoad ops must take 2 arguments, a pointer and a memory.
// They must also take an offset in Aux/AuxInt.
var pairableLoads = map[Op]pairableLoadInfo{
	OpARM64MOVDload:  {8, OpARM64LDP},
	OpARM64MOVWUload: {4, OpARM64LDPW},
	OpARM64MOVWload:  {4, OpARM64LDPSW},
	// TODO: conceivably we could pair a signed and unsigned load
	// if we knew the upper bits of one of them weren't being used.
	OpARM64FMOVDload: {8, OpARM64FLDPD},
	OpARM64FMOVSload: {4, OpARM64FLDPS},
}

type pairableStoreInfo struct {
	width int64 // width of one element in the pair, in bytes
	pair  Op
}

// All pairableStore keys must take 3 arguments, a pointer, a value, and a memory.
// All pairableStore values must take 4 arguments, a pointer, 2 values, and a memory.
// They must also take an offset in Aux/AuxInt.
var pairableStores = map[Op]pairableStoreInfo{
	OpARM64MOVDstore:  {8, OpARM64STP},
	OpARM64MOVWstore:  {4, OpARM64STPW},
	OpARM64FMOVDstore: {8, OpARM64FSTPD},
	OpARM64FMOVSstore: {4, OpARM64FSTPS},
}

// offsetOk returns true if a pair instruction should be used
// for the offset Aux+off, when the data width (of the
// unpaired instructions) is width.
// This function is best-effort. The compiled function must
// still work if offsetOk always returns true.
// TODO: this is currently arm64-specific.
func offsetOk(aux Aux, off, width int64) bool {
	if true {
		// Seems to generate slightly smaller code if we just
		// always allow this rewrite.
		//
		// Without pairing, we have 2 load instructions, like:
		//   LDR 88(R0), R1
		//   LDR 96(R0), R2
		// with pairing we have, best case:
		//   LDP 88(R0), R1, R2
		// but maybe we need an adjuster if out of range or unaligned:
		//   ADD R0, $88, R27
		//   LDP (R27), R1, R2
		// Even with the adjuster, it is at least no worse.
		//
		// A similar situation occurs when accessing globals.
		// Two loads from globals requires 4 instructions,
		// two ADRP and two LDR. With pairing, we need
		// ADRP+ADD+LDP, three instructions.
		//
		// With pairing, it looks like the critical path might
		// be a little bit longer. But it should never be more
		// instructions.
		// TODO: see if that longer critical path causes any
		// regressions.
		return true
	}
	if aux != nil {
		if _, ok := aux.(*ir.Name); !ok {
			// Offset is probably too big (globals).
			return false
		}
		// We let *ir.Names pass here, as
		// they are probably small offsets from SP.
		// There's no guarantee that we're in range
		// in that case though (we don't know the
		// stack frame size yet), so the assembler
		// might need to issue fixup instructions.
		// Assume some small frame size.
		if off >= 0 {
			off += 120
		}
		// TODO: figure out how often this helps vs. hurts.
	}
	switch width {
	case 4:
		if off >= -256 && off <= 252 && off%4 == 0 {
			return true
		}
	case 8:
		if off >= -512 && off <= 504 && off%8 == 0 {
			return true
		}
	}
	return false
}

func pairLoads(f *Func) {
	var loads []*Value

	// Registry of aux values for sorting.
	auxIDs := map[Aux]int{}
	auxID := func(aux Aux) int {
		id, ok := auxIDs[aux]
		if !ok {
			id = len(auxIDs)
			auxIDs[aux] = id
		}
		return id
	}

	for _, b := range f.Blocks {
		// Find loads.
		loads = loads[:0]
		clear(auxIDs)
		for _, v := range b.Values {
			info := pairableLoads[v.Op]
			if info.width == 0 {
				continue // not pairable
			}
			if !offsetOk(v.Aux, v.AuxInt, info.width) {
				continue // not advisable
			}
			loads = append(loads, v)
		}
		if len(loads) < 2 {
			continue
		}

		// Sort to put pairable loads together.
		slices.SortFunc(loads, func(x, y *Value) int {
			// First sort by op, ptr, and memory arg.
			if x.Op != y.Op {
				return int(x.Op - y.Op)
			}
			if x.Args[0].ID != y.Args[0].ID {
				return int(x.Args[0].ID - y.Args[0].ID)
			}
			if x.Args[1].ID != y.Args[1].ID {
				return int(x.Args[1].ID - y.Args[1].ID)
			}
			// Then sort by aux. (nil first, then by aux ID)
			if x.Aux != nil {
				if y.Aux == nil {
					return 1
				}
				a, b := auxID(x.Aux), auxID(y.Aux)
				if a != b {
					return a - b
				}
			} else if y.Aux != nil {
				return -1
			}
			// Then sort by offset, low to high.
			return int(x.AuxInt - y.AuxInt)
		})

		// Look for pairable loads.
		for i := 0; i < len(loads)-1; i++ {
			x := loads[i]
			y := loads[i+1]
			if x.Op != y.Op || x.Args[0] != y.Args[0] || x.Args[1] != y.Args[1] {
				continue
			}
			if x.Aux != y.Aux {
				continue
			}
			if x.AuxInt+pairableLoads[x.Op].width != y.AuxInt {
				continue
			}

			// Commit point.

			// Make the 2-register load.
			load := b.NewValue2IA(x.Pos, pairableLoads[x.Op].pair, types.NewTuple(x.Type, y.Type), x.AuxInt, x.Aux, x.Args[0], x.Args[1])

			// Modify x to be (Select0 load). Similar for y.
			x.reset(OpSelect0)
			x.SetArgs1(load)
			y.reset(OpSelect1)
			y.SetArgs1(load)

			i++ // Skip y next time around the loop.
		}
	}

	// Try to pair a load with a load from a subsequent block.
	// Note that this is always safe to do if the memory arguments match.
	// (But see the memory barrier case below.)
	type nextBlockKey struct {
		op     Op
		ptr    ID
		mem    ID
		auxInt int64
		aux    any
	}
	nextBlock := map[nextBlockKey]*Value{}
	for _, b := range f.Blocks {
		if memoryBarrierTest(b) {
			// TODO: Do we really need to skip write barrier test blocks?
			//     type T struct {
			//         a *byte
			//         b int
			//     }
			//     func f(t *T) int {
			//         r := t.b
			//         t.a = nil
			//         return r
			//     }
			// This would issue a single LDP for both the t.a and t.b fields,
			// *before* we check the write barrier flag. (We load the t.a field
			// to put it in the write barrier buffer.) Not sure if that is ok.
			continue
		}
		// Find loads in the next block(s) that we can move to this one.
		// TODO: could maybe look further than just one successor hop.
		clear(nextBlock)
		for _, e := range b.Succs {
			if len(e.b.Preds) > 1 {
				continue
			}
			for _, v := range e.b.Values {
				info := pairableLoads[v.Op]
				if info.width == 0 {
					continue
				}
				if !offsetOk(v.Aux, v.AuxInt, info.width) {
					continue // not advisable
				}
				nextBlock[nextBlockKey{op: v.Op, ptr: v.Args[0].ID, mem: v.Args[1].ID, auxInt: v.AuxInt, aux: v.Aux}] = v
			}
		}
		if len(nextBlock) == 0 {
			continue
		}
		// don't move too many loads. Each requires a register across a basic block boundary.
		const maxMoved = 4
		nMoved := 0
		for i := len(b.Values) - 1; i >= 0 && nMoved < maxMoved; i-- {
			x := b.Values[i]
			info := pairableLoads[x.Op]
			if info.width == 0 {
				continue
			}
			if !offsetOk(x.Aux, x.AuxInt, info.width) {
				continue // not advisable
			}
			key := nextBlockKey{op: x.Op, ptr: x.Args[0].ID, mem: x.Args[1].ID, auxInt: x.AuxInt + info.width, aux: x.Aux}
			if y := nextBlock[key]; y != nil {
				delete(nextBlock, key)

				// Make the 2-register load.
				load := b.NewValue2IA(x.Pos, info.pair, types.NewTuple(x.Type, y.Type), x.AuxInt, x.Aux, x.Args[0], x.Args[1])

				// Modify x to be (Select0 load).
				x.reset(OpSelect0)
				x.SetArgs1(load)
				// Modify y to be (Copy (Select1 load)).
				// Note: the Select* needs to live in the load's block, not y's block.
				y.reset(OpCopy)
				y.SetArgs1(b.NewValue1(y.Pos, OpSelect1, y.Type, load))
				nMoved++
				continue
			}
			key.auxInt = x.AuxInt - info.width
			if y := nextBlock[key]; y != nil {
				delete(nextBlock, key)

				// Make the 2-register load.
				load := b.NewValue2IA(x.Pos, info.pair, types.NewTuple(y.Type, x.Type), y.AuxInt, x.Aux, x.Args[0], x.Args[1])

				// Modify x to be (Select1 load).
				x.reset(OpSelect1)
				x.SetArgs1(load)
				// Modify y to be (Copy (Select0 load)).
				y.reset(OpCopy)
				y.SetArgs1(b.NewValue1(y.Pos, OpSelect0, y.Type, load))
				nMoved++
				continue
			}
		}
	}
}

func memoryBarrierTest(b *Block) bool {
	if b.Kind != BlockARM64NZW {
		return false
	}
	c := b.Controls[0]
	if c.Op != OpARM64MOVWUload {
		return false
	}
	if globl, ok := c.Aux.(*obj.LSym); ok {
		return globl.Name == "runtime.writeBarrier"
	}
	return false
}

// pairStores merges store instructions.
// It collects stores into a buffer where they can be freely reordered.
// When encountering an instruction that cannot be added to the buffer,
// it pairs the accumulated stores, flushes the buffer, and continues processing.
func pairStores(f *Func) {
	last := f.Cache.allocBoolSlice(f.NumValues())
	defer f.Cache.freeBoolSlice(last)

	// memChain contains a list of stores with the same ptr/aux pair and
	// nonoverlapping write ranges [AuxInt:AuxInt+writeSize]. All of the
	// elements of memChain can be reordered with each other.
	memChain := []*Value{}

	// Limit of length of memChain array.
	// This keeps us in O(n) territory.
	limit := 100

	// flushMemChain sorts the stores in memChain and merges them when possible.
	// Then it flushes memChain.
	flushMemChain := func() {
		if len(memChain) < 2 {
			memChain = memChain[:0]
			return
		}

		// Sort in increasing AuxInt to put pairable stores together.
		slices.SortFunc(memChain, func(x, y *Value) int {
			return int(x.AuxInt - y.AuxInt)
		})

		lastIdx := len(memChain) - 1
		for i := 0; i < lastIdx; i++ {
			v := memChain[i]
			w := memChain[i+1]
			info := pairableStores[v.Op]

			off := v.AuxInt
			mem := v.MemoryArg()
			aux := v.Aux
			pos := v.Pos
			wmem := w.MemoryArg()

			if w.Op == v.Op && w.AuxInt == off+info.width {
				// Arguments for the merged store: ptr, val1, val2, mem.
				args := []*Value{v.Args[0], v.Args[1], w.Args[1], mem}

				v.reset(info.pair)
				v.AddArgs(args...)
				v.Aux = aux
				v.AuxInt = off
				v.Pos = pos

				// Make w just a memory copy.
				w.reset(OpCopy)
				w.SetArgs1(wmem)

				// Skip merged store (w)
				i++
			}
		}

		memChain = memChain[:0]
	}

	// prevStore returns the previous store in the
	// same block, or nil if there are none.
	prevStore := func(v *Value) *Value {
		if v.Op == OpInitMem || v.Op == OpPhi {
			return nil
		}
		m := v.MemoryArg()
		if m.Block != v.Block {
			return nil
		}
		return m
	}

	// storeWidth returns the width of store,
	// or 0 if it is not a store this pass understands.
	storeWidth := func(op Op) int64 {
		var width int64
		switch op {
		case OpARM64MOVDstore, OpARM64FMOVDstore:
			width = 8
		case OpARM64MOVWstore, OpARM64FMOVSstore:
			width = 4
		case OpARM64MOVHstore:
			width = 2
		case OpARM64MOVBstore:
			width = 1
		default:
			width = 0
		}
		return width
	}

	for _, b := range f.Blocks {
		memChain = memChain[:0]

		// Find last store in block, so we can
		// walk the stores last to first.
		// Last to first helps ensure that the rewrites we
		// perform do not get in the way of subsequent rewrites.
		for _, v := range b.Values {
			if v.Type.IsMemory() {
				last[v.ID] = true
			}
		}
		for _, v := range b.Values {
			if v.Type.IsMemory() {
				if m := prevStore(v); m != nil {
					last[m.ID] = false
				}
			}
		}
		var lastMem *Value
		for _, v := range b.Values {
			if last[v.ID] {
				lastMem = v
				break
			}
		}

		// Iterate over memory stores, accumulating them in memChain for potential merging.
		// Flush the chain when reordering is unsafe or a conflict is detected.
		for v := lastMem; v != nil; v = prevStore(v) {
			writeSize := storeWidth(v.Op)

			if writeSize == 0 {
				// We can't reorder stores with calls or other instructions
				// with writeSize == 0.
				flushMemChain()
				continue
			}
			if v.Uses != 1 && len(memChain) > 0 ||
				len(memChain) > 0 && (v.Args[0] != memChain[0].Args[0] || v.Aux != memChain[0].Aux) ||
				len(memChain) == limit {
				// If v has multiple uses and it is not the latest store in the chain,
				// we cannot merge it with other store instructions.
				// If v has a different base pointer or Aux value from the current chain,
				// we need to flush memChain and start a new one with v.
				// If memChain length limit is exceeded, we also need to flush the chain
				// and start a new one with v.
				// Only look back so far.
				// This keeps us in O(n) territory, and it
				// also prevents us from keeping values
				// in registers for too long (and thus
				// needing to spill them).
				flushMemChain()
			}

			for _, w := range memChain {
				wWriteSize := storeWidth(w.Op)
				if overlap(w.AuxInt, wWriteSize, v.AuxInt, writeSize) {
					// Aliases with w's location.
					// Flush the chain and start a new one with v.
					flushMemChain()
					break
				}
			}

			memChain = append(memChain, v)
		}
		flushMemChain()
	}
}
