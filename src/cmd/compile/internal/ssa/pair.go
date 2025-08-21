// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
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
}

func pairStores(f *Func) {
	last := f.Cache.allocBoolSlice(f.NumValues())
	defer f.Cache.freeBoolSlice(last)

	type stChainElem struct {
		v *Value
		i int // Index in chain (0 == last store)
	}
	var order []stChainElem

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
	// or 0 if it is not a store
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

	const limit = 10

	for _, b := range f.Blocks {
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

		order = order[:0]
		for i, v := 0, lastMem; v != nil; v = prevStore(v) {
			order = append(order, stChainElem{v, i})
			i++
		}
	reordering:
		for i, v_elem := range order {
			v := v_elem.v
			if v.Uses != 1 {
				// We can't reorder stores if the earlier
				// store has any use besides the next one
				// in the store chain.
				// (Unless we could check the aliasing of
				// all those other uses.)
				continue
			}
			widthV := storeWidth(v.Op)
			if widthV == 0 {
				// Can't reorder with any other memory operations.
				// (atomics, calls, ...)
				continue
			}
			chain := order[i+1:]
			count := limit
			// Var 'count' keeps us in O(n) territory
			for j, w_elem := range chain {
				if count--; count == 0 {
					// Only look back so far.
					// This keeps us in O(n) territory, and it
					// also prevents us from keeping values
					// in registers for too long (and thus
					// needing to spill them).
					continue reordering
				}

				w := w_elem.v
				if w.Uses != 1 {
					// We can't reorder stores if the earlier
					// store has any use besides the next one
					// in the store chain.
					// (Unless we could check the aliasing of
					// all those other uses.)
					continue reordering
				}

				widthW := storeWidth(w.Op)
				if widthW == 0 {
					// Can't reorder with any other memory operations.
					// (atomics, calls, ...)
					continue reordering
				}

				// We only allow reordering with respect to other
				// writes to the same pointer and aux, so we can
				// compute the exact the aliasing relationship.
				if w.Args[0] != v.Args[0] ||
					w.Aux != v.Aux {
					// Can't reorder with operation with incomparable destination memory pointer.
					continue reordering
				}
				if overlap(w.AuxInt, widthW, v.AuxInt, widthV) {
					// Aliases with the same slot with v's location.
					continue reordering
				}

				// Reordering stores in increasing order of memory access
				if v.AuxInt < w.AuxInt {
					order[i], order[i+j+1] = order[i+j+1], order[i]
					v = w
					widthV = widthW
				}
			}
		}

		// Check all stores, from last to first.
	memCheck:
		for i, v_elem := range order {
			v := v_elem.v
			info := pairableStores[v.Op]
			if info.width == 0 {
				continue // Not pairable.
			}
			if !offsetOk(v.Aux, v.AuxInt, info.width) {
				continue // Not advisable to pair.
			}
			ptr := v.Args[0]
			val := v.Args[1]
			mem := v.Args[2]
			off := v.AuxInt
			aux := v.Aux

			// Look for earlier store we can combine with.
			lowerOk := true
			higherOk := true
			count := limit // max lookback distance
			chain := order[i+1:]
			for _, w_elem := range chain {
				w := w_elem.v
				if w.Uses != 1 {
					// We can't combine stores if the earlier
					// store has any use besides the next one
					// in the store chain.
					// (Unless we could check the aliasing of
					// all those other uses.)
					continue memCheck
				}
				if w.Op == v.Op &&
					w.Args[0] == ptr &&
					w.Aux == aux &&
					(lowerOk && w.AuxInt == off-info.width || higherOk && w.AuxInt == off+info.width) {
					// This op is mergeable with v.

					// Commit point.

					// ptr val1 val2 mem
					args := []*Value{ptr, val, w.Args[1], mem}
					if w.AuxInt == off-info.width {
						args[1], args[2] = args[2], args[1]
						off -= info.width
					}

					v.reset(info.pair)
					v.AddArgs(args...)
					v.Aux = aux
					v.AuxInt = off
					// Take position of earlier of the two stores
					if v_elem.i < w_elem.i {
						v.Pos = w.Pos
					} else {
						w.Pos = v.Pos
					}

					// Make w just a memory copy.
					wmem := w.MemoryArg()
					w.reset(OpCopy)
					w.SetArgs1(wmem)
					continue memCheck
				}
				if count--; count == 0 {
					// Only look back so far.
					// This keeps us in O(n) territory, and it
					// also prevents us from keeping values
					// in registers for too long (and thus
					// needing to spill them).
					continue memCheck
				}
				// We're now looking at a store w which is currently
				// between the store v that we're intending to merge into,
				// and the store we'll eventually find to merge with it.
				// Make sure this store doesn't alias with the one
				// we'll be moving.
				var width int64
				switch w.Op {
				case OpARM64MOVDstore, OpARM64FMOVDstore:
					width = 8
				case OpARM64MOVWstore, OpARM64FMOVSstore:
					width = 4
				case OpARM64MOVHstore:
					width = 2
				case OpARM64MOVBstore:
					width = 1
				case OpCopy:
					continue // this was a store we merged earlier
				default:
					// Can't reorder with any other memory operations.
					// (atomics, calls, ...)
					continue memCheck
				}

				// We only allow reordering with respect to other
				// writes to the same pointer and aux, so we can
				// compute the exact the aliasing relationship.
				if w.Args[0] != ptr || w.Aux != aux {
					continue memCheck
				}
				if overlap(w.AuxInt, width, off-info.width, info.width) {
					// Aliases with slot before v's location.
					lowerOk = false
				}
				if overlap(w.AuxInt, width, off+info.width, info.width) {
					// Aliases with slot after v's location.
					higherOk = false
				}
				if !higherOk && !lowerOk {
					continue memCheck
				}
			}
		}
	}
}
