// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// dse does dead-store elimination on the Function.
// Dead stores are those which are unconditionally followed by
// another store to the same location, with no intervening load.
// This implementation only works within a basic block. TODO: use something more global.
func dse(f *Func) {
	var stores []*Value
	loadUse := f.newSparseSet(f.NumValues())
	defer f.retSparseSet(loadUse)
	storeUse := f.newSparseSet(f.NumValues())
	defer f.retSparseSet(storeUse)
	shadowed := newSparseMap(f.NumValues()) // TODO: cache
	for _, b := range f.Blocks {
		// Find all the stores in this block. Categorize their uses:
		//  loadUse contains stores which are used by a subsequent load.
		//  storeUse contains stores which are used by a subsequent store.
		loadUse.clear()
		storeUse.clear()
		stores = stores[:0]
		for _, v := range b.Values {
			if v.Op == OpPhi {
				// Ignore phis - they will always be first and can't be eliminated
				continue
			}
			if v.Type.IsMemory() {
				stores = append(stores, v)
				if v.Op == OpSelect1 {
					// Use the args of the tuple-generating op.
					v = v.Args[0]
				}
				for _, a := range v.Args {
					if a.Block == b && a.Type.IsMemory() {
						storeUse.add(a.ID)
						if v.Op != OpStore && v.Op != OpZero && v.Op != OpVarDef && v.Op != OpVarKill {
							// CALL, DUFFCOPY, etc. are both
							// reads and writes.
							loadUse.add(a.ID)
						}
					}
				}
			} else {
				for _, a := range v.Args {
					if a.Block == b && a.Type.IsMemory() {
						loadUse.add(a.ID)
					}
				}
			}
		}
		if len(stores) == 0 {
			continue
		}

		// find last store in the block
		var last *Value
		for _, v := range stores {
			if storeUse.contains(v.ID) {
				continue
			}
			if last != nil {
				b.Fatalf("two final stores - simultaneous live stores %s %s", last, v)
			}
			last = v
		}
		if last == nil {
			b.Fatalf("no last store found - cycle?")
		}

		// Walk backwards looking for dead stores. Keep track of shadowed addresses.
		// An "address" is an SSA Value which encodes both the address and size of
		// the write. This code will not remove dead stores to the same address
		// of different types.
		shadowed.clear()
		v := last

	walkloop:
		if loadUse.contains(v.ID) {
			// Someone might be reading this memory state.
			// Clear all shadowed addresses.
			shadowed.clear()
		}
		if v.Op == OpStore || v.Op == OpZero {
			var sz int64
			if v.Op == OpStore {
				sz = v.AuxInt
			} else { // OpZero
				sz = SizeAndAlign(v.AuxInt).Size()
			}
			if shadowedSize := int64(shadowed.get(v.Args[0].ID)); shadowedSize != -1 && shadowedSize >= sz {
				// Modify store into a copy
				if v.Op == OpStore {
					// store addr value mem
					v.SetArgs1(v.Args[2])
				} else {
					// zero addr mem
					typesz := v.Args[0].Type.ElemType().Size()
					if sz != typesz {
						f.Fatalf("mismatched zero/store sizes: %d and %d [%s]",
							sz, typesz, v.LongString())
					}
					v.SetArgs1(v.Args[1])
				}
				v.Aux = nil
				v.AuxInt = 0
				v.Op = OpCopy
			} else {
				if sz > 0x7fffffff { // work around sparseMap's int32 value type
					sz = 0x7fffffff
				}
				shadowed.set(v.Args[0].ID, int32(sz), 0)
			}
		}
		// walk to previous store
		if v.Op == OpPhi {
			continue // At start of block.  Move on to next block.
		}
		for _, a := range v.Args {
			if a.Block == b && a.Type.IsMemory() {
				v = a
				goto walkloop
			}
		}
	}
}
