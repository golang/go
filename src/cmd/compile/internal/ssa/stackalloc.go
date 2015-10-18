// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: live at start of block instead?

package ssa

// stackalloc allocates storage in the stack frame for
// all Values that did not get a register.
func stackalloc(f *Func) {
	// Cache value types by ID.
	types := make([]Type, f.NumValues())
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			types[v.ID] = v.Type
		}
	}

	// Build interference graph among StoreReg and stack phi ops.
	live := f.liveSpills()
	interfere := make([][]ID, f.NumValues())
	s := newSparseSet(f.NumValues())
	for _, b := range f.Blocks {
		// Start with known live values at the end of the block.
		s.clear()
		for i := 0; i < len(b.Succs); i++ {
			s.addAll(live[b.ID][i])
		}

		// Propagate backwards to the start of the block.
		// Remember interfering sets.
		for i := len(b.Values) - 1; i >= 0; i-- {
			v := b.Values[i]
			switch {
			case v.Op == OpStoreReg, v.isStackPhi():
				s.remove(v.ID)
				for _, id := range s.contents() {
					if v.Type == types[id] {
						interfere[v.ID] = append(interfere[v.ID], id)
						interfere[id] = append(interfere[id], v.ID)
					}
				}
			case v.Op == OpLoadReg:
				s.add(v.Args[0].ID)
			}
		}
	}

	// Figure out which StoreReg ops are phi args.  We don't pick slots for
	// phi args because a stack phi and its args must all use the same stack slot.
	phiArg := make([]bool, f.NumValues())
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if !v.isStackPhi() {
				continue
			}
			for _, a := range v.Args {
				phiArg[a.ID] = true
			}
		}
	}

	// For each type, we keep track of all the stack slots we
	// have allocated for that type.
	locations := map[Type][]*LocalSlot{}

	// Each time we assign a stack slot to a value v, we remember
	// the slot we used via an index into locations[v.Type].
	slots := make([]int, f.NumValues())
	for i := f.NumValues() - 1; i >= 0; i-- {
		slots[i] = -1
	}

	// Pick a stack slot for each non-phi-arg StoreReg and each stack phi.
	used := make([]bool, f.NumValues())
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op != OpStoreReg && !v.isStackPhi() {
				continue
			}
			if phiArg[v.ID] {
				continue
			}
			// Set of stack slots we could reuse.
			locs := locations[v.Type]
			// Mark all positions in locs used by interfering values.
			for i := 0; i < len(locs); i++ {
				used[i] = false
			}
			for _, xid := range interfere[v.ID] {
				slot := slots[xid]
				if slot >= 0 {
					used[slot] = true
				}
			}
			if v.Op == OpPhi {
				// Stack phi and args must get the same stack slot, so
				// anything they interfere with is something v the phi
				// interferes with.
				for _, a := range v.Args {
					for _, xid := range interfere[a.ID] {
						slot := slots[xid]
						if slot >= 0 {
							used[slot] = true
						}
					}
				}
			}
			// Find an unused stack slot.
			var i int
			for i = 0; i < len(locs); i++ {
				if !used[i] {
					break
				}
			}
			// If there is no unused stack slot, allocate a new one.
			if i == len(locs) {
				locs = append(locs, &LocalSlot{f.Config.fe.Auto(v.Type)})
				locations[v.Type] = locs
			}
			// Use the stack variable at that index for v.
			loc := locs[i]
			f.setHome(v, loc)
			slots[v.ID] = i
			if v.Op == OpPhi {
				for _, a := range v.Args {
					f.setHome(a, loc)
					slots[a.ID] = i
				}
			}
		}
	}
}

// live returns a map from block ID and successor edge index to a list
// of StoreReg/stackphi value IDs live on that edge.
// TODO: this could be quadratic if lots of variables are live across lots of
// basic blocks.  Figure out a way to make this function (or, more precisely, the user
// of this function) require only linear size & time.
func (f *Func) liveSpills() [][][]ID {
	live := make([][][]ID, f.NumBlocks())
	for _, b := range f.Blocks {
		live[b.ID] = make([][]ID, len(b.Succs))
	}
	var phis []*Value

	s := newSparseSet(f.NumValues())
	t := newSparseSet(f.NumValues())

	// Instead of iterating over f.Blocks, iterate over their postordering.
	// Liveness information flows backward, so starting at the end
	// increases the probability that we will stabilize quickly.
	po := postorder(f)
	for {
		changed := false
		for _, b := range po {
			// Start with known live values at the end of the block
			s.clear()
			for i := 0; i < len(b.Succs); i++ {
				s.addAll(live[b.ID][i])
			}

			// Propagate backwards to the start of the block
			phis = phis[:0]
			for i := len(b.Values) - 1; i >= 0; i-- {
				v := b.Values[i]
				switch {
				case v.Op == OpStoreReg:
					s.remove(v.ID)
				case v.Op == OpLoadReg:
					s.add(v.Args[0].ID)
				case v.isStackPhi():
					s.remove(v.ID)
					// save stack phi ops for later
					phis = append(phis, v)
				}
			}

			// for each predecessor of b, expand its list of live-at-end values
			// invariant: s contains the values live at the start of b (excluding phi inputs)
			for i, p := range b.Preds {
				// Find index of b in p's successors.
				var j int
				for j = 0; j < len(p.Succs); j++ {
					if p.Succs[j] == b {
						break
					}
				}
				t.clear()
				t.addAll(live[p.ID][j])
				t.addAll(s.contents())
				for _, v := range phis {
					t.add(v.Args[i].ID)
				}
				if t.size() == len(live[p.ID][j]) {
					continue
				}
				// grow p's live set
				live[p.ID][j] = append(live[p.ID][j][:0], t.contents()...)
				changed = true
			}
		}

		if !changed {
			break
		}
	}
	return live
}

func (f *Func) getHome(v *Value) Location {
	if int(v.ID) >= len(f.RegAlloc) {
		return nil
	}
	return f.RegAlloc[v.ID]
}

func (f *Func) setHome(v *Value, loc Location) {
	for v.ID >= ID(len(f.RegAlloc)) {
		f.RegAlloc = append(f.RegAlloc, nil)
	}
	f.RegAlloc[v.ID] = loc
}

func (v *Value) isStackPhi() bool {
	if v.Op != OpPhi {
		return false
	}
	if v.Type == TypeMem {
		return false
	}
	if int(v.ID) >= len(v.Block.Func.RegAlloc) {
		return true
	}
	return v.Block.Func.RegAlloc[v.ID] == nil
	// TODO: use a separate opcode for StackPhi?
}
