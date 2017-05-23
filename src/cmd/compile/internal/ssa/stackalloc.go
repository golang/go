// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: live at start of block instead?

package ssa

import "fmt"

type stackAllocState struct {
	f *Func

	// live is the output of stackalloc.
	// live[b.id] = live values at the end of block b.
	live [][]ID

	// The following slices are reused across multiple users
	// of stackAllocState.
	values    []stackValState
	interfere [][]ID // interfere[v.id] = values that interfere with v.
	names     []LocalSlot
	slots     []int
	used      []bool

	nArgSlot, // Number of Values sourced to arg slot
	nNotNeed, // Number of Values not needing a stack slot
	nNamedSlot, // Number of Values using a named stack slot
	nReuse, // Number of values reusing a stack slot
	nAuto, // Number of autos allocated for stack slots.
	nSelfInterfere int32 // Number of self-interferences
}

func newStackAllocState(f *Func) *stackAllocState {
	s := f.Config.stackAllocState
	if s == nil {
		return new(stackAllocState)
	}
	if s.f != nil {
		f.Config.Fatalf(0, "newStackAllocState called without previous free")
	}
	return s
}

func putStackAllocState(s *stackAllocState) {
	for i := range s.values {
		s.values[i] = stackValState{}
	}
	for i := range s.interfere {
		s.interfere[i] = nil
	}
	for i := range s.names {
		s.names[i] = LocalSlot{}
	}
	for i := range s.slots {
		s.slots[i] = 0
	}
	for i := range s.used {
		s.used[i] = false
	}
	s.f.Config.stackAllocState = s
	s.f = nil
	s.live = nil
	s.nArgSlot, s.nNotNeed, s.nNamedSlot, s.nReuse, s.nAuto, s.nSelfInterfere = 0, 0, 0, 0, 0, 0
}

type stackValState struct {
	typ      Type
	spill    *Value
	needSlot bool
}

// stackalloc allocates storage in the stack frame for
// all Values that did not get a register.
// Returns a map from block ID to the stack values live at the end of that block.
func stackalloc(f *Func, spillLive [][]ID) [][]ID {
	if f.pass.debug > stackDebug {
		fmt.Println("before stackalloc")
		fmt.Println(f.String())
	}
	s := newStackAllocState(f)
	s.init(f, spillLive)
	defer putStackAllocState(s)

	s.stackalloc()
	if f.pass.stats > 0 {
		f.LogStat("stack_alloc_stats",
			s.nArgSlot, "arg_slots", s.nNotNeed, "slot_not_needed",
			s.nNamedSlot, "named_slots", s.nAuto, "auto_slots",
			s.nReuse, "reused_slots", s.nSelfInterfere, "self_interfering")
	}

	return s.live
}

func (s *stackAllocState) init(f *Func, spillLive [][]ID) {
	s.f = f

	// Initialize value information.
	if n := f.NumValues(); cap(s.values) >= n {
		s.values = s.values[:n]
	} else {
		s.values = make([]stackValState, n)
	}
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			s.values[v.ID].typ = v.Type
			s.values[v.ID].needSlot = !v.Type.IsMemory() && !v.Type.IsVoid() && !v.Type.IsFlags() && f.getHome(v.ID) == nil && !v.rematerializeable()
			if f.pass.debug > stackDebug && s.values[v.ID].needSlot {
				fmt.Printf("%s needs a stack slot\n", v)
			}
			if v.Op == OpStoreReg {
				s.values[v.Args[0].ID].spill = v
			}
		}
	}

	// Compute liveness info for values needing a slot.
	s.computeLive(spillLive)

	// Build interference graph among values needing a slot.
	s.buildInterferenceGraph()
}

func (s *stackAllocState) stackalloc() {
	f := s.f

	// Build map from values to their names, if any.
	// A value may be associated with more than one name (e.g. after
	// the assignment i=j). This step picks one name per value arbitrarily.
	if n := f.NumValues(); cap(s.names) >= n {
		s.names = s.names[:n]
	} else {
		s.names = make([]LocalSlot, n)
	}
	names := s.names
	for _, name := range f.Names {
		// Note: not "range f.NamedValues" above, because
		// that would be nondeterministic.
		for _, v := range f.NamedValues[name] {
			names[v.ID] = name
		}
	}

	// Allocate args to their assigned locations.
	for _, v := range f.Entry.Values {
		if v.Op != OpArg {
			continue
		}
		loc := LocalSlot{v.Aux.(GCNode), v.Type, v.AuxInt}
		if f.pass.debug > stackDebug {
			fmt.Printf("stackalloc %s to %s\n", v, loc.Name())
		}
		f.setHome(v, loc)
	}

	// For each type, we keep track of all the stack slots we
	// have allocated for that type.
	// TODO: share slots among equivalent types. We would need to
	// only share among types with the same GC signature. See the
	// type.Equal calls below for where this matters.
	locations := map[Type][]LocalSlot{}

	// Each time we assign a stack slot to a value v, we remember
	// the slot we used via an index into locations[v.Type].
	slots := s.slots
	if n := f.NumValues(); cap(slots) >= n {
		slots = slots[:n]
	} else {
		slots = make([]int, n)
		s.slots = slots
	}
	for i := range slots {
		slots[i] = -1
	}

	// Pick a stack slot for each value needing one.
	var used []bool
	if n := f.NumValues(); cap(s.used) >= n {
		used = s.used[:n]
	} else {
		used = make([]bool, n)
		s.used = used
	}
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if !s.values[v.ID].needSlot {
				s.nNotNeed++
				continue
			}
			if v.Op == OpArg {
				s.nArgSlot++
				continue // already picked
			}

			// If this is a named value, try to use the name as
			// the spill location.
			var name LocalSlot
			if v.Op == OpStoreReg {
				name = names[v.Args[0].ID]
			} else {
				name = names[v.ID]
			}
			if name.N != nil && v.Type.Compare(name.Type) == CMPeq {
				for _, id := range s.interfere[v.ID] {
					h := f.getHome(id)
					if h != nil && h.(LocalSlot).N == name.N && h.(LocalSlot).Off == name.Off {
						// A variable can interfere with itself.
						// It is rare, but but it can happen.
						s.nSelfInterfere++
						goto noname
					}
				}
				if f.pass.debug > stackDebug {
					fmt.Printf("stackalloc %s to %s\n", v, name.Name())
				}
				s.nNamedSlot++
				f.setHome(v, name)
				continue
			}

		noname:
			// Set of stack slots we could reuse.
			locs := locations[v.Type]
			// Mark all positions in locs used by interfering values.
			for i := 0; i < len(locs); i++ {
				used[i] = false
			}
			for _, xid := range s.interfere[v.ID] {
				slot := slots[xid]
				if slot >= 0 {
					used[slot] = true
				}
			}
			// Find an unused stack slot.
			var i int
			for i = 0; i < len(locs); i++ {
				if !used[i] {
					s.nReuse++
					break
				}
			}
			// If there is no unused stack slot, allocate a new one.
			if i == len(locs) {
				s.nAuto++
				locs = append(locs, LocalSlot{N: f.Config.fe.Auto(v.Type), Type: v.Type, Off: 0})
				locations[v.Type] = locs
			}
			// Use the stack variable at that index for v.
			loc := locs[i]
			if f.pass.debug > stackDebug {
				fmt.Printf("stackalloc %s to %s\n", v, loc.Name())
			}
			f.setHome(v, loc)
			slots[v.ID] = i
		}
	}
}

// computeLive computes a map from block ID to a list of
// stack-slot-needing value IDs live at the end of that block.
// TODO: this could be quadratic if lots of variables are live across lots of
// basic blocks. Figure out a way to make this function (or, more precisely, the user
// of this function) require only linear size & time.
func (s *stackAllocState) computeLive(spillLive [][]ID) {
	s.live = make([][]ID, s.f.NumBlocks())
	var phis []*Value
	live := s.f.newSparseSet(s.f.NumValues())
	defer s.f.retSparseSet(live)
	t := s.f.newSparseSet(s.f.NumValues())
	defer s.f.retSparseSet(t)

	// Instead of iterating over f.Blocks, iterate over their postordering.
	// Liveness information flows backward, so starting at the end
	// increases the probability that we will stabilize quickly.
	po := postorder(s.f)
	for {
		changed := false
		for _, b := range po {
			// Start with known live values at the end of the block
			live.clear()
			live.addAll(s.live[b.ID])

			// Propagate backwards to the start of the block
			phis = phis[:0]
			for i := len(b.Values) - 1; i >= 0; i-- {
				v := b.Values[i]
				live.remove(v.ID)
				if v.Op == OpPhi {
					// Save phi for later.
					// Note: its args might need a stack slot even though
					// the phi itself doesn't. So don't use needSlot.
					if !v.Type.IsMemory() && !v.Type.IsVoid() {
						phis = append(phis, v)
					}
					continue
				}
				for _, a := range v.Args {
					if s.values[a.ID].needSlot {
						live.add(a.ID)
					}
				}
			}

			// for each predecessor of b, expand its list of live-at-end values
			// invariant: s contains the values live at the start of b (excluding phi inputs)
			for i, e := range b.Preds {
				p := e.b
				t.clear()
				t.addAll(s.live[p.ID])
				t.addAll(live.contents())
				t.addAll(spillLive[p.ID])
				for _, v := range phis {
					a := v.Args[i]
					if s.values[a.ID].needSlot {
						t.add(a.ID)
					}
					if spill := s.values[a.ID].spill; spill != nil {
						//TODO: remove?  Subsumed by SpillUse?
						t.add(spill.ID)
					}
				}
				if t.size() == len(s.live[p.ID]) {
					continue
				}
				// grow p's live set
				s.live[p.ID] = append(s.live[p.ID][:0], t.contents()...)
				changed = true
			}
		}

		if !changed {
			break
		}
	}
	if s.f.pass.debug > stackDebug {
		for _, b := range s.f.Blocks {
			fmt.Printf("stacklive %s %v\n", b, s.live[b.ID])
		}
	}
}

func (f *Func) getHome(vid ID) Location {
	if int(vid) >= len(f.RegAlloc) {
		return nil
	}
	return f.RegAlloc[vid]
}

func (f *Func) setHome(v *Value, loc Location) {
	for v.ID >= ID(len(f.RegAlloc)) {
		f.RegAlloc = append(f.RegAlloc, nil)
	}
	f.RegAlloc[v.ID] = loc
}

func (s *stackAllocState) buildInterferenceGraph() {
	f := s.f
	if n := f.NumValues(); cap(s.interfere) >= n {
		s.interfere = s.interfere[:n]
	} else {
		s.interfere = make([][]ID, n)
	}
	live := f.newSparseSet(f.NumValues())
	defer f.retSparseSet(live)
	for _, b := range f.Blocks {
		// Propagate liveness backwards to the start of the block.
		// Two values interfere if one is defined while the other is live.
		live.clear()
		live.addAll(s.live[b.ID])
		for i := len(b.Values) - 1; i >= 0; i-- {
			v := b.Values[i]
			if s.values[v.ID].needSlot {
				live.remove(v.ID)
				for _, id := range live.contents() {
					if s.values[v.ID].typ.Compare(s.values[id].typ) == CMPeq {
						s.interfere[v.ID] = append(s.interfere[v.ID], id)
						s.interfere[id] = append(s.interfere[id], v.ID)
					}
				}
			}
			for _, a := range v.Args {
				if s.values[a.ID].needSlot {
					live.add(a.ID)
				}
			}
			if v.Op == OpArg && s.values[v.ID].needSlot {
				// OpArg is an input argument which is pre-spilled.
				// We add back v.ID here because we want this value
				// to appear live even before this point. Being live
				// all the way to the start of the entry block prevents other
				// values from being allocated to the same slot and clobbering
				// the input value before we have a chance to load it.
				live.add(v.ID)
			}
		}
	}
	if f.pass.debug > stackDebug {
		for vid, i := range s.interfere {
			if len(i) > 0 {
				fmt.Printf("v%d interferes with", vid)
				for _, x := range i {
					fmt.Printf(" v%d", x)
				}
				fmt.Println()
			}
		}
	}
}
