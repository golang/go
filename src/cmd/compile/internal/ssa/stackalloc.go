// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: live at start of block instead?

package ssa

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"fmt"
)

type StackAllocState struct {
	f *Func

	// Live is the output of stackalloc.
	// Live[b.id] = Live values at the end of block b.
	Live [][]ID

	// The following slices are reused across multiple users
	// of stackAllocState.
	values    []stackValState
	interfere [][]ID // interfere[v.id] = values that interfere with v.
	names     []LocalSlot

	NArgSlot, // Number of Values sourced to arg slot
	NNotNeed, // Number of Values not needing a stack slot
	NNamedSlot, // Number of Values using a named stack slot
	NReuse, // Number of values reusing a stack slot
	NAuto, // Number of autos allocated for stack slots.
	NSelfInterfere int32 // Number of self-interferences
}

func NewStackAllocState(f *Func) *StackAllocState {
	s := f.Cache.stackAllocState
	if s == nil {
		return new(StackAllocState)
	}
	if s.f != nil {
		f.Fe.Fatalf(src.NoXPos, "newStackAllocState called without previous free")
	}
	return s
}

func PutStackAllocState(s *StackAllocState) {
	clear(s.values)
	clear(s.interfere)
	clear(s.names)
	s.f.Cache.stackAllocState = s
	s.f = nil
	s.Live = nil
	s.NArgSlot, s.NNotNeed, s.NNamedSlot, s.NReuse, s.NAuto, s.NSelfInterfere = 0, 0, 0, 0, 0, 0
}

type stackValState struct {
	typ       *types.Type
	spill     *Value
	needSlot  bool
	isArg     bool
	defBlock  ID
	useBlocks []stackUseBlock
}

// addUseBlock adds a block to the set of blocks that uses this value.
// Note that we only loosely enforce the set property by checking the last
// block that was appended to the list and duplicates may occur.
// Because we add values block by block (barring phi-nodes), the number of duplicates is
// small and we deduplicate as part of the liveness algorithm later anyway.
func (sv *stackValState) addUseBlock(b *Block, liveout bool) {
	entry := stackUseBlock{
		b:       b,
		liveout: liveout,
	}
	if sv.useBlocks == nil || sv.useBlocks[len(sv.useBlocks)-1] != entry {
		sv.useBlocks = append(sv.useBlocks, stackUseBlock{
			b:       b,
			liveout: liveout,
		})
	}
}

type stackUseBlock struct {
	b       *Block
	liveout bool
}

// stackalloc allocates storage in the stack frame for
// all Values that did not get a register.
// Returns a map from block ID to the stack values live at the end of that block.
func stackalloc(f *Func, spillLive [][]ID) [][]ID {
	if f.Pass.Debug > StackDebug {
		fmt.Println("before stackalloc")
		fmt.Println(f.String())
	}
	s := NewStackAllocState(f)
	s.Init(f, spillLive)
	defer PutStackAllocState(s)

	s.Stackalloc()
	if f.Pass.Stats > 0 {
		f.LogStat("stack_alloc_stats",
			s.NArgSlot, "arg_slots", s.NNotNeed, "slot_not_needed",
			s.NNamedSlot, "named_slots", s.NAuto, "auto_slots",
			s.NReuse, "reused_slots", s.NSelfInterfere, "self_interfering")
	}

	return s.Live
}

func (s *StackAllocState) Init(f *Func, spillLive [][]ID) {
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
			s.values[v.ID].needSlot = !v.Type.IsMemory() && !v.Type.IsVoid() && !v.Type.IsFlags() && f.GetHome(v.ID) == nil && !v.Rematerializeable() && !v.OnWasmStack
			s.values[v.ID].isArg = hasAnyArgOp(v)
			s.values[v.ID].defBlock = b.ID
			if f.Pass.Debug > StackDebug && s.values[v.ID].needSlot {
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

func (s *StackAllocState) Stackalloc() {
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
	empty := LocalSlot{}
	for _, name := range f.Names {
		// Note: not "range f.NamedValues" above, because
		// that would be nondeterministic.
		for _, v := range f.NamedValues[name] {
			if v.Op == OpArgIntReg || v.Op == OpArgFloatReg {
				aux := v.Aux.(*AuxNameOffset)
				// Never let an arg be bound to a differently named thing.
				if name.N != aux.Name || name.Off != aux.Offset {
					if f.Pass.Debug > StackDebug {
						fmt.Printf("stackalloc register arg %s skipping name %s\n", v, name)
					}
					continue
				}
			} else if name.N.Class == ir.PPARAM && v.Op != OpArg {
				// PPARAM's only bind to OpArg
				if f.Pass.Debug > StackDebug {
					fmt.Printf("stackalloc PPARAM name %s skipping non-Arg %s\n", name, v)
				}
				continue
			}

			if names[v.ID] == empty {
				if f.Pass.Debug > StackDebug {
					fmt.Printf("stackalloc value %s to name %s\n", v, name)
				}
				names[v.ID] = name
			}
		}
	}

	// Allocate args to their assigned locations.
	for _, v := range f.Entry.Values {
		if !hasAnyArgOp(v) {
			continue
		}
		if v.Aux == nil {
			f.Fatalf("%s has nil Aux\n", v.LongString())
		}
		if v.Op == OpArg {
			loc := LocalSlot{N: v.Aux.(*ir.Name), Type: v.Type, Off: v.AuxInt}
			if f.Pass.Debug > StackDebug {
				fmt.Printf("stackalloc OpArg %s to %s\n", v, loc)
			}
			f.SetHome(v, loc)
			continue
		}
		// You might think this below would be the right idea, but you would be wrong.
		// It almost works; as of 105a6e9518 - 2021-04-23,
		// GOSSAHASH=11011011001011111 == cmd/compile/internal/noder.(*noder).embedded
		// is compiled incorrectly.  I believe the cause is one of those SSA-to-registers
		// puzzles that the register allocator untangles; in the event that a register
		// parameter does not end up bound to a name, "fixing" it is a bad idea.
		//
		//if f.DebugTest {
		//	if v.Op == OpArgIntReg || v.Op == OpArgFloatReg {
		//		aux := v.Aux.(*AuxNameOffset)
		//		loc := LocalSlot{N: aux.Name, Type: v.Type, Off: aux.Offset}
		//		if f.pass.debug > stackDebug {
		//			fmt.Printf("stackalloc Op%s %s to %s\n", v.Op, v, loc)
		//		}
		//		names[v.ID] = loc
		//		continue
		//	}
		//}

	}

	// For each type, we keep track of all the stack slots we
	// have allocated for that type. This map is keyed by
	// strings returned by types.LinkString. This guarantees
	// type equality, but also lets us match the same type represented
	// by two different types.Type structures. See issue 65783.
	locations := map[string][]LocalSlot{}

	// Each time we assign a stack slot to a value v, we remember
	// the slot we used via an index into locations[v.Type].
	slots := f.Cache.AllocIntSlice(f.NumValues())
	defer f.Cache.FreeIntSlice(slots)
	for i := range slots {
		slots[i] = -1
	}

	// Pick a stack slot for each value needing one.
	used := f.Cache.AllocBoolSlice(f.NumValues())
	defer f.Cache.FreeBoolSlice(used)
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if !s.values[v.ID].needSlot {
				s.NNotNeed++
				continue
			}
			if hasAnyArgOp(v) {
				s.NArgSlot++
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
			if name.N != nil && v.Type.Compare(name.Type) == types.CMPeq {
				for _, id := range s.interfere[v.ID] {
					h := f.GetHome(id)
					if h != nil && h.(LocalSlot).N == name.N && h.(LocalSlot).Off == name.Off {
						// A variable can interfere with itself.
						// It is rare, but it can happen.
						s.NSelfInterfere++
						goto noname
					}
				}
				if f.Pass.Debug > StackDebug {
					fmt.Printf("stackalloc %s to %s\n", v, name)
				}
				s.NNamedSlot++
				f.SetHome(v, name)
				continue
			}

		noname:
			// Set of stack slots we could reuse.
			typeKey := v.Type.LinkString()
			locs := locations[typeKey]
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
					s.NReuse++
					break
				}
			}
			// If there is no unused stack slot, allocate a new one.
			if i == len(locs) {
				s.NAuto++
				locs = append(locs, LocalSlot{N: f.NewLocal(v.Pos, v.Type), Type: v.Type, Off: 0})
				locations[typeKey] = locs
			}
			// Use the stack variable at that index for v.
			loc := locs[i]
			if f.Pass.Debug > StackDebug {
				fmt.Printf("stackalloc %s to %s\n", v, loc)
			}
			f.SetHome(v, loc)
			slots[v.ID] = i
		}
	}
}

// computeLive computes a map from block ID to a list of
// stack-slot-needing value IDs live at the end of that block.
func (s *StackAllocState) computeLive(spillLive [][]ID) {

	// Because values using stack slots are few and far inbetween
	// (compared to the set of all values), we use a path exploration
	// algorithm to calculate liveness here.
	f := s.f
	for _, b := range f.Blocks {
		for _, spillvid := range spillLive[b.ID] {
			val := &s.values[spillvid]
			val.addUseBlock(b, true)
		}
		for _, v := range b.Values {
			for i, a := range v.Args {
				val := &s.values[a.ID]
				useBlock := b
				forceLiveout := false
				if v.Op == OpPhi {
					useBlock = b.Preds[i].B
					forceLiveout = true
					if spill := val.spill; spill != nil {
						//TODO: remove?  Subsumed by SpillUse?
						s.values[spill.ID].addUseBlock(useBlock, true)
					}
				}
				if !val.needSlot {
					continue
				}
				val.addUseBlock(useBlock, forceLiveout)
			}
		}
	}

	s.Live = make([][]ID, f.NumBlocks())
	push := func(bid, vid ID) {
		l := s.Live[bid]
		if l == nil || l[len(l)-1] != vid {
			l = append(l, vid)
			s.Live[bid] = l
		}
	}
	// TODO: If we can help along the interference graph by calculating livein sets,
	// we can do so trivially by turning this sparse set into an array of arrays
	// and checking the top for the current value instead of inclusion in the sparse set.
	seen := f.NewSparseSet(f.NumBlocks())
	defer f.RetSparseSet(seen)
	// instead of pruning out duplicate blocks when we build the useblocks slices
	// or when we add them to the queue, rely on the seen set to stop considering
	// them. This is slightly faster than building the workqueues as sets
	//
	// However, this means that the queue can grow larger than the number of blocks,
	// usually in very short functions. Returning a slice with values appended beyond the
	// original allocation can corrupt the allocator state, so cap the queue and return
	// the originally allocated slice regardless.
	allocedBqueue := f.Cache.AllocBlockSlice(f.NumBlocks())
	defer f.Cache.FreeBlockSlice(allocedBqueue)
	bqueue := allocedBqueue[:0:f.NumBlocks()]

	for vid, v := range s.values {
		if !v.needSlot {
			continue
		}
		seen.Clear()
		bqueue = bqueue[:0]
		for _, b := range v.useBlocks {
			if b.liveout {
				push(b.b.ID, ID(vid))
			}
			bqueue = append(bqueue, b.b)
		}
		for len(bqueue) > 0 {
			work := bqueue[len(bqueue)-1]
			bqueue = bqueue[:len(bqueue)-1]
			if seen.Contains(work.ID) || work.ID == v.defBlock {
				continue
			}
			seen.Add(work.ID)
			for _, e := range work.Preds {
				push(e.B.ID, ID(vid))
				bqueue = append(bqueue, e.B)
			}
		}
	}

	if s.f.Pass.Debug > StackDebug {
		for _, b := range s.f.Blocks {
			fmt.Printf("stacklive %s %v\n", b, s.Live[b.ID])
		}
	}
}

func (f *Func) GetHome(vid ID) Location {
	if int(vid) >= len(f.RegAlloc) {
		return nil
	}
	return f.RegAlloc[vid]
}

func (f *Func) SetHome(v *Value, loc Location) {
	for v.ID >= ID(len(f.RegAlloc)) {
		f.RegAlloc = append(f.RegAlloc, nil)
	}
	f.RegAlloc[v.ID] = loc
}

func (s *StackAllocState) buildInterferenceGraph() {
	f := s.f
	if n := f.NumValues(); cap(s.interfere) >= n {
		s.interfere = s.interfere[:n]
	} else {
		s.interfere = make([][]ID, n)
	}
	live := f.NewSparseSet(f.NumValues())
	defer f.RetSparseSet(live)
	for _, b := range f.Blocks {
		// Propagate liveness backwards to the start of the block.
		// Two values interfere if one is defined while the other is live.
		live.Clear()
		live.addAll(s.Live[b.ID])
		for i := len(b.Values) - 1; i >= 0; i-- {
			v := b.Values[i]
			if s.values[v.ID].needSlot {
				live.Remove(v.ID)
				for _, id := range live.Contents() {
					// Note: args can have different types and still interfere
					// (with each other or with other values). See issue 23522.
					if s.values[v.ID].typ.Compare(s.values[id].typ) == types.CMPeq || hasAnyArgOp(v) || s.values[id].isArg {
						s.interfere[v.ID] = append(s.interfere[v.ID], id)
						s.interfere[id] = append(s.interfere[id], v.ID)
					}
				}
			}
			for _, a := range v.Args {
				if s.values[a.ID].needSlot {
					live.Add(a.ID)
				}
			}
			if hasAnyArgOp(v) && s.values[v.ID].needSlot {
				// OpArg is an input argument which is pre-spilled.
				// We add back v.ID here because we want this value
				// to appear live even before this point. Being live
				// all the way to the start of the entry block prevents other
				// values from being allocated to the same slot and clobbering
				// the input value before we have a chance to load it.

				// TODO(register args) this is apparently not wrong for register args -- is it necessary?
				live.Add(v.ID)
			}
		}
	}
	if f.Pass.Debug > StackDebug {
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

func hasAnyArgOp(v *Value) bool {
	return v.Op == OpArg || v.Op == OpArgIntReg || v.Op == OpArgFloatReg
}
