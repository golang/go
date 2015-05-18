package ssa

// stackalloc allocates storage in the stack frame for
// all Values that did not get a register.
func stackalloc(f *Func) {
	home := f.RegAlloc

	var n int64 = 8 // 8 = space for return address.  TODO: arch-dependent

	// Assign stack locations to phis first, because we
	// must also assign the same locations to the phi copies
	// introduced during regalloc.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op != OpPhi {
				continue
			}
			if v.Type.IsMemory() { // TODO: only "regallocable" types
				continue
			}
			n += v.Type.Size()
			// a := v.Type.Align()
			// n = (n + a - 1) / a * a  TODO
			loc := &LocalSlot{n}
			home = setloc(home, v, loc)
			for _, w := range v.Args {
				home = setloc(home, w, loc)
			}
		}
	}

	// Now do all other unassigned values.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.ID < ID(len(home)) && home[v.ID] != nil {
				continue
			}
			if v.Type.IsMemory() { // TODO: only "regallocable" types
				continue
			}
			if len(v.Args) == 0 {
				// v will have been materialized wherever it is needed.
				continue
			}
			if len(v.Args) == 1 && (v.Args[0].Op == OpFP || v.Args[0].Op == OpSP || v.Args[0].Op == OpGlobal) {
				continue
			}
			// a := v.Type.Align()
			// n = (n + a - 1) / a * a  TODO
			n += v.Type.Size()
			loc := &LocalSlot{n}
			home = setloc(home, v, loc)
		}
	}
	f.RegAlloc = home

	// TODO: share stack slots among noninterfering (& gc type compatible) values
	// TODO: align final n
	// TODO: compute total frame size: n + max paramout space
	// TODO: save total size somewhere
}
