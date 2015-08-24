// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// setloc sets the home location of v to loc.
func setloc(home []Location, v *Value, loc Location) []Location {
	for v.ID >= ID(len(home)) {
		home = append(home, nil)
	}
	home[v.ID] = loc
	return home
}

// stackalloc allocates storage in the stack frame for
// all Values that did not get a register.
func stackalloc(f *Func) {
	home := f.RegAlloc

	// Assign stack locations to phis first, because we
	// must also assign the same locations to the phi stores
	// introduced during regalloc.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op != OpPhi {
				continue
			}
			if v.Type.IsMemory() { // TODO: only "regallocable" types
				continue
			}
			if int(v.ID) < len(home) && home[v.ID] != nil {
				continue // register-based phi
			}
			// stack-based phi
			n := f.Config.fe.Auto(v.Type)
			f.Logf("stackalloc: %s: for %v <%v>\n", n, v, v.Type)
			loc := &LocalSlot{n}
			home = setloc(home, v, loc)
			for _, w := range v.Args {
				if w.Op != OpStoreReg {
					f.Fatalf("stack-based phi must have StoreReg args")
				}
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
			if len(v.Args) == 1 && (v.Args[0].Op == OpSP || v.Args[0].Op == OpSB) {
				continue
			}

			n := f.Config.fe.Auto(v.Type)
			f.Logf("stackalloc: %s for %v\n", n, v)
			loc := &LocalSlot{n}
			home = setloc(home, v, loc)
		}
	}

	f.RegAlloc = home

	// TODO: share stack slots among noninterfering (& gc type compatible) values
}

// align increases n to the next multiple of a.  a must be a power of 2.
func align(n int64, a int64) int64 {
	if a == 0 {
		return n
	}
	return (n + a - 1) &^ (a - 1)
}
