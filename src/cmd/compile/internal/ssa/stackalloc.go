// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// stackalloc allocates storage in the stack frame for
// all Values that did not get a register.
func stackalloc(f *Func) {
	home := f.RegAlloc

	// Start with space for callee arguments/returns.
	var n int64
	for _, b := range f.Blocks {
		if b.Kind != BlockCall {
			continue
		}
		v := b.Control
		if n < v.AuxInt {
			n = v.AuxInt
		}
	}
	f.Logf("stackalloc: 0-%d for callee arguments/returns\n", n)

	// TODO: group variables by ptr/nonptr, size, etc.  Emit ptr vars last
	// so stackmap is smaller.

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
			n = align(n, v.Type.Alignment())
			f.Logf("stackalloc: %d-%d for %v\n", n, n+v.Type.Size(), v)
			loc := &LocalSlot{n}
			n += v.Type.Size()
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
			if len(v.Args) == 1 && (v.Args[0].Op == OpSP || v.Args[0].Op == OpSB) {
				continue
			}
			n = align(n, v.Type.Alignment())
			f.Logf("stackalloc: %d-%d for %v\n", n, n+v.Type.Size(), v)
			loc := &LocalSlot{n}
			n += v.Type.Size()
			home = setloc(home, v, loc)
		}
	}

	// Finally, allocate space for all autos that we used
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			s, ok := v.Aux.(*AutoSymbol)
			if !ok || s.Offset >= 0 {
				continue
			}
			t := s.Typ
			n = align(n, t.Alignment())
			f.Logf("stackalloc: %d-%d for auto %v\n", n, n+t.Size(), v)
			s.Offset = n
			n += t.Size()
		}
	}

	n = align(n, f.Config.PtrSize)
	f.Logf("stackalloc: %d-%d for return address\n", n, n+f.Config.PtrSize)
	n += f.Config.PtrSize // space for return address.  TODO: arch-dependent
	f.RegAlloc = home
	f.FrameSize = n

	// TODO: share stack slots among noninterfering (& gc type compatible) values
}

// align increases n to the next multiple of a.  a must be a power of 2.
func align(n int64, a int64) int64 {
	if a == 0 {
		return n
	}
	return (n + a - 1) &^ (a - 1)
}
