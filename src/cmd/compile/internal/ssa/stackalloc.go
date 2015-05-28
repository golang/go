package ssa

import "log"

// stackalloc allocates storage in the stack frame for
// all Values that did not get a register.
func stackalloc(f *Func) {
	home := f.RegAlloc

	// First compute the size of the outargs section.
	n := int64(16) //TODO: compute max of all callsites

	// Include one slot for deferreturn.
	if false && n < f.Config.ptrSize { //TODO: check for deferreturn
		n = f.Config.ptrSize
	}

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

	// TODO: align n
	n += f.Config.ptrSize // space for return address.  TODO: arch-dependent
	f.RegAlloc = home
	f.FrameSize = n

	// TODO: share stack slots among noninterfering (& gc type compatible) values

	// adjust all uses of FP to SP now that we have the frame size.
	var fp *Value
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op == OpFP {
				if fp != nil {
					log.Panicf("multiple FP ops: %s %s", fp, v)
				}
				fp = v
			}
			for i, a := range v.Args {
				if a.Op != OpFP {
					continue
				}
				// TODO: do this with arch-specific rewrite rules somehow?
				switch v.Op {
				case OpADDQ:
					// (ADDQ (FP) x) -> (LEAQ [n] (SP) x)
					v.Op = OpLEAQ
					v.Aux = n
				case OpLEAQ, OpMOVQload, OpMOVQstore, OpMOVBload, OpMOVQloadidx8:
					if v.Op == OpMOVQloadidx8 && i == 1 {
						// Note: we could do it, but it is probably an error
						log.Panicf("can't do FP->SP adjust on index slot of load %s", v.Op)
					}
					// eg: (MOVQload [c] (FP) mem) -> (MOVQload [c+n] (SP) mem)
					v.Aux = addOffset(v.Aux.(int64), n)
				default:
					log.Panicf("can't do FP->SP adjust on %s", v.Op)
				}
			}
		}
	}
	if fp != nil {
		fp.Op = OpSP
		home[fp.ID] = &registers[4] // TODO: arch-dependent
	}
}
