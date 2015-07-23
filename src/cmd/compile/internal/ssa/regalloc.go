// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "sort"

func setloc(home []Location, v *Value, loc Location) []Location {
	for v.ID >= ID(len(home)) {
		home = append(home, nil)
	}
	home[v.ID] = loc
	return home
}

type register uint

type regMask uint64

// TODO: make arch-dependent
var numRegs register = 64

var registers = [...]Register{
	Register{0, "AX"},
	Register{1, "CX"},
	Register{2, "DX"},
	Register{3, "BX"},
	Register{4, "SP"},
	Register{5, "BP"},
	Register{6, "SI"},
	Register{7, "DI"},
	Register{8, "R8"},
	Register{9, "R9"},
	Register{10, "R10"},
	Register{11, "R11"},
	Register{12, "R12"},
	Register{13, "R13"},
	Register{14, "R14"},
	Register{15, "R15"},
	Register{16, "X0"},
	Register{17, "X1"},
	Register{18, "X2"},
	Register{19, "X3"},
	Register{20, "X4"},
	Register{21, "X5"},
	Register{22, "X6"},
	Register{23, "X7"},
	Register{24, "X8"},
	Register{25, "X9"},
	Register{26, "X10"},
	Register{27, "X11"},
	Register{28, "X12"},
	Register{29, "X13"},
	Register{30, "X14"},
	Register{31, "X15"},
	Register{32, "SB"}, // pseudo-register for global base pointer (aka %rip)
	Register{33, "FLAGS"},

	// TODO: make arch-dependent
}

// countRegs returns the number of set bits in the register mask.
func countRegs(r regMask) int {
	n := 0
	for r != 0 {
		n += int(r & 1)
		r >>= 1
	}
	return n
}

// pickReg picks an arbitrary register from the register mask.
func pickReg(r regMask) register {
	// pick the lowest one
	if r == 0 {
		panic("can't pick a register from an empty set")
	}
	for i := register(0); ; i++ {
		if r&1 != 0 {
			return i
		}
		r >>= 1
	}
}

// regalloc performs register allocation on f.  It sets f.RegAlloc
// to the resulting allocation.
func regalloc(f *Func) {
	// For now, a very simple allocator.  Everything has a home
	// location on the stack (TBD as a subsequent stackalloc pass).
	// Values live in the home locations at basic block boundaries.
	// We use a simple greedy allocator within a basic block.
	home := make([]Location, f.NumValues())

	addPhiCopies(f) // add copies of phi inputs in preceeding blocks

	// Compute live values at the end of each block.
	live := live(f)
	lastUse := make([]int, f.NumValues())

	var oldSched []*Value

	// Hack to find sp and sb Values and assign them a register. (TODO: make not so hacky)
	var sp, sb *Value
	for _, v := range f.Entry.Values {
		switch v.Op {
		case OpSP:
			sp = v
			home = setloc(home, v, &registers[4]) // TODO: arch-dependent
		case OpSB:
			sb = v
			home = setloc(home, v, &registers[32]) // TODO: arch-dependent
		}
	}

	// Register allocate each block separately.  All live values will live
	// in home locations (stack slots) between blocks.
	for _, b := range f.Blocks {

		// Compute the index of the last use of each Value in the Block.
		// Scheduling has already happened, so Values are totally ordered.
		// lastUse[x] = max(i) where b.Value[i] uses Value x.
		for i, v := range b.Values {
			lastUse[v.ID] = -1
			for _, w := range v.Args {
				// could condition this store on w.Block == b, but no need
				lastUse[w.ID] = i
			}
		}
		// Values which are live at block exit have a lastUse of len(b.Values).
		if b.Control != nil {
			lastUse[b.Control.ID] = len(b.Values)
		}
		// Values live after block exit have a lastUse of len(b.Values)+1.
		for _, vid := range live[b.ID] {
			lastUse[vid] = len(b.Values) + 1
		}

		// For each register, store which value it contains
		type regInfo struct {
			v     *Value // stack-homed original value (or nil if empty)
			c     *Value // the register copy of v
			dirty bool   // if the stack-homed copy is out of date
		}
		regs := make([]regInfo, numRegs)

		// TODO: hack: initialize fixed registers
		regs[4] = regInfo{sp, sp, false}
		regs[32] = regInfo{sb, sb, false}

		var used regMask  // has a 1 for each non-nil entry in regs
		var dirty regMask // has a 1 for each dirty entry in regs

		oldSched = append(oldSched[:0], b.Values...)
		b.Values = b.Values[:0]

		for idx, v := range oldSched {
			// For each instruction, do:
			//   set up inputs to v in registers
			//   pick output register
			//   run insn
			//   mark output register as dirty
			// Note that v represents the Value at "home" (on the stack), and c
			// is its register equivalent.  There are two ways to establish c:
			//   - use of v.  c will be a load from v's home.
			//   - definition of v.  c will be identical to v but will live in
			//     a register.  v will be modified into a spill of c.
			regspec := opcodeTable[v.Op].reg
			if v.Op == OpCopy || v.Op == OpConvNop {
				// TODO: make this less of a hack
				regspec = opcodeTable[OpAMD64ADDQconst].reg
			}
			inputs := regspec.inputs
			outputs := regspec.outputs
			if len(inputs) == 0 && len(outputs) == 0 {
				// No register allocation required (or none specified yet)
				b.Values = append(b.Values, v)
				continue
			}
			if v.Op == OpCopy && v.Type.IsMemory() {
				b.Values = append(b.Values, v)
				continue
			}

			// Compute a good input ordering.  Start with the most constrained input.
			order := make([]intPair, len(inputs))
			for i, input := range inputs {
				order[i] = intPair{countRegs(input), i}
			}
			sort.Sort(byKey(order))

			// nospill contains registers that we can't spill because
			// we already set them up for use by the current instruction.
			var nospill regMask
			nospill |= 0x100000010 // SP & SB can't be spilled (TODO: arch-specific)

			// Move inputs into registers
			for _, o := range order {
				w := v.Args[o.val]
				mask := inputs[o.val]
				if mask == 0 {
					// Input doesn't need a register
					continue
				}
				// TODO: 2-address overwrite instructions

				// Find registers that w is already in
				var wreg regMask
				for r := register(0); r < numRegs; r++ {
					if regs[r].v == w {
						wreg |= regMask(1) << r
					}
				}

				var r register
				if mask&wreg != 0 {
					// w is already in an allowed register.  We're done.
					r = pickReg(mask & wreg)
				} else {
					// Pick a register for w
					// Priorities (in order)
					//  - an unused register
					//  - a clean register
					//  - a dirty register
					// TODO: for used registers, pick the one whose next use is the
					// farthest in the future.
					mask &^= nospill
					if mask & ^dirty != 0 {
						mask &^= dirty
					}
					if mask & ^used != 0 {
						mask &^= used
					}
					r = pickReg(mask)

					// Kick out whomever is using this register.
					if regs[r].v != nil {
						x := regs[r].v
						c := regs[r].c
						if regs[r].dirty && lastUse[x.ID] > idx {
							// Write x back to home.  Its value is currently held in c.
							x.Op = OpStoreReg8
							x.Aux = nil
							x.resetArgs()
							x.AddArg(c)
							b.Values = append(b.Values, x)
							regs[r].dirty = false
							dirty &^= regMask(1) << r
						}
						regs[r].v = nil
						regs[r].c = nil
						used &^= regMask(1) << r
					}

					// Load w into this register
					var c *Value
					if len(w.Args) == 0 {
						// Materialize w
						if w.Op == OpSB {
							c = w
						} else if w.Op == OpSP {
							c = b.NewValue1(w.Line, OpCopy, w.Type, w)
						} else {
							c = b.NewValue0IA(w.Line, w.Op, w.Type, w.AuxInt, w.Aux)
						}
					} else if len(w.Args) == 1 && (w.Args[0].Op == OpSP || w.Args[0].Op == OpSB) {
						// Materialize offsets from SP/SB
						c = b.NewValue1IA(w.Line, w.Op, w.Type, w.AuxInt, w.Aux, w.Args[0])
					} else if wreg != 0 {
						// Copy from another register.
						// Typically just an optimization, but this is
						// required if w is dirty.
						s := pickReg(wreg)
						// inv: s != r
						c = b.NewValue1(w.Line, OpCopy, w.Type, regs[s].c)
					} else {
						// Load from home location
						c = b.NewValue1(w.Line, OpLoadReg8, w.Type, w)
					}
					home = setloc(home, c, &registers[r])
					// Remember what we did
					regs[r].v = w
					regs[r].c = c
					regs[r].dirty = false
					used |= regMask(1) << r
				}

				// Replace w with its in-register copy.
				v.SetArg(o.val, regs[r].c)

				// Remember not to undo this register assignment until after
				// the instruction is issued.
				nospill |= regMask(1) << r
			}

			// TODO: do any clobbering

			// pick a register for v itself.
			if len(outputs) > 1 {
				panic("can't do multi-output yet")
			}
			if len(outputs) == 0 || outputs[0] == 0 {
				// output doesn't need a register
				b.Values = append(b.Values, v)
			} else {
				mask := outputs[0]
				if mask & ^dirty != 0 {
					mask &^= dirty
				}
				if mask & ^used != 0 {
					mask &^= used
				}
				r := pickReg(mask)

				// Kick out whomever is using this register.
				if regs[r].v != nil {
					x := regs[r].v
					c := regs[r].c
					if regs[r].dirty && lastUse[x.ID] > idx {
						// Write x back to home.  Its value is currently held in c.
						x.Op = OpStoreReg8
						x.Aux = nil
						x.resetArgs()
						x.AddArg(c)
						b.Values = append(b.Values, x)
						regs[r].dirty = false
						dirty &^= regMask(1) << r
					}
					regs[r].v = nil
					regs[r].c = nil
					used &^= regMask(1) << r
				}

				// Reissue v with new op, with r as its home.
				c := b.NewValue0IA(v.Line, v.Op, v.Type, v.AuxInt, v.Aux)
				c.AddArgs(v.Args...)
				home = setloc(home, c, &registers[r])

				// Remember what we did
				regs[r].v = v
				regs[r].c = c
				regs[r].dirty = true
				used |= regMask(1) << r
				dirty |= regMask(1) << r
			}
		}

		// If the block ends in a call, we must put the call after the spill code.
		var call *Value
		if b.Kind == BlockCall {
			call = b.Control
			if call != b.Values[len(b.Values)-1] {
				b.Fatalf("call not at end of block %b %v", b, call)
			}
			b.Values = b.Values[:len(b.Values)-1]
			// TODO: do this for all control types?
		}

		// at the end of the block, spill any remaining dirty, live values
		for r := register(0); r < numRegs; r++ {
			if !regs[r].dirty {
				continue
			}
			v := regs[r].v
			c := regs[r].c
			if lastUse[v.ID] <= len(oldSched) {
				if v == v.Block.Control {
					// link control value to register version
					v.Block.Control = c
				}
				continue // not live after block
			}

			// change v to be a copy of c
			v.Op = OpStoreReg8
			v.Aux = nil
			v.resetArgs()
			v.AddArg(c)
			b.Values = append(b.Values, v)
		}

		// add call back after spills
		if b.Kind == BlockCall {
			b.Values = append(b.Values, call)
		}
	}
	f.RegAlloc = home
	deadcode(f) // remove values that had all of their uses rematerialized.  TODO: separate pass?
}

// addPhiCopies adds copies of phi inputs in the blocks
// immediately preceding the phi's block.
func addPhiCopies(f *Func) {
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op != OpPhi {
				break // all phis should appear first
			}
			if v.Type.IsMemory() { // TODO: only "regallocable" types
				continue
			}
			for i, w := range v.Args {
				c := b.Preds[i]
				cpy := c.NewValue1(w.Line, OpCopy, v.Type, w)
				v.Args[i] = cpy
			}
		}
	}
}

// live returns a map from block ID to a list of value IDs live at the end of that block
// TODO: this could be quadratic if lots of variables are live across lots of
// basic blocks.  Figure out a way to make this function (or, more precisely, the user
// of this function) require only linear size & time.
func live(f *Func) [][]ID {
	live := make([][]ID, f.NumBlocks())
	var phis []*Value

	s := newSparseSet(f.NumValues())
	t := newSparseSet(f.NumValues())

	// Instead of iterating over f.Blocks, iterate over their postordering.
	// Liveness information flows backward, so starting at the end
	// increases the probability that we will stabilize quickly.
	// TODO: Do a better job yet. Here's one possibility:
	// Calculate the dominator tree and locate all strongly connected components.
	// If a value is live in one block of an SCC, it is live in all.
	// Walk the dominator tree from end to beginning, just once, treating SCC
	// components as single blocks, duplicated calculated liveness information
	// out to all of them.
	po := postorder(f)
	for {
		for _, b := range po {
			f.Logf("live %s %v\n", b, live[b.ID])
		}
		changed := false

		for _, b := range po {
			// Start with known live values at the end of the block
			s.clear()
			s.addAll(live[b.ID])

			// Propagate backwards to the start of the block
			// Assumes Values have been scheduled.
			phis := phis[:0]
			for i := len(b.Values) - 1; i >= 0; i-- {
				v := b.Values[i]
				s.remove(v.ID)
				if v.Op == OpPhi {
					// save phi ops for later
					phis = append(phis, v)
					continue
				}
				s.addAllValues(v.Args)
			}

			// for each predecessor of b, expand its list of live-at-end values
			// inv: s contains the values live at the start of b (excluding phi inputs)
			for i, p := range b.Preds {
				t.clear()
				t.addAll(live[p.ID])
				t.addAll(s.contents())
				for _, v := range phis {
					t.add(v.Args[i].ID)
				}
				if t.size() == len(live[p.ID]) {
					continue
				}
				// grow p's live set
				c := make([]ID, t.size())
				copy(c, t.contents())
				live[p.ID] = c
				changed = true
			}
		}

		if !changed {
			break
		}
	}
	return live
}

// for sorting a pair of integers by key
type intPair struct {
	key, val int
}
type byKey []intPair

func (a byKey) Len() int           { return len(a) }
func (a byKey) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byKey) Less(i, j int) bool { return a[i].key < a[j].key }
