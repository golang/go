// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// nilcheckelim eliminates unnecessary nil checks.
// runs on machine-independent code.
func nilcheckelim(f *Func) {
	// A nil check is redundant if the same nil check was successful in a
	// dominating block. The efficacy of this pass depends heavily on the
	// efficacy of the cse pass.
	sdom := f.sdom()

	// TODO: Eliminate more nil checks.
	// We can recursively remove any chain of fixed offset calculations,
	// i.e. struct fields and array elements, even with non-constant
	// indices: x is non-nil iff x.a.b[i].c is.

	type walkState int
	const (
		Work     walkState = iota // process nil checks and traverse to dominees
		ClearPtr                  // forget the fact that ptr is nil
	)

	type bp struct {
		block *Block // block, or nil in ClearPtr state
		ptr   *Value // if non-nil, ptr that is to be cleared in ClearPtr state
		op    walkState
	}

	work := make([]bp, 0, 256)
	work = append(work, bp{block: f.Entry})

	// map from value ID to bool indicating if value is known to be non-nil
	// in the current dominator path being walked. This slice is updated by
	// walkStates to maintain the known non-nil values.
	nonNilValues := make([]bool, f.NumValues())

	// make an initial pass identifying any non-nil values
	for _, b := range f.Blocks {
		// a value resulting from taking the address of a
		// value, or a value constructed from an offset of a
		// non-nil ptr (OpAddPtr) implies it is non-nil
		for _, v := range b.Values {
			if v.Op == OpAddr || v.Op == OpAddPtr {
				nonNilValues[v.ID] = true
			} else if v.Op == OpPhi {
				// phis whose arguments are all non-nil
				// are non-nil
				argsNonNil := true
				for _, a := range v.Args {
					if !nonNilValues[a.ID] {
						argsNonNil = false
					}
				}
				if argsNonNil {
					nonNilValues[v.ID] = true
				}
			}
		}
	}

	// perform a depth first walk of the dominee tree
	for len(work) > 0 {
		node := work[len(work)-1]
		work = work[:len(work)-1]

		switch node.op {
		case Work:
			b := node.block

			// First, see if we're dominated by an explicit nil check.
			if len(b.Preds) == 1 {
				p := b.Preds[0].b
				if p.Kind == BlockIf && p.Control.Op == OpIsNonNil && p.Succs[0].b == b {
					ptr := p.Control.Args[0]
					if !nonNilValues[ptr.ID] {
						nonNilValues[ptr.ID] = true
						work = append(work, bp{op: ClearPtr, ptr: ptr})
					}
				}
			}

			// Next, eliminate any redundant nil checks in this block.
			i := 0
			for _, v := range b.Values {
				b.Values[i] = v
				i++
				switch v.Op {
				case OpIsNonNil:
					ptr := v.Args[0]
					if nonNilValues[ptr.ID] {
						// This is a redundant explicit nil check.
						v.reset(OpConstBool)
						v.AuxInt = 1 // true
					}
				case OpNilCheck:
					ptr := v.Args[0]
					if nonNilValues[ptr.ID] {
						// This is a redundant implicit nil check.
						// Logging in the style of the former compiler -- and omit line 1,
						// which is usually in generated code.
						if f.Config.Debug_checknil() && v.Line > 1 {
							f.Config.Warnl(v.Line, "removed nil check")
						}
						v.reset(OpUnknown)
						// TODO: f.freeValue(v)
						i--
						continue
					}
				}
			}
			for j := i; j < len(b.Values); j++ {
				b.Values[j] = nil
			}
			b.Values = b.Values[:i]

			// Finally, find redundant nil checks for subsequent blocks.
			// Note that we can't add these until the loop above is done, as the
			// values in the block are not ordered in any way when this pass runs.
			// This was the cause of issue #18725.
			for _, v := range b.Values {
				if v.Op != OpNilCheck {
					continue
				}
				ptr := v.Args[0]
				// Record the fact that we know ptr is non nil, and remember to
				// undo that information when this dominator subtree is done.
				nonNilValues[ptr.ID] = true
				work = append(work, bp{op: ClearPtr, ptr: ptr})
			}

			// Add all dominated blocks to the work list.
			for w := sdom[node.block.ID].child; w != nil; w = sdom[w.ID].sibling {
				work = append(work, bp{op: Work, block: w})
			}

		case ClearPtr:
			nonNilValues[node.ptr.ID] = false
			continue
		}
	}
}

// All platforms are guaranteed to fault if we load/store to anything smaller than this address.
//
// This should agree with minLegalPointer in the runtime.
const minZeroPage = 4096

// nilcheckelim2 eliminates unnecessary nil checks.
// Runs after lowering and scheduling.
func nilcheckelim2(f *Func) {
	unnecessary := f.newSparseSet(f.NumValues())
	defer f.retSparseSet(unnecessary)
	for _, b := range f.Blocks {
		// Walk the block backwards. Find instructions that will fault if their
		// input pointer is nil. Remove nil checks on those pointers, as the
		// faulting instruction effectively does the nil check for free.
		unnecessary.clear()
		for i := len(b.Values) - 1; i >= 0; i-- {
			v := b.Values[i]
			if opcodeTable[v.Op].nilCheck && unnecessary.contains(v.Args[0].ID) {
				if f.Config.Debug_checknil() && int(v.Line) > 1 {
					f.Config.Warnl(v.Line, "removed nil check")
				}
				v.reset(OpUnknown)
				continue
			}
			if v.Type.IsMemory() || v.Type.IsTuple() && v.Type.FieldType(1).IsMemory() {
				if v.Op == OpVarDef || v.Op == OpVarKill || v.Op == OpVarLive {
					// These ops don't really change memory.
					continue
				}
				// This op changes memory.  Any faulting instruction after v that
				// we've recorded in the unnecessary map is now obsolete.
				unnecessary.clear()
			}

			// Find any pointers that this op is guaranteed to fault on if nil.
			var ptrstore [2]*Value
			ptrs := ptrstore[:0]
			if opcodeTable[v.Op].faultOnNilArg0 {
				ptrs = append(ptrs, v.Args[0])
			}
			if opcodeTable[v.Op].faultOnNilArg1 {
				ptrs = append(ptrs, v.Args[1])
			}
			for _, ptr := range ptrs {
				// Check to make sure the offset is small.
				switch opcodeTable[v.Op].auxType {
				case auxSymOff:
					if v.Aux != nil || v.AuxInt < 0 || v.AuxInt >= minZeroPage {
						continue
					}
				case auxSymValAndOff:
					off := ValAndOff(v.AuxInt).Off()
					if v.Aux != nil || off < 0 || off >= minZeroPage {
						continue
					}
				case auxInt32:
					// Mips uses this auxType for atomic add constant. It does not affect the effective address.
				case auxInt64:
					// ARM uses this auxType for duffcopy/duffzero/alignment info.
					// It does not affect the effective address.
				case auxNone:
					// offset is zero.
				default:
					v.Fatalf("can't handle aux %s (type %d) yet\n", v.auxString(), int(opcodeTable[v.Op].auxType))
				}
				// This instruction is guaranteed to fault if ptr is nil.
				// Any previous nil check op is unnecessary.
				unnecessary.add(ptr.ID)
			}
		}
		// Remove values we've clobbered with OpUnknown.
		i := 0
		for _, v := range b.Values {
			if v.Op != OpUnknown {
				b.Values[i] = v
				i++
			}
		}
		for j := i; j < len(b.Values); j++ {
			b.Values[j] = nil
		}
		b.Values = b.Values[:i]

		// TODO: if b.Kind == BlockPlain, start the analysis in the subsequent block to find
		// more unnecessary nil checks.  Would fix test/nilptr3_ssa.go:157.
	}
}
