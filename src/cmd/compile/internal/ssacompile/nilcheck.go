// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"internal/buildcfg"

	"cmd/compile/internal/ir"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/block"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/internal/src"
)

// nilcheckelim eliminates unnecessary nil checks.
// runs on machine-independent code.
func nilcheckelim(f *ssa.Func) {
	// A nil check is redundant if the same nil check was successful in a
	// dominating block. The efficacy of this pass depends heavily on the
	// efficacy of the cse pass.
	sdom := f.Sdom()

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
		block *ssa.Block // block, or nil in ClearPtr state
		ptr   *ssa.Value // if non-nil, ptr that is to be cleared in ClearPtr state
		op    walkState
	}

	work := make([]bp, 0, 256)
	work = append(work, bp{block: f.Entry})

	// map from value ID to known non-nil version of that value ID
	// (in the current dominator path being walked). This slice is updated by
	// walkStates to maintain the known non-nil values.
	// If there is extrinsic information about non-nil-ness, this map
	// points a value to itself. If a value is known non-nil because we
	// already did a nil check on it, it points to the nil check operation.
	nonNilValues := f.Cache.AllocValueSlice(f.NumValues())
	defer f.Cache.FreeValueSlice(nonNilValues)

	// make an initial pass identifying any non-nil values
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			// a value resulting from taking the address of a
			// value, or a value constructed from an offset of a
			// non-nil ptr (OpAddPtr) implies it is non-nil
			// We also assume unsafe pointer arithmetic generates non-nil pointers. See #27180.
			// We assume that SlicePtr is non-nil because we do a bounds check
			// before the slice access (and all cap>0 slices have a non-nil ptr). See #30366.
			if v.Op == ssaop.OpAddr || v.Op == ssaop.OpLocalAddr || v.Op == ssaop.OpAddPtr || v.Op == ssaop.OpOffPtr || v.Op == ssaop.OpAdd32 || v.Op == ssaop.OpAdd64 || v.Op == ssaop.OpSub32 || v.Op == ssaop.OpSub64 || v.Op == ssaop.OpSlicePtr {
				nonNilValues[v.ID] = v
			}
		}
	}

	for changed := true; changed; {
		changed = false
		for _, b := range f.Blocks {
			for _, v := range b.Values {
				// phis whose arguments are all non-nil
				// are non-nil
				if v.Op == ssaop.OpPhi {
					argsNonNil := true
					for _, a := range v.Args {
						if nonNilValues[a.ID] == nil {
							argsNonNil = false
							break
						}
					}
					if argsNonNil {
						if nonNilValues[v.ID] == nil {
							changed = true
						}
						nonNilValues[v.ID] = v
					}
				}
			}
		}
	}

	// allocate auxiliary date structures for computing store order
	sset := f.NewSparseSet(f.NumValues())
	defer f.RetSparseSet(sset)
	storeNumber := f.Cache.AllocInt32Slice(f.NumValues())
	defer f.Cache.FreeInt32Slice(storeNumber)

	// perform a depth first walk of the dominee tree
	for len(work) > 0 {
		node := work[len(work)-1]
		work = work[:len(work)-1]

		switch node.op {
		case Work:
			b := node.block

			// First, see if we're dominated by an explicit nil check.
			if len(b.Preds) == 1 {
				p := b.Preds[0].B
				if p.Kind == block.BlockIf && p.Controls[0].Op == ssaop.OpIsNonNil && p.Succs[0].B == b {
					if ptr := p.Controls[0].Args[0]; nonNilValues[ptr.ID] == nil {
						nonNilValues[ptr.ID] = ptr
						work = append(work, bp{op: ClearPtr, ptr: ptr})
					}
				}
			}

			// Next, order values in the current block w.r.t. stores.
			b.Values = storeOrder(b.Values, sset, storeNumber)

			pendingLines := f.CachedLineStarts // Holds statement boundaries that need to be moved to a new value/block
			pendingLines.Clear()

			// Next, process values in the block.
			for _, v := range b.Values {
				switch v.Op {
				case ssaop.OpIsNonNil:
					ptr := v.Args[0]
					if nonNilValues[ptr.ID] != nil {
						if v.Pos.IsStmt() == src.PosIsStmt { // Boolean true is a terrible statement boundary.
							pendingLines.Add(v.Pos)
							v.Pos = v.Pos.WithNotStmt()
						}
						// This is a redundant explicit nil check.
						v.Reset(ssaop.OpConstBool)
						v.AuxInt = 1 // true
					}
				case ssaop.OpNilCheck:
					ptr := v.Args[0]
					if nilCheck := nonNilValues[ptr.ID]; nilCheck != nil {
						// This is a redundant implicit nil check.
						// Logging in the style of the former compiler -- and omit line 1,
						// which is usually in generated code.
						if f.Fe.Debug_checknil() && v.Pos.Line() > 1 {
							f.Warnl(v.Pos, "removed nil check")
						}
						if v.Pos.IsStmt() == src.PosIsStmt { // About to lose a statement boundary
							pendingLines.Add(v.Pos)
						}
						v.Op = ssaop.OpCopy
						v.SetArgs1(nilCheck)
						continue
					}
					// Record the fact that we know ptr is non nil, and remember to
					// undo that information when this dominator subtree is done.
					nonNilValues[ptr.ID] = v
					work = append(work, bp{op: ClearPtr, ptr: ptr})
					fallthrough // a non-eliminated nil check might be a good place for a statement boundary.
				default:
					if v.Pos.IsStmt() != src.PosNotStmt && !isPoorStatementOp(v.Op) && pendingLines.Contains(v.Pos) {
						v.Pos = v.Pos.WithIsStmt()
						pendingLines.Remove(v.Pos)
					}
				}
			}
			// This reduces the lost statement count in "go" by 5 (out of 500 total).
			for j := range b.Values { // is this an ordering problem?
				v := b.Values[j]
				if v.Pos.IsStmt() != src.PosNotStmt && !isPoorStatementOp(v.Op) && pendingLines.Contains(v.Pos) {
					v.Pos = v.Pos.WithIsStmt()
					pendingLines.Remove(v.Pos)
				}
			}
			if pendingLines.Contains(b.Pos) {
				b.Pos = b.Pos.WithIsStmt()
				pendingLines.Remove(b.Pos)
			}

			// Add all dominated blocks to the work list.
			for w := sdom[node.block.ID].Child; w != nil; w = sdom[w.ID].Sibling {
				work = append(work, bp{op: Work, block: w})
			}

		case ClearPtr:
			nonNilValues[node.ptr.ID] = nil
			continue
		}
	}
}

// All platforms are guaranteed to fault if we load/store to anything smaller than this address.
//
// This should agree with minLegalPointer in the runtime.
const minZeroPage = 4096

// faultOnLoad is true if a load to an address below minZeroPage will trigger a SIGSEGV.
var faultOnLoad = buildcfg.GOOS != "aix"

// nilcheckelim2 eliminates unnecessary nil checks.
// Runs after lowering and scheduling.
func nilcheckelim2(f *ssa.Func) {
	unnecessary := f.NewSparseMap(f.NumValues()) // map from pointer that will be dereferenced to index of dereferencing value in b.Values[]
	defer f.RetSparseMap(unnecessary)

	pendingLines := f.CachedLineStarts // Holds statement boundaries that need to be moved to a new value/block

	for _, b := range f.Blocks {
		// Walk the block backwards. Find instructions that will fault if their
		// input pointer is nil. Remove nil checks on those pointers, as the
		// faulting instruction effectively does the nil check for free.
		unnecessary.Clear()
		pendingLines.Clear()
		// Optimization: keep track of removed nilcheck with smallest index
		firstToRemove := len(b.Values)
		for i := len(b.Values) - 1; i >= 0; i-- {
			v := b.Values[i]
			if ssaop.OpcodeTable[v.Op].NilCheck && unnecessary.Contains(v.Args[0].ID) {
				if f.Fe.Debug_checknil() && v.Pos.Line() > 1 {
					f.Warnl(v.Pos, "removed nil check")
				}
				// For bug 33724, policy is that we might choose to bump an existing position
				// off the faulting load/store in favor of the one from the nil check.

				// Iteration order means that first nilcheck in the chain wins, others
				// are bumped into the ordinary statement preservation algorithm.
				uid, _ := unnecessary.Get(v.Args[0].ID)
				u := b.Values[uid]
				if !u.Pos.SameFileAndLine(v.Pos) {
					if u.Pos.IsStmt() == src.PosIsStmt {
						pendingLines.Add(u.Pos)
					}
					u.Pos = v.Pos
				} else if v.Pos.IsStmt() == src.PosIsStmt {
					pendingLines.Add(v.Pos)
				}

				v.Reset(ssaop.OpUnknown)
				firstToRemove = i
				continue
			}
			if v.Type.IsMemory() || v.Type.IsTuple() && v.Type.FieldType(1).IsMemory() {
				if v.Op == ssaop.OpVarLive || (v.Op == ssaop.OpVarDef && !v.Aux.(*ir.Name).Type().HasPointers()) {
					// These ops don't really change memory.
					continue
					// Note: OpVarDef requires that the defined variable not have pointers.
					// We need to make sure that there's no possible faulting
					// instruction between a VarDef and that variable being
					// fully initialized. If there was, then anything scanning
					// the stack during the handling of that fault will see
					// a live but uninitialized pointer variable on the stack.
					//
					// If we have:
					//
					//   NilCheck p
					//   VarDef x
					//   x = *p
					//
					// We can't rewrite that to
					//
					//   VarDef x
					//   NilCheck p
					//   x = *p
					//
					// Particularly, even though *p faults on p==nil, we still
					// have to do the explicit nil check before the VarDef.
					// See issue #32288.
				}
				// This op changes memory.  Any faulting instruction after v that
				// we've recorded in the unnecessary map is now obsolete.
				unnecessary.Clear()
			}

			// Find any pointers that this op is guaranteed to fault on if nil.
			var ptrstore [2]*ssa.Value
			ptrs := ptrstore[:0]
			if ssaop.OpcodeTable[v.Op].FaultOnNilArg0 && (faultOnLoad || v.Type.IsMemory()) {
				// On AIX, only writing will fault.
				ptrs = append(ptrs, v.Args[0])
			}
			if ssaop.OpcodeTable[v.Op].FaultOnNilArg1 && (faultOnLoad || (v.Type.IsMemory() && v.Op != ssaop.OpPPC64LoweredMove)) {
				// On AIX, only writing will fault.
				// LoweredMove is a special case because it's considered as a "mem" as it stores on arg0 but arg1 is accessed as a load and should be checked.
				ptrs = append(ptrs, v.Args[1])
			}

			for _, ptr := range ptrs {
				// Check to make sure the offset is small.
				switch ssaop.OpcodeTable[v.Op].AuxType {
				case ssaop.AuxTypeSym:
					if v.Aux != nil {
						continue
					}
				case ssaop.AuxTypeSymOff:
					if v.Aux != nil || v.AuxInt < 0 || v.AuxInt >= minZeroPage {
						continue
					}
				case ssaop.AuxTypeSymValAndOff:
					off := ssa.ValAndOff(v.AuxInt).Off()
					if v.Aux != nil || off < 0 || off >= minZeroPage {
						continue
					}
				case ssaop.AuxTypeInt32:
					// Mips uses this auxType for atomic add constant. It does not affect the effective address.
				case ssaop.AuxTypeInt64:
					// ARM uses this auxType for duffcopy/duffzero/alignment info.
					// It does not affect the effective address.
				case ssaop.AuxTypeNone:
					// offset is zero.
				default:
					v.Fatalf("can't handle aux %s (type %d) yet\n", v.AuxString(), int(ssaop.OpcodeTable[v.Op].AuxType))
				}
				// This instruction is guaranteed to fault if ptr is nil.
				// Any previous nil check op is unnecessary.
				unnecessary.Set(ptr.ID, int32(i))
			}
		}
		// Remove values we've clobbered with OpUnknown.
		i := firstToRemove
		for j := i; j < len(b.Values); j++ {
			v := b.Values[j]
			if v.Op != ssaop.OpUnknown {
				if !ssa.NotStmtBoundary(v.Op) && pendingLines.Contains(v.Pos) { // Late in compilation, so any remaining NotStmt values are probably okay now.
					v.Pos = v.Pos.WithIsStmt()
					pendingLines.Remove(v.Pos)
				}
				b.Values[i] = v
				i++
			}
		}

		if pendingLines.Contains(b.Pos) {
			b.Pos = b.Pos.WithIsStmt()
		}

		b.TruncateValues(i)

		// TODO: if b.Kind == BlockPlain, start the analysis in the subsequent block to find
		// more unnecessary nil checks.  Would fix test/nilptr3.go:159.
	}
}
