// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "cmd/compile/internal/types"

// expandCalls converts LE (Late Expansion) calls that act like they receive value args into a lower-level form
// that is more oriented to a platform's ABI.  The SelectN operations that extract results are also rewritten into
// more appropriate forms.
func expandCalls(f *Func) {
	canSSAType := f.fe.CanSSA
	sp, _ := f.spSb()
	// Calls that need lowering have some number of inputs, including a memory input,
	// and produce a tuple of (value1, value2, ..., mem) where valueK may or may not be SSA-able.

	// With the current ABI those inputs need to be converted into stores to memory,
	// rethreading the call's memory input to the first, and the new call now receiving the last.

	// With the current ABI, the outputs need to be converted to loads, which will all use the call's
	// memory output as their input.

	// Step 1: find all references to calls as values and rewrite those.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			switch v.Op {
			case OpSelectN:
				call := v.Args[0]
				aux := call.Aux.(*AuxCall)
				which := v.AuxInt
				t := v.Type
				if which == aux.NResults() { // mem is after the results.
					// rewrite v as a Copy of call -- the replacement call will produce a mem.
					v.copyOf(call)
				} else {
					pt := types.NewPtr(t)
					if canSSAType(t) {
						off := f.ConstOffPtrSP(pt, aux.OffsetOfResult(which), sp)
						v.reset(OpLoad)
						v.SetArgs2(off, call)
					} else {
						panic("Should not have non-SSA-able OpSelectN")
					}
				}
				v.Type = t // not right for the mem operand yet, but will be when call is rewritten.

			case OpSelectNAddr:
				call := v.Args[0]
				which := v.AuxInt
				aux := call.Aux.(*AuxCall)
				pt := v.Type
				off := f.ConstOffPtrSP(pt, aux.OffsetOfResult(which), sp)
				v.copyOf(off)
			}
		}
	}

	// Step 2: rewrite the calls
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			switch v.Op {
			case OpStaticLECall:
				// Thread the stores on the memory arg
				m0 := v.Args[len(v.Args)-1]
				mem := m0
				pos := v.Pos.WithNotStmt()
				aux := v.Aux.(*AuxCall)
				auxInt := v.AuxInt
				for i, a := range v.Args {
					if a == m0 {
						break
					}
					if a.Op == OpDereference {
						// "Dereference" of addressed (probably not-SSA-eligible) value becomes Move
						src := a.Args[0]
						dst := f.ConstOffPtrSP(src.Type, aux.OffsetOfArg(int64(i)), sp)
						a.reset(OpMove)
						a.Pos = pos
						a.Type = types.TypeMem
						a.Aux = aux.TypeOfArg(int64(i))
						a.AuxInt = aux.SizeOfArg(int64(i))
						a.SetArgs3(dst, src, mem)
						mem = a
					} else {
						// Add a new store.
						t := aux.TypeOfArg(int64(i))
						dst := f.ConstOffPtrSP(types.NewPtr(t), aux.OffsetOfArg(int64(i)), sp)
						mem = b.NewValue3A(pos, OpStore, types.TypeMem, t, dst, a, mem)
					}
				}
				v.reset(OpStaticCall)
				v.Type = types.TypeMem
				v.Aux = aux
				v.AuxInt = auxInt
				v.SetArgs1(mem)
			}
		}
	}
}

