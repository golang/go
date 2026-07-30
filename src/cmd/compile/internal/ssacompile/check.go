// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"math"
	"math/bits"

	"cmd/compile/internal/ir"
	"cmd/compile/internal/ssa/block"
	"cmd/compile/internal/ssa/ssacore"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/internal/obj/s390x"
)

// checkFunc checks invariants of f.
func checkFunc(f *ssacore.Func) {
	blockMark := make([]bool, f.NumBlocks())
	valueMark := make([]bool, f.NumValues())

	for _, b := range f.Blocks {
		if blockMark[b.ID] {
			f.Fatalf("block %s appears twice in %s!", b, f.Name)
		}
		blockMark[b.ID] = true
		if b.Func != f {
			f.Fatalf("%s.Func=%s, want %s", b, b.Func.Name, f.Name)
		}

		for i, e := range b.Preds {
			if se := e.B.Succs[e.I]; se.B != b || se.I != i {
				f.Fatalf("block pred/succ not crosslinked correctly %d:%s %d:%s", i, b, se.I, se.B)
			}
		}
		for i, e := range b.Succs {
			if pe := e.B.Preds[e.I]; pe.B != b || pe.I != i {
				f.Fatalf("block succ/pred not crosslinked correctly %d:%s %d:%s", i, b, pe.I, pe.B)
			}
		}

		switch b.Kind {
		case block.BlockExit:
			if len(b.Succs) != 0 {
				f.Fatalf("exit block %s has successors", b)
			}
			if b.NumControls() != 1 {
				f.Fatalf("exit block %s has no control value", b)
			}
			if !b.Controls[0].Type.IsMemory() {
				f.Fatalf("exit block %s has non-memory control value %s", b, b.Controls[0].LongString())
			}
		case block.BlockRet:
			if len(b.Succs) != 0 {
				f.Fatalf("ret block %s has successors", b)
			}
			if b.NumControls() != 1 {
				f.Fatalf("ret block %s has nil control", b)
			}
			if !b.Controls[0].Type.IsMemory() {
				f.Fatalf("ret block %s has non-memory control value %s", b, b.Controls[0].LongString())
			}
		case block.BlockRetJmp:
			if len(b.Succs) != 0 {
				f.Fatalf("retjmp block %s len(Succs)==%d, want 0", b, len(b.Succs))
			}
			if b.NumControls() != 1 {
				f.Fatalf("retjmp block %s has nil control", b)
			}
			if !b.Controls[0].Type.IsMemory() {
				f.Fatalf("retjmp block %s has non-memory control value %s", b, b.Controls[0].LongString())
			}
		case block.BlockPlain:
			if len(b.Succs) != 1 {
				f.Fatalf("plain block %s len(Succs)==%d, want 1", b, len(b.Succs))
			}
			if b.NumControls() != 0 {
				f.Fatalf("plain block %s has non-nil control %s", b, b.Controls[0].LongString())
			}
		case block.BlockIf:
			if len(b.Succs) != 2 {
				f.Fatalf("if block %s len(Succs)==%d, want 2", b, len(b.Succs))
			}
			if b.NumControls() != 1 {
				f.Fatalf("if block %s has no control value", b)
			}
			if !b.Controls[0].Type.IsBoolean() {
				f.Fatalf("if block %s has non-bool control value %s", b, b.Controls[0].LongString())
			}
		case block.BlockDefer:
			if len(b.Succs) != 2 {
				f.Fatalf("defer block %s len(Succs)==%d, want 2", b, len(b.Succs))
			}
			if b.NumControls() != 1 {
				f.Fatalf("defer block %s has no control value", b)
			}
			if !b.Controls[0].Type.IsMemory() {
				f.Fatalf("defer block %s has non-memory control value %s", b, b.Controls[0].LongString())
			}
		case block.BlockFirst:
			if len(b.Succs) != 2 {
				f.Fatalf("plain/dead block %s len(Succs)==%d, want 2", b, len(b.Succs))
			}
			if b.NumControls() != 0 {
				f.Fatalf("plain/dead block %s has a control value", b)
			}
		case block.BlockJumpTable:
			if b.NumControls() != 1 {
				f.Fatalf("jumpTable block %s has no control value", b)
			}
		}
		if len(b.Succs) != 2 && b.Likely != ssacore.BranchUnknown {
			f.Fatalf("likeliness prediction %d for block %s with %d successors", b.Likely, b, len(b.Succs))
		}

		for _, v := range b.Values {
			// Check to make sure argument count makes sense (argLen of -1 indicates
			// variable length args)
			nArgs := ssaop.OpcodeTable[v.Op].ArgLen
			if nArgs != -1 && int32(len(v.Args)) != nArgs {
				f.Fatalf("value %s has %d args, expected %d", v.LongString(),
					len(v.Args), nArgs)
			}

			// Check to make sure aux values make sense.
			canHaveAux := false
			canHaveAuxInt := false
			// TODO: enforce types of Aux in this switch (like auxString does below)
			switch ssaop.OpcodeTable[v.Op].AuxType {
			case ssaop.AuxTypeNone:
			case ssaop.AuxTypeBool:
				if v.AuxInt < 0 || v.AuxInt > 1 {
					f.Fatalf("bad bool AuxInt value for %v", v)
				}
				canHaveAuxInt = true
			case ssaop.AuxTypeInt8:
				if v.AuxInt != int64(int8(v.AuxInt)) {
					f.Fatalf("bad int8 AuxInt value for %v", v)
				}
				canHaveAuxInt = true
			case ssaop.AuxTypeInt16:
				if v.AuxInt != int64(int16(v.AuxInt)) {
					f.Fatalf("bad int16 AuxInt value for %v", v)
				}
				canHaveAuxInt = true
			case ssaop.AuxTypeInt32:
				if v.AuxInt != int64(int32(v.AuxInt)) {
					f.Fatalf("bad int32 AuxInt value for %v", v)
				}
				canHaveAuxInt = true
			case ssaop.AuxTypeInt64, ssaop.AuxTypeARM64BitField, ssaop.AuxTypeARM64ConditionalParams:
				canHaveAuxInt = true
			case ssaop.AuxTypeInt128:
				// AuxInt must be zero, so leave canHaveAuxInt set to false.
			case ssaop.AuxTypeUInt8:
				// Cast to int8 due to requirement of AuxInt, check its comment for details.
				if v.AuxInt != int64(int8(v.AuxInt)) {
					f.Fatalf("bad uint8 AuxInt value for %v, saw %d but need %d", v, v.AuxInt, int64(int8(v.AuxInt)))
				}
				canHaveAuxInt = true
			case ssaop.AuxTypeFloat32:
				canHaveAuxInt = true
				if math.IsNaN(v.AuxFloat()) {
					f.Fatalf("value %v has an AuxInt that encodes a NaN", v)
				}
				if !isExactFloat32(v.AuxFloat()) {
					f.Fatalf("value %v has an AuxInt value that is not an exact float32", v)
				}
			case ssaop.AuxTypeFloat64:
				canHaveAuxInt = true
				if math.IsNaN(v.AuxFloat()) {
					f.Fatalf("value %v has an AuxInt that encodes a NaN", v)
				}
			case ssaop.AuxTypeString:
				if _, ok := v.Aux.(ssacore.StringAux); !ok {
					f.Fatalf("value %v has Aux type %T, want string", v, v.Aux)
				}
				canHaveAux = true
			case ssaop.AuxTypeCallOff:
				canHaveAuxInt = true
				fallthrough
			case ssaop.AuxTypeCall:
				if ac, ok := v.Aux.(*ssacore.AuxCall); ok {
					if v.Op == ssaop.OpStaticCall && ac.Fn == nil {
						f.Fatalf("value %v has *AuxCall with nil Fn", v)
					}
				} else {
					f.Fatalf("value %v has Aux type %T, want *AuxCall", v, v.Aux)
				}
				canHaveAux = true
			case ssaop.AuxTypeNameOffsetInt8:
				if _, ok := v.Aux.(*ssacore.AuxNameOffset); !ok {
					f.Fatalf("value %v has Aux type %T, want *AuxNameOffset", v, v.Aux)
				}
				canHaveAux = true
				canHaveAuxInt = true
			case ssaop.AuxTypeSym, ssaop.AuxTypeTyp:
				canHaveAux = true
			case ssaop.AuxTypeSymOff, ssaop.AuxTypeSymValAndOff, ssaop.AuxTypeTypSize:
				canHaveAuxInt = true
				canHaveAux = true
			case ssaop.AuxTypeCCop:
				if ssaop.OpcodeTable[ssaop.Op(v.AuxInt)].Name == "OpInvalid" {
					f.Fatalf("value %v has an AuxInt value that is not a valid opcode", v)
				}
				canHaveAuxInt = true
			case ssaop.AuxTypeS390XCCMask:
				if _, ok := v.Aux.(s390x.CCMask); !ok {
					f.Fatalf("bad type %T for S390XCCMask in %v", v.Aux, v)
				}
				canHaveAux = true
			case ssaop.AuxTypeS390XRotateParams:
				if _, ok := v.Aux.(s390x.RotateParams); !ok {
					f.Fatalf("bad type %T for S390XRotateParams in %v", v.Aux, v)
				}
				canHaveAux = true
			case ssaop.AuxTypeFlagConstant:
				if v.AuxInt < 0 || v.AuxInt > 15 {
					f.Fatalf("bad FlagConstant AuxInt value for %v", v)
				}
				canHaveAuxInt = true
			case ssaop.AuxTypePanicBoundsC, ssaop.AuxTypePanicBoundsCC:
				canHaveAux = true
				canHaveAuxInt = true
			default:
				f.Fatalf("unknown aux type for %s", v.Op)
			}
			if !canHaveAux && v.Aux != nil {
				f.Fatalf("value %s has an Aux value %v but shouldn't", v.LongString(), v.Aux)
			}
			if !canHaveAuxInt && v.AuxInt != 0 {
				f.Fatalf("value %s has an AuxInt value %d but shouldn't", v.LongString(), v.AuxInt)
			}

			for i, arg := range v.Args {
				if arg == nil {
					f.Fatalf("value %s has nil arg", v.LongString())
				}
				if v.Op != ssaop.OpPhi {
					// For non-Phi ops, memory args must be last, if present
					if arg.Type.IsMemory() && i != len(v.Args)-1 {
						f.Fatalf("value %s has non-final memory arg (%d < %d)", v.LongString(), i, len(v.Args)-1)
					}
				}
			}

			if valueMark[v.ID] {
				f.Fatalf("value %s appears twice!", v.LongString())
			}
			valueMark[v.ID] = true

			if v.Block != b {
				f.Fatalf("%s.block != %s", v, b)
			}
			if v.Op == ssaop.OpPhi && len(v.Args) != len(b.Preds) {
				f.Fatalf("phi length %s does not match pred length %d for block %s", v.LongString(), len(b.Preds), b)
			}

			if v.Op == ssaop.OpAddr {
				if len(v.Args) == 0 {
					f.Fatalf("no args for OpAddr %s", v.LongString())
				}
				if v.Args[0].Op != ssaop.OpSB {
					f.Fatalf("bad arg to OpAddr %v", v)
				}
			}

			if v.Op == ssaop.OpLocalAddr {
				if len(v.Args) != 2 {
					f.Fatalf("wrong # of args for OpLocalAddr %s", v.LongString())
				}
				if v.Args[0].Op != ssaop.OpSP {
					f.Fatalf("bad arg 0 to OpLocalAddr %v", v)
				}
				if !v.Args[1].Type.IsMemory() {
					f.Fatalf("bad arg 1 to OpLocalAddr %v", v)
				}
			}

			if (v.Op == ssaop.OpStructMake || v.Op == ssaop.OpArrayMake1) && v.Type.Size() == 0 {
				f.Fatalf("zero-sized Make; use Empty instead %v", v)
			}

			if f.RegAlloc != nil && f.Config.SoftFloat && v.Type.IsFloat() {
				f.Fatalf("unexpected floating-point type %v", v.LongString())
			}

			// Check types.
			// TODO: more type checks?
			switch c := f.Config; v.Op {
			case ssaop.OpSP, ssaop.OpSB:
				if v.Type != c.Types.Uintptr {
					f.Fatalf("bad %s type: want uintptr, have %s",
						v.Op, v.Type.String())
				}
			case ssaop.OpStringLen:
				if v.Type != c.Types.Int {
					f.Fatalf("bad %s type: want int, have %s",
						v.Op, v.Type.String())
				}
			case ssaop.OpLoad:
				if !v.Args[1].Type.IsMemory() {
					f.Fatalf("bad arg 1 type to %s: want mem, have %s",
						v.Op, v.Args[1].Type.String())
				}
			case ssaop.OpStore:
				if !v.Type.IsMemory() {
					f.Fatalf("bad %s type: want mem, have %s",
						v.Op, v.Type.String())
				}
				if !v.Args[2].Type.IsMemory() {
					f.Fatalf("bad arg 2 type to %s: want mem, have %s",
						v.Op, v.Args[2].Type.String())
				}
			case ssaop.OpCondSelect:
				if !v.Args[2].Type.IsBoolean() {
					f.Fatalf("bad arg 2 type to %s: want boolean, have %s",
						v.Op, v.Args[2].Type.String())
				}
			case ssaop.OpAddPtr:
				if !v.Args[0].Type.IsPtrShaped() && v.Args[0].Type != c.Types.Uintptr {
					f.Fatalf("bad arg 0 type to %s: want ptr, have %s", v.Op, v.Args[0].LongString())
				}
				if !v.Args[1].Type.IsInteger() {
					f.Fatalf("bad arg 1 type to %s: want integer, have %s", v.Op, v.Args[1].LongString())
				}
			case ssaop.OpVarDef:
				n := v.Aux.(*ir.Name)
				if !n.Type().HasPointers() && !ssacore.IsMergeCandidate(n) {
					f.Fatalf("vardef must be merge candidate or have pointer type %s", v.Aux.(*ir.Name).Type().String())
				}
			case ssaop.OpNilCheck:
				// nil checks have pointer type before scheduling, and
				// void type after scheduling.
				if f.Scheduled {
					if v.Uses != 0 {
						f.Fatalf("nilcheck must have 0 uses %s", v.Uses)
					}
					if !v.Type.IsVoid() {
						f.Fatalf("nilcheck must have void type %s", v.Type.String())
					}
				} else {
					if !v.Type.IsPtrShaped() && !v.Type.IsUintptr() {
						f.Fatalf("nilcheck must have pointer type %s", v.Type.String())
					}
				}
				if !v.Args[0].Type.IsPtrShaped() && !v.Args[0].Type.IsUintptr() {
					f.Fatalf("nilcheck must have argument of pointer type %s", v.Args[0].Type.String())
				}
				if !v.Args[1].Type.IsMemory() {
					f.Fatalf("bad arg 1 type to %s: want mem, have %s",
						v.Op, v.Args[1].Type.String())
				}
			}
			// Check size of args.
			// This list isn't exhaustive, just the common ops.
			// It also can't handle ops with args of different types, like shifts.
			var argSize int64
			switch v.Op {
			case ssaop.OpAdd8, ssaop.OpSub8, ssaop.OpMul8, ssaop.OpDiv8, ssaop.OpDiv8u, ssaop.OpMod8, ssaop.OpMod8u,
				ssaop.OpAnd8, ssaop.OpOr8, ssaop.OpXor8,
				ssaop.OpEq8, ssaop.OpNeq8, ssaop.OpLess8, ssaop.OpLeq8,
				ssaop.OpNeg8, ssaop.OpCom8,
				ssaop.OpSignExt8to16, ssaop.OpSignExt8to32, ssaop.OpSignExt8to64,
				ssaop.OpZeroExt8to16, ssaop.OpZeroExt8to32, ssaop.OpZeroExt8to64:
				argSize = 1
			case ssaop.OpAdd16, ssaop.OpSub16, ssaop.OpMul16, ssaop.OpDiv16, ssaop.OpDiv16u, ssaop.OpMod16, ssaop.OpMod16u,
				ssaop.OpAnd16, ssaop.OpOr16, ssaop.OpXor16,
				ssaop.OpEq16, ssaop.OpNeq16, ssaop.OpLess16, ssaop.OpLeq16,
				ssaop.OpNeg16, ssaop.OpCom16,
				ssaop.OpSignExt16to32, ssaop.OpSignExt16to64,
				ssaop.OpZeroExt16to32, ssaop.OpZeroExt16to64,
				ssaop.OpTrunc16to8:
				argSize = 2
			case ssaop.OpAdd32, ssaop.OpSub32, ssaop.OpMul32, ssaop.OpDiv32, ssaop.OpDiv32u, ssaop.OpMod32, ssaop.OpMod32u,
				ssaop.OpAnd32, ssaop.OpOr32, ssaop.OpXor32,
				ssaop.OpEq32, ssaop.OpNeq32, ssaop.OpLess32, ssaop.OpLeq32,
				ssaop.OpNeg32, ssaop.OpCom32,
				ssaop.OpSignExt32to64, ssaop.OpZeroExt32to64,
				ssaop.OpTrunc32to8, ssaop.OpTrunc32to16:
				argSize = 4
			case ssaop.OpAdd64, ssaop.OpSub64, ssaop.OpMul64, ssaop.OpDiv64, ssaop.OpDiv64u, ssaop.OpMod64, ssaop.OpMod64u,
				ssaop.OpAnd64, ssaop.OpOr64, ssaop.OpXor64,
				ssaop.OpEq64, ssaop.OpNeq64, ssaop.OpLess64, ssaop.OpLeq64,
				ssaop.OpNeg64, ssaop.OpCom64,
				ssaop.OpTrunc64to8, ssaop.OpTrunc64to16, ssaop.OpTrunc64to32:
				argSize = 8
			}
			if argSize != 0 {
				for i, arg := range v.Args {
					if arg.Type.Size() != argSize {
						f.Fatalf("arg %d to %s (%v) should be %d bytes in size, it is %s", i, v.Op, v, argSize, arg.Type.String())
					}
				}
			}

			// TODO: check for cycles in values
		}
	}

	// Check to make sure all Blocks referenced are in the function.
	if !blockMark[f.Entry.ID] {
		f.Fatalf("entry block %v is missing", f.Entry)
	}
	for _, b := range f.Blocks {
		for _, c := range b.Preds {
			if !blockMark[c.B.ID] {
				f.Fatalf("predecessor block %v for %v is missing", c, b)
			}
		}
		for _, c := range b.Succs {
			if !blockMark[c.B.ID] {
				f.Fatalf("successor block %v for %v is missing", c, b)
			}
		}
	}

	if len(f.Entry.Preds) > 0 {
		f.Fatalf("entry block %s of %s has predecessor(s) %v", f.Entry, f.Name, f.Entry.Preds)
	}

	// Check to make sure all Values referenced are in the function.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			for i, a := range v.Args {
				if !valueMark[a.ID] {
					f.Fatalf("%v, arg %d of %s, is missing", a, i, v.LongString())
				}
			}
		}
		for _, c := range b.ControlValues() {
			if !valueMark[c.ID] {
				f.Fatalf("control value for %s is missing: %v", b, c)
			}
		}
	}
	for b := f.FreeBlocks; b != nil; b = b.Succstorage[0].B {
		if blockMark[b.ID] {
			f.Fatalf("used block b%d in free list", b.ID)
		}
	}
	for v := f.FreeValues; v != nil; v = v.Argstorage[0] {
		if valueMark[v.ID] {
			f.Fatalf("used value v%d in free list", v.ID)
		}
	}

	// Check to make sure all args dominate uses.
	if f.RegAlloc == nil {
		// Note: regalloc introduces non-dominating args.
		// See TODO in regalloc.go.
		sdom := f.Sdom()
		for _, b := range f.Blocks {
			for _, v := range b.Values {
				for i, arg := range v.Args {
					x := arg.Block
					y := b
					if v.Op == ssaop.OpPhi {
						y = b.Preds[i].B
					}
					if !domCheck(f, sdom, x, y) {
						f.Fatalf("arg %d of value %s does not dominate, arg=%s", i, v.LongString(), arg.LongString())
					}
				}
			}
			for _, c := range b.ControlValues() {
				if !domCheck(f, sdom, c.Block, b) {
					f.Fatalf("control value %s for %s doesn't dominate", c, b)
				}
			}
		}
	}

	// Check loop construction
	if f.RegAlloc == nil && f.Pass != nil { // non-nil pass allows better-targeted debug printing
		ln := f.Loopnest()
		if !ln.HasIrreducible {
			po := f.Postorder() // use po to avoid unreachable blocks.
			for _, b := range po {
				for _, s := range b.Succs {
					bb := s.Block()
					if ln.B2L[b.ID] == nil && ln.B2L[bb.ID] != nil && bb != ln.B2L[bb.ID].Header {
						f.Fatalf("block %s not in loop branches to non-header block %s in loop", b.String(), bb.String())
					}
					if ln.B2L[b.ID] != nil && ln.B2L[bb.ID] != nil && bb != ln.B2L[bb.ID].Header && !ln.B2L[b.ID].IsWithinOrEq(ln.B2L[bb.ID]) {
						f.Fatalf("block %s in loop branches to non-header block %s in non-containing loop", b.String(), bb.String())
					}
				}
			}
		}
	}

	// Check use counts
	uses := make([]int32, f.NumValues())
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			for _, a := range v.Args {
				uses[a.ID]++
			}
		}
		for _, c := range b.ControlValues() {
			uses[c.ID]++
		}
	}
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Uses != uses[v.ID] {
				f.Fatalf("%s has %d uses, but has Uses=%d", v, uses[v.ID], v.Uses)
			}
		}
	}

	memCheck(f)
}

func memCheck(f *ssacore.Func) {
	// Check that if a tuple has a memory type, it is second.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Type.IsTuple() && v.Type.FieldType(0).IsMemory() {
				f.Fatalf("memory is first in a tuple: %s\n", v.LongString())
			}
		}
	}

	// Single live memory checks.
	// These checks only work if there are no memory copies.
	// (Memory copies introduce ambiguity about which mem value is really live.
	// probably fixable, but it's easier to avoid the problem.)
	// For the same reason, disable this check if some memory ops are unused.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if (v.Op == ssaop.OpCopy || v.Uses == 0) && v.Type.IsMemory() {
				return
			}
		}
		if b != f.Entry && len(b.Preds) == 0 {
			return
		}
	}

	// Compute live memory at the end of each block.
	lastmem := make([]*ssacore.Value, f.NumBlocks())
	ss := ssacore.NewSparseSet(f.NumValues())
	for _, b := range f.Blocks {
		// Mark overwritten memory values. Those are args of other
		// ops that generate memory values.
		ss.Clear()
		for _, v := range b.Values {
			if v.Op == ssaop.OpPhi || !v.Type.IsMemory() {
				continue
			}
			if m := v.MemoryArg(); m != nil {
				ss.Add(m.ID)
			}
		}
		// There should be at most one remaining unoverwritten memory value.
		for _, v := range b.Values {
			if !v.Type.IsMemory() {
				continue
			}
			if ss.Contains(v.ID) {
				continue
			}
			if lastmem[b.ID] != nil {
				f.Fatalf("two live memory values in %s: %s and %s", b, lastmem[b.ID], v)
			}
			lastmem[b.ID] = v
		}
		// If there is no remaining memory value, that means there was no memory update.
		// Take any memory arg.
		if lastmem[b.ID] == nil {
			for _, v := range b.Values {
				if v.Op == ssaop.OpPhi {
					continue
				}
				m := v.MemoryArg()
				if m == nil {
					continue
				}
				if lastmem[b.ID] != nil && lastmem[b.ID] != m {
					f.Fatalf("two live memory values in %s: %s and %s", b, lastmem[b.ID], m)
				}
				lastmem[b.ID] = m
			}
		}
	}
	// Propagate last live memory through storeless blocks.
	for {
		changed := false
		for _, b := range f.Blocks {
			if lastmem[b.ID] != nil {
				continue
			}
			for _, e := range b.Preds {
				p := e.B
				if lastmem[p.ID] != nil {
					lastmem[b.ID] = lastmem[p.ID]
					changed = true
					break
				}
			}
		}
		if !changed {
			break
		}
	}
	// Check merge points.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op == ssaop.OpPhi && v.Type.IsMemory() {
				for i, a := range v.Args {
					if a != lastmem[b.Preds[i].B.ID] {
						f.Fatalf("inconsistent memory phi %s %d %s %s", v.LongString(), i, a, lastmem[b.Preds[i].B.ID])
					}
				}
			}
		}
	}

	// Check that only one memory is live at any point.
	if f.Scheduled {
		for _, b := range f.Blocks {
			var mem *ssacore.Value // the current live memory in the block
			for _, v := range b.Values {
				if v.Op == ssaop.OpPhi {
					if v.Type.IsMemory() {
						mem = v
					}
					continue
				}
				if mem == nil && len(b.Preds) > 0 {
					// If no mem phi, take mem of any predecessor.
					mem = lastmem[b.Preds[0].B.ID]
				}
				for _, a := range v.Args {
					if a.Type.IsMemory() && a != mem {
						f.Fatalf("two live mems @ %s: %s and %s", v, mem, a)
					}
				}
				if v.Type.IsMemory() {
					mem = v
				}
			}
		}
	}

	// Check that after scheduling, phis are always first in the block.
	if f.Scheduled {
		for _, b := range f.Blocks {
			seenNonPhi := false
			for _, v := range b.Values {
				switch v.Op {
				case ssaop.OpPhi:
					if seenNonPhi {
						f.Fatalf("phi after non-phi @ %s: %s", b, v)
					}
				default:
					seenNonPhi = true
				}
			}
		}
	}
}

// domCheck reports whether x dominates y (including x==y).
func domCheck(f *ssacore.Func, sdom ssacore.SparseTree, x, y *ssacore.Block) bool {
	if !sdom.IsAncestorEq(f.Entry, y) {
		// unreachable - ignore
		return true
	}
	return sdom.IsAncestorEq(x, y)
}

// isExactFloat32 reports whether x can be exactly represented as a float32.
func isExactFloat32(x float64) bool {
	// Check the mantissa is in range.
	if bits.TrailingZeros64(math.Float64bits(x)) < 52-23 {
		return false
	}
	// Check the exponent is in range. The mantissa check above is sufficient for NaN values.
	return math.IsNaN(x) || x == float64(float32(x))
}
