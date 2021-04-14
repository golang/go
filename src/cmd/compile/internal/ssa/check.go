// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/internal/obj/s390x"
	"math"
	"math/bits"
)

// checkFunc checks invariants of f.
func checkFunc(f *Func) {
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
			if se := e.b.Succs[e.i]; se.b != b || se.i != i {
				f.Fatalf("block pred/succ not crosslinked correctly %d:%s %d:%s", i, b, se.i, se.b)
			}
		}
		for i, e := range b.Succs {
			if pe := e.b.Preds[e.i]; pe.b != b || pe.i != i {
				f.Fatalf("block succ/pred not crosslinked correctly %d:%s %d:%s", i, b, pe.i, pe.b)
			}
		}

		switch b.Kind {
		case BlockExit:
			if len(b.Succs) != 0 {
				f.Fatalf("exit block %s has successors", b)
			}
			if b.NumControls() != 1 {
				f.Fatalf("exit block %s has no control value", b)
			}
			if !b.Controls[0].Type.IsMemory() {
				f.Fatalf("exit block %s has non-memory control value %s", b, b.Controls[0].LongString())
			}
		case BlockRet:
			if len(b.Succs) != 0 {
				f.Fatalf("ret block %s has successors", b)
			}
			if b.NumControls() != 1 {
				f.Fatalf("ret block %s has nil control", b)
			}
			if !b.Controls[0].Type.IsMemory() {
				f.Fatalf("ret block %s has non-memory control value %s", b, b.Controls[0].LongString())
			}
		case BlockRetJmp:
			if len(b.Succs) != 0 {
				f.Fatalf("retjmp block %s len(Succs)==%d, want 0", b, len(b.Succs))
			}
			if b.NumControls() != 1 {
				f.Fatalf("retjmp block %s has nil control", b)
			}
			if !b.Controls[0].Type.IsMemory() {
				f.Fatalf("retjmp block %s has non-memory control value %s", b, b.Controls[0].LongString())
			}
			if b.Aux == nil {
				f.Fatalf("retjmp block %s has nil Aux field", b)
			}
		case BlockPlain:
			if len(b.Succs) != 1 {
				f.Fatalf("plain block %s len(Succs)==%d, want 1", b, len(b.Succs))
			}
			if b.NumControls() != 0 {
				f.Fatalf("plain block %s has non-nil control %s", b, b.Controls[0].LongString())
			}
		case BlockIf:
			if len(b.Succs) != 2 {
				f.Fatalf("if block %s len(Succs)==%d, want 2", b, len(b.Succs))
			}
			if b.NumControls() != 1 {
				f.Fatalf("if block %s has no control value", b)
			}
			if !b.Controls[0].Type.IsBoolean() {
				f.Fatalf("if block %s has non-bool control value %s", b, b.Controls[0].LongString())
			}
		case BlockDefer:
			if len(b.Succs) != 2 {
				f.Fatalf("defer block %s len(Succs)==%d, want 2", b, len(b.Succs))
			}
			if b.NumControls() != 1 {
				f.Fatalf("defer block %s has no control value", b)
			}
			if !b.Controls[0].Type.IsMemory() {
				f.Fatalf("defer block %s has non-memory control value %s", b, b.Controls[0].LongString())
			}
		case BlockFirst:
			if len(b.Succs) != 2 {
				f.Fatalf("plain/dead block %s len(Succs)==%d, want 2", b, len(b.Succs))
			}
			if b.NumControls() != 0 {
				f.Fatalf("plain/dead block %s has a control value", b)
			}
		}
		if len(b.Succs) != 2 && b.Likely != BranchUnknown {
			f.Fatalf("likeliness prediction %d for block %s with %d successors", b.Likely, b, len(b.Succs))
		}

		for _, v := range b.Values {
			// Check to make sure argument count makes sense (argLen of -1 indicates
			// variable length args)
			nArgs := opcodeTable[v.Op].argLen
			if nArgs != -1 && int32(len(v.Args)) != nArgs {
				f.Fatalf("value %s has %d args, expected %d", v.LongString(),
					len(v.Args), nArgs)
			}

			// Check to make sure aux values make sense.
			canHaveAux := false
			canHaveAuxInt := false
			// TODO: enforce types of Aux in this switch (like auxString does below)
			switch opcodeTable[v.Op].auxType {
			case auxNone:
			case auxBool:
				if v.AuxInt < 0 || v.AuxInt > 1 {
					f.Fatalf("bad bool AuxInt value for %v", v)
				}
				canHaveAuxInt = true
			case auxInt8:
				if v.AuxInt != int64(int8(v.AuxInt)) {
					f.Fatalf("bad int8 AuxInt value for %v", v)
				}
				canHaveAuxInt = true
			case auxInt16:
				if v.AuxInt != int64(int16(v.AuxInt)) {
					f.Fatalf("bad int16 AuxInt value for %v", v)
				}
				canHaveAuxInt = true
			case auxInt32:
				if v.AuxInt != int64(int32(v.AuxInt)) {
					f.Fatalf("bad int32 AuxInt value for %v", v)
				}
				canHaveAuxInt = true
			case auxInt64, auxARM64BitField:
				canHaveAuxInt = true
			case auxInt128:
				// AuxInt must be zero, so leave canHaveAuxInt set to false.
			case auxFloat32:
				canHaveAuxInt = true
				if math.IsNaN(v.AuxFloat()) {
					f.Fatalf("value %v has an AuxInt that encodes a NaN", v)
				}
				if !isExactFloat32(v.AuxFloat()) {
					f.Fatalf("value %v has an AuxInt value that is not an exact float32", v)
				}
			case auxFloat64:
				canHaveAuxInt = true
				if math.IsNaN(v.AuxFloat()) {
					f.Fatalf("value %v has an AuxInt that encodes a NaN", v)
				}
			case auxString:
				if _, ok := v.Aux.(string); !ok {
					f.Fatalf("value %v has Aux type %T, want string", v, v.Aux)
				}
				canHaveAux = true
			case auxSym, auxTyp:
				canHaveAux = true
			case auxSymOff, auxSymValAndOff, auxTypSize:
				canHaveAuxInt = true
				canHaveAux = true
			case auxCCop:
				if _, ok := v.Aux.(Op); !ok {
					f.Fatalf("bad type %T for CCop in %v", v.Aux, v)
				}
				canHaveAux = true
			case auxS390XCCMask:
				if _, ok := v.Aux.(s390x.CCMask); !ok {
					f.Fatalf("bad type %T for S390XCCMask in %v", v.Aux, v)
				}
				canHaveAux = true
			case auxS390XRotateParams:
				if _, ok := v.Aux.(s390x.RotateParams); !ok {
					f.Fatalf("bad type %T for S390XRotateParams in %v", v.Aux, v)
				}
				canHaveAux = true
			case auxFlagConstant:
				if v.AuxInt < 0 || v.AuxInt > 15 {
					f.Fatalf("bad FlagConstant AuxInt value for %v", v)
				}
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
				if v.Op != OpPhi {
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
			if v.Op == OpPhi && len(v.Args) != len(b.Preds) {
				f.Fatalf("phi length %s does not match pred length %d for block %s", v.LongString(), len(b.Preds), b)
			}

			if v.Op == OpAddr {
				if len(v.Args) == 0 {
					f.Fatalf("no args for OpAddr %s", v.LongString())
				}
				if v.Args[0].Op != OpSB {
					f.Fatalf("bad arg to OpAddr %v", v)
				}
			}

			if v.Op == OpLocalAddr {
				if len(v.Args) != 2 {
					f.Fatalf("wrong # of args for OpLocalAddr %s", v.LongString())
				}
				if v.Args[0].Op != OpSP {
					f.Fatalf("bad arg 0 to OpLocalAddr %v", v)
				}
				if !v.Args[1].Type.IsMemory() {
					f.Fatalf("bad arg 1 to OpLocalAddr %v", v)
				}
			}

			if f.RegAlloc != nil && f.Config.SoftFloat && v.Type.IsFloat() {
				f.Fatalf("unexpected floating-point type %v", v.LongString())
			}

			// Check types.
			// TODO: more type checks?
			switch c := f.Config; v.Op {
			case OpSP, OpSB:
				if v.Type != c.Types.Uintptr {
					f.Fatalf("bad %s type: want uintptr, have %s",
						v.Op, v.Type.String())
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
			if !blockMark[c.b.ID] {
				f.Fatalf("predecessor block %v for %v is missing", c, b)
			}
		}
		for _, c := range b.Succs {
			if !blockMark[c.b.ID] {
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
	for b := f.freeBlocks; b != nil; b = b.succstorage[0].b {
		if blockMark[b.ID] {
			f.Fatalf("used block b%d in free list", b.ID)
		}
	}
	for v := f.freeValues; v != nil; v = v.argstorage[0] {
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
					if v.Op == OpPhi {
						y = b.Preds[i].b
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
	if f.RegAlloc == nil && f.pass != nil { // non-nil pass allows better-targeted debug printing
		ln := f.loopnest()
		if !ln.hasIrreducible {
			po := f.postorder() // use po to avoid unreachable blocks.
			for _, b := range po {
				for _, s := range b.Succs {
					bb := s.Block()
					if ln.b2l[b.ID] == nil && ln.b2l[bb.ID] != nil && bb != ln.b2l[bb.ID].header {
						f.Fatalf("block %s not in loop branches to non-header block %s in loop", b.String(), bb.String())
					}
					if ln.b2l[b.ID] != nil && ln.b2l[bb.ID] != nil && bb != ln.b2l[bb.ID].header && !ln.b2l[b.ID].isWithinOrEq(ln.b2l[bb.ID]) {
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

func memCheck(f *Func) {
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
			if (v.Op == OpCopy || v.Uses == 0) && v.Type.IsMemory() {
				return
			}
		}
		if b != f.Entry && len(b.Preds) == 0 {
			return
		}
	}

	// Compute live memory at the end of each block.
	lastmem := make([]*Value, f.NumBlocks())
	ss := newSparseSet(f.NumValues())
	for _, b := range f.Blocks {
		// Mark overwritten memory values. Those are args of other
		// ops that generate memory values.
		ss.clear()
		for _, v := range b.Values {
			if v.Op == OpPhi || !v.Type.IsMemory() {
				continue
			}
			if m := v.MemoryArg(); m != nil {
				ss.add(m.ID)
			}
		}
		// There should be at most one remaining unoverwritten memory value.
		for _, v := range b.Values {
			if !v.Type.IsMemory() {
				continue
			}
			if ss.contains(v.ID) {
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
				if v.Op == OpPhi {
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
				p := e.b
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
			if v.Op == OpPhi && v.Type.IsMemory() {
				for i, a := range v.Args {
					if a != lastmem[b.Preds[i].b.ID] {
						f.Fatalf("inconsistent memory phi %s %d %s %s", v.LongString(), i, a, lastmem[b.Preds[i].b.ID])
					}
				}
			}
		}
	}

	// Check that only one memory is live at any point.
	if f.scheduled {
		for _, b := range f.Blocks {
			var mem *Value // the current live memory in the block
			for _, v := range b.Values {
				if v.Op == OpPhi {
					if v.Type.IsMemory() {
						mem = v
					}
					continue
				}
				if mem == nil && len(b.Preds) > 0 {
					// If no mem phi, take mem of any predecessor.
					mem = lastmem[b.Preds[0].b.ID]
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
	if f.scheduled {
		for _, b := range f.Blocks {
			seenNonPhi := false
			for _, v := range b.Values {
				switch v.Op {
				case OpPhi:
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
func domCheck(f *Func, sdom SparseTree, x, y *Block) bool {
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
