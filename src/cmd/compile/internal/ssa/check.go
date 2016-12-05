// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

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
			if b.Control == nil {
				f.Fatalf("exit block %s has no control value", b)
			}
			if !b.Control.Type.IsMemory() {
				f.Fatalf("exit block %s has non-memory control value %s", b, b.Control.LongString())
			}
		case BlockRet:
			if len(b.Succs) != 0 {
				f.Fatalf("ret block %s has successors", b)
			}
			if b.Control == nil {
				f.Fatalf("ret block %s has nil control", b)
			}
			if !b.Control.Type.IsMemory() {
				f.Fatalf("ret block %s has non-memory control value %s", b, b.Control.LongString())
			}
		case BlockRetJmp:
			if len(b.Succs) != 0 {
				f.Fatalf("retjmp block %s len(Succs)==%d, want 0", b, len(b.Succs))
			}
			if b.Control == nil {
				f.Fatalf("retjmp block %s has nil control", b)
			}
			if !b.Control.Type.IsMemory() {
				f.Fatalf("retjmp block %s has non-memory control value %s", b, b.Control.LongString())
			}
			if b.Aux == nil {
				f.Fatalf("retjmp block %s has nil Aux field", b)
			}
		case BlockPlain:
			if len(b.Succs) != 1 {
				f.Fatalf("plain block %s len(Succs)==%d, want 1", b, len(b.Succs))
			}
			if b.Control != nil {
				f.Fatalf("plain block %s has non-nil control %s", b, b.Control.LongString())
			}
		case BlockIf:
			if len(b.Succs) != 2 {
				f.Fatalf("if block %s len(Succs)==%d, want 2", b, len(b.Succs))
			}
			if b.Control == nil {
				f.Fatalf("if block %s has no control value", b)
			}
			if !b.Control.Type.IsBoolean() {
				f.Fatalf("if block %s has non-bool control value %s", b, b.Control.LongString())
			}
		case BlockDefer:
			if len(b.Succs) != 2 {
				f.Fatalf("defer block %s len(Succs)==%d, want 2", b, len(b.Succs))
			}
			if b.Control == nil {
				f.Fatalf("defer block %s has no control value", b)
			}
			if !b.Control.Type.IsMemory() {
				f.Fatalf("defer block %s has non-memory control value %s", b, b.Control.LongString())
			}
		case BlockFirst:
			if len(b.Succs) != 2 {
				f.Fatalf("plain/dead block %s len(Succs)==%d, want 2", b, len(b.Succs))
			}
			if b.Control != nil {
				f.Fatalf("plain/dead block %s has a control value", b)
			}
		}
		if len(b.Succs) > 2 && b.Likely != BranchUnknown {
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
			case auxInt64, auxFloat64:
				canHaveAuxInt = true
			case auxInt128:
				// AuxInt must be zero, so leave canHaveAuxInt set to false.
			case auxFloat32:
				canHaveAuxInt = true
				if !isExactFloat32(v) {
					f.Fatalf("value %v has an AuxInt value that is not an exact float32", v)
				}
			case auxSizeAndAlign:
				canHaveAuxInt = true
			case auxString, auxSym:
				canHaveAux = true
			case auxSymOff, auxSymValAndOff, auxSymSizeAndAlign:
				canHaveAuxInt = true
				canHaveAux = true
			case auxSymInt32:
				if v.AuxInt != int64(int32(v.AuxInt)) {
					f.Fatalf("bad int32 AuxInt value for %v", v)
				}
				canHaveAuxInt = true
				canHaveAux = true
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
				if v.Args[0].Op != OpSP && v.Args[0].Op != OpSB {
					f.Fatalf("bad arg to OpAddr %v", v)
				}
			}

			// TODO: check for cycles in values
			// TODO: check type
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
		if b.Control != nil && !valueMark[b.Control.ID] {
			f.Fatalf("control value for %s is missing: %v", b, b.Control)
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
		sdom := f.sdom()
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
			if b.Control != nil && !domCheck(f, sdom, b.Control.Block, b) {
				f.Fatalf("control value %s for %s doesn't dominate", b.Control, b)
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
		if b.Control != nil {
			uses[b.Control.ID]++
		}
	}
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Uses != uses[v.ID] {
				f.Fatalf("%s has %d uses, but has Uses=%d", v, uses[v.ID], v.Uses)
			}
		}
	}
}

// domCheck reports whether x dominates y (including x==y).
func domCheck(f *Func, sdom SparseTree, x, y *Block) bool {
	if !sdom.isAncestorEq(f.Entry, y) {
		// unreachable - ignore
		return true
	}
	return sdom.isAncestorEq(x, y)
}

// isExactFloat32 reoprts whether v has an AuxInt that can be exactly represented as a float32.
func isExactFloat32(v *Value) bool {
	return v.AuxFloat() == float64(float32(v.AuxFloat()))
}
