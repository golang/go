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

		if f.RegAlloc == nil {
			for i, c := range b.Succs {
				for j, d := range b.Succs {
					if i != j && c == d {
						f.Fatalf("%s.Succs has duplicate block %s", b, c)
					}
				}
			}
		}
		// Note: duplicate successors are hard in the following case:
		//      if(...) goto x else goto x
		//   x: v = phi(a, b)
		// If the conditional is true, does v get the value of a or b?
		// We could solve this other ways, but the easiest is just to
		// require (by possibly adding empty control-flow blocks) that
		// all successors are distinct.  They will need to be distinct
		// anyway for register allocation (duplicate successors implies
		// the existence of critical edges).
		// After regalloc we can allow non-distinct predecessors.

		for _, p := range b.Preds {
			var found bool
			for _, c := range p.Succs {
				if c == b {
					found = true
					break
				}
			}
			if !found {
				f.Fatalf("block %s is not a succ of its pred block %s", b, p)
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
				f.Fatalf("ret block %s has nil control %s", b)
			}
			if !b.Control.Type.IsMemory() {
				f.Fatalf("ret block %s has non-memory control value %s", b, b.Control.LongString())
			}
		case BlockRetJmp:
			if len(b.Succs) != 0 {
				f.Fatalf("retjmp block %s len(Succs)==%d, want 0", b, len(b.Succs))
			}
			if b.Control == nil {
				f.Fatalf("retjmp block %s has nil control %s", b)
			}
			if !b.Control.Type.IsMemory() {
				f.Fatalf("retjmp block %s has non-memory control value %s", b, b.Control.LongString())
			}
			if b.Aux == nil {
				f.Fatalf("retjmp block %s has nil Aux field", b)
			}
		case BlockDead:
			if len(b.Succs) != 0 {
				f.Fatalf("dead block %s has successors", b)
			}
			if len(b.Preds) != 0 {
				f.Fatalf("dead block %s has predecessors", b)
			}
			if len(b.Values) != 0 {
				f.Fatalf("dead block %s has values", b)
			}
			if b.Control != nil {
				f.Fatalf("dead block %s has a control value", b)
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
		case BlockCall:
			if len(b.Succs) != 1 {
				f.Fatalf("call block %s len(Succs)==%d, want 1", b, len(b.Succs))
			}
			if b.Control == nil {
				f.Fatalf("call block %s has no control value", b)
			}
			if !b.Control.Type.IsMemory() {
				f.Fatalf("call block %s has non-memory control value %s", b, b.Control.LongString())
			}
		case BlockCheck:
			if len(b.Succs) != 1 {
				f.Fatalf("check block %s len(Succs)==%d, want 1", b, len(b.Succs))
			}
			if b.Control == nil {
				f.Fatalf("check block %s has no control value", b)
			}
			if !b.Control.Type.IsVoid() {
				f.Fatalf("check block %s has non-void control value %s", b, b.Control.LongString())
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
			f.Fatalf("likeliness prediction %d for block %s with %d successors: %s", b.Likely, b, len(b.Succs))
		}

		for _, v := range b.Values {
			// Check to make sure argument count makes sense (argLen of -1 indicates
			// variable length args)
			nArgs := opcodeTable[v.Op].argLen
			if nArgs != -1 && int32(len(v.Args)) != nArgs {
				f.Fatalf("value %v has %d args, expected %d", v.LongString(),
					len(v.Args), nArgs)
			}

			// Check to make sure aux values make sense.
			canHaveAux := false
			canHaveAuxInt := false
			switch opcodeTable[v.Op].auxType {
			case auxNone:
			case auxBool, auxInt8, auxInt16, auxInt32, auxInt64, auxFloat:
				canHaveAuxInt = true
			case auxString, auxSym:
				canHaveAux = true
			case auxSymOff, auxSymValAndOff:
				canHaveAuxInt = true
				canHaveAux = true
			default:
				f.Fatalf("unknown aux type for %s", v.Op)
			}
			if !canHaveAux && v.Aux != nil {
				f.Fatalf("value %v has an Aux value %v but shouldn't", v.LongString(), v.Aux)
			}
			if !canHaveAuxInt && v.AuxInt != 0 {
				f.Fatalf("value %v has an AuxInt value %d but shouldn't", v.LongString(), v.AuxInt)
			}

			for _, arg := range v.Args {
				if arg == nil {
					f.Fatalf("value %v has nil arg", v.LongString())
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
			if !blockMark[c.ID] {
				f.Fatalf("predecessor block %v for %v is missing", c, b)
			}
		}
		for _, c := range b.Succs {
			if !blockMark[c.ID] {
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
					f.Fatalf("%v, arg %d of %v, is missing", a, i, v)
				}
			}
		}
		if b.Control != nil && !valueMark[b.Control.ID] {
			f.Fatalf("control value for %s is missing: %v", b, b.Control)
		}
	}
	for b := f.freeBlocks; b != nil; b = b.succstorage[0] {
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
		idom := dominators(f)
		sdom := newSparseTree(f, idom)
		for _, b := range f.Blocks {
			for _, v := range b.Values {
				for i, arg := range v.Args {
					x := arg.Block
					y := b
					if v.Op == OpPhi {
						y = b.Preds[i]
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
}

// domCheck reports whether x dominates y (including x==y).
func domCheck(f *Func, sdom sparseTree, x, y *Block) bool {
	if !sdom.isAncestorEq(y, f.Entry) {
		// unreachable - ignore
		return true
	}
	return sdom.isAncestorEq(x, y)
}
