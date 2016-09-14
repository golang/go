// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Portable half of code generator; mainly statements and control flow.

package gc

import "fmt"

// TODO: labellist should become part of a "compilation state" for functions.
var labellist []*Label

func Sysfunc(name string) *Node {
	n := newname(Pkglookup(name, Runtimepkg))
	n.Class = PFUNC
	return n
}

// addrescapes tags node n as having had its address taken
// by "increasing" the "value" of n.Esc to EscHeap.
// Storage is allocated as necessary to allow the address
// to be taken.
func addrescapes(n *Node) {
	switch n.Op {
	// probably a type error already.
	// dump("addrescapes", n);
	default:
		break

	case ONAME:
		if n == nodfp {
			break
		}

		// if this is a tmpname (PAUTO), it was tagged by tmpname as not escaping.
		// on PPARAM it means something different.
		if n.Class == PAUTO && n.Esc == EscNever {
			break
		}

		// If a closure reference escapes, mark the outer variable as escaping.
		if n.isClosureVar() {
			addrescapes(n.Name.Defn)
			break
		}

		if n.Class != PPARAM && n.Class != PPARAMOUT && n.Class != PAUTO {
			break
		}

		// This is a plain parameter or local variable that needs to move to the heap,
		// but possibly for the function outside the one we're compiling.
		// That is, if we have:
		//
		//	func f(x int) {
		//		func() {
		//			global = &x
		//		}
		//	}
		//
		// then we're analyzing the inner closure but we need to move x to the
		// heap in f, not in the inner closure. Flip over to f before calling moveToHeap.
		oldfn := Curfn
		Curfn = n.Name.Curfn
		if Curfn.Func.Closure != nil && Curfn.Op == OCLOSURE {
			Curfn = Curfn.Func.Closure
		}
		ln := lineno
		lineno = Curfn.Lineno
		moveToHeap(n)
		Curfn = oldfn
		lineno = ln

	case OIND, ODOTPTR:
		break

	// ODOTPTR has already been introduced,
	// so these are the non-pointer ODOT and OINDEX.
	// In &x[0], if x is a slice, then x does not
	// escape--the pointer inside x does, but that
	// is always a heap pointer anyway.
	case ODOT, OINDEX, OPAREN, OCONVNOP:
		if !n.Left.Type.IsSlice() {
			addrescapes(n.Left)
		}
	}
}

// isParamStackCopy reports whether this is the on-stack copy of a
// function parameter that moved to the heap.
func (n *Node) isParamStackCopy() bool {
	return n.Op == ONAME && (n.Class == PPARAM || n.Class == PPARAMOUT) && n.Name.Heapaddr != nil
}

// isParamHeapCopy reports whether this is the on-heap copy of
// a function parameter that moved to the heap.
func (n *Node) isParamHeapCopy() bool {
	return n.Op == ONAME && n.Class == PAUTOHEAP && n.Name.Param.Stackcopy != nil
}

// moveToHeap records the parameter or local variable n as moved to the heap.
func moveToHeap(n *Node) {
	if Debug['r'] != 0 {
		Dump("MOVE", n)
	}
	if compiling_runtime {
		Yyerror("%v escapes to heap, not allowed in runtime.", n)
	}
	if n.Class == PAUTOHEAP {
		Dump("n", n)
		Fatalf("double move to heap")
	}

	// Allocate a local stack variable to hold the pointer to the heap copy.
	// temp will add it to the function declaration list automatically.
	heapaddr := temp(Ptrto(n.Type))
	heapaddr.Sym = Lookup("&" + n.Sym.Name)
	heapaddr.Orig.Sym = heapaddr.Sym

	// Parameters have a local stack copy used at function start/end
	// in addition to the copy in the heap that may live longer than
	// the function.
	if n.Class == PPARAM || n.Class == PPARAMOUT {
		if n.Xoffset == BADWIDTH {
			Fatalf("addrescapes before param assignment")
		}

		// We rewrite n below to be a heap variable (indirection of heapaddr).
		// Preserve a copy so we can still write code referring to the original,
		// and substitute that copy into the function declaration list
		// so that analyses of the local (on-stack) variables use it.
		stackcopy := Nod(ONAME, nil, nil)
		stackcopy.Sym = n.Sym
		stackcopy.Type = n.Type
		stackcopy.Xoffset = n.Xoffset
		stackcopy.Class = n.Class
		stackcopy.Name.Heapaddr = heapaddr
		if n.Class == PPARAM {
			stackcopy.SetNotLiveAtEnd(true)
		}
		if n.Class == PPARAMOUT {
			// Make sure the pointer to the heap copy is kept live throughout the function.
			// The function could panic at any point, and then a defer could recover.
			// Thus, we need the pointer to the heap copy always available so the
			// post-deferreturn code can copy the return value back to the stack.
			// See issue 16095.
			heapaddr.setIsOutputParamHeapAddr(true)
		}
		n.Name.Param.Stackcopy = stackcopy

		// Substitute the stackcopy into the function variable list so that
		// liveness and other analyses use the underlying stack slot
		// and not the now-pseudo-variable n.
		found := false
		for i, d := range Curfn.Func.Dcl {
			if d == n {
				Curfn.Func.Dcl[i] = stackcopy
				found = true
				break
			}
			// Parameters are before locals, so can stop early.
			// This limits the search even in functions with many local variables.
			if d.Class == PAUTO {
				break
			}
		}
		if !found {
			Fatalf("cannot find %v in local variable list", n)
		}
		Curfn.Func.Dcl = append(Curfn.Func.Dcl, n)
	}

	// Modify n in place so that uses of n now mean indirection of the heapaddr.
	n.Class = PAUTOHEAP
	n.Ullman = 2
	n.Xoffset = 0
	n.Name.Heapaddr = heapaddr
	n.Esc = EscHeap
	if Debug['m'] != 0 {
		fmt.Printf("%v: moved to heap: %v\n", n.Line(), n)
	}
}

func clearlabels() {
	for _, l := range labellist {
		l.Sym.Label = nil
	}
	labellist = labellist[:0]
}

func newlab(n *Node) *Label {
	s := n.Left.Sym
	lab := s.Label
	if lab == nil {
		lab = new(Label)
		lab.Sym = s
		s.Label = lab
		if n.Used {
			lab.Used = true
		}
		labellist = append(labellist, lab)
	}

	if n.Op == OLABEL {
		if lab.Def != nil {
			Yyerror("label %v already defined at %v", s, lab.Def.Line())
		} else {
			lab.Def = n
		}
	} else {
		lab.Use = append(lab.Use, n)
	}

	return lab
}

// There is a copy of checkgoto in the new SSA backend.
// Please keep them in sync.
func checkgoto(from *Node, to *Node) {
	if from.Sym == to.Sym {
		return
	}

	nf := 0
	for fs := from.Sym; fs != nil; fs = fs.Link {
		nf++
	}
	nt := 0
	for fs := to.Sym; fs != nil; fs = fs.Link {
		nt++
	}
	fs := from.Sym
	for ; nf > nt; nf-- {
		fs = fs.Link
	}
	if fs != to.Sym {
		lno := lineno
		setlineno(from)

		// decide what to complain about.
		// prefer to complain about 'into block' over declarations,
		// so scan backward to find most recent block or else dcl.
		var block *Sym

		var dcl *Sym
		ts := to.Sym
		for ; nt > nf; nt-- {
			if ts.Pkg == nil {
				block = ts
			} else {
				dcl = ts
			}
			ts = ts.Link
		}

		for ts != fs {
			if ts.Pkg == nil {
				block = ts
			} else {
				dcl = ts
			}
			ts = ts.Link
			fs = fs.Link
		}

		if block != nil {
			Yyerror("goto %v jumps into block starting at %v", from.Left.Sym, linestr(block.Lastlineno))
		} else {
			Yyerror("goto %v jumps over declaration of %v at %v", from.Left.Sym, dcl, linestr(dcl.Lastlineno))
		}
		lineno = lno
	}
}

func stmtlabel(n *Node) *Label {
	if n.Sym != nil {
		lab := n.Sym.Label
		if lab != nil {
			if lab.Def != nil {
				if lab.Def.Name.Defn == n {
					return lab
				}
			}
		}
	}
	return nil
}

// make a new off the books
func Tempname(nn *Node, t *Type) {
	if Curfn == nil {
		Fatalf("no curfn for tempname")
	}
	if Curfn.Func.Closure != nil && Curfn.Op == OCLOSURE {
		Dump("Tempname", Curfn)
		Fatalf("adding tempname to wrong closure function")
	}

	if t == nil {
		Yyerror("tempname called with nil type")
		t = Types[TINT32]
	}

	// give each tmp a different name so that there
	// a chance to registerizer them
	s := LookupN("autotmp_", statuniqgen)
	statuniqgen++
	n := Nod(ONAME, nil, nil)
	n.Sym = s
	s.Def = n
	n.Type = t
	n.Class = PAUTO
	n.Addable = true
	n.Ullman = 1
	n.Esc = EscNever
	n.Name.Curfn = Curfn
	Curfn.Func.Dcl = append(Curfn.Func.Dcl, n)

	dowidth(t)
	n.Xoffset = 0
	*nn = *n
}

func temp(t *Type) *Node {
	var n Node
	Tempname(&n, t)
	n.Sym.Def.Used = true
	return n.Orig
}

func checklabels() {
	for _, lab := range labellist {
		if lab.Def == nil {
			for _, n := range lab.Use {
				yyerrorl(n.Lineno, "label %v not defined", lab.Sym)
			}
			continue
		}

		if lab.Use == nil && !lab.Used {
			yyerrorl(lab.Def.Lineno, "label %v defined and not used", lab.Sym)
			continue
		}

		if lab.Gotopc != nil {
			Fatalf("label %v never resolved", lab.Sym)
		}
		for _, n := range lab.Use {
			checkgoto(n, lab.Def)
		}
	}
}
