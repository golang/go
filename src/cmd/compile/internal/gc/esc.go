// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/types"
	"fmt"
)

func escapes(all []*Node) {
	visitBottomUp(all, escapeFuncs)
}

const (
	EscFuncUnknown = 0 + iota
	EscFuncPlanned
	EscFuncStarted
	EscFuncTagged
)

func min8(a, b int8) int8 {
	if a < b {
		return a
	}
	return b
}

func max8(a, b int8) int8 {
	if a > b {
		return a
	}
	return b
}

const (
	EscUnknown = iota
	EscNone    // Does not escape to heap, result, or parameters.
	EscHeap    // Reachable from the heap
	EscNever   // By construction will not escape.
)

// funcSym returns fn.Func.Nname.Sym if no nils are encountered along the way.
func funcSym(fn *Node) *types.Sym {
	if fn == nil || fn.Func.Nname == nil {
		return nil
	}
	return fn.Func.Nname.Sym
}

// Mark labels that have no backjumps to them as not increasing e.loopdepth.
// Walk hasn't generated (goto|label).Left.Sym.Label yet, so we'll cheat
// and set it to one of the following two. Then in esc we'll clear it again.
var (
	looping    Node
	nonlooping Node
)

func isSliceSelfAssign(dst, src *Node) bool {
	// Detect the following special case.
	//
	//	func (b *Buffer) Foo() {
	//		n, m := ...
	//		b.buf = b.buf[n:m]
	//	}
	//
	// This assignment is a no-op for escape analysis,
	// it does not store any new pointers into b that were not already there.
	// However, without this special case b will escape, because we assign to OIND/ODOTPTR.
	// Here we assume that the statement will not contain calls,
	// that is, that order will move any calls to init.
	// Otherwise base ONAME value could change between the moments
	// when we evaluate it for dst and for src.

	// dst is ONAME dereference.
	if dst.Op != ODEREF && dst.Op != ODOTPTR || dst.Left.Op != ONAME {
		return false
	}
	// src is a slice operation.
	switch src.Op {
	case OSLICE, OSLICE3, OSLICESTR:
		// OK.
	case OSLICEARR, OSLICE3ARR:
		// Since arrays are embedded into containing object,
		// slice of non-pointer array will introduce a new pointer into b that was not already there
		// (pointer to b itself). After such assignment, if b contents escape,
		// b escapes as well. If we ignore such OSLICEARR, we will conclude
		// that b does not escape when b contents do.
		//
		// Pointer to an array is OK since it's not stored inside b directly.
		// For slicing an array (not pointer to array), there is an implicit OADDR.
		// We check that to determine non-pointer array slicing.
		if src.Left.Op == OADDR {
			return false
		}
	default:
		return false
	}
	// slice is applied to ONAME dereference.
	if src.Left.Op != ODEREF && src.Left.Op != ODOTPTR || src.Left.Left.Op != ONAME {
		return false
	}
	// dst and src reference the same base ONAME.
	return dst.Left == src.Left.Left
}

// isSelfAssign reports whether assignment from src to dst can
// be ignored by the escape analysis as it's effectively a self-assignment.
func isSelfAssign(dst, src *Node) bool {
	if isSliceSelfAssign(dst, src) {
		return true
	}

	// Detect trivial assignments that assign back to the same object.
	//
	// It covers these cases:
	//	val.x = val.y
	//	val.x[i] = val.y[j]
	//	val.x1.x2 = val.x1.y2
	//	... etc
	//
	// These assignments do not change assigned object lifetime.

	if dst == nil || src == nil || dst.Op != src.Op {
		return false
	}

	switch dst.Op {
	case ODOT, ODOTPTR:
		// Safe trailing accessors that are permitted to differ.
	case OINDEX:
		if mayAffectMemory(dst.Right) || mayAffectMemory(src.Right) {
			return false
		}
	default:
		return false
	}

	// The expression prefix must be both "safe" and identical.
	return samesafeexpr(dst.Left, src.Left)
}

// mayAffectMemory reports whether evaluation of n may affect the program's
// memory state. If the expression can't affect memory state, then it can be
// safely ignored by the escape analysis.
func mayAffectMemory(n *Node) bool {
	// We may want to use a list of "memory safe" ops instead of generally
	// "side-effect free", which would include all calls and other ops that can
	// allocate or change global state. For now, it's safer to start with the latter.
	//
	// We're ignoring things like division by zero, index out of range,
	// and nil pointer dereference here.
	switch n.Op {
	case ONAME, OCLOSUREVAR, OLITERAL:
		return false

	// Left+Right group.
	case OINDEX, OADD, OSUB, OOR, OXOR, OMUL, OLSH, ORSH, OAND, OANDNOT, ODIV, OMOD:
		return mayAffectMemory(n.Left) || mayAffectMemory(n.Right)

	// Left group.
	case ODOT, ODOTPTR, ODEREF, OCONVNOP, OCONV, OLEN, OCAP,
		ONOT, OBITNOT, OPLUS, ONEG, OALIGNOF, OOFFSETOF, OSIZEOF:
		return mayAffectMemory(n.Left)

	default:
		return true
	}
}

// heapAllocReason returns the reason the given Node must be heap
// allocated, or the empty string if it doesn't.
func heapAllocReason(n *Node) string {
	if n.Type == nil {
		return ""
	}

	// Parameters are always passed via the stack.
	if n.Op == ONAME && (n.Class() == PPARAM || n.Class() == PPARAMOUT) {
		return ""
	}

	if n.Type.Width > maxStackVarSize {
		return "too large for stack"
	}

	if (n.Op == ONEW || n.Op == OPTRLIT) && n.Type.Elem().Width >= maxImplicitStackVarSize {
		return "too large for stack"
	}

	if n.Op == OCLOSURE && closureType(n).Size() >= maxImplicitStackVarSize {
		return "too large for stack"
	}
	if n.Op == OCALLPART && partialCallType(n).Size() >= maxImplicitStackVarSize {
		return "too large for stack"
	}

	if n.Op == OMAKESLICE {
		r := n.Right
		if r == nil {
			r = n.Left
		}
		if !smallintconst(r) {
			return "non-constant size"
		}
		if t := n.Type; t.Elem().Width != 0 && r.Int64Val() >= maxImplicitStackVarSize/t.Elem().Width {
			return "too large for stack"
		}
	}

	return ""
}

// addrescapes tags node n as having had its address taken
// by "increasing" the "value" of n.Esc to EscHeap.
// Storage is allocated as necessary to allow the address
// to be taken.
func addrescapes(n *Node) {
	switch n.Op {
	default:
		// Unexpected Op, probably due to a previous type error. Ignore.

	case ODEREF, ODOTPTR:
		// Nothing to do.

	case ONAME:
		if n == nodfp {
			break
		}

		// if this is a tmpname (PAUTO), it was tagged by tmpname as not escaping.
		// on PPARAM it means something different.
		if n.Class() == PAUTO && n.Esc == EscNever {
			break
		}

		// If a closure reference escapes, mark the outer variable as escaping.
		if n.Name.IsClosureVar() {
			addrescapes(n.Name.Defn)
			break
		}

		if n.Class() != PPARAM && n.Class() != PPARAMOUT && n.Class() != PAUTO {
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
		lineno = Curfn.Pos
		moveToHeap(n)
		Curfn = oldfn
		lineno = ln

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

// moveToHeap records the parameter or local variable n as moved to the heap.
func moveToHeap(n *Node) {
	if Debug.r != 0 {
		Dump("MOVE", n)
	}
	if compiling_runtime {
		yyerror("%v escapes to heap, not allowed in runtime", n)
	}
	if n.Class() == PAUTOHEAP {
		Dump("n", n)
		Fatalf("double move to heap")
	}

	// Allocate a local stack variable to hold the pointer to the heap copy.
	// temp will add it to the function declaration list automatically.
	heapaddr := temp(types.NewPtr(n.Type))
	heapaddr.Sym = lookup("&" + n.Sym.Name)
	heapaddr.Orig.Sym = heapaddr.Sym
	heapaddr.Pos = n.Pos

	// Unset AutoTemp to persist the &foo variable name through SSA to
	// liveness analysis.
	// TODO(mdempsky/drchase): Cleaner solution?
	heapaddr.Name.SetAutoTemp(false)

	// Parameters have a local stack copy used at function start/end
	// in addition to the copy in the heap that may live longer than
	// the function.
	if n.Class() == PPARAM || n.Class() == PPARAMOUT {
		if n.Xoffset == BADWIDTH {
			Fatalf("addrescapes before param assignment")
		}

		// We rewrite n below to be a heap variable (indirection of heapaddr).
		// Preserve a copy so we can still write code referring to the original,
		// and substitute that copy into the function declaration list
		// so that analyses of the local (on-stack) variables use it.
		stackcopy := newname(n.Sym)
		stackcopy.Type = n.Type
		stackcopy.Xoffset = n.Xoffset
		stackcopy.SetClass(n.Class())
		stackcopy.Name.Param.Heapaddr = heapaddr
		if n.Class() == PPARAMOUT {
			// Make sure the pointer to the heap copy is kept live throughout the function.
			// The function could panic at any point, and then a defer could recover.
			// Thus, we need the pointer to the heap copy always available so the
			// post-deferreturn code can copy the return value back to the stack.
			// See issue 16095.
			heapaddr.Name.SetIsOutputParamHeapAddr(true)
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
			if d.Class() == PAUTO {
				break
			}
		}
		if !found {
			Fatalf("cannot find %v in local variable list", n)
		}
		Curfn.Func.Dcl = append(Curfn.Func.Dcl, n)
	}

	// Modify n in place so that uses of n now mean indirection of the heapaddr.
	n.SetClass(PAUTOHEAP)
	n.Xoffset = 0
	n.Name.Param.Heapaddr = heapaddr
	n.Esc = EscHeap
	if Debug.m != 0 {
		Warnl(n.Pos, "moved to heap: %v", n)
	}
}

// This special tag is applied to uintptr variables
// that we believe may hold unsafe.Pointers for
// calls into assembly functions.
const unsafeUintptrTag = "unsafe-uintptr"

// This special tag is applied to uintptr parameters of functions
// marked go:uintptrescapes.
const uintptrEscapesTag = "uintptr-escapes"

func (e *Escape) paramTag(fn *Node, narg int, f *types.Field) string {
	name := func() string {
		if f.Sym != nil {
			return f.Sym.Name
		}
		return fmt.Sprintf("arg#%d", narg)
	}

	if fn.Nbody.Len() == 0 {
		// Assume that uintptr arguments must be held live across the call.
		// This is most important for syscall.Syscall.
		// See golang.org/issue/13372.
		// This really doesn't have much to do with escape analysis per se,
		// but we are reusing the ability to annotate an individual function
		// argument and pass those annotations along to importing code.
		if f.Type.IsUintptr() {
			if Debug.m != 0 {
				Warnl(f.Pos, "assuming %v is unsafe uintptr", name())
			}
			return unsafeUintptrTag
		}

		if !f.Type.HasPointers() { // don't bother tagging for scalars
			return ""
		}

		var esc EscLeaks

		// External functions are assumed unsafe, unless
		// //go:noescape is given before the declaration.
		if fn.Func.Pragma&Noescape != 0 {
			if Debug.m != 0 && f.Sym != nil {
				Warnl(f.Pos, "%v does not escape", name())
			}
		} else {
			if Debug.m != 0 && f.Sym != nil {
				Warnl(f.Pos, "leaking param: %v", name())
			}
			esc.AddHeap(0)
		}

		return esc.Encode()
	}

	if fn.Func.Pragma&UintptrEscapes != 0 {
		if f.Type.IsUintptr() {
			if Debug.m != 0 {
				Warnl(f.Pos, "marking %v as escaping uintptr", name())
			}
			return uintptrEscapesTag
		}
		if f.IsDDD() && f.Type.Elem().IsUintptr() {
			// final argument is ...uintptr.
			if Debug.m != 0 {
				Warnl(f.Pos, "marking %v as escaping ...uintptr", name())
			}
			return uintptrEscapesTag
		}
	}

	if !f.Type.HasPointers() { // don't bother tagging for scalars
		return ""
	}

	// Unnamed parameters are unused and therefore do not escape.
	if f.Sym == nil || f.Sym.IsBlank() {
		var esc EscLeaks
		return esc.Encode()
	}

	n := asNode(f.Nname)
	loc := e.oldLoc(n)
	esc := loc.paramEsc
	esc.Optimize()

	if Debug.m != 0 && !loc.escapes {
		if esc.Empty() {
			Warnl(f.Pos, "%v does not escape", name())
		}
		if x := esc.Heap(); x >= 0 {
			if x == 0 {
				Warnl(f.Pos, "leaking param: %v", name())
			} else {
				// TODO(mdempsky): Mention level=x like below?
				Warnl(f.Pos, "leaking param content: %v", name())
			}
		}
		for i := 0; i < numEscResults; i++ {
			if x := esc.Result(i); x >= 0 {
				res := fn.Type.Results().Field(i).Sym
				Warnl(f.Pos, "leaking param: %v to result %v level=%d", name(), res, x)
			}
		}
	}

	return esc.Encode()
}
