// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"fmt"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

// Escape analysis.
//
// Here we analyze functions to determine which Go variables
// (including implicit allocations such as calls to "new" or "make",
// composite literals, etc.) can be allocated on the stack. The two
// key invariants we have to ensure are: (1) pointers to stack objects
// cannot be stored in the heap, and (2) pointers to a stack object
// cannot outlive that object (e.g., because the declaring function
// returned and destroyed the object's stack frame, or its space is
// reused across loop iterations for logically distinct variables).
//
// We implement this with a static data-flow analysis of the AST.
// First, we construct a directed weighted graph where vertices
// (termed "locations") represent variables allocated by statements
// and expressions, and edges represent assignments between variables
// (with weights representing addressing/dereference counts).
//
// Next we walk the graph looking for assignment paths that might
// violate the invariants stated above. If a variable v's address is
// stored in the heap or elsewhere that may outlive it, then v is
// marked as requiring heap allocation.
//
// To support interprocedural analysis, we also record data-flow from
// each function's parameters to the heap and to its result
// parameters. This information is summarized as "parameter tags",
// which are used at static call sites to improve escape analysis of
// function arguments.

// Constructing the location graph.
//
// Every allocating statement (e.g., variable declaration) or
// expression (e.g., "new" or "make") is first mapped to a unique
// "location."
//
// We also model every Go assignment as a directed edges between
// locations. The number of dereference operations minus the number of
// addressing operations is recorded as the edge's weight (termed
// "derefs"). For example:
//
//     p = &q    // -1
//     p = q     //  0
//     p = *q    //  1
//     p = **q   //  2
//
//     p = **&**&q  // 2
//
// Note that the & operator can only be applied to addressable
// expressions, and the expression &x itself is not addressable, so
// derefs cannot go below -1.
//
// Every Go language construct is lowered into this representation,
// generally without sensitivity to flow, path, or context; and
// without distinguishing elements within a compound variable. For
// example:
//
//     var x struct { f, g *int }
//     var u []*int
//
//     x.f = u[0]
//
// is modeled simply as
//
//     x = *u
//
// That is, we don't distinguish x.f from x.g, or u[0] from u[1],
// u[2], etc. However, we do record the implicit dereference involved
// in indexing a slice.

// A batch holds escape analysis state that's shared across an entire
// batch of functions being analyzed at once.
type batch struct {
	allLocs  []*location
	closures []closure

	heapLoc    location
	mutatorLoc location
	calleeLoc  location
	blankLoc   location
}

// A closure holds a closure expression and its spill hole (i.e.,
// where the hole representing storing into its closure record).
type closure struct {
	k   hole
	clo *ir.ClosureExpr
}

// An escape holds state specific to a single function being analyzed
// within a batch.
type escape struct {
	*batch

	curfn *ir.Func // function being analyzed

	labels map[*types.Sym]labelState // known labels

	// loopDepth counts the current loop nesting depth within
	// curfn. It increments within each "for" loop and at each
	// label with a corresponding backwards "goto" (i.e.,
	// unstructured loop).
	loopDepth int
}

func Funcs(all []*ir.Func) {
	ir.VisitFuncsBottomUp(all, Batch)
}

// Batch performs escape analysis on a minimal batch of
// functions.
func Batch(fns []*ir.Func, recursive bool) {
	var b batch
	b.heapLoc.attrs = attrEscapes | attrPersists | attrMutates | attrCalls
	b.mutatorLoc.attrs = attrMutates
	b.calleeLoc.attrs = attrCalls

	// Construct data-flow graph from syntax trees.
	for _, fn := range fns {
		if base.Flag.W > 1 {
			s := fmt.Sprintf("\nbefore escape %v", fn)
			ir.Dump(s, fn)
		}
		b.initFunc(fn)
	}
	for _, fn := range fns {
		if !fn.IsHiddenClosure() {
			b.walkFunc(fn)
		}
	}

	// We've walked the function bodies, so we've seen everywhere a
	// variable might be reassigned or have it's address taken. Now we
	// can decide whether closures should capture their free variables
	// by value or reference.
	for _, closure := range b.closures {
		b.flowClosure(closure.k, closure.clo)
	}
	b.closures = nil

	for _, loc := range b.allLocs {
		if why := HeapAllocReason(loc.n); why != "" {
			b.flow(b.heapHole().addr(loc.n, why), loc)
		}
	}

	b.walkAll()
	b.finish(fns)
}

func (b *batch) with(fn *ir.Func) *escape {
	return &escape{
		batch:     b,
		curfn:     fn,
		loopDepth: 1,
	}
}

func (b *batch) initFunc(fn *ir.Func) {
	e := b.with(fn)
	if fn.Esc() != escFuncUnknown {
		base.Fatalf("unexpected node: %v", fn)
	}
	fn.SetEsc(escFuncPlanned)
	if base.Flag.LowerM > 3 {
		ir.Dump("escAnalyze", fn)
	}

	// Allocate locations for local variables.
	for _, n := range fn.Dcl {
		e.newLoc(n, true)
	}

	// Also for hidden parameters (e.g., the ".this" parameter to a
	// method value wrapper).
	if fn.OClosure == nil {
		for _, n := range fn.ClosureVars {
			e.newLoc(n.Canonical(), true)
		}
	}

	// Initialize resultIndex for result parameters.
	for i, f := range fn.Type().Results() {
		e.oldLoc(f.Nname.(*ir.Name)).resultIndex = 1 + i
	}
}

func (b *batch) walkFunc(fn *ir.Func) {
	e := b.with(fn)
	fn.SetEsc(escFuncStarted)

	// Identify labels that mark the head of an unstructured loop.
	ir.Visit(fn, func { n ->
		switch n.Op() {
		case ir.OLABEL:
			n := n.(*ir.LabelStmt)
			if n.Label.IsBlank() {
				break
			}
			if e.labels == nil {
				e.labels = make(map[*types.Sym]labelState)
			}
			e.labels[n.Label] = nonlooping

		case ir.OGOTO:
			// If we visited the label before the goto,
			// then this is a looping label.
			n := n.(*ir.BranchStmt)
			if e.labels[n.Label] == nonlooping {
				e.labels[n.Label] = looping
			}
		}
	})

	e.block(fn.Body)

	if len(e.labels) != 0 {
		base.FatalfAt(fn.Pos(), "leftover labels after walkFunc")
	}
}

func (b *batch) flowClosure(k hole, clo *ir.ClosureExpr) {
	for _, cv := range clo.Func.ClosureVars {
		n := cv.Canonical()
		loc := b.oldLoc(cv)
		if !loc.captured {
			base.FatalfAt(cv.Pos(), "closure variable never captured: %v", cv)
		}

		// Capture by value for variables <= 128 bytes that are never reassigned.
		n.SetByval(!loc.addrtaken && !loc.reassigned && n.Type().Size() <= 128)
		if !n.Byval() {
			n.SetAddrtaken(true)
			if n.Sym().Name == typecheck.LocalDictName {
				base.FatalfAt(n.Pos(), "dictionary variable not captured by value")
			}
		}

		if base.Flag.LowerM > 1 {
			how := "ref"
			if n.Byval() {
				how = "value"
			}
			base.WarnfAt(n.Pos(), "%v capturing by %s: %v (addr=%v assign=%v width=%d)", n.Curfn, how, n, loc.addrtaken, loc.reassigned, n.Type().Size())
		}

		// Flow captured variables to closure.
		k := k
		if !cv.Byval() {
			k = k.addr(cv, "reference")
		}
		b.flow(k.note(cv, "captured by a closure"), loc)
	}
}

func (b *batch) finish(fns []*ir.Func) {
	// Record parameter tags for package export data.
	for _, fn := range fns {
		fn.SetEsc(escFuncTagged)

		for i, param := range fn.Type().RecvParams() {
			param.Note = b.paramTag(fn, 1+i, param)
		}
	}

	for _, loc := range b.allLocs {
		n := loc.n
		if n == nil {
			continue
		}

		if n.Op() == ir.ONAME {
			n := n.(*ir.Name)
			n.Opt = nil
		}

		// Update n.Esc based on escape analysis results.

		// Omit escape diagnostics for go/defer wrappers, at least for now.
		// Historically, we haven't printed them, and test cases don't expect them.
		// TODO(mdempsky): Update tests to expect this.
		goDeferWrapper := n.Op() == ir.OCLOSURE && n.(*ir.ClosureExpr).Func.Wrapper()

		if loc.hasAttr(attrEscapes) {
			if n.Op() == ir.ONAME {
				if base.Flag.CompilingRuntime {
					base.ErrorfAt(n.Pos(), 0, "%v escapes to heap, not allowed in runtime", n)
				}
				if base.Flag.LowerM != 0 {
					base.WarnfAt(n.Pos(), "moved to heap: %v", n)
				}
			} else {
				if base.Flag.LowerM != 0 && !goDeferWrapper {
					base.WarnfAt(n.Pos(), "%v escapes to heap", n)
				}
				if logopt.Enabled() {
					var e_curfn *ir.Func // TODO(mdempsky): Fix.
					logopt.LogOpt(n.Pos(), "escape", "escape", ir.FuncName(e_curfn))
				}
			}
			n.SetEsc(ir.EscHeap)
		} else {
			if base.Flag.LowerM != 0 && n.Op() != ir.ONAME && !goDeferWrapper {
				base.WarnfAt(n.Pos(), "%v does not escape", n)
			}
			n.SetEsc(ir.EscNone)
			if !loc.hasAttr(attrPersists) {
				switch n.Op() {
				case ir.OCLOSURE:
					n := n.(*ir.ClosureExpr)
					n.SetTransient(true)
				case ir.OMETHVALUE:
					n := n.(*ir.SelectorExpr)
					n.SetTransient(true)
				case ir.OSLICELIT:
					n := n.(*ir.CompLitExpr)
					n.SetTransient(true)
				}
			}
		}

		// If the result of a string->[]byte conversion is never mutated,
		// then it can simply reuse the string's memory directly.
		if base.Debug.ZeroCopy != 0 {
			if n, ok := n.(*ir.ConvExpr); ok && n.Op() == ir.OSTR2BYTES && !loc.hasAttr(attrMutates) {
				if base.Flag.LowerM >= 1 {
					base.WarnfAt(n.Pos(), "zero-copy string->[]byte conversion")
				}
				n.SetOp(ir.OSTR2BYTESTMP)
			}
		}
	}
}

// inMutualBatch reports whether function fn is in the batch of
// mutually recursive functions being analyzed. When this is true,
// fn has not yet been analyzed, so its parameters and results
// should be incorporated directly into the flow graph instead of
// relying on its escape analysis tagging.
func (b *batch) inMutualBatch(fn *ir.Name) bool {
	if fn.Defn != nil && fn.Defn.Esc() < escFuncTagged {
		if fn.Defn.Esc() == escFuncUnknown {
			base.FatalfAt(fn.Pos(), "graph inconsistency: %v", fn)
		}
		return true
	}
	return false
}

const (
	escFuncUnknown = 0 + iota
	escFuncPlanned
	escFuncStarted
	escFuncTagged
)

// Mark labels that have no backjumps to them as not increasing e.loopdepth.
type labelState int

const (
	looping labelState = 1 + iota
	nonlooping
)

func (b *batch) paramTag(fn *ir.Func, narg int, f *types.Field) string {
	name := func {
		if f.Nname != nil {
			return f.Nname.Sym().Name
		}
		return fmt.Sprintf("arg#%d", narg)
	}

	// Only report diagnostics for user code;
	// not for wrappers generated around them.
	// TODO(mdempsky): Generalize this.
	diagnose := base.Flag.LowerM != 0 && !(fn.Wrapper() || fn.Dupok())

	if len(fn.Body) == 0 {
		// Assume that uintptr arguments must be held live across the call.
		// This is most important for syscall.Syscall.
		// See golang.org/issue/13372.
		// This really doesn't have much to do with escape analysis per se,
		// but we are reusing the ability to annotate an individual function
		// argument and pass those annotations along to importing code.
		fn.Pragma |= ir.UintptrKeepAlive

		if f.Type.IsUintptr() {
			if diagnose {
				base.WarnfAt(f.Pos, "assuming %v is unsafe uintptr", name())
			}
			return ""
		}

		if !f.Type.HasPointers() { // don't bother tagging for scalars
			return ""
		}

		var esc leaks

		// External functions are assumed unsafe, unless
		// //go:noescape is given before the declaration.
		if fn.Pragma&ir.Noescape != 0 {
			if diagnose && f.Sym != nil {
				base.WarnfAt(f.Pos, "%v does not escape", name())
			}
			esc.AddMutator(0)
			esc.AddCallee(0)
		} else {
			if diagnose && f.Sym != nil {
				base.WarnfAt(f.Pos, "leaking param: %v", name())
			}
			esc.AddHeap(0)
		}

		return esc.Encode()
	}

	if fn.Pragma&ir.UintptrEscapes != 0 {
		if f.Type.IsUintptr() {
			if diagnose {
				base.WarnfAt(f.Pos, "marking %v as escaping uintptr", name())
			}
			return ""
		}
		if f.IsDDD() && f.Type.Elem().IsUintptr() {
			// final argument is ...uintptr.
			if diagnose {
				base.WarnfAt(f.Pos, "marking %v as escaping ...uintptr", name())
			}
			return ""
		}
	}

	if !f.Type.HasPointers() { // don't bother tagging for scalars
		return ""
	}

	// Unnamed parameters are unused and therefore do not escape.
	if f.Sym == nil || f.Sym.IsBlank() {
		var esc leaks
		return esc.Encode()
	}

	n := f.Nname.(*ir.Name)
	loc := b.oldLoc(n)
	esc := loc.paramEsc
	esc.Optimize()

	if diagnose && !loc.hasAttr(attrEscapes) {
		b.reportLeaks(f.Pos, name(), esc, fn.Type())
	}

	return esc.Encode()
}

func (b *batch) reportLeaks(pos src.XPos, name string, esc leaks, sig *types.Type) {
	warned := false
	if x := esc.Heap(); x >= 0 {
		if x == 0 {
			base.WarnfAt(pos, "leaking param: %v", name)
		} else {
			// TODO(mdempsky): Mention level=x like below?
			base.WarnfAt(pos, "leaking param content: %v", name)
		}
		warned = true
	}
	for i := 0; i < numEscResults; i++ {
		if x := esc.Result(i); x >= 0 {
			res := sig.Result(i).Nname.Sym().Name
			base.WarnfAt(pos, "leaking param: %v to result %v level=%d", name, res, x)
			warned = true
		}
	}

	if base.Debug.EscapeMutationsCalls <= 0 {
		if !warned {
			base.WarnfAt(pos, "%v does not escape", name)
		}
		return
	}

	if x := esc.Mutator(); x >= 0 {
		base.WarnfAt(pos, "mutates param: %v derefs=%v", name, x)
		warned = true
	}
	if x := esc.Callee(); x >= 0 {
		base.WarnfAt(pos, "calls param: %v derefs=%v", name, x)
		warned = true
	}

	if !warned {
		base.WarnfAt(pos, "%v does not escape, mutate, or call", name)
	}
}
