// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"fmt"
	"math"
	"strings"

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

	heapLoc  location
	blankLoc location
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

// An location represents an abstract location that stores a Go
// variable.
type location struct {
	n         ir.Node  // represented variable or expression, if any
	curfn     *ir.Func // enclosing function
	edges     []edge   // incoming edges
	loopDepth int      // loopDepth at declaration

	// resultIndex records the tuple index (starting at 1) for
	// PPARAMOUT variables within their function's result type.
	// For non-PPARAMOUT variables it's 0.
	resultIndex int

	// derefs and walkgen are used during walkOne to track the
	// minimal dereferences from the walk root.
	derefs  int // >= -1
	walkgen uint32

	// dst and dstEdgeindex track the next immediate assignment
	// destination location during walkone, along with the index
	// of the edge pointing back to this location.
	dst        *location
	dstEdgeIdx int

	// queued is used by walkAll to track whether this location is
	// in the walk queue.
	queued bool

	// escapes reports whether the represented variable's address
	// escapes; that is, whether the variable must be heap
	// allocated.
	escapes bool

	// transient reports whether the represented expression's
	// address does not outlive the statement; that is, whether
	// its storage can be immediately reused.
	transient bool

	// paramEsc records the represented parameter's leak set.
	paramEsc leaks

	captured   bool // has a closure captured this variable?
	reassigned bool // has this variable been reassigned?
	addrtaken  bool // has this variable's address been taken?
}

// An edge represents an assignment edge between two Go variables.
type edge struct {
	src    *location
	derefs int // >= -1
	notes  *note
}

// Fmt is called from node printing to print information about escape analysis results.
func Fmt(n ir.Node) string {
	text := ""
	switch n.Esc() {
	case ir.EscUnknown:
		break

	case ir.EscHeap:
		text = "esc(h)"

	case ir.EscNone:
		text = "esc(no)"

	case ir.EscNever:
		text = "esc(N)"

	default:
		text = fmt.Sprintf("esc(%d)", n.Esc())
	}

	if n.Op() == ir.ONAME {
		n := n.(*ir.Name)
		if loc, ok := n.Opt.(*location); ok && loc.loopDepth != 0 {
			if text != "" {
				text += " "
			}
			text += fmt.Sprintf("ld(%d)", loc.loopDepth)
		}
	}

	return text
}

// Batch performs escape analysis on a minimal batch of
// functions.
func Batch(fns []*ir.Func, recursive bool) {
	for _, fn := range fns {
		if fn.Op() != ir.ODCLFUNC {
			base.Fatalf("unexpected node: %v", fn)
		}
	}

	var b batch
	b.heapLoc.escapes = true

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
		if n.Op() == ir.ONAME {
			e.newLoc(n, false)
		}
	}

	// Initialize resultIndex for result parameters.
	for i, f := range fn.Type().Results().FieldSlice() {
		e.oldLoc(f.Nname.(*ir.Name)).resultIndex = 1 + i
	}
}

func (b *batch) walkFunc(fn *ir.Func) {
	e := b.with(fn)
	fn.SetEsc(escFuncStarted)

	// Identify labels that mark the head of an unstructured loop.
	ir.Visit(fn, func(n ir.Node) {
		switch n.Op() {
		case ir.OLABEL:
			n := n.(*ir.LabelStmt)
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

// Below we implement the methods for walking the AST and recording
// data flow edges. Note that because a sub-expression might have
// side-effects, it's important to always visit the entire AST.
//
// For example, write either:
//
//     if x {
//         e.discard(n.Left)
//     } else {
//         e.value(k, n.Left)
//     }
//
// or
//
//     if x {
//         k = e.discardHole()
//     }
//     e.value(k, n.Left)
//
// Do NOT write:
//
//    // BAD: possibly loses side-effects within n.Left
//    if !x {
//        e.value(k, n.Left)
//    }

// stmt evaluates a single Go statement.
func (e *escape) stmt(n ir.Node) {
	if n == nil {
		return
	}

	lno := ir.SetPos(n)
	defer func() {
		base.Pos = lno
	}()

	if base.Flag.LowerM > 2 {
		fmt.Printf("%v:[%d] %v stmt: %v\n", base.FmtPos(base.Pos), e.loopDepth, e.curfn, n)
	}

	e.stmts(n.Init())

	switch n.Op() {
	default:
		base.Fatalf("unexpected stmt: %v", n)

	case ir.ODCLCONST, ir.ODCLTYPE, ir.OFALL, ir.OINLMARK:
		// nop

	case ir.OBREAK, ir.OCONTINUE, ir.OGOTO:
		// TODO(mdempsky): Handle dead code?

	case ir.OBLOCK:
		n := n.(*ir.BlockStmt)
		e.stmts(n.List)

	case ir.ODCL:
		// Record loop depth at declaration.
		n := n.(*ir.Decl)
		if !ir.IsBlank(n.X) {
			e.dcl(n.X)
		}

	case ir.OLABEL:
		n := n.(*ir.LabelStmt)
		switch e.labels[n.Label] {
		case nonlooping:
			if base.Flag.LowerM > 2 {
				fmt.Printf("%v:%v non-looping label\n", base.FmtPos(base.Pos), n)
			}
		case looping:
			if base.Flag.LowerM > 2 {
				fmt.Printf("%v: %v looping label\n", base.FmtPos(base.Pos), n)
			}
			e.loopDepth++
		default:
			base.Fatalf("label missing tag")
		}
		delete(e.labels, n.Label)

	case ir.OIF:
		n := n.(*ir.IfStmt)
		e.discard(n.Cond)
		e.block(n.Body)
		e.block(n.Else)

	case ir.OFOR, ir.OFORUNTIL:
		n := n.(*ir.ForStmt)
		e.loopDepth++
		e.discard(n.Cond)
		e.stmt(n.Post)
		e.block(n.Body)
		e.loopDepth--

	case ir.ORANGE:
		// for Key, Value = range X { Body }
		n := n.(*ir.RangeStmt)

		// X is evaluated outside the loop.
		tmp := e.newLoc(nil, false)
		e.expr(tmp.asHole(), n.X)

		e.loopDepth++
		ks := e.addrs([]ir.Node{n.Key, n.Value})
		if n.X.Type().IsArray() {
			e.flow(ks[1].note(n, "range"), tmp)
		} else {
			e.flow(ks[1].deref(n, "range-deref"), tmp)
		}
		e.reassigned(ks, n)

		e.block(n.Body)
		e.loopDepth--

	case ir.OSWITCH:
		n := n.(*ir.SwitchStmt)

		if guard, ok := n.Tag.(*ir.TypeSwitchGuard); ok {
			var ks []hole
			if guard.Tag != nil {
				for _, cas := range n.Cases {
					cv := cas.Var
					k := e.dcl(cv) // type switch variables have no ODCL.
					if cv.Type().HasPointers() {
						ks = append(ks, k.dotType(cv.Type(), cas, "switch case"))
					}
				}
			}
			e.expr(e.teeHole(ks...), n.Tag.(*ir.TypeSwitchGuard).X)
		} else {
			e.discard(n.Tag)
		}

		for _, cas := range n.Cases {
			e.discards(cas.List)
			e.block(cas.Body)
		}

	case ir.OSELECT:
		n := n.(*ir.SelectStmt)
		for _, cas := range n.Cases {
			e.stmt(cas.Comm)
			e.block(cas.Body)
		}
	case ir.ORECV:
		// TODO(mdempsky): Consider e.discard(n.Left).
		n := n.(*ir.UnaryExpr)
		e.exprSkipInit(e.discardHole(), n) // already visited n.Ninit
	case ir.OSEND:
		n := n.(*ir.SendStmt)
		e.discard(n.Chan)
		e.assignHeap(n.Value, "send", n)

	case ir.OAS:
		n := n.(*ir.AssignStmt)
		e.assignList([]ir.Node{n.X}, []ir.Node{n.Y}, "assign", n)
	case ir.OASOP:
		n := n.(*ir.AssignOpStmt)
		// TODO(mdempsky): Worry about OLSH/ORSH?
		e.assignList([]ir.Node{n.X}, []ir.Node{n.Y}, "assign", n)
	case ir.OAS2:
		n := n.(*ir.AssignListStmt)
		e.assignList(n.Lhs, n.Rhs, "assign-pair", n)

	case ir.OAS2DOTTYPE: // v, ok = x.(type)
		n := n.(*ir.AssignListStmt)
		e.assignList(n.Lhs, n.Rhs, "assign-pair-dot-type", n)
	case ir.OAS2MAPR: // v, ok = m[k]
		n := n.(*ir.AssignListStmt)
		e.assignList(n.Lhs, n.Rhs, "assign-pair-mapr", n)
	case ir.OAS2RECV, ir.OSELRECV2: // v, ok = <-ch
		n := n.(*ir.AssignListStmt)
		e.assignList(n.Lhs, n.Rhs, "assign-pair-receive", n)

	case ir.OAS2FUNC:
		n := n.(*ir.AssignListStmt)
		e.stmts(n.Rhs[0].Init())
		ks := e.addrs(n.Lhs)
		e.call(ks, n.Rhs[0], nil)
		e.reassigned(ks, n)
	case ir.ORETURN:
		n := n.(*ir.ReturnStmt)
		results := e.curfn.Type().Results().FieldSlice()
		dsts := make([]ir.Node, len(results))
		for i, res := range results {
			dsts[i] = res.Nname.(*ir.Name)
		}
		e.assignList(dsts, n.Results, "return", n)
	case ir.OCALLFUNC, ir.OCALLMETH, ir.OCALLINTER, ir.OCLOSE, ir.OCOPY, ir.ODELETE, ir.OPANIC, ir.OPRINT, ir.OPRINTN, ir.ORECOVER:
		e.call(nil, n, nil)
	case ir.OGO, ir.ODEFER:
		n := n.(*ir.GoDeferStmt)
		e.stmts(n.Call.Init())
		e.call(nil, n.Call, n)

	case ir.OTAILCALL:
		// TODO(mdempsky): Treat like a normal call? esc.go used to just ignore it.
	}
}

func (e *escape) stmts(l ir.Nodes) {
	for _, n := range l {
		e.stmt(n)
	}
}

// block is like stmts, but preserves loopDepth.
func (e *escape) block(l ir.Nodes) {
	old := e.loopDepth
	e.stmts(l)
	e.loopDepth = old
}

// expr models evaluating an expression n and flowing the result into
// hole k.
func (e *escape) expr(k hole, n ir.Node) {
	if n == nil {
		return
	}
	e.stmts(n.Init())
	e.exprSkipInit(k, n)
}

func (e *escape) exprSkipInit(k hole, n ir.Node) {
	if n == nil {
		return
	}

	lno := ir.SetPos(n)
	defer func() {
		base.Pos = lno
	}()

	uintptrEscapesHack := k.uintptrEscapesHack
	k.uintptrEscapesHack = false

	if uintptrEscapesHack && n.Op() == ir.OCONVNOP && n.(*ir.ConvExpr).X.Type().IsUnsafePtr() {
		// nop
	} else if k.derefs >= 0 && !n.Type().HasPointers() {
		k.dst = &e.blankLoc
	}

	switch n.Op() {
	default:
		base.Fatalf("unexpected expr: %s %v", n.Op().String(), n)

	case ir.OLITERAL, ir.ONIL, ir.OGETG, ir.OTYPE, ir.OMETHEXPR, ir.OLINKSYMOFFSET:
		// nop

	case ir.ONAME:
		n := n.(*ir.Name)
		if n.Class == ir.PFUNC || n.Class == ir.PEXTERN {
			return
		}
		if n.IsClosureVar() && n.Defn == nil {
			return // ".this" from method value wrapper
		}
		e.flow(k, e.oldLoc(n))

	case ir.OPLUS, ir.ONEG, ir.OBITNOT, ir.ONOT:
		n := n.(*ir.UnaryExpr)
		e.discard(n.X)
	case ir.OADD, ir.OSUB, ir.OOR, ir.OXOR, ir.OMUL, ir.ODIV, ir.OMOD, ir.OLSH, ir.ORSH, ir.OAND, ir.OANDNOT, ir.OEQ, ir.ONE, ir.OLT, ir.OLE, ir.OGT, ir.OGE:
		n := n.(*ir.BinaryExpr)
		e.discard(n.X)
		e.discard(n.Y)
	case ir.OANDAND, ir.OOROR:
		n := n.(*ir.LogicalExpr)
		e.discard(n.X)
		e.discard(n.Y)
	case ir.OADDR:
		n := n.(*ir.AddrExpr)
		e.expr(k.addr(n, "address-of"), n.X) // "address-of"
	case ir.ODEREF:
		n := n.(*ir.StarExpr)
		e.expr(k.deref(n, "indirection"), n.X) // "indirection"
	case ir.ODOT, ir.ODOTMETH, ir.ODOTINTER:
		n := n.(*ir.SelectorExpr)
		e.expr(k.note(n, "dot"), n.X)
	case ir.ODOTPTR:
		n := n.(*ir.SelectorExpr)
		e.expr(k.deref(n, "dot of pointer"), n.X) // "dot of pointer"
	case ir.ODOTTYPE, ir.ODOTTYPE2:
		n := n.(*ir.TypeAssertExpr)
		e.expr(k.dotType(n.Type(), n, "dot"), n.X)
	case ir.OINDEX:
		n := n.(*ir.IndexExpr)
		if n.X.Type().IsArray() {
			e.expr(k.note(n, "fixed-array-index-of"), n.X)
		} else {
			// TODO(mdempsky): Fix why reason text.
			e.expr(k.deref(n, "dot of pointer"), n.X)
		}
		e.discard(n.Index)
	case ir.OINDEXMAP:
		n := n.(*ir.IndexExpr)
		e.discard(n.X)
		e.discard(n.Index)
	case ir.OSLICE, ir.OSLICEARR, ir.OSLICE3, ir.OSLICE3ARR, ir.OSLICESTR:
		n := n.(*ir.SliceExpr)
		e.expr(k.note(n, "slice"), n.X)
		e.discard(n.Low)
		e.discard(n.High)
		e.discard(n.Max)

	case ir.OCONV, ir.OCONVNOP:
		n := n.(*ir.ConvExpr)
		if ir.ShouldCheckPtr(e.curfn, 2) && n.Type().IsUnsafePtr() && n.X.Type().IsPtr() {
			// When -d=checkptr=2 is enabled, treat
			// conversions to unsafe.Pointer as an
			// escaping operation. This allows better
			// runtime instrumentation, since we can more
			// easily detect object boundaries on the heap
			// than the stack.
			e.assignHeap(n.X, "conversion to unsafe.Pointer", n)
		} else if n.Type().IsUnsafePtr() && n.X.Type().IsUintptr() {
			e.unsafeValue(k, n.X)
		} else {
			e.expr(k, n.X)
		}
	case ir.OCONVIFACE:
		n := n.(*ir.ConvExpr)
		if !n.X.Type().IsInterface() && !types.IsDirectIface(n.X.Type()) {
			k = e.spill(k, n)
		}
		e.expr(k.note(n, "interface-converted"), n.X)
	case ir.OSLICE2ARRPTR:
		// the slice pointer flows directly to the result
		n := n.(*ir.ConvExpr)
		e.expr(k, n.X)
	case ir.ORECV:
		n := n.(*ir.UnaryExpr)
		e.discard(n.X)

	case ir.OCALLMETH, ir.OCALLFUNC, ir.OCALLINTER, ir.OLEN, ir.OCAP, ir.OCOMPLEX, ir.OREAL, ir.OIMAG, ir.OAPPEND, ir.OCOPY, ir.OUNSAFEADD, ir.OUNSAFESLICE:
		e.call([]hole{k}, n, nil)

	case ir.ONEW:
		n := n.(*ir.UnaryExpr)
		e.spill(k, n)

	case ir.OMAKESLICE:
		n := n.(*ir.MakeExpr)
		e.spill(k, n)
		e.discard(n.Len)
		e.discard(n.Cap)
	case ir.OMAKECHAN:
		n := n.(*ir.MakeExpr)
		e.discard(n.Len)
	case ir.OMAKEMAP:
		n := n.(*ir.MakeExpr)
		e.spill(k, n)
		e.discard(n.Len)

	case ir.ORECOVER:
		// nop

	case ir.OCALLPART:
		// Flow the receiver argument to both the closure and
		// to the receiver parameter.

		n := n.(*ir.SelectorExpr)
		closureK := e.spill(k, n)

		m := n.Selection

		// We don't know how the method value will be called
		// later, so conservatively assume the result
		// parameters all flow to the heap.
		//
		// TODO(mdempsky): Change ks into a callback, so that
		// we don't have to create this slice?
		var ks []hole
		for i := m.Type.NumResults(); i > 0; i-- {
			ks = append(ks, e.heapHole())
		}
		name, _ := m.Nname.(*ir.Name)
		paramK := e.tagHole(ks, name, m.Type.Recv())

		e.expr(e.teeHole(paramK, closureK), n.X)

	case ir.OPTRLIT:
		n := n.(*ir.AddrExpr)
		e.expr(e.spill(k, n), n.X)

	case ir.OARRAYLIT:
		n := n.(*ir.CompLitExpr)
		for _, elt := range n.List {
			if elt.Op() == ir.OKEY {
				elt = elt.(*ir.KeyExpr).Value
			}
			e.expr(k.note(n, "array literal element"), elt)
		}

	case ir.OSLICELIT:
		n := n.(*ir.CompLitExpr)
		k = e.spill(k, n)
		k.uintptrEscapesHack = uintptrEscapesHack // for ...uintptr parameters

		for _, elt := range n.List {
			if elt.Op() == ir.OKEY {
				elt = elt.(*ir.KeyExpr).Value
			}
			e.expr(k.note(n, "slice-literal-element"), elt)
		}

	case ir.OSTRUCTLIT:
		n := n.(*ir.CompLitExpr)
		for _, elt := range n.List {
			e.expr(k.note(n, "struct literal element"), elt.(*ir.StructKeyExpr).Value)
		}

	case ir.OMAPLIT:
		n := n.(*ir.CompLitExpr)
		e.spill(k, n)

		// Map keys and values are always stored in the heap.
		for _, elt := range n.List {
			elt := elt.(*ir.KeyExpr)
			e.assignHeap(elt.Key, "map literal key", n)
			e.assignHeap(elt.Value, "map literal value", n)
		}

	case ir.OCLOSURE:
		n := n.(*ir.ClosureExpr)
		k = e.spill(k, n)
		e.closures = append(e.closures, closure{k, n})

		if fn := n.Func; fn.IsHiddenClosure() {
			for _, cv := range fn.ClosureVars {
				if loc := e.oldLoc(cv); !loc.captured {
					loc.captured = true

					// Ignore reassignments to the variable in straightline code
					// preceding the first capture by a closure.
					if loc.loopDepth == e.loopDepth {
						loc.reassigned = false
					}
				}
			}

			for _, n := range fn.Dcl {
				// Add locations for local variables of the
				// closure, if needed, in case we're not including
				// the closure func in the batch for escape
				// analysis (happens for escape analysis called
				// from reflectdata.methodWrapper)
				if n.Op() == ir.ONAME && n.Opt == nil {
					e.with(fn).newLoc(n, false)
				}
			}
			e.walkFunc(fn)
		}

	case ir.ORUNES2STR, ir.OBYTES2STR, ir.OSTR2RUNES, ir.OSTR2BYTES, ir.ORUNESTR:
		n := n.(*ir.ConvExpr)
		e.spill(k, n)
		e.discard(n.X)

	case ir.OADDSTR:
		n := n.(*ir.AddStringExpr)
		e.spill(k, n)

		// Arguments of OADDSTR never escape;
		// runtime.concatstrings makes sure of that.
		e.discards(n.List)
	}
}

// unsafeValue evaluates a uintptr-typed arithmetic expression looking
// for conversions from an unsafe.Pointer.
func (e *escape) unsafeValue(k hole, n ir.Node) {
	if n.Type().Kind() != types.TUINTPTR {
		base.Fatalf("unexpected type %v for %v", n.Type(), n)
	}
	if k.addrtaken {
		base.Fatalf("unexpected addrtaken")
	}

	e.stmts(n.Init())

	switch n.Op() {
	case ir.OCONV, ir.OCONVNOP:
		n := n.(*ir.ConvExpr)
		if n.X.Type().IsUnsafePtr() {
			e.expr(k, n.X)
		} else {
			e.discard(n.X)
		}
	case ir.ODOTPTR:
		n := n.(*ir.SelectorExpr)
		if ir.IsReflectHeaderDataField(n) {
			e.expr(k.deref(n, "reflect.Header.Data"), n.X)
		} else {
			e.discard(n.X)
		}
	case ir.OPLUS, ir.ONEG, ir.OBITNOT:
		n := n.(*ir.UnaryExpr)
		e.unsafeValue(k, n.X)
	case ir.OADD, ir.OSUB, ir.OOR, ir.OXOR, ir.OMUL, ir.ODIV, ir.OMOD, ir.OAND, ir.OANDNOT:
		n := n.(*ir.BinaryExpr)
		e.unsafeValue(k, n.X)
		e.unsafeValue(k, n.Y)
	case ir.OLSH, ir.ORSH:
		n := n.(*ir.BinaryExpr)
		e.unsafeValue(k, n.X)
		// RHS need not be uintptr-typed (#32959) and can't meaningfully
		// flow pointers anyway.
		e.discard(n.Y)
	default:
		e.exprSkipInit(e.discardHole(), n)
	}
}

// discard evaluates an expression n for side-effects, but discards
// its value.
func (e *escape) discard(n ir.Node) {
	e.expr(e.discardHole(), n)
}

func (e *escape) discards(l ir.Nodes) {
	for _, n := range l {
		e.discard(n)
	}
}

// addr evaluates an addressable expression n and returns a hole
// that represents storing into the represented location.
func (e *escape) addr(n ir.Node) hole {
	if n == nil || ir.IsBlank(n) {
		// Can happen in select case, range, maybe others.
		return e.discardHole()
	}

	k := e.heapHole()

	switch n.Op() {
	default:
		base.Fatalf("unexpected addr: %v", n)
	case ir.ONAME:
		n := n.(*ir.Name)
		if n.Class == ir.PEXTERN {
			break
		}
		k = e.oldLoc(n).asHole()
	case ir.OLINKSYMOFFSET:
		break
	case ir.ODOT:
		n := n.(*ir.SelectorExpr)
		k = e.addr(n.X)
	case ir.OINDEX:
		n := n.(*ir.IndexExpr)
		e.discard(n.Index)
		if n.X.Type().IsArray() {
			k = e.addr(n.X)
		} else {
			e.discard(n.X)
		}
	case ir.ODEREF, ir.ODOTPTR:
		e.discard(n)
	case ir.OINDEXMAP:
		n := n.(*ir.IndexExpr)
		e.discard(n.X)
		e.assignHeap(n.Index, "key of map put", n)
	}

	return k
}

func (e *escape) addrs(l ir.Nodes) []hole {
	var ks []hole
	for _, n := range l {
		ks = append(ks, e.addr(n))
	}
	return ks
}

// reassigned marks the locations associated with the given holes as
// reassigned, unless the location represents a variable declared and
// assigned exactly once by where.
func (e *escape) reassigned(ks []hole, where ir.Node) {
	if as, ok := where.(*ir.AssignStmt); ok && as.Op() == ir.OAS && as.Y == nil {
		if dst, ok := as.X.(*ir.Name); ok && dst.Op() == ir.ONAME && dst.Defn == nil {
			// Zero-value assignment for variable declared without an
			// explicit initial value. Assume this is its initialization
			// statement.
			return
		}
	}

	for _, k := range ks {
		loc := k.dst
		// Variables declared by range statements are assigned on every iteration.
		if n, ok := loc.n.(*ir.Name); ok && n.Defn == where && where.Op() != ir.ORANGE {
			continue
		}
		loc.reassigned = true
	}
}

// assignList evaluates the assignment dsts... = srcs....
func (e *escape) assignList(dsts, srcs []ir.Node, why string, where ir.Node) {
	ks := e.addrs(dsts)
	for i, k := range ks {
		var src ir.Node
		if i < len(srcs) {
			src = srcs[i]
		}

		if dst := dsts[i]; dst != nil {
			// Detect implicit conversion of uintptr to unsafe.Pointer when
			// storing into reflect.{Slice,String}Header.
			if dst.Op() == ir.ODOTPTR && ir.IsReflectHeaderDataField(dst) {
				e.unsafeValue(e.heapHole().note(where, why), src)
				continue
			}

			// Filter out some no-op assignments for escape analysis.
			if src != nil && isSelfAssign(dst, src) {
				if base.Flag.LowerM != 0 {
					base.WarnfAt(where.Pos(), "%v ignoring self-assignment in %v", e.curfn, where)
				}
				k = e.discardHole()
			}
		}

		e.expr(k.note(where, why), src)
	}

	e.reassigned(ks, where)
}

func (e *escape) assignHeap(src ir.Node, why string, where ir.Node) {
	e.expr(e.heapHole().note(where, why), src)
}

// call evaluates a call expressions, including builtin calls. ks
// should contain the holes representing where the function callee's
// results flows; where is the OGO/ODEFER context of the call, if any.
func (e *escape) call(ks []hole, call, where ir.Node) {
	topLevelDefer := where != nil && where.Op() == ir.ODEFER && e.loopDepth == 1
	if topLevelDefer {
		// force stack allocation of defer record, unless
		// open-coded defers are used (see ssa.go)
		where.SetEsc(ir.EscNever)
	}

	argument := func(k hole, arg ir.Node) {
		if topLevelDefer {
			// Top level defers arguments don't escape to
			// heap, but they do need to last until end of
			// function.
			k = e.later(k)
		} else if where != nil {
			k = e.heapHole()
		}

		e.expr(k.note(call, "call parameter"), arg)
	}

	switch call.Op() {
	default:
		ir.Dump("esc", call)
		base.Fatalf("unexpected call op: %v", call.Op())

	case ir.OCALLFUNC, ir.OCALLMETH, ir.OCALLINTER:
		call := call.(*ir.CallExpr)
		typecheck.FixVariadicCall(call)

		// Pick out the function callee, if statically known.
		var fn *ir.Name
		switch call.Op() {
		case ir.OCALLFUNC:
			switch v := ir.StaticValue(call.X); {
			case v.Op() == ir.ONAME && v.(*ir.Name).Class == ir.PFUNC:
				fn = v.(*ir.Name)
			case v.Op() == ir.OCLOSURE:
				fn = v.(*ir.ClosureExpr).Func.Nname
			}
		case ir.OCALLMETH:
			fn = ir.MethodExprName(call.X)
		}

		fntype := call.X.Type()
		if fn != nil {
			fntype = fn.Type()
		}

		if ks != nil && fn != nil && e.inMutualBatch(fn) {
			for i, result := range fn.Type().Results().FieldSlice() {
				e.expr(ks[i], ir.AsNode(result.Nname))
			}
		}

		if r := fntype.Recv(); r != nil {
			argument(e.tagHole(ks, fn, r), call.X.(*ir.SelectorExpr).X)
		} else {
			// Evaluate callee function expression.
			argument(e.discardHole(), call.X)
		}

		args := call.Args
		for i, param := range fntype.Params().FieldSlice() {
			argument(e.tagHole(ks, fn, param), args[i])
		}

	case ir.OAPPEND:
		call := call.(*ir.CallExpr)
		args := call.Args

		// Appendee slice may flow directly to the result, if
		// it has enough capacity. Alternatively, a new heap
		// slice might be allocated, and all slice elements
		// might flow to heap.
		appendeeK := ks[0]
		if args[0].Type().Elem().HasPointers() {
			appendeeK = e.teeHole(appendeeK, e.heapHole().deref(call, "appendee slice"))
		}
		argument(appendeeK, args[0])

		if call.IsDDD {
			appendedK := e.discardHole()
			if args[1].Type().IsSlice() && args[1].Type().Elem().HasPointers() {
				appendedK = e.heapHole().deref(call, "appended slice...")
			}
			argument(appendedK, args[1])
		} else {
			for _, arg := range args[1:] {
				argument(e.heapHole(), arg)
			}
		}

	case ir.OCOPY:
		call := call.(*ir.BinaryExpr)
		argument(e.discardHole(), call.X)

		copiedK := e.discardHole()
		if call.Y.Type().IsSlice() && call.Y.Type().Elem().HasPointers() {
			copiedK = e.heapHole().deref(call, "copied slice")
		}
		argument(copiedK, call.Y)

	case ir.OPANIC:
		call := call.(*ir.UnaryExpr)
		argument(e.heapHole(), call.X)

	case ir.OCOMPLEX:
		call := call.(*ir.BinaryExpr)
		argument(e.discardHole(), call.X)
		argument(e.discardHole(), call.Y)
	case ir.ODELETE, ir.OPRINT, ir.OPRINTN, ir.ORECOVER:
		call := call.(*ir.CallExpr)
		for _, arg := range call.Args {
			argument(e.discardHole(), arg)
		}
	case ir.OLEN, ir.OCAP, ir.OREAL, ir.OIMAG, ir.OCLOSE:
		call := call.(*ir.UnaryExpr)
		argument(e.discardHole(), call.X)

	case ir.OUNSAFEADD, ir.OUNSAFESLICE:
		call := call.(*ir.BinaryExpr)
		argument(ks[0], call.X)
		argument(e.discardHole(), call.Y)
	}
}

// tagHole returns a hole for evaluating an argument passed to param.
// ks should contain the holes representing where the function
// callee's results flows. fn is the statically-known callee function,
// if any.
func (e *escape) tagHole(ks []hole, fn *ir.Name, param *types.Field) hole {
	// If this is a dynamic call, we can't rely on param.Note.
	if fn == nil {
		return e.heapHole()
	}

	if e.inMutualBatch(fn) {
		return e.addr(ir.AsNode(param.Nname))
	}

	// Call to previously tagged function.

	if param.Note == UintptrEscapesNote {
		k := e.heapHole()
		k.uintptrEscapesHack = true
		return k
	}

	var tagKs []hole

	esc := parseLeaks(param.Note)
	if x := esc.Heap(); x >= 0 {
		tagKs = append(tagKs, e.heapHole().shift(x))
	}

	if ks != nil {
		for i := 0; i < numEscResults; i++ {
			if x := esc.Result(i); x >= 0 {
				tagKs = append(tagKs, ks[i].shift(x))
			}
		}
	}

	return e.teeHole(tagKs...)
}

// inMutualBatch reports whether function fn is in the batch of
// mutually recursive functions being analyzed. When this is true,
// fn has not yet been analyzed, so its parameters and results
// should be incorporated directly into the flow graph instead of
// relying on its escape analysis tagging.
func (e *escape) inMutualBatch(fn *ir.Name) bool {
	if fn.Defn != nil && fn.Defn.Esc() < escFuncTagged {
		if fn.Defn.Esc() == escFuncUnknown {
			base.Fatalf("graph inconsistency: %v", fn)
		}
		return true
	}
	return false
}

// An hole represents a context for evaluation a Go
// expression. E.g., when evaluating p in "x = **p", we'd have a hole
// with dst==x and derefs==2.
type hole struct {
	dst    *location
	derefs int // >= -1
	notes  *note

	// addrtaken indicates whether this context is taking the address of
	// the expression, independent of whether the address will actually
	// be stored into a variable.
	addrtaken bool

	// uintptrEscapesHack indicates this context is evaluating an
	// argument for a //go:uintptrescapes function.
	uintptrEscapesHack bool
}

type note struct {
	next  *note
	where ir.Node
	why   string
}

func (k hole) note(where ir.Node, why string) hole {
	if where == nil || why == "" {
		base.Fatalf("note: missing where/why")
	}
	if base.Flag.LowerM >= 2 || logopt.Enabled() {
		k.notes = &note{
			next:  k.notes,
			where: where,
			why:   why,
		}
	}
	return k
}

func (k hole) shift(delta int) hole {
	k.derefs += delta
	if k.derefs < -1 {
		base.Fatalf("derefs underflow: %v", k.derefs)
	}
	k.addrtaken = delta < 0
	return k
}

func (k hole) deref(where ir.Node, why string) hole { return k.shift(1).note(where, why) }
func (k hole) addr(where ir.Node, why string) hole  { return k.shift(-1).note(where, why) }

func (k hole) dotType(t *types.Type, where ir.Node, why string) hole {
	if !t.IsInterface() && !types.IsDirectIface(t) {
		k = k.shift(1)
	}
	return k.note(where, why)
}

// teeHole returns a new hole that flows into each hole of ks,
// similar to the Unix tee(1) command.
func (e *escape) teeHole(ks ...hole) hole {
	if len(ks) == 0 {
		return e.discardHole()
	}
	if len(ks) == 1 {
		return ks[0]
	}
	// TODO(mdempsky): Optimize if there's only one non-discard hole?

	// Given holes "l1 = _", "l2 = **_", "l3 = *_", ..., create a
	// new temporary location ltmp, wire it into place, and return
	// a hole for "ltmp = _".
	loc := e.newLoc(nil, true)
	for _, k := range ks {
		// N.B., "p = &q" and "p = &tmp; tmp = q" are not
		// semantically equivalent. To combine holes like "l1
		// = _" and "l2 = &_", we'd need to wire them as "l1 =
		// *ltmp" and "l2 = ltmp" and return "ltmp = &_"
		// instead.
		if k.derefs < 0 {
			base.Fatalf("teeHole: negative derefs")
		}

		e.flow(k, loc)
	}
	return loc.asHole()
}

func (e *escape) dcl(n *ir.Name) hole {
	if n.Curfn != e.curfn || n.IsClosureVar() {
		base.Fatalf("bad declaration of %v", n)
	}
	loc := e.oldLoc(n)
	loc.loopDepth = e.loopDepth
	return loc.asHole()
}

// spill allocates a new location associated with expression n, flows
// its address to k, and returns a hole that flows values to it. It's
// intended for use with most expressions that allocate storage.
func (e *escape) spill(k hole, n ir.Node) hole {
	loc := e.newLoc(n, true)
	e.flow(k.addr(n, "spill"), loc)
	return loc.asHole()
}

// later returns a new hole that flows into k, but some time later.
// Its main effect is to prevent immediate reuse of temporary
// variables introduced during Order.
func (e *escape) later(k hole) hole {
	loc := e.newLoc(nil, false)
	e.flow(k, loc)
	return loc.asHole()
}

func (e *escape) newLoc(n ir.Node, transient bool) *location {
	if e.curfn == nil {
		base.Fatalf("e.curfn isn't set")
	}
	if n != nil && n.Type() != nil && n.Type().NotInHeap() {
		base.ErrorfAt(n.Pos(), "%v is incomplete (or unallocatable); stack allocation disallowed", n.Type())
	}

	if n != nil && n.Op() == ir.ONAME {
		n = n.(*ir.Name).Canonical()
	}
	loc := &location{
		n:         n,
		curfn:     e.curfn,
		loopDepth: e.loopDepth,
		transient: transient,
	}
	e.allLocs = append(e.allLocs, loc)
	if n != nil {
		if n.Op() == ir.ONAME {
			n := n.(*ir.Name)
			if n.Curfn != e.curfn {
				base.Fatalf("curfn mismatch: %v != %v for %v", n.Curfn, e.curfn, n)
			}

			if n.Opt != nil {
				base.Fatalf("%v already has a location", n)
			}
			n.Opt = loc
		}
	}
	return loc
}

func (b *batch) oldLoc(n *ir.Name) *location {
	if n.Canonical().Opt == nil {
		base.Fatalf("%v has no location", n)
	}
	return n.Canonical().Opt.(*location)
}

func (l *location) asHole() hole {
	return hole{dst: l}
}

func (b *batch) flow(k hole, src *location) {
	if k.addrtaken {
		src.addrtaken = true
	}

	dst := k.dst
	if dst == &b.blankLoc {
		return
	}
	if dst == src && k.derefs >= 0 { // dst = dst, dst = *dst, ...
		return
	}
	if dst.escapes && k.derefs < 0 { // dst = &src
		if base.Flag.LowerM >= 2 || logopt.Enabled() {
			pos := base.FmtPos(src.n.Pos())
			if base.Flag.LowerM >= 2 {
				fmt.Printf("%s: %v escapes to heap:\n", pos, src.n)
			}
			explanation := b.explainFlow(pos, dst, src, k.derefs, k.notes, []*logopt.LoggedOpt{})
			if logopt.Enabled() {
				var e_curfn *ir.Func // TODO(mdempsky): Fix.
				logopt.LogOpt(src.n.Pos(), "escapes", "escape", ir.FuncName(e_curfn), fmt.Sprintf("%v escapes to heap", src.n), explanation)
			}

		}
		src.escapes = true
		return
	}

	// TODO(mdempsky): Deduplicate edges?
	dst.edges = append(dst.edges, edge{src: src, derefs: k.derefs, notes: k.notes})
}

func (b *batch) heapHole() hole    { return b.heapLoc.asHole() }
func (b *batch) discardHole() hole { return b.blankLoc.asHole() }

// walkAll computes the minimal dereferences between all pairs of
// locations.
func (b *batch) walkAll() {
	// We use a work queue to keep track of locations that we need
	// to visit, and repeatedly walk until we reach a fixed point.
	//
	// We walk once from each location (including the heap), and
	// then re-enqueue each location on its transition from
	// transient->!transient and !escapes->escapes, which can each
	// happen at most once. So we take Θ(len(e.allLocs)) walks.

	// LIFO queue, has enough room for e.allLocs and e.heapLoc.
	todo := make([]*location, 0, len(b.allLocs)+1)
	enqueue := func(loc *location) {
		if !loc.queued {
			todo = append(todo, loc)
			loc.queued = true
		}
	}

	for _, loc := range b.allLocs {
		enqueue(loc)
	}
	enqueue(&b.heapLoc)

	var walkgen uint32
	for len(todo) > 0 {
		root := todo[len(todo)-1]
		todo = todo[:len(todo)-1]
		root.queued = false

		walkgen++
		b.walkOne(root, walkgen, enqueue)
	}
}

// walkOne computes the minimal number of dereferences from root to
// all other locations.
func (b *batch) walkOne(root *location, walkgen uint32, enqueue func(*location)) {
	// The data flow graph has negative edges (from addressing
	// operations), so we use the Bellman-Ford algorithm. However,
	// we don't have to worry about infinite negative cycles since
	// we bound intermediate dereference counts to 0.

	root.walkgen = walkgen
	root.derefs = 0
	root.dst = nil

	todo := []*location{root} // LIFO queue
	for len(todo) > 0 {
		l := todo[len(todo)-1]
		todo = todo[:len(todo)-1]

		derefs := l.derefs

		// If l.derefs < 0, then l's address flows to root.
		addressOf := derefs < 0
		if addressOf {
			// For a flow path like "root = &l; l = x",
			// l's address flows to root, but x's does
			// not. We recognize this by lower bounding
			// derefs at 0.
			derefs = 0

			// If l's address flows to a non-transient
			// location, then l can't be transiently
			// allocated.
			if !root.transient && l.transient {
				l.transient = false
				enqueue(l)
			}
		}

		if b.outlives(root, l) {
			// l's value flows to root. If l is a function
			// parameter and root is the heap or a
			// corresponding result parameter, then record
			// that value flow for tagging the function
			// later.
			if l.isName(ir.PPARAM) {
				if (logopt.Enabled() || base.Flag.LowerM >= 2) && !l.escapes {
					if base.Flag.LowerM >= 2 {
						fmt.Printf("%s: parameter %v leaks to %s with derefs=%d:\n", base.FmtPos(l.n.Pos()), l.n, b.explainLoc(root), derefs)
					}
					explanation := b.explainPath(root, l)
					if logopt.Enabled() {
						var e_curfn *ir.Func // TODO(mdempsky): Fix.
						logopt.LogOpt(l.n.Pos(), "leak", "escape", ir.FuncName(e_curfn),
							fmt.Sprintf("parameter %v leaks to %s with derefs=%d", l.n, b.explainLoc(root), derefs), explanation)
					}
				}
				l.leakTo(root, derefs)
			}

			// If l's address flows somewhere that
			// outlives it, then l needs to be heap
			// allocated.
			if addressOf && !l.escapes {
				if logopt.Enabled() || base.Flag.LowerM >= 2 {
					if base.Flag.LowerM >= 2 {
						fmt.Printf("%s: %v escapes to heap:\n", base.FmtPos(l.n.Pos()), l.n)
					}
					explanation := b.explainPath(root, l)
					if logopt.Enabled() {
						var e_curfn *ir.Func // TODO(mdempsky): Fix.
						logopt.LogOpt(l.n.Pos(), "escape", "escape", ir.FuncName(e_curfn), fmt.Sprintf("%v escapes to heap", l.n), explanation)
					}
				}
				l.escapes = true
				enqueue(l)
				continue
			}
		}

		for i, edge := range l.edges {
			if edge.src.escapes {
				continue
			}
			d := derefs + edge.derefs
			if edge.src.walkgen != walkgen || edge.src.derefs > d {
				edge.src.walkgen = walkgen
				edge.src.derefs = d
				edge.src.dst = l
				edge.src.dstEdgeIdx = i
				todo = append(todo, edge.src)
			}
		}
	}
}

// explainPath prints an explanation of how src flows to the walk root.
func (b *batch) explainPath(root, src *location) []*logopt.LoggedOpt {
	visited := make(map[*location]bool)
	pos := base.FmtPos(src.n.Pos())
	var explanation []*logopt.LoggedOpt
	for {
		// Prevent infinite loop.
		if visited[src] {
			if base.Flag.LowerM >= 2 {
				fmt.Printf("%s:   warning: truncated explanation due to assignment cycle; see golang.org/issue/35518\n", pos)
			}
			break
		}
		visited[src] = true
		dst := src.dst
		edge := &dst.edges[src.dstEdgeIdx]
		if edge.src != src {
			base.Fatalf("path inconsistency: %v != %v", edge.src, src)
		}

		explanation = b.explainFlow(pos, dst, src, edge.derefs, edge.notes, explanation)

		if dst == root {
			break
		}
		src = dst
	}

	return explanation
}

func (b *batch) explainFlow(pos string, dst, srcloc *location, derefs int, notes *note, explanation []*logopt.LoggedOpt) []*logopt.LoggedOpt {
	ops := "&"
	if derefs >= 0 {
		ops = strings.Repeat("*", derefs)
	}
	print := base.Flag.LowerM >= 2

	flow := fmt.Sprintf("   flow: %s = %s%v:", b.explainLoc(dst), ops, b.explainLoc(srcloc))
	if print {
		fmt.Printf("%s:%s\n", pos, flow)
	}
	if logopt.Enabled() {
		var epos src.XPos
		if notes != nil {
			epos = notes.where.Pos()
		} else if srcloc != nil && srcloc.n != nil {
			epos = srcloc.n.Pos()
		}
		var e_curfn *ir.Func // TODO(mdempsky): Fix.
		explanation = append(explanation, logopt.NewLoggedOpt(epos, "escflow", "escape", ir.FuncName(e_curfn), flow))
	}

	for note := notes; note != nil; note = note.next {
		if print {
			fmt.Printf("%s:     from %v (%v) at %s\n", pos, note.where, note.why, base.FmtPos(note.where.Pos()))
		}
		if logopt.Enabled() {
			var e_curfn *ir.Func // TODO(mdempsky): Fix.
			explanation = append(explanation, logopt.NewLoggedOpt(note.where.Pos(), "escflow", "escape", ir.FuncName(e_curfn),
				fmt.Sprintf("     from %v (%v)", note.where, note.why)))
		}
	}
	return explanation
}

func (b *batch) explainLoc(l *location) string {
	if l == &b.heapLoc {
		return "{heap}"
	}
	if l.n == nil {
		// TODO(mdempsky): Omit entirely.
		return "{temp}"
	}
	if l.n.Op() == ir.ONAME {
		return fmt.Sprintf("%v", l.n)
	}
	return fmt.Sprintf("{storage for %v}", l.n)
}

// outlives reports whether values stored in l may survive beyond
// other's lifetime if stack allocated.
func (b *batch) outlives(l, other *location) bool {
	// The heap outlives everything.
	if l.escapes {
		return true
	}

	// We don't know what callers do with returned values, so
	// pessimistically we need to assume they flow to the heap and
	// outlive everything too.
	if l.isName(ir.PPARAMOUT) {
		// Exception: Directly called closures can return
		// locations allocated outside of them without forcing
		// them to the heap. For example:
		//
		//    var u int  // okay to stack allocate
		//    *(func() *int { return &u }()) = 42
		if containsClosure(other.curfn, l.curfn) && l.curfn.ClosureCalled() {
			return false
		}

		return true
	}

	// If l and other are within the same function, then l
	// outlives other if it was declared outside other's loop
	// scope. For example:
	//
	//    var l *int
	//    for {
	//        l = new(int)
	//    }
	if l.curfn == other.curfn && l.loopDepth < other.loopDepth {
		return true
	}

	// If other is declared within a child closure of where l is
	// declared, then l outlives it. For example:
	//
	//    var l *int
	//    func() {
	//        l = new(int)
	//    }
	if containsClosure(l.curfn, other.curfn) {
		return true
	}

	return false
}

// containsClosure reports whether c is a closure contained within f.
func containsClosure(f, c *ir.Func) bool {
	// Common case.
	if f == c {
		return false
	}

	// Closures within function Foo are named like "Foo.funcN..."
	// TODO(mdempsky): Better way to recognize this.
	fn := f.Sym().Name
	cn := c.Sym().Name
	return len(cn) > len(fn) && cn[:len(fn)] == fn && cn[len(fn)] == '.'
}

// leak records that parameter l leaks to sink.
func (l *location) leakTo(sink *location, derefs int) {
	// If sink is a result parameter that doesn't escape (#44614)
	// and we can fit return bits into the escape analysis tag,
	// then record as a result leak.
	if !sink.escapes && sink.isName(ir.PPARAMOUT) && sink.curfn == l.curfn {
		ri := sink.resultIndex - 1
		if ri < numEscResults {
			// Leak to result parameter.
			l.paramEsc.AddResult(ri, derefs)
			return
		}
	}

	// Otherwise, record as heap leak.
	l.paramEsc.AddHeap(derefs)
}

func (b *batch) finish(fns []*ir.Func) {
	// Record parameter tags for package export data.
	for _, fn := range fns {
		fn.SetEsc(escFuncTagged)

		narg := 0
		for _, fs := range &types.RecvsParams {
			for _, f := range fs(fn.Type()).Fields().Slice() {
				narg++
				f.Note = b.paramTag(fn, narg, f)
			}
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

		if loc.escapes {
			if n.Op() == ir.ONAME {
				if base.Flag.CompilingRuntime {
					base.ErrorfAt(n.Pos(), "%v escapes to heap, not allowed in runtime", n)
				}
				if base.Flag.LowerM != 0 {
					base.WarnfAt(n.Pos(), "moved to heap: %v", n)
				}
			} else {
				if base.Flag.LowerM != 0 {
					base.WarnfAt(n.Pos(), "%v escapes to heap", n)
				}
				if logopt.Enabled() {
					var e_curfn *ir.Func // TODO(mdempsky): Fix.
					logopt.LogOpt(n.Pos(), "escape", "escape", ir.FuncName(e_curfn))
				}
			}
			n.SetEsc(ir.EscHeap)
		} else {
			if base.Flag.LowerM != 0 && n.Op() != ir.ONAME {
				base.WarnfAt(n.Pos(), "%v does not escape", n)
			}
			n.SetEsc(ir.EscNone)
			if loc.transient {
				switch n.Op() {
				case ir.OCLOSURE:
					n := n.(*ir.ClosureExpr)
					n.SetTransient(true)
				case ir.OCALLPART:
					n := n.(*ir.SelectorExpr)
					n.SetTransient(true)
				case ir.OSLICELIT:
					n := n.(*ir.CompLitExpr)
					n.SetTransient(true)
				}
			}
		}
	}
}

func (l *location) isName(c ir.Class) bool {
	return l.n != nil && l.n.Op() == ir.ONAME && l.n.(*ir.Name).Class == c
}

const numEscResults = 7

// An leaks represents a set of assignment flows from a parameter
// to the heap or to any of its function's (first numEscResults)
// result parameters.
type leaks [1 + numEscResults]uint8

// Empty reports whether l is an empty set (i.e., no assignment flows).
func (l leaks) Empty() bool { return l == leaks{} }

// Heap returns the minimum deref count of any assignment flow from l
// to the heap. If no such flows exist, Heap returns -1.
func (l leaks) Heap() int { return l.get(0) }

// Result returns the minimum deref count of any assignment flow from
// l to its function's i'th result parameter. If no such flows exist,
// Result returns -1.
func (l leaks) Result(i int) int { return l.get(1 + i) }

// AddHeap adds an assignment flow from l to the heap.
func (l *leaks) AddHeap(derefs int) { l.add(0, derefs) }

// AddResult adds an assignment flow from l to its function's i'th
// result parameter.
func (l *leaks) AddResult(i, derefs int) { l.add(1+i, derefs) }

func (l *leaks) setResult(i, derefs int) { l.set(1+i, derefs) }

func (l leaks) get(i int) int { return int(l[i]) - 1 }

func (l *leaks) add(i, derefs int) {
	if old := l.get(i); old < 0 || derefs < old {
		l.set(i, derefs)
	}
}

func (l *leaks) set(i, derefs int) {
	v := derefs + 1
	if v < 0 {
		base.Fatalf("invalid derefs count: %v", derefs)
	}
	if v > math.MaxUint8 {
		v = math.MaxUint8
	}

	l[i] = uint8(v)
}

// Optimize removes result flow paths that are equal in length or
// longer than the shortest heap flow path.
func (l *leaks) Optimize() {
	// If we have a path to the heap, then there's no use in
	// keeping equal or longer paths elsewhere.
	if x := l.Heap(); x >= 0 {
		for i := 0; i < numEscResults; i++ {
			if l.Result(i) >= x {
				l.setResult(i, -1)
			}
		}
	}
}

var leakTagCache = map[leaks]string{}

// Encode converts l into a binary string for export data.
func (l leaks) Encode() string {
	if l.Heap() == 0 {
		// Space optimization: empty string encodes more
		// efficiently in export data.
		return ""
	}
	if s, ok := leakTagCache[l]; ok {
		return s
	}

	n := len(l)
	for n > 0 && l[n-1] == 0 {
		n--
	}
	s := "esc:" + string(l[:n])
	leakTagCache[l] = s
	return s
}

// parseLeaks parses a binary string representing a leaks
func parseLeaks(s string) leaks {
	var l leaks
	if !strings.HasPrefix(s, "esc:") {
		l.AddHeap(0)
		return l
	}
	copy(l[:], s[4:])
	return l
}

func Funcs(all []ir.Node) {
	ir.VisitFuncsBottomUp(all, Batch)
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

func isSliceSelfAssign(dst, src ir.Node) bool {
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
	var dstX ir.Node
	switch dst.Op() {
	default:
		return false
	case ir.ODEREF:
		dst := dst.(*ir.StarExpr)
		dstX = dst.X
	case ir.ODOTPTR:
		dst := dst.(*ir.SelectorExpr)
		dstX = dst.X
	}
	if dstX.Op() != ir.ONAME {
		return false
	}
	// src is a slice operation.
	switch src.Op() {
	case ir.OSLICE, ir.OSLICE3, ir.OSLICESTR:
		// OK.
	case ir.OSLICEARR, ir.OSLICE3ARR:
		// Since arrays are embedded into containing object,
		// slice of non-pointer array will introduce a new pointer into b that was not already there
		// (pointer to b itself). After such assignment, if b contents escape,
		// b escapes as well. If we ignore such OSLICEARR, we will conclude
		// that b does not escape when b contents do.
		//
		// Pointer to an array is OK since it's not stored inside b directly.
		// For slicing an array (not pointer to array), there is an implicit OADDR.
		// We check that to determine non-pointer array slicing.
		src := src.(*ir.SliceExpr)
		if src.X.Op() == ir.OADDR {
			return false
		}
	default:
		return false
	}
	// slice is applied to ONAME dereference.
	var baseX ir.Node
	switch base := src.(*ir.SliceExpr).X; base.Op() {
	default:
		return false
	case ir.ODEREF:
		base := base.(*ir.StarExpr)
		baseX = base.X
	case ir.ODOTPTR:
		base := base.(*ir.SelectorExpr)
		baseX = base.X
	}
	if baseX.Op() != ir.ONAME {
		return false
	}
	// dst and src reference the same base ONAME.
	return dstX.(*ir.Name) == baseX.(*ir.Name)
}

// isSelfAssign reports whether assignment from src to dst can
// be ignored by the escape analysis as it's effectively a self-assignment.
func isSelfAssign(dst, src ir.Node) bool {
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

	if dst == nil || src == nil || dst.Op() != src.Op() {
		return false
	}

	// The expression prefix must be both "safe" and identical.
	switch dst.Op() {
	case ir.ODOT, ir.ODOTPTR:
		// Safe trailing accessors that are permitted to differ.
		dst := dst.(*ir.SelectorExpr)
		src := src.(*ir.SelectorExpr)
		return ir.SameSafeExpr(dst.X, src.X)
	case ir.OINDEX:
		dst := dst.(*ir.IndexExpr)
		src := src.(*ir.IndexExpr)
		if mayAffectMemory(dst.Index) || mayAffectMemory(src.Index) {
			return false
		}
		return ir.SameSafeExpr(dst.X, src.X)
	default:
		return false
	}
}

// mayAffectMemory reports whether evaluation of n may affect the program's
// memory state. If the expression can't affect memory state, then it can be
// safely ignored by the escape analysis.
func mayAffectMemory(n ir.Node) bool {
	// We may want to use a list of "memory safe" ops instead of generally
	// "side-effect free", which would include all calls and other ops that can
	// allocate or change global state. For now, it's safer to start with the latter.
	//
	// We're ignoring things like division by zero, index out of range,
	// and nil pointer dereference here.

	// TODO(rsc): It seems like it should be possible to replace this with
	// an ir.Any looking for any op that's not the ones in the case statement.
	// But that produces changes in the compiled output detected by buildall.
	switch n.Op() {
	case ir.ONAME, ir.OLITERAL, ir.ONIL:
		return false

	case ir.OADD, ir.OSUB, ir.OOR, ir.OXOR, ir.OMUL, ir.OLSH, ir.ORSH, ir.OAND, ir.OANDNOT, ir.ODIV, ir.OMOD:
		n := n.(*ir.BinaryExpr)
		return mayAffectMemory(n.X) || mayAffectMemory(n.Y)

	case ir.OINDEX:
		n := n.(*ir.IndexExpr)
		return mayAffectMemory(n.X) || mayAffectMemory(n.Index)

	case ir.OCONVNOP, ir.OCONV:
		n := n.(*ir.ConvExpr)
		return mayAffectMemory(n.X)

	case ir.OLEN, ir.OCAP, ir.ONOT, ir.OBITNOT, ir.OPLUS, ir.ONEG, ir.OALIGNOF, ir.OOFFSETOF, ir.OSIZEOF:
		n := n.(*ir.UnaryExpr)
		return mayAffectMemory(n.X)

	case ir.ODOT, ir.ODOTPTR:
		n := n.(*ir.SelectorExpr)
		return mayAffectMemory(n.X)

	case ir.ODEREF:
		n := n.(*ir.StarExpr)
		return mayAffectMemory(n.X)

	default:
		return true
	}
}

// HeapAllocReason returns the reason the given Node must be heap
// allocated, or the empty string if it doesn't.
func HeapAllocReason(n ir.Node) string {
	if n == nil || n.Type() == nil {
		return ""
	}

	// Parameters are always passed via the stack.
	if n.Op() == ir.ONAME {
		n := n.(*ir.Name)
		if n.Class == ir.PPARAM || n.Class == ir.PPARAMOUT {
			return ""
		}
	}

	if n.Type().Width > ir.MaxStackVarSize {
		return "too large for stack"
	}

	if (n.Op() == ir.ONEW || n.Op() == ir.OPTRLIT) && n.Type().Elem().Width >= ir.MaxImplicitStackVarSize {
		return "too large for stack"
	}

	if n.Op() == ir.OCLOSURE && typecheck.ClosureType(n.(*ir.ClosureExpr)).Size() >= ir.MaxImplicitStackVarSize {
		return "too large for stack"
	}
	if n.Op() == ir.OCALLPART && typecheck.PartialCallType(n.(*ir.SelectorExpr)).Size() >= ir.MaxImplicitStackVarSize {
		return "too large for stack"
	}

	if n.Op() == ir.OMAKESLICE {
		n := n.(*ir.MakeExpr)
		r := n.Cap
		if r == nil {
			r = n.Len
		}
		if !ir.IsSmallIntConst(r) {
			return "non-constant size"
		}
		if t := n.Type(); t.Elem().Width != 0 && ir.Int64Val(r) >= ir.MaxImplicitStackVarSize/t.Elem().Width {
			return "too large for stack"
		}
	}

	return ""
}

// This special tag is applied to uintptr variables
// that we believe may hold unsafe.Pointers for
// calls into assembly functions.
const UnsafeUintptrNote = "unsafe-uintptr"

// This special tag is applied to uintptr parameters of functions
// marked go:uintptrescapes.
const UintptrEscapesNote = "uintptr-escapes"

func (b *batch) paramTag(fn *ir.Func, narg int, f *types.Field) string {
	name := func() string {
		if f.Sym != nil {
			return f.Sym.Name
		}
		return fmt.Sprintf("arg#%d", narg)
	}

	if len(fn.Body) == 0 {
		// Assume that uintptr arguments must be held live across the call.
		// This is most important for syscall.Syscall.
		// See golang.org/issue/13372.
		// This really doesn't have much to do with escape analysis per se,
		// but we are reusing the ability to annotate an individual function
		// argument and pass those annotations along to importing code.
		if f.Type.IsUintptr() {
			if base.Flag.LowerM != 0 {
				base.WarnfAt(f.Pos, "assuming %v is unsafe uintptr", name())
			}
			return UnsafeUintptrNote
		}

		if !f.Type.HasPointers() { // don't bother tagging for scalars
			return ""
		}

		var esc leaks

		// External functions are assumed unsafe, unless
		// //go:noescape is given before the declaration.
		if fn.Pragma&ir.Noescape != 0 {
			if base.Flag.LowerM != 0 && f.Sym != nil {
				base.WarnfAt(f.Pos, "%v does not escape", name())
			}
		} else {
			if base.Flag.LowerM != 0 && f.Sym != nil {
				base.WarnfAt(f.Pos, "leaking param: %v", name())
			}
			esc.AddHeap(0)
		}

		return esc.Encode()
	}

	if fn.Pragma&ir.UintptrEscapes != 0 {
		if f.Type.IsUintptr() {
			if base.Flag.LowerM != 0 {
				base.WarnfAt(f.Pos, "marking %v as escaping uintptr", name())
			}
			return UintptrEscapesNote
		}
		if f.IsDDD() && f.Type.Elem().IsUintptr() {
			// final argument is ...uintptr.
			if base.Flag.LowerM != 0 {
				base.WarnfAt(f.Pos, "marking %v as escaping ...uintptr", name())
			}
			return UintptrEscapesNote
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

	if base.Flag.LowerM != 0 && !loc.escapes {
		if esc.Empty() {
			base.WarnfAt(f.Pos, "%v does not escape", name())
		}
		if x := esc.Heap(); x >= 0 {
			if x == 0 {
				base.WarnfAt(f.Pos, "leaking param: %v", name())
			} else {
				// TODO(mdempsky): Mention level=x like below?
				base.WarnfAt(f.Pos, "leaking param content: %v", name())
			}
		}
		for i := 0; i < numEscResults; i++ {
			if x := esc.Result(i); x >= 0 {
				res := fn.Type().Results().Field(i).Sym
				base.WarnfAt(f.Pos, "leaking param: %v to result %v level=%d", name(), res, x)
			}
		}
	}

	return esc.Encode()
}
