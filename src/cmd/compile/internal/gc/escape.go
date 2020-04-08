// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/types"
	"fmt"
	"math"
	"strings"
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

type Escape struct {
	allLocs []*EscLocation

	curfn *Node

	// loopDepth counts the current loop nesting depth within
	// curfn. It increments within each "for" loop and at each
	// label with a corresponding backwards "goto" (i.e.,
	// unstructured loop).
	loopDepth int

	heapLoc  EscLocation
	blankLoc EscLocation
}

// An EscLocation represents an abstract location that stores a Go
// variable.
type EscLocation struct {
	n         *Node     // represented variable or expression, if any
	curfn     *Node     // enclosing function
	edges     []EscEdge // incoming edges
	loopDepth int       // loopDepth at declaration

	// derefs and walkgen are used during walkOne to track the
	// minimal dereferences from the walk root.
	derefs  int // >= -1
	walkgen uint32

	// dst and dstEdgeindex track the next immediate assignment
	// destination location during walkone, along with the index
	// of the edge pointing back to this location.
	dst        *EscLocation
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
	paramEsc EscLeaks
}

// An EscEdge represents an assignment edge between two Go variables.
type EscEdge struct {
	src    *EscLocation
	derefs int // >= -1
	notes  *EscNote
}

// escapeFuncs performs escape analysis on a minimal batch of
// functions.
func escapeFuncs(fns []*Node, recursive bool) {
	for _, fn := range fns {
		if fn.Op != ODCLFUNC {
			Fatalf("unexpected node: %v", fn)
		}
	}

	var e Escape
	e.heapLoc.escapes = true

	// Construct data-flow graph from syntax trees.
	for _, fn := range fns {
		e.initFunc(fn)
	}
	for _, fn := range fns {
		e.walkFunc(fn)
	}
	e.curfn = nil

	e.walkAll()
	e.finish(fns)
}

func (e *Escape) initFunc(fn *Node) {
	if fn.Op != ODCLFUNC || fn.Esc != EscFuncUnknown {
		Fatalf("unexpected node: %v", fn)
	}
	fn.Esc = EscFuncPlanned
	if Debug['m'] > 3 {
		Dump("escAnalyze", fn)
	}

	e.curfn = fn
	e.loopDepth = 1

	// Allocate locations for local variables.
	for _, dcl := range fn.Func.Dcl {
		if dcl.Op == ONAME {
			e.newLoc(dcl, false)
		}
	}
}

func (e *Escape) walkFunc(fn *Node) {
	fn.Esc = EscFuncStarted

	// Identify labels that mark the head of an unstructured loop.
	inspectList(fn.Nbody, func(n *Node) bool {
		switch n.Op {
		case OLABEL:
			n.Sym.Label = asTypesNode(&nonlooping)

		case OGOTO:
			// If we visited the label before the goto,
			// then this is a looping label.
			if n.Sym.Label == asTypesNode(&nonlooping) {
				n.Sym.Label = asTypesNode(&looping)
			}
		}

		return true
	})

	e.curfn = fn
	e.loopDepth = 1
	e.block(fn.Nbody)
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
func (e *Escape) stmt(n *Node) {
	if n == nil {
		return
	}

	lno := setlineno(n)
	defer func() {
		lineno = lno
	}()

	if Debug['m'] > 2 {
		fmt.Printf("%v:[%d] %v stmt: %v\n", linestr(lineno), e.loopDepth, funcSym(e.curfn), n)
	}

	e.stmts(n.Ninit)

	switch n.Op {
	default:
		Fatalf("unexpected stmt: %v", n)

	case ODCLCONST, ODCLTYPE, OEMPTY, OFALL, OINLMARK:
		// nop

	case OBREAK, OCONTINUE, OGOTO:
		// TODO(mdempsky): Handle dead code?

	case OBLOCK:
		e.stmts(n.List)

	case ODCL:
		// Record loop depth at declaration.
		if !n.Left.isBlank() {
			e.dcl(n.Left)
		}

	case OLABEL:
		switch asNode(n.Sym.Label) {
		case &nonlooping:
			if Debug['m'] > 2 {
				fmt.Printf("%v:%v non-looping label\n", linestr(lineno), n)
			}
		case &looping:
			if Debug['m'] > 2 {
				fmt.Printf("%v: %v looping label\n", linestr(lineno), n)
			}
			e.loopDepth++
		default:
			Fatalf("label missing tag")
		}
		n.Sym.Label = nil

	case OIF:
		e.discard(n.Left)
		e.block(n.Nbody)
		e.block(n.Rlist)

	case OFOR, OFORUNTIL:
		e.loopDepth++
		e.discard(n.Left)
		e.stmt(n.Right)
		e.block(n.Nbody)
		e.loopDepth--

	case ORANGE:
		// for List = range Right { Nbody }
		e.loopDepth++
		ks := e.addrs(n.List)
		e.block(n.Nbody)
		e.loopDepth--

		// Right is evaluated outside the loop.
		k := e.discardHole()
		if len(ks) >= 2 {
			if n.Right.Type.IsArray() {
				k = ks[1].note(n, "range")
			} else {
				k = ks[1].deref(n, "range-deref")
			}
		}
		e.expr(e.later(k), n.Right)

	case OSWITCH:
		typesw := n.Left != nil && n.Left.Op == OTYPESW

		var ks []EscHole
		for _, cas := range n.List.Slice() { // cases
			if typesw && n.Left.Left != nil {
				cv := cas.Rlist.First()
				k := e.dcl(cv) // type switch variables have no ODCL.
				if types.Haspointers(cv.Type) {
					ks = append(ks, k.dotType(cv.Type, cas, "switch case"))
				}
			}

			e.discards(cas.List)
			e.block(cas.Nbody)
		}

		if typesw {
			e.expr(e.teeHole(ks...), n.Left.Right)
		} else {
			e.discard(n.Left)
		}

	case OSELECT:
		for _, cas := range n.List.Slice() {
			e.stmt(cas.Left)
			e.block(cas.Nbody)
		}
	case OSELRECV:
		e.assign(n.Left, n.Right, "selrecv", n)
	case OSELRECV2:
		e.assign(n.Left, n.Right, "selrecv", n)
		e.assign(n.List.First(), nil, "selrecv", n)
	case ORECV:
		// TODO(mdempsky): Consider e.discard(n.Left).
		e.exprSkipInit(e.discardHole(), n) // already visited n.Ninit
	case OSEND:
		e.discard(n.Left)
		e.assignHeap(n.Right, "send", n)

	case OAS, OASOP:
		e.assign(n.Left, n.Right, "assign", n)

	case OAS2:
		for i, nl := range n.List.Slice() {
			e.assign(nl, n.Rlist.Index(i), "assign-pair", n)
		}

	case OAS2DOTTYPE: // v, ok = x.(type)
		e.assign(n.List.First(), n.Right, "assign-pair-dot-type", n)
		e.assign(n.List.Second(), nil, "assign-pair-dot-type", n)
	case OAS2MAPR: // v, ok = m[k]
		e.assign(n.List.First(), n.Right, "assign-pair-mapr", n)
		e.assign(n.List.Second(), nil, "assign-pair-mapr", n)
	case OAS2RECV: // v, ok = <-ch
		e.assign(n.List.First(), n.Right, "assign-pair-receive", n)
		e.assign(n.List.Second(), nil, "assign-pair-receive", n)

	case OAS2FUNC:
		e.stmts(n.Right.Ninit)
		e.call(e.addrs(n.List), n.Right, nil)
	case ORETURN:
		results := e.curfn.Type.Results().FieldSlice()
		for i, v := range n.List.Slice() {
			e.assign(asNode(results[i].Nname), v, "return", n)
		}
	case OCALLFUNC, OCALLMETH, OCALLINTER, OCLOSE, OCOPY, ODELETE, OPANIC, OPRINT, OPRINTN, ORECOVER:
		e.call(nil, n, nil)
	case OGO, ODEFER:
		e.stmts(n.Left.Ninit)
		e.call(nil, n.Left, n)

	case ORETJMP:
		// TODO(mdempsky): What do? esc.go just ignores it.
	}
}

func (e *Escape) stmts(l Nodes) {
	for _, n := range l.Slice() {
		e.stmt(n)
	}
}

// block is like stmts, but preserves loopDepth.
func (e *Escape) block(l Nodes) {
	old := e.loopDepth
	e.stmts(l)
	e.loopDepth = old
}

// expr models evaluating an expression n and flowing the result into
// hole k.
func (e *Escape) expr(k EscHole, n *Node) {
	if n == nil {
		return
	}
	e.stmts(n.Ninit)
	e.exprSkipInit(k, n)
}

func (e *Escape) exprSkipInit(k EscHole, n *Node) {
	if n == nil {
		return
	}

	lno := setlineno(n)
	defer func() {
		lineno = lno
	}()

	if k.derefs >= 0 && !types.Haspointers(n.Type) {
		k = e.discardHole()
	}

	switch n.Op {
	default:
		Fatalf("unexpected expr: %v", n)

	case OLITERAL, OGETG, OCLOSUREVAR, OTYPE:
		// nop

	case ONAME:
		if n.Class() == PFUNC || n.Class() == PEXTERN {
			return
		}
		e.flow(k, e.oldLoc(n))

	case OPLUS, ONEG, OBITNOT, ONOT:
		e.discard(n.Left)
	case OADD, OSUB, OOR, OXOR, OMUL, ODIV, OMOD, OLSH, ORSH, OAND, OANDNOT, OEQ, ONE, OLT, OLE, OGT, OGE, OANDAND, OOROR:
		e.discard(n.Left)
		e.discard(n.Right)

	case OADDR:
		e.expr(k.addr(n, "address-of"), n.Left) // "address-of"
	case ODEREF:
		e.expr(k.deref(n, "indirection"), n.Left) // "indirection"
	case ODOT, ODOTMETH, ODOTINTER:
		e.expr(k.note(n, "dot"), n.Left)
	case ODOTPTR:
		e.expr(k.deref(n, "dot of pointer"), n.Left) // "dot of pointer"
	case ODOTTYPE, ODOTTYPE2:
		e.expr(k.dotType(n.Type, n, "dot"), n.Left)
	case OINDEX:
		if n.Left.Type.IsArray() {
			e.expr(k.note(n, "fixed-array-index-of"), n.Left)
		} else {
			// TODO(mdempsky): Fix why reason text.
			e.expr(k.deref(n, "dot of pointer"), n.Left)
		}
		e.discard(n.Right)
	case OINDEXMAP:
		e.discard(n.Left)
		e.discard(n.Right)
	case OSLICE, OSLICEARR, OSLICE3, OSLICE3ARR, OSLICESTR:
		e.expr(k.note(n, "slice"), n.Left)
		low, high, max := n.SliceBounds()
		e.discard(low)
		e.discard(high)
		e.discard(max)

	case OCONV, OCONVNOP:
		if checkPtr(e.curfn, 2) && n.Type.Etype == TUNSAFEPTR && n.Left.Type.IsPtr() {
			// When -d=checkptr=2 is enabled, treat
			// conversions to unsafe.Pointer as an
			// escaping operation. This allows better
			// runtime instrumentation, since we can more
			// easily detect object boundaries on the heap
			// than the stack.
			e.assignHeap(n.Left, "conversion to unsafe.Pointer", n)
		} else if n.Type.Etype == TUNSAFEPTR && n.Left.Type.Etype == TUINTPTR {
			e.unsafeValue(k, n.Left)
		} else {
			e.expr(k, n.Left)
		}
	case OCONVIFACE:
		if !n.Left.Type.IsInterface() && !isdirectiface(n.Left.Type) {
			k = e.spill(k, n)
		}
		e.expr(k.note(n, "interface-converted"), n.Left)

	case ORECV:
		e.discard(n.Left)

	case OCALLMETH, OCALLFUNC, OCALLINTER, OLEN, OCAP, OCOMPLEX, OREAL, OIMAG, OAPPEND, OCOPY:
		e.call([]EscHole{k}, n, nil)

	case ONEW:
		e.spill(k, n)

	case OMAKESLICE:
		e.spill(k, n)
		e.discard(n.Left)
		e.discard(n.Right)
	case OMAKECHAN:
		e.discard(n.Left)
	case OMAKEMAP:
		e.spill(k, n)
		e.discard(n.Left)

	case ORECOVER:
		// nop

	case OCALLPART:
		e.spill(k, n)

		// TODO(mdempsky): We can do better here. See #27557.
		e.assignHeap(n.Left, "call part", n)

	case OPTRLIT:
		e.expr(e.spill(k, n), n.Left)

	case OARRAYLIT:
		for _, elt := range n.List.Slice() {
			if elt.Op == OKEY {
				elt = elt.Right
			}
			e.expr(k.note(n, "array literal element"), elt)
		}

	case OSLICELIT:
		k = e.spill(k, n)

		for _, elt := range n.List.Slice() {
			if elt.Op == OKEY {
				elt = elt.Right
			}
			e.expr(k.note(n, "slice-literal-element"), elt)
		}

	case OSTRUCTLIT:
		for _, elt := range n.List.Slice() {
			e.expr(k.note(n, "struct literal element"), elt.Left)
		}

	case OMAPLIT:
		e.spill(k, n)

		// Map keys and values are always stored in the heap.
		for _, elt := range n.List.Slice() {
			e.assignHeap(elt.Left, "map literal key", n)
			e.assignHeap(elt.Right, "map literal value", n)
		}

	case OCLOSURE:
		k = e.spill(k, n)

		// Link addresses of captured variables to closure.
		for _, v := range n.Func.Closure.Func.Cvars.Slice() {
			if v.Op == OXXX { // unnamed out argument; see dcl.go:/^funcargs
				continue
			}

			k := k
			if !v.Name.Byval() {
				k = k.addr(v, "reference")
			}

			e.expr(k.note(n, "captured by a closure"), v.Name.Defn)
		}

	case ORUNES2STR, OBYTES2STR, OSTR2RUNES, OSTR2BYTES, ORUNESTR:
		e.spill(k, n)
		e.discard(n.Left)

	case OADDSTR:
		e.spill(k, n)

		// Arguments of OADDSTR never escape;
		// runtime.concatstrings makes sure of that.
		e.discards(n.List)
	}
}

// unsafeValue evaluates a uintptr-typed arithmetic expression looking
// for conversions from an unsafe.Pointer.
func (e *Escape) unsafeValue(k EscHole, n *Node) {
	if n.Type.Etype != TUINTPTR {
		Fatalf("unexpected type %v for %v", n.Type, n)
	}

	e.stmts(n.Ninit)

	switch n.Op {
	case OCONV, OCONVNOP:
		if n.Left.Type.Etype == TUNSAFEPTR {
			e.expr(k, n.Left)
		} else {
			e.discard(n.Left)
		}
	case ODOTPTR:
		if isReflectHeaderDataField(n) {
			e.expr(k.deref(n, "reflect.Header.Data"), n.Left)
		} else {
			e.discard(n.Left)
		}
	case OPLUS, ONEG, OBITNOT:
		e.unsafeValue(k, n.Left)
	case OADD, OSUB, OOR, OXOR, OMUL, ODIV, OMOD, OAND, OANDNOT:
		e.unsafeValue(k, n.Left)
		e.unsafeValue(k, n.Right)
	case OLSH, ORSH:
		e.unsafeValue(k, n.Left)
		// RHS need not be uintptr-typed (#32959) and can't meaningfully
		// flow pointers anyway.
		e.discard(n.Right)
	default:
		e.exprSkipInit(e.discardHole(), n)
	}
}

// discard evaluates an expression n for side-effects, but discards
// its value.
func (e *Escape) discard(n *Node) {
	e.expr(e.discardHole(), n)
}

func (e *Escape) discards(l Nodes) {
	for _, n := range l.Slice() {
		e.discard(n)
	}
}

// addr evaluates an addressable expression n and returns an EscHole
// that represents storing into the represented location.
func (e *Escape) addr(n *Node) EscHole {
	if n == nil || n.isBlank() {
		// Can happen at least in OSELRECV.
		// TODO(mdempsky): Anywhere else?
		return e.discardHole()
	}

	k := e.heapHole()

	switch n.Op {
	default:
		Fatalf("unexpected addr: %v", n)
	case ONAME:
		if n.Class() == PEXTERN {
			break
		}
		k = e.oldLoc(n).asHole()
	case ODOT:
		k = e.addr(n.Left)
	case OINDEX:
		e.discard(n.Right)
		if n.Left.Type.IsArray() {
			k = e.addr(n.Left)
		} else {
			e.discard(n.Left)
		}
	case ODEREF, ODOTPTR:
		e.discard(n)
	case OINDEXMAP:
		e.discard(n.Left)
		e.assignHeap(n.Right, "key of map put", n)
	}

	if !types.Haspointers(n.Type) {
		k = e.discardHole()
	}

	return k
}

func (e *Escape) addrs(l Nodes) []EscHole {
	var ks []EscHole
	for _, n := range l.Slice() {
		ks = append(ks, e.addr(n))
	}
	return ks
}

// assign evaluates the assignment dst = src.
func (e *Escape) assign(dst, src *Node, why string, where *Node) {
	// Filter out some no-op assignments for escape analysis.
	ignore := dst != nil && src != nil && isSelfAssign(dst, src)
	if ignore && Debug['m'] != 0 {
		Warnl(where.Pos, "%v ignoring self-assignment in %S", funcSym(e.curfn), where)
	}

	k := e.addr(dst)
	if dst != nil && dst.Op == ODOTPTR && isReflectHeaderDataField(dst) {
		e.unsafeValue(e.heapHole().note(where, why), src)
	} else {
		if ignore {
			k = e.discardHole()
		}
		e.expr(k.note(where, why), src)
	}
}

func (e *Escape) assignHeap(src *Node, why string, where *Node) {
	e.expr(e.heapHole().note(where, why), src)
}

// call evaluates a call expressions, including builtin calls. ks
// should contain the holes representing where the function callee's
// results flows; where is the OGO/ODEFER context of the call, if any.
func (e *Escape) call(ks []EscHole, call, where *Node) {
	// First, pick out the function callee, its type, and receiver
	// (if any) and normal arguments list.
	var fn, recv *Node
	var fntype *types.Type
	args := call.List.Slice()
	switch call.Op {
	case OCALLFUNC:
		fn = call.Left
		if fn.Op == OCLOSURE {
			fn = fn.Func.Closure.Func.Nname
		}
		fntype = fn.Type
	case OCALLMETH:
		fn = asNode(call.Left.Type.FuncType().Nname)
		fntype = fn.Type
		recv = call.Left.Left
	case OCALLINTER:
		fntype = call.Left.Type
		recv = call.Left.Left
	case OAPPEND, ODELETE, OPRINT, OPRINTN, ORECOVER:
		// ok
	case OLEN, OCAP, OREAL, OIMAG, OCLOSE, OPANIC:
		args = []*Node{call.Left}
	case OCOMPLEX, OCOPY:
		args = []*Node{call.Left, call.Right}
	default:
		Fatalf("unexpected call op: %v", call.Op)
	}

	static := fn != nil && fn.Op == ONAME && fn.Class() == PFUNC

	// Setup evaluation holes for each receiver/argument.
	var recvK EscHole
	var paramKs []EscHole

	if static && fn.Name.Defn != nil && fn.Name.Defn.Esc < EscFuncTagged {
		// Static call to function in same mutually recursive
		// group; incorporate into data flow graph.

		if fn.Name.Defn.Esc == EscFuncUnknown {
			Fatalf("graph inconsistency")
		}

		if ks != nil {
			for i, result := range fntype.Results().FieldSlice() {
				e.expr(ks[i], asNode(result.Nname))
			}
		}

		if r := fntype.Recv(); r != nil {
			recvK = e.addr(asNode(r.Nname))
		}
		for _, param := range fntype.Params().FieldSlice() {
			paramKs = append(paramKs, e.addr(asNode(param.Nname)))
		}
	} else if call.Op == OCALLFUNC || call.Op == OCALLMETH || call.Op == OCALLINTER {
		// Dynamic call, or call to previously tagged
		// function. Setup flows to heap and/or ks according
		// to parameter tags.
		if r := fntype.Recv(); r != nil {
			recvK = e.tagHole(ks, r, static)
		}
		for _, param := range fntype.Params().FieldSlice() {
			paramKs = append(paramKs, e.tagHole(ks, param, static))
		}
	} else {
		// Handle escape analysis for builtins.
		// By default, we just discard everything.
		for range args {
			paramKs = append(paramKs, e.discardHole())
		}

		switch call.Op {
		case OAPPEND:
			// Appendee slice may flow directly to the
			// result, if it has enough capacity.
			// Alternatively, a new heap slice might be
			// allocated, and all slice elements might
			// flow to heap.
			paramKs[0] = e.teeHole(paramKs[0], ks[0])
			if types.Haspointers(args[0].Type.Elem()) {
				paramKs[0] = e.teeHole(paramKs[0], e.heapHole().deref(call, "appendee slice"))
			}

			if call.IsDDD() {
				if args[1].Type.IsSlice() && types.Haspointers(args[1].Type.Elem()) {
					paramKs[1] = e.teeHole(paramKs[1], e.heapHole().deref(call, "appended slice..."))
				}
			} else {
				for i := 1; i < len(args); i++ {
					paramKs[i] = e.heapHole()
				}
			}

		case OCOPY:
			if call.Right.Type.IsSlice() && types.Haspointers(call.Right.Type.Elem()) {
				paramKs[1] = e.teeHole(paramKs[1], e.heapHole().deref(call, "copied slice"))
			}

		case OPANIC:
			paramKs[0] = e.heapHole()
		}
	}

	if call.Op == OCALLFUNC {
		// Evaluate callee function expression.
		e.expr(e.augmentParamHole(e.discardHole(), call, where), call.Left)
	}

	if recv != nil {
		// TODO(mdempsky): Handle go:uintptrescapes here too?
		e.expr(e.augmentParamHole(recvK, call, where), recv)
	}

	// Apply augmentParamHole before ODDDARG so that it affects
	// the implicit slice allocation for variadic calls, if any.
	for i, paramK := range paramKs {
		paramKs[i] = e.augmentParamHole(paramK, call, where)
	}

	// TODO(mdempsky): Remove after early ddd-ification.
	if fntype != nil && fntype.IsVariadic() && !call.IsDDD() {
		vi := fntype.NumParams() - 1

		elt := fntype.Params().Field(vi).Type.Elem()
		nva := call.List.Len()
		nva -= vi

		// Introduce ODDDARG node to represent ... allocation.
		ddd := nodl(call.Pos, ODDDARG, nil, nil)
		ddd.Type = types.NewPtr(types.NewArray(elt, int64(nva)))
		call.Right = ddd

		dddK := e.spill(paramKs[vi], ddd)
		paramKs = paramKs[:vi]
		for i := 0; i < nva; i++ {
			paramKs = append(paramKs, dddK)
		}
	}

	for i, arg := range args {
		// For arguments to go:uintptrescapes, peel
		// away an unsafe.Pointer->uintptr conversion,
		// if present.
		if static && arg.Op == OCONVNOP && arg.Type.Etype == TUINTPTR && arg.Left.Type.Etype == TUNSAFEPTR {
			x := i
			if fntype.IsVariadic() && x >= fntype.NumParams() {
				x = fntype.NumParams() - 1
			}
			if fntype.Params().Field(x).Note == uintptrEscapesTag {
				arg = arg.Left
			}
		}

		// no augmentParamHole here; handled in loop before ODDDARG
		e.expr(paramKs[i], arg)
	}
}

// augmentParamHole augments parameter holes as necessary for use in
// go/defer statements.
func (e *Escape) augmentParamHole(k EscHole, call, where *Node) EscHole {
	k = k.note(call, "call parameter")
	if where == nil {
		return k
	}

	// Top level defers arguments don't escape to heap, but they
	// do need to last until end of function. Tee with a
	// non-transient location to avoid arguments from being
	// transiently allocated.
	if where.Op == ODEFER && e.loopDepth == 1 {
		// force stack allocation of defer record, unless open-coded
		// defers are used (see ssa.go)
		where.Esc = EscNever
		return e.later(k)
	}

	return e.heapHole().note(where, "call parameter")
}

// tagHole returns a hole for evaluating an argument passed to param.
// ks should contain the holes representing where the function
// callee's results flows; static indicates whether this is a static
// call.
func (e *Escape) tagHole(ks []EscHole, param *types.Field, static bool) EscHole {
	// If this is a dynamic call, we can't rely on param.Note.
	if !static {
		return e.heapHole()
	}

	var tagKs []EscHole

	esc := ParseLeaks(param.Note)
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

// An EscHole represents a context for evaluation a Go
// expression. E.g., when evaluating p in "x = **p", we'd have a hole
// with dst==x and derefs==2.
type EscHole struct {
	dst    *EscLocation
	derefs int // >= -1
	notes  *EscNote
}

type EscNote struct {
	next  *EscNote
	where *Node
	why   string
}

func (k EscHole) note(where *Node, why string) EscHole {
	if where == nil || why == "" {
		Fatalf("note: missing where/why")
	}
	if Debug['m'] >= 2 {
		k.notes = &EscNote{
			next:  k.notes,
			where: where,
			why:   why,
		}
	}
	return k
}

func (k EscHole) shift(delta int) EscHole {
	k.derefs += delta
	if k.derefs < -1 {
		Fatalf("derefs underflow: %v", k.derefs)
	}
	return k
}

func (k EscHole) deref(where *Node, why string) EscHole { return k.shift(1).note(where, why) }
func (k EscHole) addr(where *Node, why string) EscHole  { return k.shift(-1).note(where, why) }

func (k EscHole) dotType(t *types.Type, where *Node, why string) EscHole {
	if !t.IsInterface() && !isdirectiface(t) {
		k = k.shift(1)
	}
	return k.note(where, why)
}

// teeHole returns a new hole that flows into each hole of ks,
// similar to the Unix tee(1) command.
func (e *Escape) teeHole(ks ...EscHole) EscHole {
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
			Fatalf("teeHole: negative derefs")
		}

		e.flow(k, loc)
	}
	return loc.asHole()
}

func (e *Escape) dcl(n *Node) EscHole {
	loc := e.oldLoc(n)
	loc.loopDepth = e.loopDepth
	return loc.asHole()
}

// spill allocates a new location associated with expression n, flows
// its address to k, and returns a hole that flows values to it. It's
// intended for use with most expressions that allocate storage.
func (e *Escape) spill(k EscHole, n *Node) EscHole {
	loc := e.newLoc(n, true)
	e.flow(k.addr(n, "spill"), loc)
	return loc.asHole()
}

// later returns a new hole that flows into k, but some time later.
// Its main effect is to prevent immediate reuse of temporary
// variables introduced during Order.
func (e *Escape) later(k EscHole) EscHole {
	loc := e.newLoc(nil, false)
	e.flow(k, loc)
	return loc.asHole()
}

// canonicalNode returns the canonical *Node that n logically
// represents.
func canonicalNode(n *Node) *Node {
	if n != nil && n.Op == ONAME && n.Name.IsClosureVar() {
		n = n.Name.Defn
		if n.Name.IsClosureVar() {
			Fatalf("still closure var")
		}
	}

	return n
}

func (e *Escape) newLoc(n *Node, transient bool) *EscLocation {
	if e.curfn == nil {
		Fatalf("e.curfn isn't set")
	}

	n = canonicalNode(n)
	loc := &EscLocation{
		n:         n,
		curfn:     e.curfn,
		loopDepth: e.loopDepth,
		transient: transient,
	}
	e.allLocs = append(e.allLocs, loc)
	if n != nil {
		if n.Op == ONAME && n.Name.Curfn != e.curfn {
			Fatalf("curfn mismatch: %v != %v", n.Name.Curfn, e.curfn)
		}

		if n.HasOpt() {
			Fatalf("%v already has a location", n)
		}
		n.SetOpt(loc)

		if mustHeapAlloc(n) {
			why := "too large for stack"
			if n.Op == OMAKESLICE && (!Isconst(n.Left, CTINT) || !Isconst(n.Right, CTINT)) {
				why = "non-constant size"
			}
			e.flow(e.heapHole().addr(n, why), loc)
		}
	}
	return loc
}

func (e *Escape) oldLoc(n *Node) *EscLocation {
	n = canonicalNode(n)
	return n.Opt().(*EscLocation)
}

func (l *EscLocation) asHole() EscHole {
	return EscHole{dst: l}
}

func (e *Escape) flow(k EscHole, src *EscLocation) {
	dst := k.dst
	if dst == &e.blankLoc {
		return
	}
	if dst == src && k.derefs >= 0 { // dst = dst, dst = *dst, ...
		return
	}
	if dst.escapes && k.derefs < 0 { // dst = &src
		if Debug['m'] >= 2 {
			pos := linestr(src.n.Pos)
			fmt.Printf("%s: %v escapes to heap:\n", pos, src.n)
			e.explainFlow(pos, dst, src, k.derefs, k.notes)
		}
		src.escapes = true
		return
	}

	// TODO(mdempsky): Deduplicate edges?
	dst.edges = append(dst.edges, EscEdge{src: src, derefs: k.derefs, notes: k.notes})
}

func (e *Escape) heapHole() EscHole    { return e.heapLoc.asHole() }
func (e *Escape) discardHole() EscHole { return e.blankLoc.asHole() }

// walkAll computes the minimal dereferences between all pairs of
// locations.
func (e *Escape) walkAll() {
	// We use a work queue to keep track of locations that we need
	// to visit, and repeatedly walk until we reach a fixed point.
	//
	// We walk once from each location (including the heap), and
	// then re-enqueue each location on its transition from
	// transient->!transient and !escapes->escapes, which can each
	// happen at most once. So we take Θ(len(e.allLocs)) walks.

	var todo []*EscLocation // LIFO queue
	enqueue := func(loc *EscLocation) {
		if !loc.queued {
			todo = append(todo, loc)
			loc.queued = true
		}
	}

	for _, loc := range e.allLocs {
		enqueue(loc)
	}
	enqueue(&e.heapLoc)

	var walkgen uint32
	for len(todo) > 0 {
		root := todo[len(todo)-1]
		todo = todo[:len(todo)-1]
		root.queued = false

		walkgen++
		e.walkOne(root, walkgen, enqueue)
	}
}

// walkOne computes the minimal number of dereferences from root to
// all other locations.
func (e *Escape) walkOne(root *EscLocation, walkgen uint32, enqueue func(*EscLocation)) {
	// The data flow graph has negative edges (from addressing
	// operations), so we use the Bellman-Ford algorithm. However,
	// we don't have to worry about infinite negative cycles since
	// we bound intermediate dereference counts to 0.

	root.walkgen = walkgen
	root.derefs = 0
	root.dst = nil

	todo := []*EscLocation{root} // LIFO queue
	for len(todo) > 0 {
		l := todo[len(todo)-1]
		todo = todo[:len(todo)-1]

		base := l.derefs

		// If l.derefs < 0, then l's address flows to root.
		addressOf := base < 0
		if addressOf {
			// For a flow path like "root = &l; l = x",
			// l's address flows to root, but x's does
			// not. We recognize this by lower bounding
			// base at 0.
			base = 0

			// If l's address flows to a non-transient
			// location, then l can't be transiently
			// allocated.
			if !root.transient && l.transient {
				l.transient = false
				enqueue(l)
			}
		}

		if e.outlives(root, l) {
			// l's value flows to root. If l is a function
			// parameter and root is the heap or a
			// corresponding result parameter, then record
			// that value flow for tagging the function
			// later.
			if l.isName(PPARAM) {
				if Debug['m'] >= 2 && !l.escapes {
					fmt.Printf("%s: parameter %v leaks to %s with derefs=%d:\n", linestr(l.n.Pos), l.n, e.explainLoc(root), base)
					e.explainPath(root, l)
				}
				l.leakTo(root, base)
			}

			// If l's address flows somewhere that
			// outlives it, then l needs to be heap
			// allocated.
			if addressOf && !l.escapes {
				if Debug['m'] >= 2 {
					fmt.Printf("%s: %v escapes to heap:\n", linestr(l.n.Pos), l.n)
					e.explainPath(root, l)
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
			derefs := base + edge.derefs
			if edge.src.walkgen != walkgen || edge.src.derefs > derefs {
				edge.src.walkgen = walkgen
				edge.src.derefs = derefs
				edge.src.dst = l
				edge.src.dstEdgeIdx = i
				todo = append(todo, edge.src)
			}
		}
	}
}

// explainPath prints an explanation of how src flows to the walk root.
func (e *Escape) explainPath(root, src *EscLocation) {
	visited := make(map[*EscLocation]bool)

	pos := linestr(src.n.Pos)
	for {
		// Prevent infinite loop.
		if visited[src] {
			fmt.Printf("%s:   warning: truncated explanation due to assignment cycle; see golang.org/issue/35518\n", pos)
			break
		}
		visited[src] = true

		dst := src.dst
		edge := &dst.edges[src.dstEdgeIdx]
		if edge.src != src {
			Fatalf("path inconsistency: %v != %v", edge.src, src)
		}

		e.explainFlow(pos, dst, src, edge.derefs, edge.notes)

		if dst == root {
			break
		}
		src = dst
	}
}

func (e *Escape) explainFlow(pos string, dst, src *EscLocation, derefs int, notes *EscNote) {
	ops := "&"
	if derefs >= 0 {
		ops = strings.Repeat("*", derefs)
	}

	fmt.Printf("%s:   flow: %s = %s%v:\n", pos, e.explainLoc(dst), ops, e.explainLoc(src))
	for note := notes; note != nil; note = note.next {
		fmt.Printf("%s:     from %v (%v) at %s\n", pos, note.where, note.why, linestr(note.where.Pos))
	}
}

func (e *Escape) explainLoc(l *EscLocation) string {
	if l == &e.heapLoc {
		return "{heap}"
	}
	if l.n == nil {
		// TODO(mdempsky): Omit entirely.
		return "{temp}"
	}
	if l.n.Op == ONAME {
		return fmt.Sprintf("%v", l.n)
	}
	return fmt.Sprintf("{storage for %v}", l.n)
}

// outlives reports whether values stored in l may survive beyond
// other's lifetime if stack allocated.
func (e *Escape) outlives(l, other *EscLocation) bool {
	// The heap outlives everything.
	if l.escapes {
		return true
	}

	// We don't know what callers do with returned values, so
	// pessimistically we need to assume they flow to the heap and
	// outlive everything too.
	if l.isName(PPARAMOUT) {
		// Exception: Directly called closures can return
		// locations allocated outside of them without forcing
		// them to the heap. For example:
		//
		//    var u int  // okay to stack allocate
		//    *(func() *int { return &u }()) = 42
		if containsClosure(other.curfn, l.curfn) && l.curfn.Func.Closure.Func.Top&ctxCallee != 0 {
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
func containsClosure(f, c *Node) bool {
	if f.Op != ODCLFUNC || c.Op != ODCLFUNC {
		Fatalf("bad containsClosure: %v, %v", f, c)
	}

	// Common case.
	if f == c {
		return false
	}

	// Closures within function Foo are named like "Foo.funcN..."
	// TODO(mdempsky): Better way to recognize this.
	fn := f.Func.Nname.Sym.Name
	cn := c.Func.Nname.Sym.Name
	return len(cn) > len(fn) && cn[:len(fn)] == fn && cn[len(fn)] == '.'
}

// leak records that parameter l leaks to sink.
func (l *EscLocation) leakTo(sink *EscLocation, derefs int) {
	// If sink is a result parameter and we can fit return bits
	// into the escape analysis tag, then record a return leak.
	if sink.isName(PPARAMOUT) && sink.curfn == l.curfn {
		// TODO(mdempsky): Eliminate dependency on Vargen here.
		ri := int(sink.n.Name.Vargen) - 1
		if ri < numEscResults {
			// Leak to result parameter.
			l.paramEsc.AddResult(ri, derefs)
			return
		}
	}

	// Otherwise, record as heap leak.
	l.paramEsc.AddHeap(derefs)
}

func (e *Escape) finish(fns []*Node) {
	// Record parameter tags for package export data.
	for _, fn := range fns {
		fn.Esc = EscFuncTagged

		narg := 0
		for _, fs := range &types.RecvsParams {
			for _, f := range fs(fn.Type).Fields().Slice() {
				narg++
				f.Note = e.paramTag(fn, narg, f)
			}
		}
	}

	for _, loc := range e.allLocs {
		n := loc.n
		if n == nil {
			continue
		}
		n.SetOpt(nil)

		// Update n.Esc based on escape analysis results.

		if loc.escapes {
			if n.Op != ONAME {
				if Debug['m'] != 0 {
					Warnl(n.Pos, "%S escapes to heap", n)
				}
				if logopt.Enabled() {
					logopt.LogOpt(n.Pos, "escape", "escape", e.curfn.funcname())
				}
			}
			n.Esc = EscHeap
			addrescapes(n)
		} else {
			if Debug['m'] != 0 && n.Op != ONAME {
				Warnl(n.Pos, "%S does not escape", n)
			}
			n.Esc = EscNone
			if loc.transient {
				n.SetTransient(true)
			}
		}
	}
}

func (l *EscLocation) isName(c Class) bool {
	return l.n != nil && l.n.Op == ONAME && l.n.Class() == c
}

const numEscResults = 7

// An EscLeaks represents a set of assignment flows from a parameter
// to the heap or to any of its function's (first numEscResults)
// result parameters.
type EscLeaks [1 + numEscResults]uint8

// Empty reports whether l is an empty set (i.e., no assignment flows).
func (l EscLeaks) Empty() bool { return l == EscLeaks{} }

// Heap returns the minimum deref count of any assignment flow from l
// to the heap. If no such flows exist, Heap returns -1.
func (l EscLeaks) Heap() int { return l.get(0) }

// Result returns the minimum deref count of any assignment flow from
// l to its function's i'th result parameter. If no such flows exist,
// Result returns -1.
func (l EscLeaks) Result(i int) int { return l.get(1 + i) }

// AddHeap adds an assignment flow from l to the heap.
func (l *EscLeaks) AddHeap(derefs int) { l.add(0, derefs) }

// AddResult adds an assignment flow from l to its function's i'th
// result parameter.
func (l *EscLeaks) AddResult(i, derefs int) { l.add(1+i, derefs) }

func (l *EscLeaks) setResult(i, derefs int) { l.set(1+i, derefs) }

func (l EscLeaks) get(i int) int { return int(l[i]) - 1 }

func (l *EscLeaks) add(i, derefs int) {
	if old := l.get(i); old < 0 || derefs < old {
		l.set(i, derefs)
	}
}

func (l *EscLeaks) set(i, derefs int) {
	v := derefs + 1
	if v < 0 {
		Fatalf("invalid derefs count: %v", derefs)
	}
	if v > math.MaxUint8 {
		v = math.MaxUint8
	}

	l[i] = uint8(v)
}

// Optimize removes result flow paths that are equal in length or
// longer than the shortest heap flow path.
func (l *EscLeaks) Optimize() {
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

var leakTagCache = map[EscLeaks]string{}

// Encode converts l into a binary string for export data.
func (l EscLeaks) Encode() string {
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

// ParseLeaks parses a binary string representing an EscLeaks.
func ParseLeaks(s string) EscLeaks {
	var l EscLeaks
	if !strings.HasPrefix(s, "esc:") {
		l.AddHeap(0)
		return l
	}
	copy(l[:], s[4:])
	return l
}
