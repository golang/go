// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"fmt"
	"strconv"
	"strings"
)

// Run analysis on minimal sets of mutually recursive functions
// or single non-recursive functions, bottom up.
//
// Finding these sets is finding strongly connected components
// in the static call graph. The algorithm for doing that is taken
// from Sedgewick, Algorithms, Second Edition, p. 482, with two
// adaptations.
//
// First, a hidden closure function (n->curfn != N) cannot be the
// root of a connected component. Refusing to use it as a root
// forces it into the component of the function in which it appears.
// This is more convenient for escape analysis.
//
// Second, each function becomes two virtual nodes in the graph,
// with numbers n and n+1. We record the function's node number as n
// but search from node n+1. If the search tells us that the component
// number (min) is n+1, we know that this is a trivial component: one function
// plus its closures. If the search tells us that the component number is
// n, then there was a path from node n+1 back to node n, meaning that
// the function set is mutually recursive. The escape analysis can be
// more precise when analyzing a single non-recursive function than
// when analyzing a set of mutually recursive functions.

type bottomUpVisitor struct {
	analyze  func([]*Node, bool)
	visitgen uint32
	nodeID   map[*Node]uint32
	stack    []*Node
}

// visitBottomUp invokes analyze on the ODCLFUNC nodes listed in list.
// It calls analyze with successive groups of functions, working from
// the bottom of the call graph upward. Each time analyze is called with
// a list of functions, every function on that list only calls other functions
// on the list or functions that have been passed in previous invocations of
// analyze. Closures appear in the same list as their outer functions.
// The lists are as short as possible while preserving those requirements.
// (In a typical program, many invocations of analyze will be passed just
// a single function.) The boolean argument 'recursive' passed to analyze
// specifies whether the functions on the list are mutually recursive.
// If recursive is false, the list consists of only a single function and its closures.
// If recursive is true, the list may still contain only a single function,
// if that function is itself recursive.
func visitBottomUp(list []*Node, analyze func(list []*Node, recursive bool)) {
	var v bottomUpVisitor
	v.analyze = analyze
	v.nodeID = make(map[*Node]uint32)
	for _, n := range list {
		if n.Op == ODCLFUNC && n.Func.FCurfn == nil {
			v.visit(n)
		}
	}
}

func (v *bottomUpVisitor) visit(n *Node) uint32 {
	if id := v.nodeID[n]; id > 0 {
		// already visited
		return id
	}

	v.visitgen++
	id := v.visitgen
	v.nodeID[n] = id
	v.visitgen++
	min := v.visitgen

	v.stack = append(v.stack, n)
	min = v.visitcodelist(n.Nbody, min)
	if (min == id || min == id+1) && n.Func.FCurfn == nil {
		// This node is the root of a strongly connected component.

		// The original min passed to visitcodelist was n->walkgen+1.
		// If visitcodelist found its way back to n->walkgen, then this
		// block is a set of mutually recursive functions.
		// Otherwise it's just a lone function that does not recurse.
		recursive := min == id

		// Remove connected component from stack.
		// Mark walkgen so that future visits return a large number
		// so as not to affect the caller's min.

		var i int
		for i = len(v.stack) - 1; i >= 0; i-- {
			x := v.stack[i]
			if x == n {
				break
			}
			v.nodeID[x] = ^uint32(0)
		}
		v.nodeID[n] = ^uint32(0)
		block := v.stack[i:]
		// Run escape analysis on this set of functions.
		v.stack = v.stack[:i]
		v.analyze(block, recursive)
	}

	return min
}

func (v *bottomUpVisitor) visitcodelist(l Nodes, min uint32) uint32 {
	for _, n := range l.Slice() {
		min = v.visitcode(n, min)
	}
	return min
}

func (v *bottomUpVisitor) visitcode(n *Node, min uint32) uint32 {
	if n == nil {
		return min
	}

	min = v.visitcodelist(n.Ninit, min)
	min = v.visitcode(n.Left, min)
	min = v.visitcode(n.Right, min)
	min = v.visitcodelist(n.List, min)
	min = v.visitcodelist(n.Nbody, min)
	min = v.visitcodelist(n.Rlist, min)

	if n.Op == OCALLFUNC || n.Op == OCALLMETH {
		fn := n.Left
		if n.Op == OCALLMETH {
			fn = n.Left.Sym.Def
		}
		if fn != nil && fn.Op == ONAME && fn.Class == PFUNC && fn.Name.Defn != nil {
			m := v.visit(fn.Name.Defn)
			if m < min {
				min = m
			}
		}
	}

	if n.Op == OCLOSURE {
		m := v.visit(n.Func.Closure)
		if m < min {
			min = m
		}
	}

	return min
}

// Escape analysis.

// An escape analysis pass for a set of functions.
// The analysis assumes that closures and the functions in which they
// appear are analyzed together, so that the aliasing between their
// variables can be modeled more precisely.
//
// First escfunc, esc and escassign recurse over the ast of each
// function to dig out flow(dst,src) edges between any
// pointer-containing nodes and store them in dst->escflowsrc. For
// variables assigned to a variable in an outer scope or used as a
// return value, they store a flow(theSink, src) edge to a fake node
// 'the Sink'.  For variables referenced in closures, an edge
// flow(closure, &var) is recorded and the flow of a closure itself to
// an outer scope is tracked the same way as other variables.
//
// Then escflood walks the graph starting at theSink and tags all
// variables of it can reach an & node as escaping and all function
// parameters it can reach as leaking.
//
// If a value's address is taken but the address does not escape,
// then the value can stay on the stack. If the value new(T) does
// not escape, then new(T) can be rewritten into a stack allocation.
// The same is true of slice literals.
//
// If optimizations are disabled (-N), this code is not used.
// Instead, the compiler assumes that any value whose address
// is taken without being immediately dereferenced
// needs to be moved to the heap, and new(T) and slice
// literals are always real allocations.

func escapes(all []*Node) {
	visitBottomUp(all, escAnalyze)
}

const (
	EscFuncUnknown = 0 + iota
	EscFuncPlanned
	EscFuncStarted
	EscFuncTagged
)

// There appear to be some loops in the escape graph, causing
// arbitrary recursion into deeper and deeper levels.
// Cut this off safely by making minLevel sticky: once you
// get that deep, you cannot go down any further but you also
// cannot go up any further. This is a conservative fix.
// Making minLevel smaller (more negative) would handle more
// complex chains of indirections followed by address-of operations,
// at the cost of repeating the traversal once for each additional
// allowed level when a loop is encountered. Using -2 suffices to
// pass all the tests we have written so far, which we assume matches
// the level of complexity we want the escape analysis code to handle.
const (
	MinLevel = -2
)

// A Level encodes the reference state and context applied to
// (stack, heap) allocated memory.
//
// value is the overall sum of *(1) and &(-1) operations encountered
// along a path from a destination (sink, return value) to a source
// (allocation, parameter).
//
// suffixValue is the maximum-copy-started-suffix-level applied to a sink.
// For example:
// sink = x.left.left --> level=2, x is dereferenced twice and does not escape to sink.
// sink = &Node{x} --> level=-1, x is accessible from sink via one "address of"
// sink = &Node{&Node{x}} --> level=-2, x is accessible from sink via two "address of"
// sink = &Node{&Node{x.left}} --> level=-1, but x is NOT accessible from sink because it was indirected and then copied.
// (The copy operations are sometimes implicit in the source code; in this case,
// value of x.left was copied into a field of a newly allocated Node)
//
// There's one of these for each Node, and the integer values
// rarely exceed even what can be stored in 4 bits, never mind 8.
type Level struct {
	value, suffixValue int8
}

func (l Level) int() int {
	return int(l.value)
}

func levelFrom(i int) Level {
	if i <= MinLevel {
		return Level{value: MinLevel}
	}
	return Level{value: int8(i)}
}

func satInc8(x int8) int8 {
	if x == 127 {
		return 127
	}
	return x + 1
}

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

// inc returns the level l + 1, representing the effect of an indirect (*) operation.
func (l Level) inc() Level {
	if l.value <= MinLevel {
		return Level{value: MinLevel}
	}
	return Level{value: satInc8(l.value), suffixValue: satInc8(l.suffixValue)}
}

// dec returns the level l - 1, representing the effect of an address-of (&) operation.
func (l Level) dec() Level {
	if l.value <= MinLevel {
		return Level{value: MinLevel}
	}
	return Level{value: l.value - 1, suffixValue: l.suffixValue - 1}
}

// copy returns the level for a copy of a value with level l.
func (l Level) copy() Level {
	return Level{value: l.value, suffixValue: max8(l.suffixValue, 0)}
}

func (l1 Level) min(l2 Level) Level {
	return Level{
		value:       min8(l1.value, l2.value),
		suffixValue: min8(l1.suffixValue, l2.suffixValue)}
}

// guaranteedDereference returns the number of dereferences
// applied to a pointer before addresses are taken/generated.
// This is the maximum level computed from path suffixes starting
// with copies where paths flow from destination to source.
func (l Level) guaranteedDereference() int {
	return int(l.suffixValue)
}

// An EscStep documents one step in the path from memory
// that is heap allocated to the (alleged) reason for the
// heap allocation.
type EscStep struct {
	src, dst *Node    // the endpoints of this edge in the escape-to-heap chain.
	parent   *EscStep // used in flood to record path
	why      string   // explanation for this step in the escape-to-heap chain
	busy     bool     // used in prevent to snip cycles.
}

type NodeEscState struct {
	Curfn             *Node
	Escflowsrc        []EscStep // flow(this, src)
	Escretval         Nodes     // on OCALLxxx, list of dummy return values
	Escloopdepth      int32     // -1: global, 0: return variables, 1:function top level, increased inside function for every loop or label to mark scopes
	Esclevel          Level
	Walkgen           uint32
	Maxextraloopdepth int32
}

func (e *EscState) nodeEscState(n *Node) *NodeEscState {
	if nE, ok := n.Opt().(*NodeEscState); ok {
		return nE
	}
	if n.Opt() != nil {
		Fatalf("nodeEscState: opt in use (%T)", n.Opt())
	}
	nE := new(NodeEscState)
	nE.Curfn = Curfn
	n.SetOpt(nE)
	e.opts = append(e.opts, n)
	return nE
}

func (e *EscState) track(n *Node) {
	if Curfn == nil {
		Fatalf("EscState.track: Curfn nil")
	}
	n.Esc = EscNone // until proven otherwise
	nE := e.nodeEscState(n)
	nE.Escloopdepth = e.loopdepth
	e.noesc = append(e.noesc, n)
}

// Escape constants are numbered in order of increasing "escapiness"
// to help make inferences be monotonic. With the exception of
// EscNever which is sticky, eX < eY means that eY is more exposed
// than eX, and hence replaces it in a conservative analysis.
const (
	EscUnknown = iota
	EscNone    // Does not escape to heap, result, or parameters.
	EscReturn  // Is returned or reachable from returned.
	EscScope   // Allocated in an inner loop scope, assigned to an outer loop scope,
	// which allows the construction of non-escaping but arbitrarily large linked
	// data structures (i.e., not eligible for allocation in a fixed-size stack frame).
	EscHeap           // Reachable from the heap
	EscNever          // By construction will not escape.
	EscBits           = 3
	EscMask           = (1 << EscBits) - 1
	EscContentEscapes = 1 << EscBits // value obtained by indirect of parameter escapes to heap
	EscReturnBits     = EscBits + 1
	// Node.esc encoding = | escapeReturnEncoding:(width-4) | contentEscapes:1 | escEnum:3
)

// escMax returns the maximum of an existing escape value
// (and its additional parameter flow flags) and a new escape type.
func escMax(e, etype uint16) uint16 {
	if e&EscMask >= EscScope {
		// normalize
		if e&^EscMask != 0 {
			Fatalf("Escape information had unexpected return encoding bits (w/ EscScope, EscHeap, EscNever), e&EscMask=%v", e&EscMask)
		}
	}
	if e&EscMask > etype {
		return e
	}
	if etype == EscNone || etype == EscReturn {
		return (e &^ EscMask) | etype
	}
	return etype
}

// For each input parameter to a function, the escapeReturnEncoding describes
// how the parameter may leak to the function's outputs. This is currently the
// "level" of the leak where level is 0 or larger (negative level means stored into
// something whose address is returned -- but that implies stored into the heap,
// hence EscHeap, which means that the details are not currently relevant. )
const (
	bitsPerOutputInTag = 3                                 // For each output, the number of bits for a tag
	bitsMaskForTag     = uint16(1<<bitsPerOutputInTag) - 1 // The bit mask to extract a single tag.
	maxEncodedLevel    = int(bitsMaskForTag - 1)           // The largest level that can be stored in a tag.
)

type EscState struct {
	// Fake node that all
	//   - return values and output variables
	//   - parameters on imported functions not marked 'safe'
	//   - assignments to global variables
	// flow to.
	theSink Node

	dsts      []*Node // all dst nodes
	loopdepth int32   // for detecting nested loop scopes
	pdepth    int     // for debug printing in recursions.
	dstcount  int     // diagnostic
	edgecount int     // diagnostic
	noesc     []*Node // list of possible non-escaping nodes, for printing
	recursive bool    // recursive function or group of mutually recursive functions.
	opts      []*Node // nodes with .Opt initialized
	walkgen   uint32
}

func (e *EscState) stepWalk(dst, src *Node, why string, parent *EscStep) *EscStep {
	// TODO: keep a cache of these, mark entry/exit in escwalk to avoid allocation
	// Or perhaps never mind, since it is disabled unless printing is on.
	// We may want to revisit this, since the EscStep nodes would make
	// an excellent replacement for the poorly-separated graph-build/graph-flood
	// stages.
	if Debug['m'] == 0 {
		return nil
	}
	return &EscStep{src: src, dst: dst, why: why, parent: parent}
}

func (e *EscState) stepAssign(step *EscStep, dst, src *Node, why string) *EscStep {
	if Debug['m'] == 0 {
		return nil
	}
	if step != nil { // Caller may have known better.
		return step
	}
	return &EscStep{src: src, dst: dst, why: why}
}

// funcSym returns fn.Func.Nname.Sym if no nils are encountered along the way.
func funcSym(fn *Node) *Sym {
	if fn == nil || fn.Func.Nname == nil {
		return nil
	}
	return fn.Func.Nname.Sym
}

// curfnSym returns n.Curfn.Nname.Sym if no nils are encountered along the way.
func (e *EscState) curfnSym(n *Node) *Sym {
	nE := e.nodeEscState(n)
	return funcSym(nE.Curfn)
}

func escAnalyze(all []*Node, recursive bool) {
	var es EscState
	e := &es
	e.theSink.Op = ONAME
	e.theSink.Orig = &e.theSink
	e.theSink.Class = PEXTERN
	e.theSink.Sym = Lookup(".sink")
	e.nodeEscState(&e.theSink).Escloopdepth = -1
	e.recursive = recursive

	for _, n := range all {
		if n.Op == ODCLFUNC {
			n.Esc = EscFuncPlanned
		}
	}

	// flow-analyze functions
	for _, n := range all {
		if n.Op == ODCLFUNC {
			escfunc(e, n)
		}
	}

	// print("escapes: %d e->dsts, %d edges\n", e->dstcount, e->edgecount);

	// visit the upstream of each dst, mark address nodes with
	// addrescapes, mark parameters unsafe
	for _, n := range e.dsts {
		escflood(e, n)
	}

	// for all top level functions, tag the typenodes corresponding to the param nodes
	for _, n := range all {
		if n.Op == ODCLFUNC {
			esctag(e, n)
		}
	}

	if Debug['m'] != 0 {
		for _, n := range e.noesc {
			if n.Esc == EscNone {
				Warnl(n.Lineno, "%v %v does not escape", e.curfnSym(n), Nconv(n, FmtShort))
			}
		}
	}
	for _, x := range e.opts {
		x.SetOpt(nil)
	}
}

func escfunc(e *EscState, func_ *Node) {
	//	print("escfunc %N %s\n", func->nname, e->recursive?"(recursive)":"");
	if func_.Esc != 1 {
		Fatalf("repeat escfunc %v", func_.Func.Nname)
	}
	func_.Esc = EscFuncStarted

	saveld := e.loopdepth
	e.loopdepth = 1
	savefn := Curfn
	Curfn = func_

	for _, ln := range Curfn.Func.Dcl {
		if ln.Op != ONAME {
			continue
		}
		llNE := e.nodeEscState(ln)
		switch ln.Class {
		// out params are in a loopdepth between the sink and all local variables
		case PPARAMOUT:
			llNE.Escloopdepth = 0

		case PPARAM:
			llNE.Escloopdepth = 1
			if ln.Type != nil && !haspointers(ln.Type) {
				break
			}
			if Curfn.Nbody.Len() == 0 && !Curfn.Noescape {
				ln.Esc = EscHeap
			} else {
				ln.Esc = EscNone // prime for escflood later
			}
			e.noesc = append(e.noesc, ln)
		}
	}

	// in a mutually recursive group we lose track of the return values
	if e.recursive {
		for _, ln := range Curfn.Func.Dcl {
			if ln.Op == ONAME && ln.Class == PPARAMOUT {
				escflows(e, &e.theSink, ln, e.stepAssign(nil, ln, ln, "returned from recursive function"))
			}
		}
	}

	escloopdepthlist(e, Curfn.Nbody)
	esclist(e, Curfn.Nbody, Curfn)
	Curfn = savefn
	e.loopdepth = saveld
}

// Mark labels that have no backjumps to them as not increasing e->loopdepth.
// Walk hasn't generated (goto|label)->left->sym->label yet, so we'll cheat
// and set it to one of the following two. Then in esc we'll clear it again.
var looping Label

var nonlooping Label

func escloopdepthlist(e *EscState, l Nodes) {
	for _, n := range l.Slice() {
		escloopdepth(e, n)
	}
}

func escloopdepth(e *EscState, n *Node) {
	if n == nil {
		return
	}

	escloopdepthlist(e, n.Ninit)

	switch n.Op {
	case OLABEL:
		if n.Left == nil || n.Left.Sym == nil {
			Fatalf("esc:label without label: %v", Nconv(n, FmtSign))
		}

		// Walk will complain about this label being already defined, but that's not until
		// after escape analysis. in the future, maybe pull label & goto analysis out of walk and put before esc
		// if(n->left->sym->label != nil)
		//	fatal("escape analysis messed up analyzing label: %+N", n);
		n.Left.Sym.Label = &nonlooping

	case OGOTO:
		if n.Left == nil || n.Left.Sym == nil {
			Fatalf("esc:goto without label: %v", Nconv(n, FmtSign))
		}

		// If we come past one that's uninitialized, this must be a (harmless) forward jump
		// but if it's set to nonlooping the label must have preceded this goto.
		if n.Left.Sym.Label == &nonlooping {
			n.Left.Sym.Label = &looping
		}
	}

	escloopdepth(e, n.Left)
	escloopdepth(e, n.Right)
	escloopdepthlist(e, n.List)
	escloopdepthlist(e, n.Nbody)
	escloopdepthlist(e, n.Rlist)
}

func esclist(e *EscState, l Nodes, up *Node) {
	for _, n := range l.Slice() {
		esc(e, n, up)
	}
}

func esc(e *EscState, n *Node, up *Node) {
	if n == nil {
		return
	}
	if n.Type == structkey {
		// This is the left side of x:y in a struct literal.
		// x is syntax, not an expression.
		// See #14405.
		return
	}

	lno := setlineno(n)

	// ninit logically runs at a different loopdepth than the rest of the for loop.
	esclist(e, n.Ninit, n)

	if n.Op == OFOR || n.Op == ORANGE {
		e.loopdepth++
	}

	// type switch variables have no ODCL.
	// process type switch as declaration.
	// must happen before processing of switch body,
	// so before recursion.
	if n.Op == OSWITCH && n.Left != nil && n.Left.Op == OTYPESW {
		for _, n1 := range n.List.Slice() { // cases
			// it.N().Rlist is the variable per case
			if n1.Rlist.Len() != 0 {
				e.nodeEscState(n1.Rlist.First()).Escloopdepth = e.loopdepth
			}
		}
	}

	// Big stuff escapes unconditionally
	// "Big" conditions that were scattered around in walk have been gathered here
	if n.Esc != EscHeap && n.Type != nil &&
		(n.Type.Width > MaxStackVarSize ||
			(n.Op == ONEW || n.Op == OPTRLIT) && n.Type.Elem().Width >= 1<<16 ||
			n.Op == OMAKESLICE && !isSmallMakeSlice(n)) {
		if Debug['m'] > 2 {
			Warnl(n.Lineno, "%v is too large for stack", n)
		}
		n.Esc = EscHeap
		addrescapes(n)
		escassignSinkNilWhy(e, n, n, "too large for stack") // TODO category: tooLarge
	}

	esc(e, n.Left, n)
	esc(e, n.Right, n)
	esclist(e, n.Nbody, n)
	esclist(e, n.List, n)
	esclist(e, n.Rlist, n)

	if n.Op == OFOR || n.Op == ORANGE {
		e.loopdepth--
	}

	if Debug['m'] > 2 {
		fmt.Printf("%v:[%d] %v esc: %v\n", linestr(lineno), e.loopdepth, funcSym(Curfn), n)
	}

	switch n.Op {
	// Record loop depth at declaration.
	case ODCL:
		if n.Left != nil {
			e.nodeEscState(n.Left).Escloopdepth = e.loopdepth
		}

	case OLABEL:
		if n.Left.Sym.Label == &nonlooping {
			if Debug['m'] > 2 {
				fmt.Printf("%v:%v non-looping label\n", linestr(lineno), n)
			}
		} else if n.Left.Sym.Label == &looping {
			if Debug['m'] > 2 {
				fmt.Printf("%v: %v looping label\n", linestr(lineno), n)
			}
			e.loopdepth++
		}

		// See case OLABEL in escloopdepth above
		// else if(n->left->sym->label == nil)
		//	fatal("escape analysis missed or messed up a label: %+N", n);

		n.Left.Sym.Label = nil

	case ORANGE:
		if n.List.Len() >= 2 {
			// Everything but fixed array is a dereference.

			// If fixed array is really the address of fixed array,
			// it is also a dereference, because it is implicitly
			// dereferenced (see #12588)
			if n.Type.IsArray() &&
				!(n.Right.Type.IsPtr() && Eqtype(n.Right.Type.Elem(), n.Type)) {
				escassignNilWhy(e, n.List.Second(), n.Right, "range")
			} else {
				escassignDereference(e, n.List.Second(), n.Right, e.stepAssign(nil, n.List.Second(), n.Right, "range-deref"))
			}
		}

	case OSWITCH:
		if n.Left != nil && n.Left.Op == OTYPESW {
			for _, n2 := range n.List.Slice() {
				// cases
				// n.Left.Right is the argument of the .(type),
				// it.N().Rlist is the variable per case
				if n2.Rlist.Len() != 0 {
					escassignNilWhy(e, n2.Rlist.First(), n.Left.Right, "switch case")
				}
			}
		}

		// Filter out the following special case.
	//
	//	func (b *Buffer) Foo() {
	//		n, m := ...
	//		b.buf = b.buf[n:m]
	//	}
	//
	// This assignment is a no-op for escape analysis,
	// it does not store any new pointers into b that were not already there.
	// However, without this special case b will escape, because we assign to OIND/ODOTPTR.
	case OAS, OASOP, OASWB:
		if (n.Left.Op == OIND || n.Left.Op == ODOTPTR) && n.Left.Left.Op == ONAME && // dst is ONAME dereference
			(n.Right.Op == OSLICE || n.Right.Op == OSLICE3 || n.Right.Op == OSLICESTR) && // src is slice operation
			(n.Right.Left.Op == OIND || n.Right.Left.Op == ODOTPTR) && n.Right.Left.Left.Op == ONAME && // slice is applied to ONAME dereference
			n.Left.Left == n.Right.Left.Left { // dst and src reference the same base ONAME

			// Here we also assume that the statement will not contain calls,
			// that is, that order will move any calls to init.
			// Otherwise base ONAME value could change between the moments
			// when we evaluate it for dst and for src.
			//
			// Note, this optimization does not apply to OSLICEARR,
			// because it does introduce a new pointer into b that was not already there
			// (pointer to b itself). After such assignment, if b contents escape,
			// b escapes as well. If we ignore such OSLICEARR, we will conclude
			// that b does not escape when b contents do.
			if Debug['m'] != 0 {
				Warnl(n.Lineno, "%v ignoring self-assignment to %v", e.curfnSym(n), Nconv(n.Left, FmtShort))
			}

			break
		}

		escassign(e, n.Left, n.Right, nil)

	case OAS2: // x,y = a,b
		if n.List.Len() == n.Rlist.Len() {
			rs := n.Rlist.Slice()
			for i, n := range n.List.Slice() {
				escassignNilWhy(e, n, rs[i], "assign-pair")
			}
		}

	case OAS2RECV: // v, ok = <-ch
		escassignNilWhy(e, n.List.First(), n.Rlist.First(), "assign-pair-receive")
	case OAS2MAPR: // v, ok = m[k]
		escassignNilWhy(e, n.List.First(), n.Rlist.First(), "assign-pair-mapr")
	case OAS2DOTTYPE: // v, ok = x.(type)
		escassignNilWhy(e, n.List.First(), n.Rlist.First(), "assign-pair-dot-type")

	case OSEND: // ch <- x
		escassignSinkNilWhy(e, n, n.Right, "send")

	case ODEFER:
		if e.loopdepth == 1 { // top level
			break
		}
		// arguments leak out of scope
		// TODO: leak to a dummy node instead
		// defer f(x) - f and x escape
		escassignSinkNilWhy(e, n, n.Left.Left, "defer func")

		escassignSinkNilWhy(e, n, n.Left.Right, "defer func ...") // ODDDARG for call
		for _, n4 := range n.Left.List.Slice() {
			escassignSinkNilWhy(e, n, n4, "defer func arg")
		}

	case OPROC:
		// go f(x) - f and x escape
		escassignSinkNilWhy(e, n, n.Left.Left, "go func")

		escassignSinkNilWhy(e, n, n.Left.Right, "go func ...") // ODDDARG for call
		for _, n4 := range n.Left.List.Slice() {
			escassignSinkNilWhy(e, n, n4, "go func arg")
		}

	case OCALLMETH, OCALLFUNC, OCALLINTER:
		esccall(e, n, up)

		// esccall already done on n->rlist->n. tie it's escretval to n->list
	case OAS2FUNC: // x,y = f()
		rs := e.nodeEscState(n.Rlist.First()).Escretval.Slice()
		for i, n := range n.List.Slice() {
			if i >= len(rs) {
				break
			}
			escassignNilWhy(e, n, rs[i], "assign-pair-func-call")
		}
		if n.List.Len() != len(rs) {
			Fatalf("esc oas2func")
		}

	case ORETURN:
		ll := n.List
		if n.List.Len() == 1 && Curfn.Type.Results().NumFields() > 1 {
			// OAS2FUNC in disguise
			// esccall already done on n->list->n
			// tie n->list->n->escretval to curfn->dcl PPARAMOUT's
			ll = e.nodeEscState(n.List.First()).Escretval
		}

		i := 0
		for _, lrn := range Curfn.Func.Dcl {
			if i >= ll.Len() {
				break
			}
			if lrn.Op != ONAME || lrn.Class != PPARAMOUT {
				continue
			}
			escassignNilWhy(e, lrn, ll.Index(i), "return")
			i++
		}

		if i < ll.Len() {
			Fatalf("esc return list")
		}

		// Argument could leak through recover.
	case OPANIC:
		escassignSinkNilWhy(e, n, n.Left, "panic")

	case OAPPEND:
		if !n.Isddd {
			for _, nn := range n.List.Slice()[1:] {
				escassignSinkNilWhy(e, n, nn, "appended to slice") // lose track of assign to dereference
			}
		} else {
			// append(slice1, slice2...) -- slice2 itself does not escape, but contents do.
			slice2 := n.List.Second()
			escassignDereference(e, &e.theSink, slice2, e.stepAssign(nil, n, slice2, "appended slice...")) // lose track of assign of dereference
			if Debug['m'] > 3 {
				Warnl(n.Lineno, "%v special treatment of append(slice1, slice2...) %v", e.curfnSym(n), Nconv(n, FmtShort))
			}
		}
		escassignDereference(e, &e.theSink, n.List.First(), e.stepAssign(nil, n, n.List.First(), "appendee slice")) // The original elements are now leaked, too

	case OCOPY:
		escassignDereference(e, &e.theSink, n.Right, e.stepAssign(nil, n, n.Right, "copied slice")) // lose track of assign of dereference

	case OCONV, OCONVNOP:
		escassignNilWhy(e, n, n.Left, "converted")

	case OCONVIFACE:
		e.track(n)
		escassignNilWhy(e, n, n.Left, "interface-converted")

	case OARRAYLIT:
		why := "array literal element"
		if n.Type.IsSlice() {
			// Slice itself is not leaked until proven otherwise
			e.track(n)
			why = "slice literal element"
		}
		// Link values to array/slice
		for _, n5 := range n.List.Slice() {
			escassign(e, n, n5.Right, e.stepAssign(nil, n, n5.Right, why))
		}

		// Link values to struct.
	case OSTRUCTLIT:
		for _, n6 := range n.List.Slice() {
			escassignNilWhy(e, n, n6.Right, "struct literal element")
		}

	case OPTRLIT:
		e.track(n)

		// Link OSTRUCTLIT to OPTRLIT; if OPTRLIT escapes, OSTRUCTLIT elements do too.
		escassignNilWhy(e, n, n.Left, "pointer literal [assign]")

	case OCALLPART:
		e.track(n)

		// Contents make it to memory, lose track.
		escassignSinkNilWhy(e, n, n.Left, "call part")

	case OMAPLIT:
		e.track(n)
		// Keys and values make it to memory, lose track.
		for _, n7 := range n.List.Slice() {
			escassignSinkNilWhy(e, n, n7.Left, "map literal key")
			escassignSinkNilWhy(e, n, n7.Right, "map literal value")
		}

	case OCLOSURE:
		// Link addresses of captured variables to closure.
		for _, v := range n.Func.Cvars.Slice() {
			if v.Op == OXXX { // unnamed out argument; see dcl.go:/^funcargs
				continue
			}
			a := v.Name.Defn
			if !v.Name.Byval {
				a = Nod(OADDR, a, nil)
				a.Lineno = v.Lineno
				e.nodeEscState(a).Escloopdepth = e.loopdepth
				a = typecheck(a, Erv)
			}

			escassignNilWhy(e, n, a, "captured by a closure")
		}
		fallthrough

	case OMAKECHAN,
		OMAKEMAP,
		OMAKESLICE,
		ONEW,
		OARRAYRUNESTR,
		OARRAYBYTESTR,
		OSTRARRAYRUNE,
		OSTRARRAYBYTE,
		ORUNESTR:
		e.track(n)

	case OADDSTR:
		e.track(n)
		// Arguments of OADDSTR do not escape.

	case OADDR:
		// current loop depth is an upper bound on actual loop depth
		// of addressed value.
		e.track(n)

		// for &x, use loop depth of x if known.
		// it should always be known, but if not, be conservative
		// and keep the current loop depth.
		if n.Left.Op == ONAME {
			switch n.Left.Class {
			case PAUTO:
				nE := e.nodeEscState(n)
				leftE := e.nodeEscState(n.Left)
				if leftE.Escloopdepth != 0 {
					nE.Escloopdepth = leftE.Escloopdepth
				}

				// PPARAM is loop depth 1 always.
			// PPARAMOUT is loop depth 0 for writes
			// but considered loop depth 1 for address-of,
			// so that writing the address of one result
			// to another (or the same) result makes the
			// first result move to the heap.
			case PPARAM, PPARAMOUT:
				nE := e.nodeEscState(n)
				nE.Escloopdepth = 1
			}
		}
	}

	lineno = lno
}

// escassignNilWhy bundles a common case of
// escassign(e, dst, src, e.stepAssign(nil, dst, src, reason))
func escassignNilWhy(e *EscState, dst, src *Node, reason string) {
	var step *EscStep
	if Debug['m'] != 0 {
		step = e.stepAssign(nil, dst, src, reason)
	}
	escassign(e, dst, src, step)
}

// escassignSinkNilWhy bundles a common case of
// escassign(e, &e.theSink, src, e.stepAssign(nil, dst, src, reason))
func escassignSinkNilWhy(e *EscState, dst, src *Node, reason string) {
	var step *EscStep
	if Debug['m'] != 0 {
		step = e.stepAssign(nil, dst, src, reason)
	}
	escassign(e, &e.theSink, src, step)
}

// Assert that expr somehow gets assigned to dst, if non nil.  for
// dst==nil, any name node expr still must be marked as being
// evaluated in curfn.	For expr==nil, dst must still be examined for
// evaluations inside it (e.g *f(x) = y)
func escassign(e *EscState, dst, src *Node, step *EscStep) {
	if isblank(dst) || dst == nil || src == nil || src.Op == ONONAME || src.Op == OXXX {
		return
	}

	if Debug['m'] > 2 {
		fmt.Printf("%v:[%d] %v escassign: %v(%v)[%v] = %v(%v)[%v]\n",
			linestr(lineno), e.loopdepth, funcSym(Curfn),
			Nconv(dst, FmtShort), jconv(dst, FmtShort), dst.Op,
			Nconv(src, FmtShort), jconv(src, FmtShort), src.Op)
	}

	setlineno(dst)

	originalDst := dst
	dstwhy := "assigned"

	// Analyze lhs of assignment.
	// Replace dst with e->theSink if we can't track it.
	switch dst.Op {
	default:
		Dump("dst", dst)
		Fatalf("escassign: unexpected dst")

	case OARRAYLIT,
		OCLOSURE,
		OCONV,
		OCONVIFACE,
		OCONVNOP,
		OMAPLIT,
		OSTRUCTLIT,
		OPTRLIT,
		ODDDARG,
		OCALLPART:
		break

	case ONAME:
		if dst.Class == PEXTERN {
			dstwhy = "assigned to top level variable"
			dst = &e.theSink
		}

	case ODOT: // treat "dst.x = src" as "dst = src"
		escassign(e, dst.Left, src, e.stepAssign(step, originalDst, src, "dot-equals"))
		return

	case OINDEX:
		if dst.Left.Type.IsArray() {
			escassign(e, dst.Left, src, e.stepAssign(step, originalDst, src, "array-element-equals"))
			return
		}

		dstwhy = "slice-element-equals"
		dst = &e.theSink // lose track of dereference

	case OIND:
		dstwhy = "star-equals"
		dst = &e.theSink // lose track of dereference

	case ODOTPTR:
		dstwhy = "star-dot-equals"
		dst = &e.theSink // lose track of dereference

		// lose track of key and value
	case OINDEXMAP:
		escassign(e, &e.theSink, dst.Right, e.stepAssign(nil, originalDst, src, "key of map put"))
		dstwhy = "value of map put"
		dst = &e.theSink
	}

	lno := setlineno(src)
	e.pdepth++

	switch src.Op {
	case OADDR, // dst = &x
		OIND,    // dst = *x
		ODOTPTR, // dst = (*x).f
		ONAME,
		ODDDARG,
		OPTRLIT,
		OARRAYLIT,
		OMAPLIT,
		OSTRUCTLIT,
		OMAKECHAN,
		OMAKEMAP,
		OMAKESLICE,
		OARRAYRUNESTR,
		OARRAYBYTESTR,
		OSTRARRAYRUNE,
		OSTRARRAYBYTE,
		OADDSTR,
		ONEW,
		OCALLPART,
		ORUNESTR,
		OCONVIFACE:
		escflows(e, dst, src, e.stepAssign(step, originalDst, src, dstwhy))

	case OCLOSURE:
		// OCLOSURE is lowered to OPTRLIT,
		// insert OADDR to account for the additional indirection.
		a := Nod(OADDR, src, nil)
		a.Lineno = src.Lineno
		e.nodeEscState(a).Escloopdepth = e.nodeEscState(src).Escloopdepth
		a.Type = Ptrto(src.Type)
		escflows(e, dst, a, e.stepAssign(nil, originalDst, src, dstwhy))

	// Flowing multiple returns to a single dst happens when
	// analyzing "go f(g())": here g() flows to sink (issue 4529).
	case OCALLMETH, OCALLFUNC, OCALLINTER:
		for _, n := range e.nodeEscState(src).Escretval.Slice() {
			escflows(e, dst, n, e.stepAssign(nil, originalDst, n, dstwhy))
		}

		// A non-pointer escaping from a struct does not concern us.
	case ODOT:
		if src.Type != nil && !haspointers(src.Type) {
			break
		}
		fallthrough

		// Conversions, field access, slice all preserve the input value.
	case OCONV,
		OCONVNOP,
		ODOTMETH,
		// treat recv.meth as a value with recv in it, only happens in ODEFER and OPROC
		// iface.method already leaks iface in esccall, no need to put in extra ODOTINTER edge here
		ODOTTYPE2,
		OSLICE,
		OSLICE3,
		OSLICEARR,
		OSLICE3ARR,
		OSLICESTR:
		// Conversions, field access, slice all preserve the input value.
		escassign(e, dst, src.Left, e.stepAssign(step, originalDst, src, dstwhy))

	case ODOTTYPE:
		if src.Type != nil && !haspointers(src.Type) {
			break
		}
		escassign(e, dst, src.Left, e.stepAssign(step, originalDst, src, dstwhy))

	case OAPPEND:
		// Append returns first argument.
		// Subsequent arguments are already leaked because they are operands to append.
		escassign(e, dst, src.List.First(), e.stepAssign(step, dst, src.List.First(), dstwhy))

	case OINDEX:
		// Index of array preserves input value.
		if src.Left.Type.IsArray() {
			escassign(e, dst, src.Left, e.stepAssign(step, originalDst, src, dstwhy))
		} else {
			escflows(e, dst, src, e.stepAssign(step, originalDst, src, dstwhy))
		}

		// Might be pointer arithmetic, in which case
	// the operands flow into the result.
	// TODO(rsc): Decide what the story is here. This is unsettling.
	case OADD,
		OSUB,
		OOR,
		OXOR,
		OMUL,
		ODIV,
		OMOD,
		OLSH,
		ORSH,
		OAND,
		OANDNOT,
		OPLUS,
		OMINUS,
		OCOM:
		escassign(e, dst, src.Left, e.stepAssign(step, originalDst, src, dstwhy))

		escassign(e, dst, src.Right, e.stepAssign(step, originalDst, src, dstwhy))
	}

	e.pdepth--
	lineno = lno
}

// Common case for escapes is 16 bits 000000000xxxEEEE
// where commonest cases for xxx encoding in-to-out pointer
//  flow are 000, 001, 010, 011  and EEEE is computed Esc bits.
// Note width of xxx depends on value of constant
// bitsPerOutputInTag -- expect 2 or 3, so in practice the
// tag cache array is 64 or 128 long. Some entries will
// never be populated.
var tags [1 << (bitsPerOutputInTag + EscReturnBits)]string

// mktag returns the string representation for an escape analysis tag.
func mktag(mask int) string {
	switch mask & EscMask {
	case EscNone, EscReturn:
		break

	default:
		Fatalf("escape mktag")
	}

	if mask < len(tags) && tags[mask] != "" {
		return tags[mask]
	}

	s := fmt.Sprintf("esc:0x%x", mask)
	if mask < len(tags) {
		tags[mask] = s
	}
	return s
}

// parsetag decodes an escape analysis tag and returns the esc value.
func parsetag(note string) uint16 {
	if !strings.HasPrefix(note, "esc:") {
		return EscUnknown
	}
	n, _ := strconv.ParseInt(note[4:], 0, 0)
	em := uint16(n)
	if em == 0 {
		return EscNone
	}
	return em
}

// describeEscape returns a string describing the escape tag.
// The result is either one of {EscUnknown, EscNone, EscHeap} which all have no further annotation
// or a description of parameter flow, which takes the form of an optional "contentToHeap"
// indicating that the content of this parameter is leaked to the heap, followed by a sequence
// of level encodings separated by spaces, one for each parameter, where _ means no flow,
// = means direct flow, and N asterisks (*) encodes content (obtained by indirection) flow.
// e.g., "contentToHeap _ =" means that a parameter's content (one or more dereferences)
// escapes to the heap, the parameter does not leak to the first output, but does leak directly
// to the second output (and if there are more than two outputs, there is no flow to those.)
func describeEscape(em uint16) string {
	var s string
	if em&EscMask == EscUnknown {
		s = "EscUnknown"
	}
	if em&EscMask == EscNone {
		s = "EscNone"
	}
	if em&EscMask == EscHeap {
		s = "EscHeap"
	}
	if em&EscMask == EscReturn {
		s = "EscReturn"
	}
	if em&EscMask == EscScope {
		s = "EscScope"
	}
	if em&EscContentEscapes != 0 {
		if s != "" {
			s += " "
		}
		s += "contentToHeap"
	}
	for em >>= EscReturnBits; em != 0; em = em >> bitsPerOutputInTag {
		// See encoding description above
		if s != "" {
			s += " "
		}
		switch embits := em & bitsMaskForTag; embits {
		case 0:
			s += "_"
		case 1:
			s += "="
		default:
			for i := uint16(0); i < embits-1; i++ {
				s += "*"
			}
		}

	}
	return s
}

// escassignfromtag models the input-to-output assignment flow of one of a function
// calls arguments, where the flow is encoded in "note".
func escassignfromtag(e *EscState, note string, dsts Nodes, src *Node) uint16 {
	em := parsetag(note)
	if src.Op == OLITERAL {
		return em
	}

	if Debug['m'] > 3 {
		fmt.Printf("%v::assignfromtag:: src=%v, em=%s\n",
			linestr(lineno), Nconv(src, FmtShort), describeEscape(em))
	}

	if em == EscUnknown {
		escassignSinkNilWhy(e, src, src, "passed to function[unknown]")
		return em
	}

	if em == EscNone {
		return em
	}

	// If content inside parameter (reached via indirection)
	// escapes to heap, mark as such.
	if em&EscContentEscapes != 0 {
		escassign(e, &e.theSink, e.addDereference(src), e.stepAssign(nil, src, src, "passed to function[content escapes]"))
	}

	em0 := em
	dstsi := 0
	for em >>= EscReturnBits; em != 0 && dstsi < dsts.Len(); em = em >> bitsPerOutputInTag {
		// Prefer the lowest-level path to the reference (for escape purposes).
		// Two-bit encoding (for example. 1, 3, and 4 bits are other options)
		//  01 = 0-level
		//  10 = 1-level, (content escapes),
		//  11 = 2-level, (content of content escapes),
		embits := em & bitsMaskForTag
		if embits > 0 {
			n := src
			for i := uint16(0); i < embits-1; i++ {
				n = e.addDereference(n) // encode level>0 as indirections
			}
			escassign(e, dsts.Index(dstsi), n, e.stepAssign(nil, dsts.Index(dstsi), src, "passed-to-and-returned-from-function"))
		}
		dstsi++
	}
	// If there are too many outputs to fit in the tag,
	// that is handled at the encoding end as EscHeap,
	// so there is no need to check here.

	if em != 0 && dstsi >= dsts.Len() {
		Fatalf("corrupt esc tag %q or messed up escretval list\n", note)
	}
	return em0
}

func escassignDereference(e *EscState, dst *Node, src *Node, step *EscStep) {
	if src.Op == OLITERAL {
		return
	}
	escassign(e, dst, e.addDereference(src), step)
}

// addDereference constructs a suitable OIND note applied to src.
// Because this is for purposes of escape accounting, not execution,
// some semantically dubious node combinations are (currently) possible.
func (e *EscState) addDereference(n *Node) *Node {
	ind := Nod(OIND, n, nil)
	e.nodeEscState(ind).Escloopdepth = e.nodeEscState(n).Escloopdepth
	ind.Lineno = n.Lineno
	t := n.Type
	if t.IsKind(Tptr) {
		// This should model our own sloppy use of OIND to encode
		// decreasing levels of indirection; i.e., "indirecting" an array
		// might yield the type of an element. To be enhanced...
		t = t.Elem()
	}
	ind.Type = t
	return ind
}

// escNoteOutputParamFlow encodes maxEncodedLevel/.../1/0-level flow to the vargen'th parameter.
// Levels greater than maxEncodedLevel are replaced with maxEncodedLevel.
// If the encoding cannot describe the modified input level and output number, then EscHeap is returned.
func escNoteOutputParamFlow(e uint16, vargen int32, level Level) uint16 {
	// Flow+level is encoded in two bits.
	// 00 = not flow, xx = level+1 for 0 <= level <= maxEncodedLevel
	// 16 bits for Esc allows 6x2bits or 4x3bits or 3x4bits if additional information would be useful.
	if level.int() <= 0 && level.guaranteedDereference() > 0 {
		return escMax(e|EscContentEscapes, EscNone) // At least one deref, thus only content.
	}
	if level.int() < 0 {
		return EscHeap
	}
	if level.int() > maxEncodedLevel {
		// Cannot encode larger values than maxEncodedLevel.
		level = levelFrom(maxEncodedLevel)
	}
	encoded := uint16(level.int() + 1)

	shift := uint(bitsPerOutputInTag*(vargen-1) + EscReturnBits)
	old := (e >> shift) & bitsMaskForTag
	if old == 0 || encoded != 0 && encoded < old {
		old = encoded
	}

	encodedFlow := old << shift
	if (encodedFlow>>shift)&bitsMaskForTag != old {
		// Encoding failure defaults to heap.
		return EscHeap
	}

	return (e &^ (bitsMaskForTag << shift)) | encodedFlow
}

func initEscretval(e *EscState, n *Node, fntype *Type) {
	i := 0
	nE := e.nodeEscState(n)
	nE.Escretval.Set(nil) // Suspect this is not nil for indirect calls.
	for _, t := range fntype.Results().Fields().Slice() {
		src := Nod(ONAME, nil, nil)
		buf := fmt.Sprintf(".out%d", i)
		i++
		src.Sym = Lookup(buf)
		src.Type = t.Type
		src.Class = PAUTO
		src.Name.Curfn = Curfn
		e.nodeEscState(src).Escloopdepth = e.loopdepth
		src.Used = true
		src.Lineno = n.Lineno
		nE.Escretval.Append(src)
	}
}

// This is a bit messier than fortunate, pulled out of esc's big
// switch for clarity.	We either have the paramnodes, which may be
// connected to other things through flows or we have the parameter type
// nodes, which may be marked "noescape". Navigating the ast is slightly
// different for methods vs plain functions and for imported vs
// this-package
func esccall(e *EscState, n *Node, up *Node) {
	var fntype *Type
	var indirect bool
	var fn *Node
	switch n.Op {
	default:
		Fatalf("esccall")

	case OCALLFUNC:
		fn = n.Left
		fntype = fn.Type
		indirect = fn.Op != ONAME || fn.Class != PFUNC

	case OCALLMETH:
		fn = n.Left.Sym.Def
		if fn != nil {
			fntype = fn.Type
		} else {
			fntype = n.Left.Type
		}

	case OCALLINTER:
		fntype = n.Left.Type
		indirect = true
	}

	ll := n.List
	if n.List.Len() == 1 {
		a := n.List.First()
		if a.Type.IsFuncArgStruct() { // f(g())
			ll = e.nodeEscState(a).Escretval
		}
	}

	if indirect {
		// We know nothing!
		// Leak all the parameters
		for _, n1 := range ll.Slice() {
			escassignSinkNilWhy(e, n, n1, "parameter to indirect call")
			if Debug['m'] > 3 {
				fmt.Printf("%v::esccall:: indirect call <- %v, untracked\n", linestr(lineno), Nconv(n1, FmtShort))
			}
		}
		// Set up bogus outputs
		initEscretval(e, n, fntype)
		// If there is a receiver, it also leaks to heap.
		if n.Op != OCALLFUNC {
			t := fntype.Recv()
			src := n.Left.Left
			if haspointers(t.Type) {
				escassignSinkNilWhy(e, n, src, "receiver in indirect call")
			}
		} else { // indirect and OCALLFUNC = could be captured variables, too. (#14409)
			ll := e.nodeEscState(n).Escretval.Slice()
			for _, llN := range ll {
				escassignDereference(e, llN, fn, e.stepAssign(nil, llN, fn, "captured by called closure"))
			}
		}
		return
	}

	nE := e.nodeEscState(n)
	if fn != nil && fn.Op == ONAME && fn.Class == PFUNC &&
		fn.Name.Defn != nil && fn.Name.Defn.Nbody.Len() != 0 && fn.Name.Param.Ntype != nil && fn.Name.Defn.Esc < EscFuncTagged {
		if Debug['m'] > 3 {
			fmt.Printf("%v::esccall:: %v in recursive group\n", linestr(lineno), Nconv(n, FmtShort))
		}

		// function in same mutually recursive group. Incorporate into flow graph.
		//		print("esc local fn: %N\n", fn->ntype);
		if fn.Name.Defn.Esc == EscFuncUnknown || nE.Escretval.Len() != 0 {
			Fatalf("graph inconsistency")
		}

		lls := ll.Slice()
		sawRcvr := false
		var src *Node
		for _, n2 := range fn.Name.Defn.Func.Dcl {
			switch n2.Class {
			case PPARAM:
				if n.Op != OCALLFUNC && !sawRcvr {
					escassignNilWhy(e, n2, n.Left.Left, "call receiver")
					sawRcvr = true
					continue
				}
				if len(lls) == 0 {
					continue
				}
				src = lls[0]
				if n2.Isddd && !n.Isddd {
					// Introduce ODDDARG node to represent ... allocation.
					src = Nod(ODDDARG, nil, nil)
					arr := typArray(n2.Type.Elem(), int64(len(lls)))
					src.Type = Ptrto(arr) // make pointer so it will be tracked
					src.Lineno = n.Lineno
					e.track(src)
					n.Right = src
				}
				escassignNilWhy(e, n2, src, "arg to recursive call")
				if src != lls[0] {
					// "..." arguments are untracked
					for _, n2 := range lls {
						if Debug['m'] > 3 {
							fmt.Printf("%v::esccall:: ... <- %v, untracked\n", linestr(lineno), Nconv(n2, FmtShort))
						}
						escassignSinkNilWhy(e, src, n2, "... arg to recursive call")
					}
					// No more PPARAM processing, but keep
					// going for PPARAMOUT.
					lls = nil
					continue
				}
				lls = lls[1:]

			case PPARAMOUT:
				nE.Escretval.Append(n2)
			}
		}

		return
	}

	// Imported or completely analyzed function. Use the escape tags.
	if nE.Escretval.Len() != 0 {
		Fatalf("esc already decorated call %v\n", Nconv(n, FmtSign))
	}

	if Debug['m'] > 3 {
		fmt.Printf("%v::esccall:: %v not recursive\n", linestr(lineno), Nconv(n, FmtShort))
	}

	// set up out list on this call node with dummy auto ONAMES in the current (calling) function.
	initEscretval(e, n, fntype)

	//	print("esc analyzed fn: %#N (%+T) returning (%+H)\n", fn, fntype, n->escretval);

	// Receiver.
	if n.Op != OCALLFUNC {
		t := fntype.Recv()
		src := n.Left.Left
		if haspointers(t.Type) {
			escassignfromtag(e, t.Note, nE.Escretval, src)
		}
	}

	var src *Node
	note := ""
	i := 0
	lls := ll.Slice()
	for t, it := IterFields(fntype.Params()); i < len(lls); i++ {
		src = lls[i]
		note = t.Note
		if t.Isddd && !n.Isddd {
			// Introduce ODDDARG node to represent ... allocation.
			src = Nod(ODDDARG, nil, nil)
			src.Lineno = n.Lineno
			arr := typArray(t.Type.Elem(), int64(len(lls)-i))
			src.Type = Ptrto(arr) // make pointer so it will be tracked
			e.track(src)
			n.Right = src
		}

		if haspointers(t.Type) {
			if escassignfromtag(e, note, nE.Escretval, src) == EscNone && up.Op != ODEFER && up.Op != OPROC {
				a := src
				for a.Op == OCONVNOP {
					a = a.Left
				}
				switch a.Op {
				// The callee has already been analyzed, so its arguments have esc tags.
				// The argument is marked as not escaping at all.
				// Record that fact so that any temporary used for
				// synthesizing this expression can be reclaimed when
				// the function returns.
				// This 'noescape' is even stronger than the usual esc == EscNone.
				// src->esc == EscNone means that src does not escape the current function.
				// src->noescape = 1 here means that src does not escape this statement
				// in the current function.
				case OCALLPART,
					OCLOSURE,
					ODDDARG,
					OARRAYLIT,
					OPTRLIT,
					OSTRUCTLIT:
					a.Noescape = true
				}
			}
		}

		if src != lls[i] {
			// This occurs when function parameter type Isddd and n not Isddd
			break
		}

		if note == uintptrEscapesTag {
			escassignSinkNilWhy(e, src, src, "escaping uintptr")
		}

		t = it.Next()
	}

	// Store arguments into slice for ... arg.
	for ; i < len(lls); i++ {
		if Debug['m'] > 3 {
			fmt.Printf("%v::esccall:: ... <- %v\n", linestr(lineno), Nconv(lls[i], FmtShort))
		}
		if note == uintptrEscapesTag {
			escassignSinkNilWhy(e, src, lls[i], "arg to uintptrescapes ...")
		} else {
			escassignNilWhy(e, src, lls[i], "arg to ...")
		}
	}
}

// escflows records the link src->dst in dst, throwing out some quick wins,
// and also ensuring that dst is noted as a flow destination.
func escflows(e *EscState, dst, src *Node, why *EscStep) {
	if dst == nil || src == nil || dst == src {
		return
	}

	// Don't bother building a graph for scalars.
	if src.Type != nil && !haspointers(src.Type) {
		return
	}

	if Debug['m'] > 3 {
		fmt.Printf("%v::flows:: %v <- %v\n", linestr(lineno), Nconv(dst, FmtShort), Nconv(src, FmtShort))
	}

	dstE := e.nodeEscState(dst)
	if len(dstE.Escflowsrc) == 0 {
		e.dsts = append(e.dsts, dst)
		e.dstcount++
	}

	e.edgecount++

	if why == nil {
		dstE.Escflowsrc = append(dstE.Escflowsrc, EscStep{src: src})
	} else {
		starwhy := *why
		starwhy.src = src // TODO: need to reconcile this w/ needs of explanations.
		dstE.Escflowsrc = append(dstE.Escflowsrc, starwhy)
	}
}

// Whenever we hit a reference node, the level goes up by one, and whenever
// we hit an OADDR, the level goes down by one. as long as we're on a level > 0
// finding an OADDR just means we're following the upstream of a dereference,
// so this address doesn't leak (yet).
// If level == 0, it means the /value/ of this node can reach the root of this flood.
// so if this node is an OADDR, its argument should be marked as escaping iff
// its currfn/e->loopdepth are different from the flood's root.
// Once an object has been moved to the heap, all of its upstream should be considered
// escaping to the global scope.
func escflood(e *EscState, dst *Node) {
	switch dst.Op {
	case ONAME, OCLOSURE:
		break

	default:
		return
	}

	dstE := e.nodeEscState(dst)
	if Debug['m'] > 2 {
		fmt.Printf("\nescflood:%d: dst %v scope:%v[%d]\n", e.walkgen, Nconv(dst, FmtShort), e.curfnSym(dst), dstE.Escloopdepth)
	}

	for i, l := range dstE.Escflowsrc {
		e.walkgen++
		dstE.Escflowsrc[i].parent = nil
		escwalk(e, levelFrom(0), dst, l.src, &dstE.Escflowsrc[i])
	}
}

// funcOutputAndInput reports whether dst and src correspond to output and input parameters of the same function.
func funcOutputAndInput(dst, src *Node) bool {
	// Note if dst is marked as escaping, then "returned" is too weak.
	return dst.Op == ONAME && dst.Class == PPARAMOUT &&
		src.Op == ONAME && src.Class == PPARAM && src.Name.Curfn == dst.Name.Curfn
}

func (es *EscStep) describe(src *Node) {
	if Debug['m'] < 2 {
		return
	}
	step0 := es
	for step := step0; step != nil && !step.busy; step = step.parent {
		// TODO: We get cycles. Trigger is i = &i (where var i interface{})
		step.busy = true
		// The trail is a little odd because of how the
		// graph is constructed.  The link to the current
		// Node is parent.src unless parent is nil in which
		// case it is step.dst.
		nextDest := step.parent
		dst := step.dst
		if nextDest != nil {
			dst = nextDest.src
		}
		Warnl(src.Lineno, "\tfrom %s (%s) at %s", dst, step.why, dst.Line())
	}
	for step := step0; step != nil && step.busy; step = step.parent {
		step.busy = false
	}
}

const NOTALOOPDEPTH = -1

func escwalk(e *EscState, level Level, dst *Node, src *Node, step *EscStep) {
	escwalkBody(e, level, dst, src, step, NOTALOOPDEPTH)
}

func escwalkBody(e *EscState, level Level, dst *Node, src *Node, step *EscStep, extraloopdepth int32) {
	if src.Op == OLITERAL {
		return
	}
	srcE := e.nodeEscState(src)
	if srcE.Walkgen == e.walkgen {
		// Esclevels are vectors, do not compare as integers,
		// and must use "min" of old and new to guarantee
		// convergence.
		level = level.min(srcE.Esclevel)
		if level == srcE.Esclevel {
			// Have we been here already with an extraloopdepth,
			// or is the extraloopdepth provided no improvement on
			// what's already been seen?
			if srcE.Maxextraloopdepth >= extraloopdepth || srcE.Escloopdepth >= extraloopdepth {
				return
			}
			srcE.Maxextraloopdepth = extraloopdepth
		}
	} else { // srcE.Walkgen < e.walkgen -- first time, reset this.
		srcE.Maxextraloopdepth = NOTALOOPDEPTH
	}

	srcE.Walkgen = e.walkgen
	srcE.Esclevel = level
	modSrcLoopdepth := srcE.Escloopdepth

	if extraloopdepth > modSrcLoopdepth {
		modSrcLoopdepth = extraloopdepth
	}

	if Debug['m'] > 2 {
		fmt.Printf("escwalk: level:%d depth:%d %.*s op=%v %v(%v) scope:%v[%d] extraloopdepth=%v\n",
			level, e.pdepth, e.pdepth, "\t\t\t\t\t\t\t\t\t\t", src.Op, Nconv(src, FmtShort), jconv(src, FmtShort), e.curfnSym(src), srcE.Escloopdepth, extraloopdepth)
	}

	e.pdepth++

	// Input parameter flowing to output parameter?
	var leaks bool
	var osrcesc uint16 // used to prevent duplicate error messages

	dstE := e.nodeEscState(dst)
	if funcOutputAndInput(dst, src) && src.Esc&EscMask < EscScope && dst.Esc != EscHeap {
		// This case handles:
		// 1. return in
		// 2. return &in
		// 3. tmp := in; return &tmp
		// 4. return *in
		if Debug['m'] != 0 {
			if Debug['m'] <= 2 {
				Warnl(src.Lineno, "leaking param: %v to result %v level=%v", Nconv(src, FmtShort), dst.Sym, level.int())
				step.describe(src)
			} else {
				Warnl(src.Lineno, "leaking param: %v to result %v level=%v", Nconv(src, FmtShort), dst.Sym, level)
			}
		}
		if src.Esc&EscMask != EscReturn {
			src.Esc = EscReturn | src.Esc&EscContentEscapes
		}
		src.Esc = escNoteOutputParamFlow(src.Esc, dst.Name.Vargen, level)
		goto recurse
	}

	// If parameter content escapes to heap, set EscContentEscapes
	// Note minor confusion around escape from pointer-to-struct vs escape from struct
	if dst.Esc == EscHeap &&
		src.Op == ONAME && src.Class == PPARAM && src.Esc&EscMask < EscScope &&
		level.int() > 0 {
		src.Esc = escMax(EscContentEscapes|src.Esc, EscNone)
		if Debug['m'] != 0 {
			Warnl(src.Lineno, "mark escaped content: %v", Nconv(src, FmtShort))
			step.describe(src)
		}
	}

	leaks = level.int() <= 0 && level.guaranteedDereference() <= 0 && dstE.Escloopdepth < modSrcLoopdepth

	osrcesc = src.Esc
	switch src.Op {
	case ONAME:
		if src.Class == PPARAM && (leaks || dstE.Escloopdepth < 0) && src.Esc&EscMask < EscScope {
			if level.guaranteedDereference() > 0 {
				src.Esc = escMax(EscContentEscapes|src.Esc, EscNone)
				if Debug['m'] != 0 {
					if Debug['m'] <= 2 {
						if osrcesc != src.Esc {
							Warnl(src.Lineno, "leaking param content: %v", Nconv(src, FmtShort))
							step.describe(src)
						}
					} else {
						Warnl(src.Lineno, "leaking param content: %v level=%v dst.eld=%v src.eld=%v dst=%v",
							Nconv(src, FmtShort), level, dstE.Escloopdepth, modSrcLoopdepth, Nconv(dst, FmtShort))
					}
				}
			} else {
				src.Esc = EscScope
				if Debug['m'] != 0 {

					if Debug['m'] <= 2 {
						Warnl(src.Lineno, "leaking param: %v", Nconv(src, FmtShort))
						step.describe(src)
					} else {
						Warnl(src.Lineno, "leaking param: %v level=%v dst.eld=%v src.eld=%v dst=%v",
							Nconv(src, FmtShort), level, dstE.Escloopdepth, modSrcLoopdepth, Nconv(dst, FmtShort))
					}
				}
			}
		}

		// Treat a captured closure variable as equivalent to the
		// original variable.
		if src.isClosureVar() {
			if leaks && Debug['m'] != 0 {
				Warnl(src.Lineno, "leaking closure reference %v", Nconv(src, FmtShort))
				step.describe(src)
			}
			escwalk(e, level, dst, src.Name.Defn, e.stepWalk(dst, src.Name.Defn, "closure-var", step))
		}

	case OPTRLIT, OADDR:
		why := "pointer literal"
		if src.Op == OADDR {
			why = "address-of"
		}
		if leaks {
			src.Esc = EscHeap
			if Debug['m'] != 0 && osrcesc != src.Esc {
				p := src
				if p.Left.Op == OCLOSURE {
					p = p.Left // merely to satisfy error messages in tests
				}
				if Debug['m'] > 2 {
					Warnl(src.Lineno, "%v escapes to heap, level=%v, dst=%v dst.eld=%v, src.eld=%v",
						Nconv(p, FmtShort), level, dst, dstE.Escloopdepth, modSrcLoopdepth)
				} else {
					Warnl(src.Lineno, "%v escapes to heap", Nconv(p, FmtShort))
					step.describe(src)
				}
			}
			addrescapes(src.Left)
			escwalkBody(e, level.dec(), dst, src.Left, e.stepWalk(dst, src.Left, why, step), modSrcLoopdepth)
			extraloopdepth = modSrcLoopdepth // passes to recursive case, seems likely a no-op
		} else {
			escwalk(e, level.dec(), dst, src.Left, e.stepWalk(dst, src.Left, why, step))
		}

	case OAPPEND:
		escwalk(e, level, dst, src.List.First(), e.stepWalk(dst, src.List.First(), "append-first-arg", step))

	case ODDDARG:
		if leaks {
			src.Esc = EscHeap
			if Debug['m'] != 0 && osrcesc != src.Esc {
				Warnl(src.Lineno, "%v escapes to heap", Nconv(src, FmtShort))
				step.describe(src)
			}
			extraloopdepth = modSrcLoopdepth
		}
		// similar to a slice arraylit and its args.
		level = level.dec()

	case OARRAYLIT:
		if src.Type.IsArray() {
			break
		}
		for _, n1 := range src.List.Slice() {
			escwalk(e, level.dec(), dst, n1.Right, e.stepWalk(dst, n1.Right, "slice-literal-element", step))
		}

		fallthrough

	case OMAKECHAN,
		OMAKEMAP,
		OMAKESLICE,
		OARRAYRUNESTR,
		OARRAYBYTESTR,
		OSTRARRAYRUNE,
		OSTRARRAYBYTE,
		OADDSTR,
		OMAPLIT,
		ONEW,
		OCLOSURE,
		OCALLPART,
		ORUNESTR,
		OCONVIFACE:
		if leaks {
			src.Esc = EscHeap
			if Debug['m'] != 0 && osrcesc != src.Esc {
				Warnl(src.Lineno, "%v escapes to heap", Nconv(src, FmtShort))
				step.describe(src)
			}
			extraloopdepth = modSrcLoopdepth
		}

	case ODOT,
		ODOTTYPE:
		escwalk(e, level, dst, src.Left, e.stepWalk(dst, src.Left, "dot", step))

	case
		OSLICE,
		OSLICEARR,
		OSLICE3,
		OSLICE3ARR,
		OSLICESTR:
		escwalk(e, level, dst, src.Left, e.stepWalk(dst, src.Left, "slice", step))

	case OINDEX:
		if src.Left.Type.IsArray() {
			escwalk(e, level, dst, src.Left, e.stepWalk(dst, src.Left, "fixed-array-index-of", step))
			break
		}
		fallthrough

	case ODOTPTR:
		escwalk(e, level.inc(), dst, src.Left, e.stepWalk(dst, src.Left, "dot of pointer", step))
	case OINDEXMAP:
		escwalk(e, level.inc(), dst, src.Left, e.stepWalk(dst, src.Left, "map index", step))
	case OIND:
		escwalk(e, level.inc(), dst, src.Left, e.stepWalk(dst, src.Left, "indirection", step))

	// In this case a link went directly to a call, but should really go
	// to the dummy .outN outputs that were created for the call that
	// themselves link to the inputs with levels adjusted.
	// See e.g. #10466
	// This can only happen with functions returning a single result.
	case OCALLMETH, OCALLFUNC, OCALLINTER:
		if srcE.Escretval.Len() != 0 {
			if Debug['m'] > 2 {
				fmt.Printf("%v:[%d] dst %v escwalk replace src: %v with %v\n",
					linestr(lineno), e.loopdepth,
					Nconv(dst, FmtShort), Nconv(src, FmtShort), Nconv(srcE.Escretval.First(), FmtShort))
			}
			src = srcE.Escretval.First()
			srcE = e.nodeEscState(src)
		}
	}

recurse:
	level = level.copy()
	for i, ll := range srcE.Escflowsrc {
		srcE.Escflowsrc[i].parent = step
		escwalkBody(e, level, dst, ll.src, &srcE.Escflowsrc[i], extraloopdepth)
		srcE.Escflowsrc[i].parent = nil
	}

	e.pdepth--
}

// This special tag is applied to uintptr variables
// that we believe may hold unsafe.Pointers for
// calls into assembly functions.
// It is logically a constant, but using a var
// lets us take the address below to get a *string.
var unsafeUintptrTag = "unsafe-uintptr"

// This special tag is applied to uintptr parameters of functions
// marked go:uintptrescapes.
const uintptrEscapesTag = "uintptr-escapes"

func esctag(e *EscState, func_ *Node) {
	func_.Esc = EscFuncTagged

	name := func(s *Sym, narg int) string {
		if s != nil {
			return s.Name
		}
		return fmt.Sprintf("arg#%d", narg)
	}

	// External functions are assumed unsafe,
	// unless //go:noescape is given before the declaration.
	if func_.Nbody.Len() == 0 {
		if func_.Noescape {
			for _, t := range func_.Type.Params().Fields().Slice() {
				if haspointers(t.Type) {
					t.Note = mktag(EscNone)
				}
			}
		}

		// Assume that uintptr arguments must be held live across the call.
		// This is most important for syscall.Syscall.
		// See golang.org/issue/13372.
		// This really doesn't have much to do with escape analysis per se,
		// but we are reusing the ability to annotate an individual function
		// argument and pass those annotations along to importing code.
		narg := 0
		for _, t := range func_.Type.Params().Fields().Slice() {
			narg++
			if t.Type.Etype == TUINTPTR {
				if Debug['m'] != 0 {
					Warnl(func_.Lineno, "%v assuming %v is unsafe uintptr", funcSym(func_), name(t.Sym, narg))
				}
				t.Note = unsafeUintptrTag
			}
		}

		return
	}

	if func_.Func.Pragma&UintptrEscapes != 0 {
		narg := 0
		for _, t := range func_.Type.Params().Fields().Slice() {
			narg++
			if t.Type.Etype == TUINTPTR {
				if Debug['m'] != 0 {
					Warnl(func_.Lineno, "%v marking %v as escaping uintptr", funcSym(func_), name(t.Sym, narg))
				}
				t.Note = uintptrEscapesTag
			}

			if t.Isddd && t.Type.Elem().Etype == TUINTPTR {
				// final argument is ...uintptr.
				if Debug['m'] != 0 {
					Warnl(func_.Lineno, "%v marking %v as escaping ...uintptr", funcSym(func_), name(t.Sym, narg))
				}
				t.Note = uintptrEscapesTag
			}
		}
	}

	savefn := Curfn
	Curfn = func_

	for _, ln := range Curfn.Func.Dcl {
		if ln.Op != ONAME {
			continue
		}

		switch ln.Esc & EscMask {
		case EscNone, // not touched by escflood
			EscReturn:
			if haspointers(ln.Type) { // don't bother tagging for scalars
				if ln.Name.Param.Field.Note != uintptrEscapesTag {
					ln.Name.Param.Field.Note = mktag(int(ln.Esc))
				}
			}

		case EscHeap, // touched by escflood, moved to heap
			EscScope: // touched by escflood, value leaves scope
			break
		}
	}

	Curfn = savefn
}
