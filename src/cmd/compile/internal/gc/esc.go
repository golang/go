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
// by reverse topological order in the static call graph.
// The algorithm (known as Tarjan's algorithm) for doing that is taken from
// Sedgewick, Algorithms, Second Edition, p. 482, with two adaptations.
//
// First, a hidden closure function (n.Func.IsHiddenClosure) cannot be the
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
		if n.Op == ODCLFUNC && !n.Func.IsHiddenClosure {
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
	if (min == id || min == id+1) && !n.Func.IsHiddenClosure {
		// This node is the root of a strongly connected component.

		// The original min passed to visitcodelist was v.nodeID[n]+1.
		// If visitcodelist found its way back to v.nodeID[n], then this
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
// pointer-containing nodes and store them in e.nodeEscState(dst).Flowsrc. For
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
	where    *Node    // sometimes the endpoints don't match source locations; set 'where' to make that right
	parent   *EscStep // used in flood to record path
	why      string   // explanation for this step in the escape-to-heap chain
	busy     bool     // used in prevent to snip cycles.
}

type NodeEscState struct {
	Curfn             *Node
	Flowsrc           []EscStep // flow(this, src)
	Retval            Nodes     // on OCALLxxx, list of dummy return values
	Loopdepth         int32     // -1: global, 0: return variables, 1:function top level, increased inside function for every loop or label to mark scopes
	Level             Level
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
	nE := &NodeEscState{
		Curfn: Curfn,
	}
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
	nE.Loopdepth = e.loopdepth
	e.noesc = append(e.noesc, n)
}

// Escape constants are numbered in order of increasing "escapiness"
// to help make inferences be monotonic. With the exception of
// EscNever which is sticky, eX < eY means that eY is more exposed
// than eX, and hence replaces it in a conservative analysis.
const (
	EscUnknown        = iota
	EscNone           // Does not escape to heap, result, or parameters.
	EscReturn         // Is returned or reachable from returned.
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
	if e&EscMask >= EscHeap {
		// normalize
		if e&^EscMask != 0 {
			Fatalf("Escape information had unexpected return encoding bits (w/ EscHeap, EscNever), e&EscMask=%v", e&EscMask)
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

func newEscState(recursive bool) *EscState {
	e := new(EscState)
	e.theSink.Op = ONAME
	e.theSink.Orig = &e.theSink
	e.theSink.Class = PEXTERN
	e.theSink.Sym = lookup(".sink")
	e.nodeEscState(&e.theSink).Loopdepth = -1
	e.recursive = recursive
	return e
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
		if step.why == "" {
			step.why = why
		}
		if step.dst == nil {
			step.dst = dst
		}
		if step.src == nil {
			step.src = src
		}
		return step
	}
	return &EscStep{src: src, dst: dst, why: why}
}

func (e *EscState) stepAssignWhere(dst, src *Node, why string, where *Node) *EscStep {
	if Debug['m'] == 0 {
		return nil
	}
	return &EscStep{src: src, dst: dst, why: why, where: where}
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
	e := newEscState(recursive)

	for _, n := range all {
		if n.Op == ODCLFUNC {
			n.Esc = EscFuncPlanned
		}
	}

	// flow-analyze functions
	for _, n := range all {
		if n.Op == ODCLFUNC {
			e.escfunc(n)
		}
	}

	// print("escapes: %d e.dsts, %d edges\n", e.dstcount, e.edgecount);

	// visit the upstream of each dst, mark address nodes with
	// addrescapes, mark parameters unsafe
	escapes := make([]uint16, len(e.dsts))
	for i, n := range e.dsts {
		escapes[i] = n.Esc
	}
	for _, n := range e.dsts {
		e.escflood(n)
	}
	for {
		done := true
		for i, n := range e.dsts {
			if n.Esc != escapes[i] {
				done = false
				if Debug['m'] > 2 {
					Warnl(n.Lineno, "Reflooding %v %S", e.curfnSym(n), n)
				}
				escapes[i] = n.Esc
				e.escflood(n)
			}
		}
		if done {
			break
		}
	}

	// for all top level functions, tag the typenodes corresponding to the param nodes
	for _, n := range all {
		if n.Op == ODCLFUNC {
			e.esctag(n)
		}
	}

	if Debug['m'] != 0 {
		for _, n := range e.noesc {
			if n.Esc == EscNone {
				Warnl(n.Lineno, "%v %S does not escape", e.curfnSym(n), n)
			}
		}
	}

	for _, x := range e.opts {
		x.SetOpt(nil)
	}
}

func (e *EscState) escfunc(fn *Node) {
	//	print("escfunc %N %s\n", fn.Func.Nname, e.recursive?"(recursive)":"");
	if fn.Esc != EscFuncPlanned {
		Fatalf("repeat escfunc %v", fn.Func.Nname)
	}
	fn.Esc = EscFuncStarted

	saveld := e.loopdepth
	e.loopdepth = 1
	savefn := Curfn
	Curfn = fn

	for _, ln := range Curfn.Func.Dcl {
		if ln.Op != ONAME {
			continue
		}
		lnE := e.nodeEscState(ln)
		switch ln.Class {
		// out params are in a loopdepth between the sink and all local variables
		case PPARAMOUT:
			lnE.Loopdepth = 0

		case PPARAM:
			lnE.Loopdepth = 1
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
				e.escflows(&e.theSink, ln, e.stepAssign(nil, ln, ln, "returned from recursive function"))
			}
		}
	}

	e.escloopdepthlist(Curfn.Nbody)
	e.esclist(Curfn.Nbody, Curfn)
	Curfn = savefn
	e.loopdepth = saveld
}

// Mark labels that have no backjumps to them as not increasing e.loopdepth.
// Walk hasn't generated (goto|label).Left.Sym.Label yet, so we'll cheat
// and set it to one of the following two. Then in esc we'll clear it again.
var (
	looping    Node
	nonlooping Node
)

func (e *EscState) escloopdepthlist(l Nodes) {
	for _, n := range l.Slice() {
		e.escloopdepth(n)
	}
}

func (e *EscState) escloopdepth(n *Node) {
	if n == nil {
		return
	}

	e.escloopdepthlist(n.Ninit)

	switch n.Op {
	case OLABEL:
		if n.Left == nil || n.Left.Sym == nil {
			Fatalf("esc:label without label: %+v", n)
		}

		// Walk will complain about this label being already defined, but that's not until
		// after escape analysis. in the future, maybe pull label & goto analysis out of walk and put before esc
		// if(n.Left.Sym.Label != nil)
		//	fatal("escape analysis messed up analyzing label: %+N", n);
		n.Left.Sym.Label = &nonlooping

	case OGOTO:
		if n.Left == nil || n.Left.Sym == nil {
			Fatalf("esc:goto without label: %+v", n)
		}

		// If we come past one that's uninitialized, this must be a (harmless) forward jump
		// but if it's set to nonlooping the label must have preceded this goto.
		if n.Left.Sym.Label == &nonlooping {
			n.Left.Sym.Label = &looping
		}
	}

	e.escloopdepth(n.Left)
	e.escloopdepth(n.Right)
	e.escloopdepthlist(n.List)
	e.escloopdepthlist(n.Nbody)
	e.escloopdepthlist(n.Rlist)
}

func (e *EscState) esclist(l Nodes, parent *Node) {
	for _, n := range l.Slice() {
		e.esc(n, parent)
	}
}

func (e *EscState) esc(n *Node, parent *Node) {
	if n == nil {
		return
	}

	lno := setlineno(n)

	// ninit logically runs at a different loopdepth than the rest of the for loop.
	e.esclist(n.Ninit, n)

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
				e.nodeEscState(n1.Rlist.First()).Loopdepth = e.loopdepth
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
		e.escassignSinkWhy(n, n, "too large for stack") // TODO category: tooLarge
	}

	e.esc(n.Left, n)
	e.esc(n.Right, n)
	e.esclist(n.Nbody, n)
	e.esclist(n.List, n)
	e.esclist(n.Rlist, n)

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
			e.nodeEscState(n.Left).Loopdepth = e.loopdepth
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
		// else if(n.Left.Sym.Label == nil)
		//	fatal("escape analysis missed or messed up a label: %+N", n);

		n.Left.Sym.Label = nil

	case ORANGE:
		if n.List.Len() >= 2 {
			// Everything but fixed array is a dereference.

			// If fixed array is really the address of fixed array,
			// it is also a dereference, because it is implicitly
			// dereferenced (see #12588)
			if n.Type.IsArray() &&
				!(n.Right.Type.IsPtr() && eqtype(n.Right.Type.Elem(), n.Type)) {
				e.escassignWhyWhere(n.List.Second(), n.Right, "range", n)
			} else {
				e.escassignDereference(n.List.Second(), n.Right, e.stepAssignWhere(n.List.Second(), n.Right, "range-deref", n))
			}
		}

	case OSWITCH:
		if n.Left != nil && n.Left.Op == OTYPESW {
			for _, n2 := range n.List.Slice() {
				// cases
				// n.Left.Right is the argument of the .(type),
				// it.N().Rlist is the variable per case
				if n2.Rlist.Len() != 0 {
					e.escassignWhyWhere(n2.Rlist.First(), n.Left.Right, "switch case", n)
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
				Warnl(n.Lineno, "%v ignoring self-assignment to %S", e.curfnSym(n), n.Left)
			}

			break
		}

		e.escassign(n.Left, n.Right, e.stepAssignWhere(nil, nil, "", n))

	case OAS2: // x,y = a,b
		if n.List.Len() == n.Rlist.Len() {
			rs := n.Rlist.Slice()
			for i, n := range n.List.Slice() {
				e.escassignWhyWhere(n, rs[i], "assign-pair", n)
			}
		}

	case OAS2RECV: // v, ok = <-ch
		e.escassignWhyWhere(n.List.First(), n.Rlist.First(), "assign-pair-receive", n)
	case OAS2MAPR: // v, ok = m[k]
		e.escassignWhyWhere(n.List.First(), n.Rlist.First(), "assign-pair-mapr", n)
	case OAS2DOTTYPE: // v, ok = x.(type)
		e.escassignWhyWhere(n.List.First(), n.Rlist.First(), "assign-pair-dot-type", n)

	case OSEND: // ch <- x
		e.escassignSinkWhy(n, n.Right, "send")

	case ODEFER:
		if e.loopdepth == 1 { // top level
			break
		}
		// arguments leak out of scope
		// TODO: leak to a dummy node instead
		// defer f(x) - f and x escape
		e.escassignSinkWhy(n, n.Left.Left, "defer func")

		e.escassignSinkWhy(n, n.Left.Right, "defer func ...") // ODDDARG for call
		for _, n4 := range n.Left.List.Slice() {
			e.escassignSinkWhy(n, n4, "defer func arg")
		}

	case OPROC:
		// go f(x) - f and x escape
		e.escassignSinkWhy(n, n.Left.Left, "go func")

		e.escassignSinkWhy(n, n.Left.Right, "go func ...") // ODDDARG for call
		for _, n4 := range n.Left.List.Slice() {
			e.escassignSinkWhy(n, n4, "go func arg")
		}

	case OCALLMETH, OCALLFUNC, OCALLINTER:
		e.esccall(n, parent)

		// esccall already done on n.Rlist.First(). tie it's Retval to n.List
	case OAS2FUNC: // x,y = f()
		rs := e.nodeEscState(n.Rlist.First()).Retval.Slice()
		for i, n := range n.List.Slice() {
			if i >= len(rs) {
				break
			}
			e.escassignWhyWhere(n, rs[i], "assign-pair-func-call", n)
		}
		if n.List.Len() != len(rs) {
			Fatalf("esc oas2func")
		}

	case ORETURN:
		retList := n.List
		if retList.Len() == 1 && Curfn.Type.Results().NumFields() > 1 {
			// OAS2FUNC in disguise
			// esccall already done on n.List.First()
			// tie e.nodeEscState(n.List.First()).Retval to Curfn.Func.Dcl PPARAMOUT's
			retList = e.nodeEscState(n.List.First()).Retval
		}

		i := 0
		for _, lrn := range Curfn.Func.Dcl {
			if i >= retList.Len() {
				break
			}
			if lrn.Op != ONAME || lrn.Class != PPARAMOUT {
				continue
			}
			e.escassignWhyWhere(lrn, retList.Index(i), "return", n)
			i++
		}

		if i < retList.Len() {
			Fatalf("esc return list")
		}

		// Argument could leak through recover.
	case OPANIC:
		e.escassignSinkWhy(n, n.Left, "panic")

	case OAPPEND:
		if !n.Isddd {
			for _, nn := range n.List.Slice()[1:] {
				e.escassignSinkWhy(n, nn, "appended to slice") // lose track of assign to dereference
			}
		} else {
			// append(slice1, slice2...) -- slice2 itself does not escape, but contents do.
			slice2 := n.List.Second()
			e.escassignDereference(&e.theSink, slice2, e.stepAssignWhere(n, slice2, "appended slice...", n)) // lose track of assign of dereference
			if Debug['m'] > 3 {
				Warnl(n.Lineno, "%v special treatment of append(slice1, slice2...) %S", e.curfnSym(n), n)
			}
		}
		e.escassignDereference(&e.theSink, n.List.First(), e.stepAssignWhere(n, n.List.First(), "appendee slice", n)) // The original elements are now leaked, too

	case OCOPY:
		e.escassignDereference(&e.theSink, n.Right, e.stepAssignWhere(n, n.Right, "copied slice", n)) // lose track of assign of dereference

	case OCONV, OCONVNOP:
		e.escassignWhyWhere(n, n.Left, "converted", n)

	case OCONVIFACE:
		e.track(n)
		e.escassignWhyWhere(n, n.Left, "interface-converted", n)

	case OARRAYLIT:
		// Link values to array
		for _, n2 := range n.List.Slice() {
			if n2.Op == OKEY {
				n2 = n2.Right
			}
			e.escassign(n, n2, e.stepAssignWhere(n, n2, "array literal element", n))
		}

	case OSLICELIT:
		// Slice is not leaked until proven otherwise
		e.track(n)
		// Link values to slice
		for _, n2 := range n.List.Slice() {
			if n2.Op == OKEY {
				n2 = n2.Right
			}
			e.escassign(n, n2, e.stepAssignWhere(n, n2, "slice literal element", n))
		}

		// Link values to struct.
	case OSTRUCTLIT:
		for _, n6 := range n.List.Slice() {
			e.escassignWhyWhere(n, n6.Left, "struct literal element", n)
		}

	case OPTRLIT:
		e.track(n)

		// Link OSTRUCTLIT to OPTRLIT; if OPTRLIT escapes, OSTRUCTLIT elements do too.
		e.escassignWhyWhere(n, n.Left, "pointer literal [assign]", n)

	case OCALLPART:
		e.track(n)

		// Contents make it to memory, lose track.
		e.escassignSinkWhy(n, n.Left, "call part")

	case OMAPLIT:
		e.track(n)
		// Keys and values make it to memory, lose track.
		for _, n7 := range n.List.Slice() {
			e.escassignSinkWhy(n, n7.Left, "map literal key")
			e.escassignSinkWhy(n, n7.Right, "map literal value")
		}

	case OCLOSURE:
		// Link addresses of captured variables to closure.
		for _, v := range n.Func.Cvars.Slice() {
			if v.Op == OXXX { // unnamed out argument; see dcl.go:/^funcargs
				continue
			}
			a := v.Name.Defn
			if !v.Name.Byval {
				a = nod(OADDR, a, nil)
				a.Lineno = v.Lineno
				e.nodeEscState(a).Loopdepth = e.loopdepth
				a = typecheck(a, Erv)
			}

			e.escassignWhyWhere(n, a, "captured by a closure", n)
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
				if leftE.Loopdepth != 0 {
					nE.Loopdepth = leftE.Loopdepth
				}

			// PPARAM is loop depth 1 always.
			// PPARAMOUT is loop depth 0 for writes
			// but considered loop depth 1 for address-of,
			// so that writing the address of one result
			// to another (or the same) result makes the
			// first result move to the heap.
			case PPARAM, PPARAMOUT:
				nE := e.nodeEscState(n)
				nE.Loopdepth = 1
			}
		}
	}

	lineno = lno
}

// escassignWhyWhere bundles a common case of
// escassign(e, dst, src, e.stepAssignWhere(dst, src, reason, where))
func (e *EscState) escassignWhyWhere(dst, src *Node, reason string, where *Node) {
	var step *EscStep
	if Debug['m'] != 0 {
		step = e.stepAssignWhere(dst, src, reason, where)
	}
	e.escassign(dst, src, step)
}

// escassignSinkWhy bundles a common case of
// escassign(e, &e.theSink, src, e.stepAssign(nil, dst, src, reason))
func (e *EscState) escassignSinkWhy(dst, src *Node, reason string) {
	var step *EscStep
	if Debug['m'] != 0 {
		step = e.stepAssign(nil, dst, src, reason)
	}
	e.escassign(&e.theSink, src, step)
}

// escassignSinkWhyWhere is escassignSinkWhy but includes a call site
// for accurate location reporting.
func (e *EscState) escassignSinkWhyWhere(dst, src *Node, reason string, call *Node) {
	var step *EscStep
	if Debug['m'] != 0 {
		step = e.stepAssignWhere(dst, src, reason, call)
	}
	e.escassign(&e.theSink, src, step)
}

// Assert that expr somehow gets assigned to dst, if non nil.  for
// dst==nil, any name node expr still must be marked as being
// evaluated in curfn.	For expr==nil, dst must still be examined for
// evaluations inside it (e.g *f(x) = y)
func (e *EscState) escassign(dst, src *Node, step *EscStep) {
	if isblank(dst) || dst == nil || src == nil || src.Op == ONONAME || src.Op == OXXX {
		return
	}

	if Debug['m'] > 2 {
		fmt.Printf("%v:[%d] %v escassign: %S(%0j)[%v] = %S(%0j)[%v]\n",
			linestr(lineno), e.loopdepth, funcSym(Curfn),
			dst, dst, dst.Op,
			src, src, src.Op)
	}

	setlineno(dst)

	originalDst := dst
	dstwhy := "assigned"

	// Analyze lhs of assignment.
	// Replace dst with &e.theSink if we can't track it.
	switch dst.Op {
	default:
		Dump("dst", dst)
		Fatalf("escassign: unexpected dst")

	case OARRAYLIT,
		OSLICELIT,
		OCLOSURE,
		OCONV,
		OCONVIFACE,
		OCONVNOP,
		OMAPLIT,
		OSTRUCTLIT,
		OPTRLIT,
		ODDDARG,
		OCALLPART:

	case ONAME:
		if dst.Class == PEXTERN {
			dstwhy = "assigned to top level variable"
			dst = &e.theSink
		}

	case ODOT: // treat "dst.x = src" as "dst = src"
		e.escassign(dst.Left, src, e.stepAssign(step, originalDst, src, "dot-equals"))
		return

	case OINDEX:
		if dst.Left.Type.IsArray() {
			e.escassign(dst.Left, src, e.stepAssign(step, originalDst, src, "array-element-equals"))
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
		e.escassign(&e.theSink, dst.Right, e.stepAssign(nil, originalDst, src, "key of map put"))
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
		OSLICELIT,
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
		e.escflows(dst, src, e.stepAssign(step, originalDst, src, dstwhy))

	case OCLOSURE:
		// OCLOSURE is lowered to OPTRLIT,
		// insert OADDR to account for the additional indirection.
		a := nod(OADDR, src, nil)
		a.Lineno = src.Lineno
		e.nodeEscState(a).Loopdepth = e.nodeEscState(src).Loopdepth
		a.Type = ptrto(src.Type)
		e.escflows(dst, a, e.stepAssign(nil, originalDst, src, dstwhy))

	// Flowing multiple returns to a single dst happens when
	// analyzing "go f(g())": here g() flows to sink (issue 4529).
	case OCALLMETH, OCALLFUNC, OCALLINTER:
		for _, n := range e.nodeEscState(src).Retval.Slice() {
			e.escflows(dst, n, e.stepAssign(nil, originalDst, n, dstwhy))
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
		OSLICE,
		OSLICE3,
		OSLICEARR,
		OSLICE3ARR,
		OSLICESTR:
		// Conversions, field access, slice all preserve the input value.
		e.escassign(dst, src.Left, e.stepAssign(step, originalDst, src, dstwhy))

	case ODOTTYPE,
		ODOTTYPE2:
		if src.Type != nil && !haspointers(src.Type) {
			break
		}
		e.escassign(dst, src.Left, e.stepAssign(step, originalDst, src, dstwhy))

	case OAPPEND:
		// Append returns first argument.
		// Subsequent arguments are already leaked because they are operands to append.
		e.escassign(dst, src.List.First(), e.stepAssign(step, dst, src.List.First(), dstwhy))

	case OINDEX:
		// Index of array preserves input value.
		if src.Left.Type.IsArray() {
			e.escassign(dst, src.Left, e.stepAssign(step, originalDst, src, dstwhy))
		} else {
			e.escflows(dst, src, e.stepAssign(step, originalDst, src, dstwhy))
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
		e.escassign(dst, src.Left, e.stepAssign(step, originalDst, src, dstwhy))

		e.escassign(dst, src.Right, e.stepAssign(step, originalDst, src, dstwhy))
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
func (e *EscState) escassignfromtag(note string, dsts Nodes, src, call *Node) uint16 {
	em := parsetag(note)
	if src.Op == OLITERAL {
		return em
	}

	if Debug['m'] > 3 {
		fmt.Printf("%v::assignfromtag:: src=%S, em=%s\n",
			linestr(lineno), src, describeEscape(em))
	}

	if em == EscUnknown {
		e.escassignSinkWhyWhere(src, src, "passed to call[argument escapes]", call)
		return em
	}

	if em == EscNone {
		return em
	}

	// If content inside parameter (reached via indirection)
	// escapes to heap, mark as such.
	if em&EscContentEscapes != 0 {
		e.escassign(&e.theSink, e.addDereference(src), e.stepAssignWhere(src, src, "passed to call[argument content escapes]", call))
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
			e.escassign(dsts.Index(dstsi), n, e.stepAssignWhere(dsts.Index(dstsi), src, "passed-to-and-returned-from-call", call))
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

func (e *EscState) escassignDereference(dst *Node, src *Node, step *EscStep) {
	if src.Op == OLITERAL {
		return
	}
	e.escassign(dst, e.addDereference(src), step)
}

// addDereference constructs a suitable OIND note applied to src.
// Because this is for purposes of escape accounting, not execution,
// some semantically dubious node combinations are (currently) possible.
func (e *EscState) addDereference(n *Node) *Node {
	ind := nod(OIND, n, nil)
	e.nodeEscState(ind).Loopdepth = e.nodeEscState(n).Loopdepth
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

func (e *EscState) initEscRetval(call *Node, fntype *Type) {
	cE := e.nodeEscState(call)
	cE.Retval.Set(nil) // Suspect this is not nil for indirect calls.
	for i, f := range fntype.Results().Fields().Slice() {
		ret := nod(ONAME, nil, nil)
		buf := fmt.Sprintf(".out%d", i)
		ret.Sym = lookup(buf)
		ret.Type = f.Type
		ret.Class = PAUTO
		ret.Name.Curfn = Curfn
		e.nodeEscState(ret).Loopdepth = e.loopdepth
		ret.Used = true
		ret.Lineno = call.Lineno
		cE.Retval.Append(ret)
	}
}

// This is a bit messier than fortunate, pulled out of esc's big
// switch for clarity. We either have the paramnodes, which may be
// connected to other things through flows or we have the parameter type
// nodes, which may be marked "noescape". Navigating the ast is slightly
// different for methods vs plain functions and for imported vs
// this-package
func (e *EscState) esccall(call *Node, parent *Node) {
	var fntype *Type
	var indirect bool
	var fn *Node
	switch call.Op {
	default:
		Fatalf("esccall")

	case OCALLFUNC:
		fn = call.Left
		fntype = fn.Type
		indirect = fn.Op != ONAME || fn.Class != PFUNC

	case OCALLMETH:
		fn = call.Left.Sym.Def
		if fn != nil {
			fntype = fn.Type
		} else {
			fntype = call.Left.Type
		}

	case OCALLINTER:
		fntype = call.Left.Type
		indirect = true
	}

	argList := call.List
	if argList.Len() == 1 {
		arg := argList.First()
		if arg.Type.IsFuncArgStruct() { // f(g())
			argList = e.nodeEscState(arg).Retval
		}
	}

	args := argList.Slice()

	if indirect {
		// We know nothing!
		// Leak all the parameters
		for _, arg := range args {
			e.escassignSinkWhy(call, arg, "parameter to indirect call")
			if Debug['m'] > 3 {
				fmt.Printf("%v::esccall:: indirect call <- %S, untracked\n", linestr(lineno), arg)
			}
		}
		// Set up bogus outputs
		e.initEscRetval(call, fntype)
		// If there is a receiver, it also leaks to heap.
		if call.Op != OCALLFUNC {
			rf := fntype.Recv()
			r := call.Left.Left
			if haspointers(rf.Type) {
				e.escassignSinkWhy(call, r, "receiver in indirect call")
			}
		} else { // indirect and OCALLFUNC = could be captured variables, too. (#14409)
			rets := e.nodeEscState(call).Retval.Slice()
			for _, ret := range rets {
				e.escassignDereference(ret, fn, e.stepAssignWhere(ret, fn, "captured by called closure", call))
			}
		}
		return
	}

	cE := e.nodeEscState(call)
	if fn != nil && fn.Op == ONAME && fn.Class == PFUNC &&
		fn.Name.Defn != nil && fn.Name.Defn.Nbody.Len() != 0 && fn.Name.Param.Ntype != nil && fn.Name.Defn.Esc < EscFuncTagged {
		if Debug['m'] > 3 {
			fmt.Printf("%v::esccall:: %S in recursive group\n", linestr(lineno), call)
		}

		// function in same mutually recursive group. Incorporate into flow graph.
		//		print("esc local fn: %N\n", fn.Func.Ntype);
		if fn.Name.Defn.Esc == EscFuncUnknown || cE.Retval.Len() != 0 {
			Fatalf("graph inconsistency")
		}

		sawRcvr := false
		for _, n := range fn.Name.Defn.Func.Dcl {
			switch n.Class {
			case PPARAM:
				if call.Op != OCALLFUNC && !sawRcvr {
					e.escassignWhyWhere(n, call.Left.Left, "call receiver", call)
					sawRcvr = true
					continue
				}
				if len(args) == 0 {
					continue
				}
				arg := args[0]
				if n.Isddd && !call.Isddd {
					// Introduce ODDDARG node to represent ... allocation.
					arg = nod(ODDDARG, nil, nil)
					arr := typArray(n.Type.Elem(), int64(len(args)))
					arg.Type = ptrto(arr) // make pointer so it will be tracked
					arg.Lineno = call.Lineno
					e.track(arg)
					call.Right = arg
				}
				e.escassignWhyWhere(n, arg, "arg to recursive call", call) // TODO this message needs help.
				if arg != args[0] {
					// "..." arguments are untracked
					for _, a := range args {
						if Debug['m'] > 3 {
							fmt.Printf("%v::esccall:: ... <- %S, untracked\n", linestr(lineno), a)
						}
						e.escassignSinkWhyWhere(arg, a, "... arg to recursive call", call)
					}
					// No more PPARAM processing, but keep
					// going for PPARAMOUT.
					args = nil
					continue
				}
				args = args[1:]

			case PPARAMOUT:
				cE.Retval.Append(n)
			}
		}

		return
	}

	// Imported or completely analyzed function. Use the escape tags.
	if cE.Retval.Len() != 0 {
		Fatalf("esc already decorated call %+v\n", call)
	}

	if Debug['m'] > 3 {
		fmt.Printf("%v::esccall:: %S not recursive\n", linestr(lineno), call)
	}

	// set up out list on this call node with dummy auto ONAMES in the current (calling) function.
	e.initEscRetval(call, fntype)

	//	print("esc analyzed fn: %#N (%+T) returning (%+H)\n", fn, fntype, e.nodeEscState(call).Retval);

	// Receiver.
	if call.Op != OCALLFUNC {
		rf := fntype.Recv()
		r := call.Left.Left
		if haspointers(rf.Type) {
			e.escassignfromtag(rf.Note, cE.Retval, r, call)
		}
	}

	var arg *Node
	var note string
	param, it := iterFields(fntype.Params())
	i := 0
	for ; i < len(args); i++ {
		arg = args[i]
		note = param.Note
		if param.Isddd && !call.Isddd {
			// Introduce ODDDARG node to represent ... allocation.
			arg = nod(ODDDARG, nil, nil)
			arg.Lineno = call.Lineno
			arr := typArray(param.Type.Elem(), int64(len(args)-i))
			arg.Type = ptrto(arr) // make pointer so it will be tracked
			e.track(arg)
			call.Right = arg
		}

		if haspointers(param.Type) {
			if e.escassignfromtag(note, cE.Retval, arg, call)&EscMask == EscNone && parent.Op != ODEFER && parent.Op != OPROC {
				a := arg
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
				// arg.Esc == EscNone means that arg does not escape the current function.
				// arg.Noescape = true here means that arg does not escape this statement
				// in the current function.
				case OCALLPART,
					OCLOSURE,
					ODDDARG,
					OARRAYLIT,
					OSLICELIT,
					OPTRLIT,
					OSTRUCTLIT:
					a.Noescape = true
				}
			}
		}

		if arg != args[i] {
			// This occurs when function parameter field Isddd and call not Isddd
			break
		}

		if note == uintptrEscapesTag {
			e.escassignSinkWhy(arg, arg, "escaping uintptr")
		}

		param = it.Next()
	}

	// Store arguments into slice for ... arg.
	for ; i < len(args); i++ {
		if Debug['m'] > 3 {
			fmt.Printf("%v::esccall:: ... <- %S\n", linestr(lineno), args[i])
		}
		if note == uintptrEscapesTag {
			e.escassignSinkWhyWhere(arg, args[i], "arg to uintptrescapes ...", call)
		} else {
			e.escassignWhyWhere(arg, args[i], "arg to ...", call)
		}
	}
}

// escflows records the link src->dst in dst, throwing out some quick wins,
// and also ensuring that dst is noted as a flow destination.
func (e *EscState) escflows(dst, src *Node, why *EscStep) {
	if dst == nil || src == nil || dst == src {
		return
	}

	// Don't bother building a graph for scalars.
	if src.Type != nil && !haspointers(src.Type) {
		return
	}

	if Debug['m'] > 3 {
		fmt.Printf("%v::flows:: %S <- %S\n", linestr(lineno), dst, src)
	}

	dstE := e.nodeEscState(dst)
	if len(dstE.Flowsrc) == 0 {
		e.dsts = append(e.dsts, dst)
		e.dstcount++
	}

	e.edgecount++

	if why == nil {
		dstE.Flowsrc = append(dstE.Flowsrc, EscStep{src: src})
	} else {
		starwhy := *why
		starwhy.src = src // TODO: need to reconcile this w/ needs of explanations.
		dstE.Flowsrc = append(dstE.Flowsrc, starwhy)
	}
}

// Whenever we hit a reference node, the level goes up by one, and whenever
// we hit an OADDR, the level goes down by one. as long as we're on a level > 0
// finding an OADDR just means we're following the upstream of a dereference,
// so this address doesn't leak (yet).
// If level == 0, it means the /value/ of this node can reach the root of this flood.
// so if this node is an OADDR, its argument should be marked as escaping iff
// its currfn/e.loopdepth are different from the flood's root.
// Once an object has been moved to the heap, all of its upstream should be considered
// escaping to the global scope.
func (e *EscState) escflood(dst *Node) {
	switch dst.Op {
	case ONAME, OCLOSURE:
	default:
		return
	}

	dstE := e.nodeEscState(dst)
	if Debug['m'] > 2 {
		fmt.Printf("\nescflood:%d: dst %S scope:%v[%d]\n", e.walkgen, dst, e.curfnSym(dst), dstE.Loopdepth)
	}

	for i := range dstE.Flowsrc {
		e.walkgen++
		s := &dstE.Flowsrc[i]
		s.parent = nil
		e.escwalk(levelFrom(0), dst, s.src, s)
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
		where := step.where
		if nextDest != nil {
			dst = nextDest.src
		}
		if where == nil {
			where = dst
		}
		Warnl(src.Lineno, "\tfrom %v (%s) at %s", dst, step.why, where.Line())
	}
	for step := step0; step != nil && step.busy; step = step.parent {
		step.busy = false
	}
}

const NOTALOOPDEPTH = -1

func (e *EscState) escwalk(level Level, dst *Node, src *Node, step *EscStep) {
	e.escwalkBody(level, dst, src, step, NOTALOOPDEPTH)
}

func (e *EscState) escwalkBody(level Level, dst *Node, src *Node, step *EscStep, extraloopdepth int32) {
	if src.Op == OLITERAL {
		return
	}
	srcE := e.nodeEscState(src)
	if srcE.Walkgen == e.walkgen {
		// Esclevels are vectors, do not compare as integers,
		// and must use "min" of old and new to guarantee
		// convergence.
		level = level.min(srcE.Level)
		if level == srcE.Level {
			// Have we been here already with an extraloopdepth,
			// or is the extraloopdepth provided no improvement on
			// what's already been seen?
			if srcE.Maxextraloopdepth >= extraloopdepth || srcE.Loopdepth >= extraloopdepth {
				return
			}
			srcE.Maxextraloopdepth = extraloopdepth
		}
	} else { // srcE.Walkgen < e.walkgen -- first time, reset this.
		srcE.Maxextraloopdepth = NOTALOOPDEPTH
	}

	srcE.Walkgen = e.walkgen
	srcE.Level = level
	modSrcLoopdepth := srcE.Loopdepth

	if extraloopdepth > modSrcLoopdepth {
		modSrcLoopdepth = extraloopdepth
	}

	if Debug['m'] > 2 {
		fmt.Printf("escwalk: level:%d depth:%d %.*s op=%v %S(%0j) scope:%v[%d] extraloopdepth=%v\n",
			level, e.pdepth, e.pdepth, "\t\t\t\t\t\t\t\t\t\t", src.Op, src, src, e.curfnSym(src), srcE.Loopdepth, extraloopdepth)
	}

	e.pdepth++

	// Input parameter flowing to output parameter?
	var leaks bool
	var osrcesc uint16 // used to prevent duplicate error messages

	dstE := e.nodeEscState(dst)
	if funcOutputAndInput(dst, src) && src.Esc&EscMask < EscHeap && dst.Esc != EscHeap {
		// This case handles:
		// 1. return in
		// 2. return &in
		// 3. tmp := in; return &tmp
		// 4. return *in
		if Debug['m'] != 0 {
			if Debug['m'] <= 2 {
				Warnl(src.Lineno, "leaking param: %S to result %v level=%v", src, dst.Sym, level.int())
				step.describe(src)
			} else {
				Warnl(src.Lineno, "leaking param: %S to result %v level=%v", src, dst.Sym, level)
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
		src.Op == ONAME && src.Class == PPARAM && src.Esc&EscMask < EscHeap &&
		level.int() > 0 {
		src.Esc = escMax(EscContentEscapes|src.Esc, EscNone)
		if Debug['m'] != 0 {
			Warnl(src.Lineno, "mark escaped content: %S", src)
			step.describe(src)
		}
	}

	leaks = level.int() <= 0 && level.guaranteedDereference() <= 0 && dstE.Loopdepth < modSrcLoopdepth
	leaks = leaks || level.int() <= 0 && dst.Esc&EscMask == EscHeap

	osrcesc = src.Esc
	switch src.Op {
	case ONAME:
		if src.Class == PPARAM && (leaks || dstE.Loopdepth < 0) && src.Esc&EscMask < EscHeap {
			if level.guaranteedDereference() > 0 {
				src.Esc = escMax(EscContentEscapes|src.Esc, EscNone)
				if Debug['m'] != 0 {
					if Debug['m'] <= 2 {
						if osrcesc != src.Esc {
							Warnl(src.Lineno, "leaking param content: %S", src)
							step.describe(src)
						}
					} else {
						Warnl(src.Lineno, "leaking param content: %S level=%v dst.eld=%v src.eld=%v dst=%S",
							src, level, dstE.Loopdepth, modSrcLoopdepth, dst)
					}
				}
			} else {
				src.Esc = EscHeap
				if Debug['m'] != 0 {
					if Debug['m'] <= 2 {
						Warnl(src.Lineno, "leaking param: %S", src)
						step.describe(src)
					} else {
						Warnl(src.Lineno, "leaking param: %S level=%v dst.eld=%v src.eld=%v dst=%S",
							src, level, dstE.Loopdepth, modSrcLoopdepth, dst)
					}
				}
			}
		}

		// Treat a captured closure variable as equivalent to the
		// original variable.
		if src.isClosureVar() {
			if leaks && Debug['m'] != 0 {
				Warnl(src.Lineno, "leaking closure reference %S", src)
				step.describe(src)
			}
			e.escwalk(level, dst, src.Name.Defn, e.stepWalk(dst, src.Name.Defn, "closure-var", step))
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
					Warnl(src.Lineno, "%S escapes to heap, level=%v, dst=%v dst.eld=%v, src.eld=%v",
						p, level, dst, dstE.Loopdepth, modSrcLoopdepth)
				} else {
					Warnl(src.Lineno, "%S escapes to heap", p)
					step.describe(src)
				}
			}
			addrescapes(src.Left)
			e.escwalkBody(level.dec(), dst, src.Left, e.stepWalk(dst, src.Left, why, step), modSrcLoopdepth)
			extraloopdepth = modSrcLoopdepth // passes to recursive case, seems likely a no-op
		} else {
			e.escwalk(level.dec(), dst, src.Left, e.stepWalk(dst, src.Left, why, step))
		}

	case OAPPEND:
		e.escwalk(level, dst, src.List.First(), e.stepWalk(dst, src.List.First(), "append-first-arg", step))

	case ODDDARG:
		if leaks {
			src.Esc = EscHeap
			if Debug['m'] != 0 && osrcesc != src.Esc {
				Warnl(src.Lineno, "%S escapes to heap", src)
				step.describe(src)
			}
			extraloopdepth = modSrcLoopdepth
		}
		// similar to a slice arraylit and its args.
		level = level.dec()

	case OSLICELIT:
		for _, n1 := range src.List.Slice() {
			if n1.Op == OKEY {
				n1 = n1.Right
			}
			e.escwalk(level.dec(), dst, n1, e.stepWalk(dst, n1, "slice-literal-element", step))
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
				Warnl(src.Lineno, "%S escapes to heap", src)
				step.describe(src)
			}
			extraloopdepth = modSrcLoopdepth
		}

	case ODOT,
		ODOTTYPE:
		e.escwalk(level, dst, src.Left, e.stepWalk(dst, src.Left, "dot", step))

	case
		OSLICE,
		OSLICEARR,
		OSLICE3,
		OSLICE3ARR,
		OSLICESTR:
		e.escwalk(level, dst, src.Left, e.stepWalk(dst, src.Left, "slice", step))

	case OINDEX:
		if src.Left.Type.IsArray() {
			e.escwalk(level, dst, src.Left, e.stepWalk(dst, src.Left, "fixed-array-index-of", step))
			break
		}
		fallthrough

	case ODOTPTR:
		e.escwalk(level.inc(), dst, src.Left, e.stepWalk(dst, src.Left, "dot of pointer", step))
	case OINDEXMAP:
		e.escwalk(level.inc(), dst, src.Left, e.stepWalk(dst, src.Left, "map index", step))
	case OIND:
		e.escwalk(level.inc(), dst, src.Left, e.stepWalk(dst, src.Left, "indirection", step))

	// In this case a link went directly to a call, but should really go
	// to the dummy .outN outputs that were created for the call that
	// themselves link to the inputs with levels adjusted.
	// See e.g. #10466
	// This can only happen with functions returning a single result.
	case OCALLMETH, OCALLFUNC, OCALLINTER:
		if srcE.Retval.Len() != 0 {
			if Debug['m'] > 2 {
				fmt.Printf("%v:[%d] dst %S escwalk replace src: %S with %S\n",
					linestr(lineno), e.loopdepth,
					dst, src, srcE.Retval.First())
			}
			src = srcE.Retval.First()
			srcE = e.nodeEscState(src)
		}
	}

recurse:
	level = level.copy()

	for i := range srcE.Flowsrc {
		s := &srcE.Flowsrc[i]
		s.parent = step
		e.escwalkBody(level, dst, s.src, s, extraloopdepth)
		s.parent = nil
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

func (e *EscState) esctag(fn *Node) {
	fn.Esc = EscFuncTagged

	name := func(s *Sym, narg int) string {
		if s != nil {
			return s.Name
		}
		return fmt.Sprintf("arg#%d", narg)
	}

	// External functions are assumed unsafe,
	// unless //go:noescape is given before the declaration.
	if fn.Nbody.Len() == 0 {
		if fn.Noescape {
			for _, f := range fn.Type.Params().Fields().Slice() {
				if haspointers(f.Type) {
					f.Note = mktag(EscNone)
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
		for _, f := range fn.Type.Params().Fields().Slice() {
			narg++
			if f.Type.Etype == TUINTPTR {
				if Debug['m'] != 0 {
					Warnl(fn.Lineno, "%v assuming %v is unsafe uintptr", funcSym(fn), name(f.Sym, narg))
				}
				f.Note = unsafeUintptrTag
			}
		}

		return
	}

	if fn.Func.Pragma&UintptrEscapes != 0 {
		narg := 0
		for _, f := range fn.Type.Params().Fields().Slice() {
			narg++
			if f.Type.Etype == TUINTPTR {
				if Debug['m'] != 0 {
					Warnl(fn.Lineno, "%v marking %v as escaping uintptr", funcSym(fn), name(f.Sym, narg))
				}
				f.Note = uintptrEscapesTag
			}

			if f.Isddd && f.Type.Elem().Etype == TUINTPTR {
				// final argument is ...uintptr.
				if Debug['m'] != 0 {
					Warnl(fn.Lineno, "%v marking %v as escaping ...uintptr", funcSym(fn), name(f.Sym, narg))
				}
				f.Note = uintptrEscapesTag
			}
		}
	}

	for _, ln := range fn.Func.Dcl {
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

		case EscHeap: // touched by escflood, moved to heap
		}
	}
}
