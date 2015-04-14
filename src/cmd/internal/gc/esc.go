// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/obj"
	"fmt"
	"strings"
)

// Run analysis on minimal sets of mutually recursive functions
// or single non-recursive functions, bottom up.
//
// Finding these sets is finding strongly connected components
// in the static call graph.  The algorithm for doing that is taken
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

// TODO(rsc): Look into using a map[*Node]bool instead of walkgen,
// to allow analysis passes to use walkgen themselves.

type bottomUpVisitor struct {
	analyze  func(*NodeList, bool)
	visitgen uint32
	stack    *NodeList
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
func visitBottomUp(list *NodeList, analyze func(list *NodeList, recursive bool)) {
	for l := list; l != nil; l = l.Next {
		l.N.Walkgen = 0
	}

	var v bottomUpVisitor
	v.analyze = analyze
	for l := list; l != nil; l = l.Next {
		if l.N.Op == ODCLFUNC && l.N.Curfn == nil {
			v.visit(l.N)
		}
	}

	for l := list; l != nil; l = l.Next {
		l.N.Walkgen = 0
	}
}

func (v *bottomUpVisitor) visit(n *Node) uint32 {
	if n.Walkgen > 0 {
		// already visited
		return n.Walkgen
	}

	v.visitgen++
	n.Walkgen = v.visitgen
	v.visitgen++
	min := v.visitgen

	l := new(NodeList)
	l.Next = v.stack
	l.N = n
	v.stack = l
	min = v.visitcodelist(n.Nbody, min)
	if (min == n.Walkgen || min == n.Walkgen+1) && n.Curfn == nil {
		// This node is the root of a strongly connected component.

		// The original min passed to visitcodelist was n->walkgen+1.
		// If visitcodelist found its way back to n->walkgen, then this
		// block is a set of mutually recursive functions.
		// Otherwise it's just a lone function that does not recurse.
		recursive := min == n.Walkgen

		// Remove connected component from stack.
		// Mark walkgen so that future visits return a large number
		// so as not to affect the caller's min.
		block := v.stack

		var l *NodeList
		for l = v.stack; l.N != n; l = l.Next {
			l.N.Walkgen = ^uint32(0)
		}
		n.Walkgen = ^uint32(0)
		v.stack = l.Next
		l.Next = nil

		// Run escape analysis on this set of functions.
		v.analyze(block, recursive)
	}

	return min
}

func (v *bottomUpVisitor) visitcodelist(l *NodeList, min uint32) uint32 {
	for ; l != nil; l = l.Next {
		min = v.visitcode(l.N, min)
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
	min = v.visitcode(n.Ntest, min)
	min = v.visitcode(n.Nincr, min)
	min = v.visitcodelist(n.Nbody, min)
	min = v.visitcodelist(n.Nelse, min)
	min = v.visitcodelist(n.Rlist, min)

	if n.Op == OCALLFUNC || n.Op == OCALLMETH {
		fn := n.Left
		if n.Op == OCALLMETH {
			fn = n.Left.Right.Sym.Def
		}
		if fn != nil && fn.Op == ONAME && fn.Class == PFUNC && fn.Defn != nil {
			m := v.visit(fn.Defn)
			if m < min {
				min = m
			}
		}
	}

	if n.Op == OCLOSURE {
		m := v.visit(n.Closure)
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
// pointer-containing nodes and store them in dst->escflowsrc.  For
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
// then the value can stay on the stack.  If the value new(T) does
// not escape, then new(T) can be rewritten into a stack allocation.
// The same is true of slice literals.
//
// If optimizations are disabled (-N), this code is not used.
// Instead, the compiler assumes that any value whose address
// is taken without being immediately dereferenced
// needs to be moved to the heap, and new(T) and slice
// literals are always real allocations.

func escapes(all *NodeList) {
	visitBottomUp(all, escAnalyze)
}

const (
	EscFuncUnknown = 0 + iota
	EscFuncPlanned
	EscFuncStarted
	EscFuncTagged
)

type EscState struct {
	// Fake node that all
	//   - return values and output variables
	//   - parameters on imported functions not marked 'safe'
	//   - assignments to global variables
	// flow to.
	theSink Node

	// If an analyzed function is recorded to return
	// pieces obtained via indirection from a parameter,
	// and later there is a call f(x) to that function,
	// we create a link funcParam <- x to record that fact.
	// The funcParam node is handled specially in escflood.
	funcParam Node

	dsts      *NodeList // all dst nodes
	loopdepth int       // for detecting nested loop scopes
	pdepth    int       // for debug printing in recursions.
	dstcount  int       // diagnostic
	edgecount int       // diagnostic
	noesc     *NodeList // list of possible non-escaping nodes, for printing
	recursive bool      // recursive function or group of mutually recursive functions.
}

var tags [16]*string

// mktag returns the string representation for an escape analysis tag.
func mktag(mask int) *string {
	switch mask & EscMask {
	case EscNone, EscReturn:
		break

	default:
		Fatal("escape mktag")
	}

	mask >>= EscBits

	if mask < len(tags) && tags[mask] != nil {
		return tags[mask]
	}

	s := fmt.Sprintf("esc:0x%x", mask)
	if mask < len(tags) {
		tags[mask] = &s
	}
	return &s
}

func parsetag(note *string) int {
	if note == nil || !strings.HasPrefix(*note, "esc:") {
		return EscUnknown
	}
	em := atoi((*note)[4:])
	if em == 0 {
		return EscNone
	}
	return EscReturn | em<<EscBits
}

func escAnalyze(all *NodeList, recursive bool) {
	var es EscState
	e := &es
	e.theSink.Op = ONAME
	e.theSink.Orig = &e.theSink
	e.theSink.Class = PEXTERN
	e.theSink.Sym = Lookup(".sink")
	e.theSink.Escloopdepth = -1
	e.recursive = recursive

	e.funcParam.Op = ONAME
	e.funcParam.Orig = &e.funcParam
	e.funcParam.Class = PAUTO
	e.funcParam.Sym = Lookup(".param")
	e.funcParam.Escloopdepth = 10000000

	for l := all; l != nil; l = l.Next {
		if l.N.Op == ODCLFUNC {
			l.N.Esc = EscFuncPlanned
		}
	}

	// flow-analyze functions
	for l := all; l != nil; l = l.Next {
		if l.N.Op == ODCLFUNC {
			escfunc(e, l.N)
		}
	}

	// print("escapes: %d e->dsts, %d edges\n", e->dstcount, e->edgecount);

	// visit the upstream of each dst, mark address nodes with
	// addrescapes, mark parameters unsafe
	for l := e.dsts; l != nil; l = l.Next {
		escflood(e, l.N)
	}

	// for all top level functions, tag the typenodes corresponding to the param nodes
	for l := all; l != nil; l = l.Next {
		if l.N.Op == ODCLFUNC {
			esctag(e, l.N)
		}
	}

	if Debug['m'] != 0 {
		for l := e.noesc; l != nil; l = l.Next {
			if l.N.Esc == EscNone {
				var tmp *Sym
				if l.N.Curfn != nil && l.N.Curfn.Nname != nil {
					tmp = l.N.Curfn.Nname.Sym
				} else {
					tmp = nil
				}
				Warnl(int(l.N.Lineno), "%v %v does not escape", Sconv(tmp, 0), Nconv(l.N, obj.FmtShort))
			}
		}
	}
}

func escfunc(e *EscState, func_ *Node) {
	//	print("escfunc %N %s\n", func->nname, e->recursive?"(recursive)":"");

	if func_.Esc != 1 {
		Fatal("repeat escfunc %v", Nconv(func_.Nname, 0))
	}
	func_.Esc = EscFuncStarted

	saveld := e.loopdepth
	e.loopdepth = 1
	savefn := Curfn
	Curfn = func_

	for ll := Curfn.Func.Dcl; ll != nil; ll = ll.Next {
		if ll.N.Op != ONAME {
			continue
		}
		switch ll.N.Class {
		// out params are in a loopdepth between the sink and all local variables
		case PPARAMOUT:
			ll.N.Escloopdepth = 0

		case PPARAM:
			ll.N.Escloopdepth = 1
			if ll.N.Type != nil && !haspointers(ll.N.Type) {
				break
			}
			if Curfn.Nbody == nil && !Curfn.Noescape {
				ll.N.Esc = EscHeap
			} else {
				ll.N.Esc = EscNone // prime for escflood later
			}
			e.noesc = list(e.noesc, ll.N)
		}
	}

	// in a mutually recursive group we lose track of the return values
	if e.recursive {
		for ll := Curfn.Func.Dcl; ll != nil; ll = ll.Next {
			if ll.N.Op == ONAME && ll.N.Class == PPARAMOUT {
				escflows(e, &e.theSink, ll.N)
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
// and set it to one of the following two.  Then in esc we'll clear it again.
var looping Label

var nonlooping Label

func escloopdepthlist(e *EscState, l *NodeList) {
	for ; l != nil; l = l.Next {
		escloopdepth(e, l.N)
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
			Fatal("esc:label without label: %v", Nconv(n, obj.FmtSign))
		}

		// Walk will complain about this label being already defined, but that's not until
		// after escape analysis. in the future, maybe pull label & goto analysis out of walk and put before esc
		// if(n->left->sym->label != nil)
		//	fatal("escape analysis messed up analyzing label: %+N", n);
		n.Left.Sym.Label = &nonlooping

	case OGOTO:
		if n.Left == nil || n.Left.Sym == nil {
			Fatal("esc:goto without label: %v", Nconv(n, obj.FmtSign))
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
	escloopdepth(e, n.Ntest)
	escloopdepth(e, n.Nincr)
	escloopdepthlist(e, n.Nbody)
	escloopdepthlist(e, n.Nelse)
	escloopdepthlist(e, n.Rlist)
}

func esclist(e *EscState, l *NodeList, up *Node) {
	for ; l != nil; l = l.Next {
		esc(e, l.N, up)
	}
}

func esc(e *EscState, n *Node, up *Node) {
	if n == nil {
		return
	}

	lno := int(setlineno(n))

	// ninit logically runs at a different loopdepth than the rest of the for loop.
	esclist(e, n.Ninit, n)

	if n.Op == OFOR || n.Op == ORANGE {
		e.loopdepth++
	}

	// type switch variables have no ODCL.
	// process type switch as declaration.
	// must happen before processing of switch body,
	// so before recursion.
	if n.Op == OSWITCH && n.Ntest != nil && n.Ntest.Op == OTYPESW {
		for ll := n.List; ll != nil; ll = ll.Next { // cases

			// ll->n->nname is the variable per case
			if ll.N.Nname != nil {
				ll.N.Nname.Escloopdepth = e.loopdepth
			}
		}
	}

	esc(e, n.Left, n)
	esc(e, n.Right, n)
	esc(e, n.Ntest, n)
	esc(e, n.Nincr, n)
	esclist(e, n.Nbody, n)
	esclist(e, n.Nelse, n)
	esclist(e, n.List, n)
	esclist(e, n.Rlist, n)

	if n.Op == OFOR || n.Op == ORANGE {
		e.loopdepth--
	}

	if Debug['m'] > 1 {
		var tmp *Sym
		if Curfn != nil && Curfn.Nname != nil {
			tmp = Curfn.Nname.Sym
		} else {
			tmp = nil
		}
		fmt.Printf("%v:[%d] %v esc: %v\n", Ctxt.Line(int(lineno)), e.loopdepth, Sconv(tmp, 0), Nconv(n, 0))
	}

	switch n.Op {
	// Record loop depth at declaration.
	case ODCL:
		if n.Left != nil {
			n.Left.Escloopdepth = e.loopdepth
		}

	case OLABEL:
		if n.Left.Sym.Label == &nonlooping {
			if Debug['m'] > 1 {
				fmt.Printf("%v:%v non-looping label\n", Ctxt.Line(int(lineno)), Nconv(n, 0))
			}
		} else if n.Left.Sym.Label == &looping {
			if Debug['m'] > 1 {
				fmt.Printf("%v: %v looping label\n", Ctxt.Line(int(lineno)), Nconv(n, 0))
			}
			e.loopdepth++
		}

		// See case OLABEL in escloopdepth above
		// else if(n->left->sym->label == nil)
		//	fatal("escape analysis missed or messed up a label: %+N", n);

		n.Left.Sym.Label = nil

		// Everything but fixed array is a dereference.
	case ORANGE:
		if Isfixedarray(n.Type) && n.List != nil && n.List.Next != nil {
			escassign(e, n.List.Next.N, n.Right)
		}

	case OSWITCH:
		if n.Ntest != nil && n.Ntest.Op == OTYPESW {
			for ll := n.List; ll != nil; ll = ll.Next { // cases

				// ntest->right is the argument of the .(type),
				// ll->n->nname is the variable per case
				escassign(e, ll.N.Nname, n.Ntest.Right)
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
	case OAS, OASOP:
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
				var tmp *Sym
				if n.Curfn != nil && n.Curfn.Nname != nil {
					tmp = n.Curfn.Nname.Sym
				} else {
					tmp = nil
				}
				Warnl(int(n.Lineno), "%v ignoring self-assignment to %v", Sconv(tmp, 0), Nconv(n.Left, obj.FmtShort))
			}

			break
		}

		escassign(e, n.Left, n.Right)

	case OAS2: // x,y = a,b
		if count(n.List) == count(n.Rlist) {
			ll := n.List
			lr := n.Rlist
			for ; ll != nil; ll, lr = ll.Next, lr.Next {
				escassign(e, ll.N, lr.N)
			}
		}

	case OAS2RECV, // v, ok = <-ch
		OAS2MAPR,    // v, ok = m[k]
		OAS2DOTTYPE: // v, ok = x.(type)
		escassign(e, n.List.N, n.Rlist.N)

	case OSEND: // ch <- x
		escassign(e, &e.theSink, n.Right)

	case ODEFER:
		if e.loopdepth == 1 { // top level
			break
		}
		// arguments leak out of scope
		// TODO: leak to a dummy node instead
		fallthrough

	case OPROC:
		// go f(x) - f and x escape
		escassign(e, &e.theSink, n.Left.Left)

		escassign(e, &e.theSink, n.Left.Right) // ODDDARG for call
		for ll := n.Left.List; ll != nil; ll = ll.Next {
			escassign(e, &e.theSink, ll.N)
		}

	case OCALLMETH, OCALLFUNC, OCALLINTER:
		esccall(e, n, up)

		// esccall already done on n->rlist->n. tie it's escretval to n->list
	case OAS2FUNC: // x,y = f()
		lr := n.Rlist.N.Escretval

		var ll *NodeList
		for ll = n.List; lr != nil && ll != nil; lr, ll = lr.Next, ll.Next {
			escassign(e, ll.N, lr.N)
		}
		if lr != nil || ll != nil {
			Fatal("esc oas2func")
		}

	case ORETURN:
		ll := n.List
		if count(n.List) == 1 && Curfn.Type.Outtuple > 1 {
			// OAS2FUNC in disguise
			// esccall already done on n->list->n
			// tie n->list->n->escretval to curfn->dcl PPARAMOUT's
			ll = n.List.N.Escretval
		}

		for lr := Curfn.Func.Dcl; lr != nil && ll != nil; lr = lr.Next {
			if lr.N.Op != ONAME || lr.N.Class != PPARAMOUT {
				continue
			}
			escassign(e, lr.N, ll.N)
			ll = ll.Next
		}

		if ll != nil {
			Fatal("esc return list")
		}

		// Argument could leak through recover.
	case OPANIC:
		escassign(e, &e.theSink, n.Left)

	case OAPPEND:
		if !n.Isddd {
			for ll := n.List.Next; ll != nil; ll = ll.Next {
				escassign(e, &e.theSink, ll.N) // lose track of assign to dereference
			}
		}

	case OCONV, OCONVNOP:
		escassign(e, n, n.Left)

	case OCONVIFACE:
		n.Esc = EscNone // until proven otherwise
		e.noesc = list(e.noesc, n)
		n.Escloopdepth = e.loopdepth
		escassign(e, n, n.Left)

	case OARRAYLIT:
		if Isslice(n.Type) {
			n.Esc = EscNone // until proven otherwise
			e.noesc = list(e.noesc, n)
			n.Escloopdepth = e.loopdepth

			// Values make it to memory, lose track.
			for ll := n.List; ll != nil; ll = ll.Next {
				escassign(e, &e.theSink, ll.N.Right)
			}
		} else {
			// Link values to array.
			for ll := n.List; ll != nil; ll = ll.Next {
				escassign(e, n, ll.N.Right)
			}
		}

		// Link values to struct.
	case OSTRUCTLIT:
		for ll := n.List; ll != nil; ll = ll.Next {
			escassign(e, n, ll.N.Right)
		}

	case OPTRLIT:
		n.Esc = EscNone // until proven otherwise
		e.noesc = list(e.noesc, n)
		n.Escloopdepth = e.loopdepth

		// Link OSTRUCTLIT to OPTRLIT; if OPTRLIT escapes, OSTRUCTLIT elements do too.
		escassign(e, n, n.Left)

	case OCALLPART:
		n.Esc = EscNone // until proven otherwise
		e.noesc = list(e.noesc, n)
		n.Escloopdepth = e.loopdepth

		// Contents make it to memory, lose track.
		escassign(e, &e.theSink, n.Left)

	case OMAPLIT:
		n.Esc = EscNone // until proven otherwise
		e.noesc = list(e.noesc, n)
		n.Escloopdepth = e.loopdepth

		// Keys and values make it to memory, lose track.
		for ll := n.List; ll != nil; ll = ll.Next {
			escassign(e, &e.theSink, ll.N.Left)
			escassign(e, &e.theSink, ll.N.Right)
		}

		// Link addresses of captured variables to closure.
	case OCLOSURE:
		var a *Node
		var v *Node
		for ll := n.Func.Cvars; ll != nil; ll = ll.Next {
			v = ll.N
			if v.Op == OXXX { // unnamed out argument; see dcl.c:/^funcargs
				continue
			}
			a = v.Closure
			if !v.Byval {
				a = Nod(OADDR, a, nil)
				a.Lineno = v.Lineno
				a.Escloopdepth = e.loopdepth
				typecheck(&a, Erv)
			}

			escassign(e, n, a)
		}
		fallthrough

		// fallthrough
	case OMAKECHAN,
		OMAKEMAP,
		OMAKESLICE,
		ONEW,
		OARRAYRUNESTR,
		OARRAYBYTESTR,
		OSTRARRAYRUNE,
		OSTRARRAYBYTE,
		ORUNESTR:
		n.Escloopdepth = e.loopdepth

		n.Esc = EscNone // until proven otherwise
		e.noesc = list(e.noesc, n)

	case OADDSTR:
		n.Escloopdepth = e.loopdepth
		n.Esc = EscNone // until proven otherwise
		e.noesc = list(e.noesc, n)

	// Arguments of OADDSTR do not escape.

	case OADDR:
		n.Esc = EscNone // until proven otherwise
		e.noesc = list(e.noesc, n)

		// current loop depth is an upper bound on actual loop depth
		// of addressed value.
		n.Escloopdepth = e.loopdepth

		// for &x, use loop depth of x if known.
		// it should always be known, but if not, be conservative
		// and keep the current loop depth.
		if n.Left.Op == ONAME {
			switch n.Left.Class {
			case PAUTO:
				if n.Left.Escloopdepth != 0 {
					n.Escloopdepth = n.Left.Escloopdepth
				}

				// PPARAM is loop depth 1 always.
			// PPARAMOUT is loop depth 0 for writes
			// but considered loop depth 1 for address-of,
			// so that writing the address of one result
			// to another (or the same) result makes the
			// first result move to the heap.
			case PPARAM, PPARAMOUT:
				n.Escloopdepth = 1
			}
		}
	}

	lineno = int32(lno)
}

// Assert that expr somehow gets assigned to dst, if non nil.  for
// dst==nil, any name node expr still must be marked as being
// evaluated in curfn.	For expr==nil, dst must still be examined for
// evaluations inside it (e.g *f(x) = y)
func escassign(e *EscState, dst *Node, src *Node) {
	if isblank(dst) || dst == nil || src == nil || src.Op == ONONAME || src.Op == OXXX {
		return
	}

	if Debug['m'] > 1 {
		var tmp *Sym
		if Curfn != nil && Curfn.Nname != nil {
			tmp = Curfn.Nname.Sym
		} else {
			tmp = nil
		}
		fmt.Printf("%v:[%d] %v escassign: %v(%v) = %v(%v)\n", Ctxt.Line(int(lineno)), e.loopdepth, Sconv(tmp, 0), Nconv(dst, obj.FmtShort), Jconv(dst, obj.FmtShort), Nconv(src, obj.FmtShort), Jconv(src, obj.FmtShort))
	}

	setlineno(dst)

	// Analyze lhs of assignment.
	// Replace dst with e->theSink if we can't track it.
	switch dst.Op {
	default:
		Dump("dst", dst)
		Fatal("escassign: unexpected dst")

	case OARRAYLIT,
		OCLOSURE,
		OCONV,
		OCONVIFACE,
		OCONVNOP,
		OMAPLIT,
		OSTRUCTLIT,
		OPTRLIT,
		OCALLPART:
		break

	case ONAME:
		if dst.Class == PEXTERN {
			dst = &e.theSink
		}

	case ODOT: // treat "dst.x  = src" as "dst = src"
		escassign(e, dst.Left, src)

		return

	case OINDEX:
		if Isfixedarray(dst.Left.Type) {
			escassign(e, dst.Left, src)
			return
		}

		dst = &e.theSink // lose track of dereference

	case OIND, ODOTPTR:
		dst = &e.theSink // lose track of dereference

		// lose track of key and value
	case OINDEXMAP:
		escassign(e, &e.theSink, dst.Right)

		dst = &e.theSink
	}

	lno := int(setlineno(src))
	e.pdepth++

	switch src.Op {
	case OADDR, // dst = &x
		OIND,    // dst = *x
		ODOTPTR, // dst = (*x).f
		ONAME,
		OPARAM,
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
		escflows(e, dst, src)

	case OCLOSURE:
		// OCLOSURE is lowered to OPTRLIT,
		// insert OADDR to account for the additional indirection.
		a := Nod(OADDR, src, nil)
		a.Lineno = src.Lineno
		a.Escloopdepth = src.Escloopdepth
		a.Type = Ptrto(src.Type)
		escflows(e, dst, a)

		// Flowing multiple returns to a single dst happens when
	// analyzing "go f(g())": here g() flows to sink (issue 4529).
	case OCALLMETH, OCALLFUNC, OCALLINTER:
		for ll := src.Escretval; ll != nil; ll = ll.Next {
			escflows(e, dst, ll.N)
		}

		// A non-pointer escaping from a struct does not concern us.
	case ODOT:
		if src.Type != nil && !haspointers(src.Type) {
			break
		}
		fallthrough

		// Conversions, field access, slice all preserve the input value.
	// fallthrough
	case OCONV,
		OCONVNOP,
		ODOTMETH,
		// treat recv.meth as a value with recv in it, only happens in ODEFER and OPROC
		// iface.method already leaks iface in esccall, no need to put in extra ODOTINTER edge here
		ODOTTYPE,
		ODOTTYPE2,
		OSLICE,
		OSLICE3,
		OSLICEARR,
		OSLICE3ARR,
		OSLICESTR:
		// Conversions, field access, slice all preserve the input value.
		escassign(e, dst, src.Left)

	case OAPPEND:
		// Append returns first argument.
		escassign(e, dst, src.List.N)

	case OINDEX:
		// Index of array preserves input value.
		if Isfixedarray(src.Left.Type) {
			escassign(e, dst, src.Left)
		}

		// Might be pointer arithmetic, in which case
	// the operands flow into the result.
	// TODO(rsc): Decide what the story is here.  This is unsettling.
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
		escassign(e, dst, src.Left)

		escassign(e, dst, src.Right)
	}

	e.pdepth--
	lineno = int32(lno)
}

func escassignfromtag(e *EscState, note *string, dsts *NodeList, src *Node) int {
	em := parsetag(note)

	if em == EscUnknown {
		escassign(e, &e.theSink, src)
		return em
	}

	if em == EscNone {
		return em
	}

	// If content inside parameter (reached via indirection)
	// escapes back to results, mark as such.
	if em&EscContentEscapes != 0 {
		escassign(e, &e.funcParam, src)
	}

	em0 := em
	for em >>= EscReturnBits; em != 0 && dsts != nil; em, dsts = em>>1, dsts.Next {
		if em&1 != 0 {
			escassign(e, dsts.N, src)
		}
	}

	if em != 0 && dsts == nil {
		Fatal("corrupt esc tag %q or messed up escretval list\n", note)
	}
	return em0
}

// This is a bit messier than fortunate, pulled out of esc's big
// switch for clarity.	We either have the paramnodes, which may be
// connected to other things through flows or we have the parameter type
// nodes, which may be marked "noescape". Navigating the ast is slightly
// different for methods vs plain functions and for imported vs
// this-package
func esccall(e *EscState, n *Node, up *Node) {
	var fntype *Type

	var fn *Node
	switch n.Op {
	default:
		Fatal("esccall")

	case OCALLFUNC:
		fn = n.Left
		fntype = fn.Type

	case OCALLMETH:
		fn = n.Left.Right.Sym.Def
		if fn != nil {
			fntype = fn.Type
		} else {
			fntype = n.Left.Type
		}

	case OCALLINTER:
		fntype = n.Left.Type
	}

	ll := n.List
	if n.List != nil && n.List.Next == nil {
		a := n.List.N
		if a.Type.Etype == TSTRUCT && a.Type.Funarg != 0 { // f(g()).
			ll = a.Escretval
		}
	}

	if fn != nil && fn.Op == ONAME && fn.Class == PFUNC && fn.Defn != nil && fn.Defn.Nbody != nil && fn.Ntype != nil && fn.Defn.Esc < EscFuncTagged {
		// function in same mutually recursive group.  Incorporate into flow graph.
		//		print("esc local fn: %N\n", fn->ntype);
		if fn.Defn.Esc == EscFuncUnknown || n.Escretval != nil {
			Fatal("graph inconsistency")
		}

		// set up out list on this call node
		for lr := fn.Ntype.Rlist; lr != nil; lr = lr.Next {
			n.Escretval = list(n.Escretval, lr.N.Left) // type.rlist ->  dclfield -> ONAME (PPARAMOUT)
		}

		// Receiver.
		if n.Op != OCALLFUNC {
			escassign(e, fn.Ntype.Left.Left, n.Left.Left)
		}

		var src *Node
		for lr := fn.Ntype.List; ll != nil && lr != nil; ll, lr = ll.Next, lr.Next {
			src = ll.N
			if lr.N.Isddd && !n.Isddd {
				// Introduce ODDDARG node to represent ... allocation.
				src = Nod(ODDDARG, nil, nil)

				src.Type = typ(TARRAY)
				src.Type.Type = lr.N.Type.Type
				src.Type.Bound = int64(count(ll))
				src.Type = Ptrto(src.Type) // make pointer so it will be tracked
				src.Escloopdepth = e.loopdepth
				src.Lineno = n.Lineno
				src.Esc = EscNone // until we find otherwise
				e.noesc = list(e.noesc, src)
				n.Right = src
			}

			if lr.N.Left != nil {
				escassign(e, lr.N.Left, src)
			}
			if src != ll.N {
				break
			}
		}

		// "..." arguments are untracked
		for ; ll != nil; ll = ll.Next {
			escassign(e, &e.theSink, ll.N)
		}

		return
	}

	// Imported or completely analyzed function.  Use the escape tags.
	if n.Escretval != nil {
		Fatal("esc already decorated call %v\n", Nconv(n, obj.FmtSign))
	}

	// set up out list on this call node with dummy auto ONAMES in the current (calling) function.
	i := 0

	var src *Node
	var buf string
	for t := getoutargx(fntype).Type; t != nil; t = t.Down {
		src = Nod(ONAME, nil, nil)
		buf = fmt.Sprintf(".dum%d", i)
		i++
		src.Sym = Lookup(buf)
		src.Type = t.Type
		src.Class = PAUTO
		src.Curfn = Curfn
		src.Escloopdepth = e.loopdepth
		src.Used = true
		src.Lineno = n.Lineno
		n.Escretval = list(n.Escretval, src)
	}

	//	print("esc analyzed fn: %#N (%+T) returning (%+H)\n", fn, fntype, n->escretval);

	// Receiver.
	if n.Op != OCALLFUNC {
		t := getthisx(fntype).Type
		src := n.Left.Left
		if haspointers(t.Type) {
			escassignfromtag(e, t.Note, n.Escretval, src)
		}
	}

	var a *Node
	for t := getinargx(fntype).Type; ll != nil; ll = ll.Next {
		src = ll.N
		if t.Isddd && !n.Isddd {
			// Introduce ODDDARG node to represent ... allocation.
			src = Nod(ODDDARG, nil, nil)

			src.Escloopdepth = e.loopdepth
			src.Lineno = n.Lineno
			src.Type = typ(TARRAY)
			src.Type.Type = t.Type.Type
			src.Type.Bound = int64(count(ll))
			src.Type = Ptrto(src.Type) // make pointer so it will be tracked
			src.Esc = EscNone          // until we find otherwise
			e.noesc = list(e.noesc, src)
			n.Right = src
		}

		if haspointers(t.Type) {
			if escassignfromtag(e, t.Note, n.Escretval, src) == EscNone && up.Op != ODEFER && up.Op != OPROC {
				a = src
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

		if src != ll.N {
			break
		}
		t = t.Down
	}

	// "..." arguments are untracked
	for ; ll != nil; ll = ll.Next {
		escassign(e, &e.theSink, ll.N)
	}
}

// Store the link src->dst in dst, throwing out some quick wins.
func escflows(e *EscState, dst *Node, src *Node) {
	if dst == nil || src == nil || dst == src {
		return
	}

	// Don't bother building a graph for scalars.
	if src.Type != nil && !haspointers(src.Type) {
		return
	}

	if Debug['m'] > 2 {
		fmt.Printf("%v::flows:: %v <- %v\n", Ctxt.Line(int(lineno)), Nconv(dst, obj.FmtShort), Nconv(src, obj.FmtShort))
	}

	if dst.Escflowsrc == nil {
		e.dsts = list(e.dsts, dst)
		e.dstcount++
	}

	e.edgecount++

	dst.Escflowsrc = list(dst.Escflowsrc, src)
}

// Whenever we hit a reference node, the level goes up by one, and whenever
// we hit an OADDR, the level goes down by one. as long as we're on a level > 0
// finding an OADDR just means we're following the upstream of a dereference,
// so this address doesn't leak (yet).
// If level == 0, it means the /value/ of this node can reach the root of this flood.
// so if this node is an OADDR, it's argument should be marked as escaping iff
// it's currfn/e->loopdepth are different from the flood's root.
// Once an object has been moved to the heap, all of it's upstream should be considered
// escaping to the global scope.
func escflood(e *EscState, dst *Node) {
	switch dst.Op {
	case ONAME, OCLOSURE:
		break

	default:
		return
	}

	if Debug['m'] > 1 {
		var tmp *Sym
		if dst.Curfn != nil && dst.Curfn.Nname != nil {
			tmp = dst.Curfn.Nname.Sym
		} else {
			tmp = nil
		}
		fmt.Printf("\nescflood:%d: dst %v scope:%v[%d]\n", walkgen, Nconv(dst, obj.FmtShort), Sconv(tmp, 0), dst.Escloopdepth)
	}

	for l := dst.Escflowsrc; l != nil; l = l.Next {
		walkgen++
		escwalk(e, 0, dst, l.N)
	}
}

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

func escwalk(e *EscState, level int, dst *Node, src *Node) {
	if src.Walkgen == walkgen && src.Esclevel <= int32(level) {
		return
	}
	src.Walkgen = walkgen
	src.Esclevel = int32(level)

	if Debug['m'] > 1 {
		var tmp *Sym
		if src.Curfn != nil && src.Curfn.Nname != nil {
			tmp = src.Curfn.Nname.Sym
		} else {
			tmp = nil
		}
		fmt.Printf("escwalk: level:%d depth:%d %.*s %v(%v) scope:%v[%d]\n", level, e.pdepth, e.pdepth, "\t\t\t\t\t\t\t\t\t\t", Nconv(src, obj.FmtShort), Jconv(src, obj.FmtShort), Sconv(tmp, 0), src.Escloopdepth)
	}

	e.pdepth++

	// Input parameter flowing to output parameter?
	var leaks bool
	if dst.Op == ONAME && dst.Class == PPARAMOUT && dst.Vargen <= 20 {
		if src.Op == ONAME && src.Class == PPARAM && src.Curfn == dst.Curfn && src.Esc != EscScope && src.Esc != EscHeap {
			if level == 0 {
				if Debug['m'] != 0 {
					Warnl(int(src.Lineno), "leaking param: %v to result %v", Nconv(src, obj.FmtShort), Sconv(dst.Sym, 0))
				}
				if src.Esc&EscMask != EscReturn {
					src.Esc = EscReturn
				}
				src.Esc |= 1 << uint((dst.Vargen-1)+EscReturnBits)
				goto recurse
			} else if level > 0 {
				if Debug['m'] != 0 {
					Warnl(int(src.Lineno), "%v leaking param %v content to result %v", Nconv(src.Curfn.Nname, 0), Nconv(src, obj.FmtShort), Sconv(dst.Sym, 0))
				}
				if src.Esc&EscMask != EscReturn {
					src.Esc = EscReturn
				}
				src.Esc |= EscContentEscapes
				goto recurse
			}
		}
	}

	// The second clause is for values pointed at by an object passed to a call
	// that returns something reached via indirect from the object.
	// We don't know which result it is or how many indirects, so we treat it as leaking.
	leaks = level <= 0 && dst.Escloopdepth < src.Escloopdepth || level < 0 && dst == &e.funcParam && haspointers(src.Type)

	switch src.Op {
	case ONAME:
		if src.Class == PPARAM && (leaks || dst.Escloopdepth < 0) && src.Esc != EscHeap {
			src.Esc = EscScope
			if Debug['m'] != 0 {
				Warnl(int(src.Lineno), "leaking param: %v", Nconv(src, obj.FmtShort))
			}
		}

		// Treat a PPARAMREF closure variable as equivalent to the
		// original variable.
		if src.Class == PPARAMREF {
			if leaks && Debug['m'] != 0 {
				Warnl(int(src.Lineno), "leaking closure reference %v", Nconv(src, obj.FmtShort))
			}
			escwalk(e, level, dst, src.Closure)
		}

	case OPTRLIT, OADDR:
		if leaks {
			src.Esc = EscHeap
			addrescapes(src.Left)
			if Debug['m'] != 0 {
				p := src
				if p.Left.Op == OCLOSURE {
					p = p.Left // merely to satisfy error messages in tests
				}
				Warnl(int(src.Lineno), "%v escapes to heap", Nconv(p, obj.FmtShort))
			}
		}

		newlevel := level
		if level > MinLevel {
			newlevel--
		}
		escwalk(e, newlevel, dst, src.Left)

	case OARRAYLIT:
		if Isfixedarray(src.Type) {
			break
		}
		fallthrough

		// fall through
	case ODDDARG,
		OMAKECHAN,
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
			if Debug['m'] != 0 {
				Warnl(int(src.Lineno), "%v escapes to heap", Nconv(src, obj.FmtShort))
			}
		}

	case ODOT,
		OSLICE,
		OSLICEARR,
		OSLICE3,
		OSLICE3ARR,
		OSLICESTR:
		escwalk(e, level, dst, src.Left)

	case OINDEX:
		if Isfixedarray(src.Left.Type) {
			escwalk(e, level, dst, src.Left)
			break
		}
		fallthrough

		// fall through
	case ODOTPTR, OINDEXMAP, OIND:
		newlevel := level

		if level > MinLevel {
			newlevel++
		}
		escwalk(e, newlevel, dst, src.Left)
	}

recurse:
	for ll := src.Escflowsrc; ll != nil; ll = ll.Next {
		escwalk(e, level, dst, ll.N)
	}

	e.pdepth--
}

func esctag(e *EscState, func_ *Node) {
	func_.Esc = EscFuncTagged

	// External functions are assumed unsafe,
	// unless //go:noescape is given before the declaration.
	if func_.Nbody == nil {
		if func_.Noescape {
			for t := getinargx(func_.Type).Type; t != nil; t = t.Down {
				if haspointers(t.Type) {
					t.Note = mktag(EscNone)
				}
			}
		}

		return
	}

	savefn := Curfn
	Curfn = func_

	for ll := Curfn.Func.Dcl; ll != nil; ll = ll.Next {
		if ll.N.Op != ONAME || ll.N.Class != PPARAM {
			continue
		}

		switch ll.N.Esc & EscMask {
		case EscNone, // not touched by escflood
			EscReturn:
			if haspointers(ll.N.Type) { // don't bother tagging for scalars
				ll.N.Paramfld.Note = mktag(int(ll.N.Esc))
			}

		case EscHeap, // touched by escflood, moved to heap
			EscScope: // touched by escflood, value leaves scope
			break
		}
	}

	Curfn = savefn
}
