// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inlheur

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"fmt"
	"os"
)

// funcFlagsAnalyzer computes the "Flags" value for the FuncProps
// object we're computing. The main item of interest here is "nstate",
// which stores the disposition of a given ir Node with respect to the
// flags/properties we're trying to compute.
type funcFlagsAnalyzer struct {
	fn     *ir.Func
	nstate map[ir.Node]pstate
	noInfo bool // set if we see something inscrutable/un-analyzable
}

// pstate keeps track of the disposition of a given node and its
// children with respect to panic/exit calls.
type pstate int

const (
	psNoInfo     pstate = iota // nothing interesting about this node
	psCallsPanic               // node causes call to panic or os.Exit
	psMayReturn                // executing node may trigger a "return" stmt
	psTop                      // dataflow lattice "top" element
)

func makeFuncFlagsAnalyzer(fn *ir.Func) *funcFlagsAnalyzer {
	return &funcFlagsAnalyzer{
		fn:     fn,
		nstate: make(map[ir.Node]pstate),
	}
}

// setResults transfers func flag results to 'funcProps'.
func (ffa *funcFlagsAnalyzer) setResults(funcProps *FuncProps) {
	var rv FuncPropBits
	if !ffa.noInfo && ffa.stateForList(ffa.fn.Body) == psCallsPanic {
		rv = FuncPropNeverReturns
	}
	// This is slightly hacky and not at all required, but include a
	// special case for main.main, which often ends in a call to
	// os.Exit. People who write code like this (very common I
	// imagine)
	//
	//   func main() {
	//     rc = perform()
	//     ...
	//     foo()
	//     os.Exit(rc)
	//   }
	//
	// will be constantly surprised when foo() is inlined in many
	// other spots in the program but not in main().
	if isMainMain(ffa.fn) {
		rv &^= FuncPropNeverReturns
	}
	funcProps.Flags = rv
}

func (ffa *funcFlagsAnalyzer) getState(n ir.Node) pstate {
	return ffa.nstate[n]
}

func (ffa *funcFlagsAnalyzer) setState(n ir.Node, st pstate) {
	if st != psNoInfo {
		ffa.nstate[n] = st
	}
}

func (ffa *funcFlagsAnalyzer) updateState(n ir.Node, st pstate) {
	if st == psNoInfo {
		delete(ffa.nstate, n)
	} else {
		ffa.nstate[n] = st
	}
}

func (ffa *funcFlagsAnalyzer) panicPathTable() map[ir.Node]pstate {
	return ffa.nstate
}

// blockCombine merges together states as part of a linear sequence of
// statements, where 'pred' and 'succ' are analysis results for a pair
// of consecutive statements. Examples:
//
//	case 1:             case 2:
//	    panic("foo")      if q { return x }        <-pred
//	    return x          panic("boo")             <-succ
//
// In case 1, since the pred state is "always panic" it doesn't matter
// what the succ state is, hence the state for the combination of the
// two blocks is "always panics". In case 2, because there is a path
// to return that avoids the panic in succ, the state for the
// combination of the two statements is "may return".
func blockCombine(pred, succ pstate) pstate {
	switch succ {
	case psTop:
		return pred
	case psMayReturn:
		if pred == psCallsPanic {
			return psCallsPanic
		}
		return psMayReturn
	case psNoInfo:
		return pred
	case psCallsPanic:
		if pred == psMayReturn {
			return psMayReturn
		}
		return psCallsPanic
	}
	panic("should never execute")
}

// branchCombine combines two states at a control flow branch point where
// either p1 or p2 executes (as in an "if" statement).
func branchCombine(p1, p2 pstate) pstate {
	if p1 == psCallsPanic && p2 == psCallsPanic {
		return psCallsPanic
	}
	if p1 == psMayReturn || p2 == psMayReturn {
		return psMayReturn
	}
	return psNoInfo
}

// stateForList walks through a list of statements and computes the
// state/disposition for the entire list as a whole, as well
// as updating disposition of intermediate nodes.
func (ffa *funcFlagsAnalyzer) stateForList(list ir.Nodes) pstate {
	st := psTop
	// Walk the list backwards so that we can update the state for
	// earlier list elements based on what we find out about their
	// successors. Example:
	//
	//        if ... {
	//  L10:    foo()
	//  L11:    <stmt>
	//  L12:    panic(...)
	//        }
	//
	// After combining the dispositions for line 11 and 12, we want to
	// update the state for the call at line 10 based on that combined
	// disposition (if L11 has no path to "return", then the call at
	// line 10 will be on a panic path).
	for i := len(list) - 1; i >= 0; i-- {
		n := list[i]
		psi := ffa.getState(n)
		if debugTrace&debugTraceFuncFlags != 0 {
			fmt.Fprintf(os.Stderr, "=-= %v: stateForList n=%s ps=%s\n",
				ir.Line(n), n.Op().String(), psi.String())
		}
		st = blockCombine(psi, st)
		ffa.updateState(n, st)
	}
	if st == psTop {
		st = psNoInfo
	}
	return st
}

func isMainMain(fn *ir.Func) bool {
	s := fn.Sym()
	return (s.Pkg.Name == "main" && s.Name == "main")
}

func isWellKnownFunc(s *types.Sym, pkg, name string) bool {
	return s.Pkg.Path == pkg && s.Name == name
}

// isExitCall reports TRUE if the node itself is an unconditional
// call to os.Exit(), a panic, or a function that does likewise.
func isExitCall(n ir.Node) bool {
	if n.Op() != ir.OCALLFUNC {
		return false
	}
	cx := n.(*ir.CallExpr)
	name := ir.StaticCalleeName(cx.Fun)
	if name == nil {
		return false
	}
	s := name.Sym()
	if isWellKnownFunc(s, "os", "Exit") ||
		isWellKnownFunc(s, "runtime", "throw") {
		return true
	}
	if funcProps := propsForFunc(name.Func); funcProps != nil {
		if funcProps.Flags&FuncPropNeverReturns != 0 {
			return true
		}
	}
	return name.Func.NeverReturns()
}

// pessimize is called to record the fact that we saw something in the
// function that renders it entirely impossible to analyze.
func (ffa *funcFlagsAnalyzer) pessimize() {
	ffa.noInfo = true
}

// shouldVisit reports TRUE if this is an interesting node from the
// perspective of computing function flags. NB: due to the fact that
// ir.CallExpr implements the Stmt interface, we wind up visiting
// a lot of nodes that we don't really need to, but these can
// simply be screened out as part of the visit.
func shouldVisit(n ir.Node) bool {
	_, isStmt := n.(ir.Stmt)
	return n.Op() != ir.ODCL &&
		(isStmt || n.Op() == ir.OCALLFUNC || n.Op() == ir.OPANIC)
}

// nodeVisitPost helps implement the propAnalyzer interface; when
// called on a given node, it decides the disposition of that node
// based on the state(s) of the node's children.
func (ffa *funcFlagsAnalyzer) nodeVisitPost(n ir.Node) {
	if debugTrace&debugTraceFuncFlags != 0 {
		fmt.Fprintf(os.Stderr, "=+= nodevis %v %s should=%v\n",
			ir.Line(n), n.Op().String(), shouldVisit(n))
	}
	if !shouldVisit(n) {
		return
	}
	var st pstate
	switch n.Op() {
	case ir.OCALLFUNC:
		if isExitCall(n) {
			st = psCallsPanic
		}
	case ir.OPANIC:
		st = psCallsPanic
	case ir.ORETURN:
		st = psMayReturn
	case ir.OBREAK, ir.OCONTINUE:
		// FIXME: this handling of break/continue is sub-optimal; we
		// have them as "mayReturn" in order to help with this case:
		//
		//   for {
		//     if q() { break }
		//     panic(...)
		//   }
		//
		// where the effect of the 'break' is to cause the subsequent
		// panic to be skipped. One possible improvement would be to
		// track whether the currently enclosing loop is a "for {" or
		// a for/range with condition, then use mayReturn only for the
		// former. Note also that "break X" or "continue X" is treated
		// the same as "goto", since we don't have a good way to track
		// the target of the branch.
		st = psMayReturn
		n := n.(*ir.BranchStmt)
		if n.Label != nil {
			ffa.pessimize()
		}
	case ir.OBLOCK:
		n := n.(*ir.BlockStmt)
		st = ffa.stateForList(n.List)
	case ir.OCASE:
		if ccst, ok := n.(*ir.CaseClause); ok {
			st = ffa.stateForList(ccst.Body)
		} else if ccst, ok := n.(*ir.CommClause); ok {
			st = ffa.stateForList(ccst.Body)
		} else {
			panic("unexpected")
		}
	case ir.OIF:
		n := n.(*ir.IfStmt)
		st = branchCombine(ffa.stateForList(n.Body), ffa.stateForList(n.Else))
	case ir.OFOR:
		// Treat for { XXX } like a block.
		// Treat for <cond> { XXX } like an if statement with no else.
		n := n.(*ir.ForStmt)
		bst := ffa.stateForList(n.Body)
		if n.Cond == nil {
			st = bst
		} else {
			if bst == psMayReturn {
				st = psMayReturn
			}
		}
	case ir.ORANGE:
		// Treat for range { XXX } like an if statement with no else.
		n := n.(*ir.RangeStmt)
		if ffa.stateForList(n.Body) == psMayReturn {
			st = psMayReturn
		}
	case ir.OGOTO:
		// punt if we see even one goto. if we built a control
		// flow graph we could do more, but this is just a tree walk.
		ffa.pessimize()
	case ir.OSELECT:
		// process selects for "may return" but not "always panics",
		// the latter case seems very improbable.
		n := n.(*ir.SelectStmt)
		if len(n.Cases) != 0 {
			st = psTop
			for _, c := range n.Cases {
				st = branchCombine(ffa.stateForList(c.Body), st)
			}
		}
	case ir.OSWITCH:
		n := n.(*ir.SwitchStmt)
		if len(n.Cases) != 0 {
			st = psTop
			for _, c := range n.Cases {
				st = branchCombine(ffa.stateForList(c.Body), st)
			}
		}

		st, fall := psTop, psNoInfo
		for i := len(n.Cases) - 1; i >= 0; i-- {
			cas := n.Cases[i]
			cst := ffa.stateForList(cas.Body)
			endsInFallthrough := false
			if len(cas.Body) != 0 {
				endsInFallthrough = cas.Body[0].Op() == ir.OFALL
			}
			if endsInFallthrough {
				cst = blockCombine(cst, fall)
			}
			st = branchCombine(st, cst)
			fall = cst
		}
	case ir.OFALL:
		// Not important.
	case ir.ODCLFUNC, ir.ORECOVER, ir.OAS, ir.OAS2, ir.OAS2FUNC, ir.OASOP,
		ir.OPRINTLN, ir.OPRINT, ir.OLABEL, ir.OCALLINTER, ir.ODEFER,
		ir.OSEND, ir.ORECV, ir.OSELRECV2, ir.OGO, ir.OAPPEND, ir.OAS2DOTTYPE,
		ir.OAS2MAPR, ir.OGETG, ir.ODELETE, ir.OINLMARK, ir.OAS2RECV,
		ir.OMIN, ir.OMAX, ir.OMAKE, ir.OGETCALLERSP:
		// these should all be benign/uninteresting
	case ir.OTAILCALL, ir.OJUMPTABLE, ir.OTYPESW:
		// don't expect to see these at all.
		base.Fatalf("unexpected op %s in func %s",
			n.Op().String(), ir.FuncName(ffa.fn))
	default:
		base.Fatalf("%v: unhandled op %s in func %v",
			ir.Line(n), n.Op().String(), ir.FuncName(ffa.fn))
	}
	if debugTrace&debugTraceFuncFlags != 0 {
		fmt.Fprintf(os.Stderr, "=-= %v: visit n=%s returns %s\n",
			ir.Line(n), n.Op().String(), st.String())
	}
	ffa.setState(n, st)
}

func (ffa *funcFlagsAnalyzer) nodeVisitPre(n ir.Node) {
}
