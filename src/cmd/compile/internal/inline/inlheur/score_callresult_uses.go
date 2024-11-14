// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inlheur

import (
	"cmd/compile/internal/ir"
	"fmt"
	"os"
)

// This file contains code to re-score callsites based on how the
// results of the call were used.  Example:
//
//    func foo() {
//       x, fptr := bar()
//       switch x {
//         case 10: fptr = baz()
//         default: blix()
//       }
//       fptr(100)
//     }
//
// The initial scoring pass will assign a score to "bar()" based on
// various criteria, however once the first pass of scoring is done,
// we look at the flags on the result from bar, and check to see
// how those results are used. If bar() always returns the same constant
// for its first result, and if the variable receiving that result
// isn't redefined, and if that variable feeds into an if/switch
// condition, then we will try to adjust the score for "bar" (on the
// theory that if we inlined, we can constant fold / deadcode).

type resultPropAndCS struct {
	defcs *CallSite
	props ResultPropBits
}

type resultUseAnalyzer struct {
	resultNameTab map[*ir.Name]resultPropAndCS
	fn            *ir.Func
	cstab         CallSiteTab
	*condLevelTracker
}

// rescoreBasedOnCallResultUses examines how call results are used,
// and tries to update the scores of calls based on how their results
// are used in the function.
func (csa *callSiteAnalyzer) rescoreBasedOnCallResultUses(fn *ir.Func, resultNameTab map[*ir.Name]resultPropAndCS, cstab CallSiteTab) {
	enableDebugTraceIfEnv()
	rua := &resultUseAnalyzer{
		resultNameTab:    resultNameTab,
		fn:               fn,
		cstab:            cstab,
		condLevelTracker: new(condLevelTracker),
	}
	var doNode func(ir.Node) bool
	doNode = func { n ->
		rua.nodeVisitPre(n)
		ir.DoChildren(n, doNode)
		rua.nodeVisitPost(n)
		return false
	}
	doNode(fn)
	disableDebugTrace()
}

func (csa *callSiteAnalyzer) examineCallResults(cs *CallSite, resultNameTab map[*ir.Name]resultPropAndCS) map[*ir.Name]resultPropAndCS {
	if debugTrace&debugTraceScoring != 0 {
		fmt.Fprintf(os.Stderr, "=-= examining call results for %q\n",
			EncodeCallSiteKey(cs))
	}

	// Invoke a helper to pick out the specific ir.Name's the results
	// from this call are assigned into, e.g. "x, y := fooBar()". If
	// the call is not part of an assignment statement, or if the
	// variables in question are not newly defined, then we'll receive
	// an empty list here.
	//
	names, autoTemps, props := namesDefined(cs)
	if len(names) == 0 {
		return resultNameTab
	}

	if debugTrace&debugTraceScoring != 0 {
		fmt.Fprintf(os.Stderr, "=-= %d names defined\n", len(names))
	}

	// For each returned value, if the value has interesting
	// properties (ex: always returns the same constant), and the name
	// in question is never redefined, then make an entry in the
	// result table for it.
	const interesting = (ResultIsConcreteTypeConvertedToInterface |
		ResultAlwaysSameConstant | ResultAlwaysSameInlinableFunc | ResultAlwaysSameFunc)
	for idx, n := range names {
		rprop := props.ResultFlags[idx]

		if debugTrace&debugTraceScoring != 0 {
			fmt.Fprintf(os.Stderr, "=-= props for ret %d %q: %s\n",
				idx, n.Sym().Name, rprop.String())
		}

		if rprop&interesting == 0 {
			continue
		}
		if csa.nameFinder.reassigned(n) {
			continue
		}
		if resultNameTab == nil {
			resultNameTab = make(map[*ir.Name]resultPropAndCS)
		} else if _, ok := resultNameTab[n]; ok {
			panic("should never happen")
		}
		entry := resultPropAndCS{
			defcs: cs,
			props: rprop,
		}
		resultNameTab[n] = entry
		if autoTemps[idx] != nil {
			resultNameTab[autoTemps[idx]] = entry
		}
		if debugTrace&debugTraceScoring != 0 {
			fmt.Fprintf(os.Stderr, "=-= add resultNameTab table entry n=%v autotemp=%v props=%s\n", n, autoTemps[idx], rprop.String())
		}
	}
	return resultNameTab
}

// namesDefined returns a list of ir.Name's corresponding to locals
// that receive the results from the call at site 'cs', plus the
// properties object for the called function. If a given result
// isn't cleanly assigned to a newly defined local, the
// slot for that result in the returned list will be nil. Example:
//
//	call                             returned name list
//
//	x := foo()                       [ x ]
//	z, y := bar()                    [ nil, nil ]
//	_, q := baz()                    [ nil, q ]
//
// In the case of a multi-return call, such as "x, y := foo()",
// the pattern we see from the front end will be a call op
// assigning to auto-temps, and then an assignment of the auto-temps
// to the user-level variables. In such cases we return
// first the user-level variable (in the first func result)
// and then the auto-temp name in the second result.
func namesDefined(cs *CallSite) ([]*ir.Name, []*ir.Name, *FuncProps) {
	// If this call doesn't feed into an assignment (and of course not
	// all calls do), then we don't have anything to work with here.
	if cs.Assign == nil {
		return nil, nil, nil
	}
	funcInlHeur, ok := fpmap[cs.Callee]
	if !ok {
		// TODO: add an assert/panic here.
		return nil, nil, nil
	}
	if len(funcInlHeur.props.ResultFlags) == 0 {
		return nil, nil, nil
	}

	// Single return case.
	if len(funcInlHeur.props.ResultFlags) == 1 {
		asgn, ok := cs.Assign.(*ir.AssignStmt)
		if !ok {
			return nil, nil, nil
		}
		// locate name being assigned
		aname, ok := asgn.X.(*ir.Name)
		if !ok {
			return nil, nil, nil
		}
		return []*ir.Name{aname}, []*ir.Name{nil}, funcInlHeur.props
	}

	// Multi-return case
	asgn, ok := cs.Assign.(*ir.AssignListStmt)
	if !ok || !asgn.Def {
		return nil, nil, nil
	}
	userVars := make([]*ir.Name, len(funcInlHeur.props.ResultFlags))
	autoTemps := make([]*ir.Name, len(funcInlHeur.props.ResultFlags))
	for idx, x := range asgn.Lhs {
		if n, ok := x.(*ir.Name); ok {
			userVars[idx] = n
			r := asgn.Rhs[idx]
			if r.Op() == ir.OCONVNOP {
				r = r.(*ir.ConvExpr).X
			}
			if ir.IsAutoTmp(r) {
				autoTemps[idx] = r.(*ir.Name)
			}
			if debugTrace&debugTraceScoring != 0 {
				fmt.Fprintf(os.Stderr, "=-= multi-ret namedef uv=%v at=%v\n",
					x, autoTemps[idx])
			}
		} else {
			return nil, nil, nil
		}
	}
	return userVars, autoTemps, funcInlHeur.props
}

func (rua *resultUseAnalyzer) nodeVisitPost(n ir.Node) {
	rua.condLevelTracker.post(n)
}

func (rua *resultUseAnalyzer) nodeVisitPre(n ir.Node) {
	rua.condLevelTracker.pre(n)
	switch n.Op() {
	case ir.OCALLINTER:
		if debugTrace&debugTraceScoring != 0 {
			fmt.Fprintf(os.Stderr, "=-= rescore examine iface call %v:\n", n)
		}
		rua.callTargetCheckResults(n)
	case ir.OCALLFUNC:
		if debugTrace&debugTraceScoring != 0 {
			fmt.Fprintf(os.Stderr, "=-= rescore examine call %v:\n", n)
		}
		rua.callTargetCheckResults(n)
	case ir.OIF:
		ifst := n.(*ir.IfStmt)
		rua.foldCheckResults(ifst.Cond)
	case ir.OSWITCH:
		swst := n.(*ir.SwitchStmt)
		if swst.Tag != nil {
			rua.foldCheckResults(swst.Tag)
		}

	}
}

// callTargetCheckResults examines a given call to see whether the
// callee expression is potentially an inlinable function returned
// from a potentially inlinable call. Examples:
//
//	Scenario 1: named intermediate
//
//	   fn1 := foo()         conc := bar()
//	   fn1("blah")          conc.MyMethod()
//
//	Scenario 2: returned func or concrete object feeds directly to call
//
//	   foo()("blah")        bar().MyMethod()
//
// In the second case although at the source level the result of the
// direct call feeds right into the method call or indirect call,
// we're relying on the front end having inserted an auto-temp to
// capture the value.
func (rua *resultUseAnalyzer) callTargetCheckResults(call ir.Node) {
	ce := call.(*ir.CallExpr)
	rname := rua.getCallResultName(ce)
	if rname == nil {
		return
	}
	if debugTrace&debugTraceScoring != 0 {
		fmt.Fprintf(os.Stderr, "=-= staticvalue returns %v:\n",
			rname)
	}
	if rname.Class != ir.PAUTO {
		return
	}
	switch call.Op() {
	case ir.OCALLINTER:
		if debugTrace&debugTraceScoring != 0 {
			fmt.Fprintf(os.Stderr, "=-= in %s checking %v for cci prop:\n",
				rua.fn.Sym().Name, rname)
		}
		if cs := rua.returnHasProp(rname, ResultIsConcreteTypeConvertedToInterface); cs != nil {

			adj := returnFeedsConcreteToInterfaceCallAdj
			cs.Score, cs.ScoreMask = adjustScore(adj, cs.Score, cs.ScoreMask)
		}
	case ir.OCALLFUNC:
		if debugTrace&debugTraceScoring != 0 {
			fmt.Fprintf(os.Stderr, "=-= in %s checking %v for samefunc props:\n",
				rua.fn.Sym().Name, rname)
			v, ok := rua.resultNameTab[rname]
			if !ok {
				fmt.Fprintf(os.Stderr, "=-= no entry for %v in rt\n", rname)
			} else {
				fmt.Fprintf(os.Stderr, "=-= props for %v: %q\n", rname, v.props.String())
			}
		}
		if cs := rua.returnHasProp(rname, ResultAlwaysSameInlinableFunc); cs != nil {
			adj := returnFeedsInlinableFuncToIndCallAdj
			cs.Score, cs.ScoreMask = adjustScore(adj, cs.Score, cs.ScoreMask)
		} else if cs := rua.returnHasProp(rname, ResultAlwaysSameFunc); cs != nil {
			adj := returnFeedsFuncToIndCallAdj
			cs.Score, cs.ScoreMask = adjustScore(adj, cs.Score, cs.ScoreMask)

		}
	}
}

// foldCheckResults examines the specified if/switch condition 'cond'
// to see if it refers to locals defined by a (potentially inlinable)
// function call at call site C, and if so, whether 'cond' contains
// only combinations of simple references to all of the names in
// 'names' with selected constants + operators. If these criteria are
// met, then we adjust the score for call site C to reflect the
// fact that inlining will enable deadcode and/or constant propagation.
// Note: for this heuristic to kick in, the names in question have to
// be all from the same callsite. Examples:
//
//	  q, r := baz()	    x, y := foo()
//	  switch q+r {		a, b, c := bar()
//		...			    if x && y && a && b && c {
//	  }					   ...
//					    }
//
// For the call to "baz" above we apply a score adjustment, but not
// for the calls to "foo" or "bar".
func (rua *resultUseAnalyzer) foldCheckResults(cond ir.Node) {
	namesUsed := collectNamesUsed(cond)
	if len(namesUsed) == 0 {
		return
	}
	var cs *CallSite
	for _, n := range namesUsed {
		rpcs, found := rua.resultNameTab[n]
		if !found {
			return
		}
		if cs != nil && rpcs.defcs != cs {
			return
		}
		cs = rpcs.defcs
		if rpcs.props&ResultAlwaysSameConstant == 0 {
			return
		}
	}
	if debugTrace&debugTraceScoring != 0 {
		nls := func(nl []*ir.Name) string {
			r := ""
			for _, n := range nl {
				r += " " + n.Sym().Name
			}
			return r
		}
		fmt.Fprintf(os.Stderr, "=-= calling ShouldFoldIfNameConstant on names={%s} cond=%v\n", nls(namesUsed), cond)
	}

	if !ShouldFoldIfNameConstant(cond, namesUsed) {
		return
	}
	adj := returnFeedsConstToIfAdj
	cs.Score, cs.ScoreMask = adjustScore(adj, cs.Score, cs.ScoreMask)
}

func collectNamesUsed(expr ir.Node) []*ir.Name {
	res := []*ir.Name{}
	ir.Visit(expr, func { n ->
		if n.Op() != ir.ONAME {
			return
		}
		nn := n.(*ir.Name)
		if nn.Class != ir.PAUTO {
			return
		}
		res = append(res, nn)
	})
	return res
}

func (rua *resultUseAnalyzer) returnHasProp(name *ir.Name, prop ResultPropBits) *CallSite {
	v, ok := rua.resultNameTab[name]
	if !ok {
		return nil
	}
	if v.props&prop == 0 {
		return nil
	}
	return v.defcs
}

func (rua *resultUseAnalyzer) getCallResultName(ce *ir.CallExpr) *ir.Name {
	var callTarg ir.Node
	if sel, ok := ce.Fun.(*ir.SelectorExpr); ok {
		// method call
		callTarg = sel.X
	} else if ctarg, ok := ce.Fun.(*ir.Name); ok {
		// regular call
		callTarg = ctarg
	} else {
		return nil
	}
	r := ir.StaticValue(callTarg)
	if debugTrace&debugTraceScoring != 0 {
		fmt.Fprintf(os.Stderr, "=-= staticname on %v returns %v:\n",
			callTarg, r)
	}
	if r.Op() == ir.OCALLFUNC {
		// This corresponds to the "x := foo()" case; here
		// ir.StaticValue has brought us all the way back to
		// the call expression itself. We need to back off to
		// the name defined by the call; do this by looking up
		// the callsite.
		ce := r.(*ir.CallExpr)
		cs, ok := rua.cstab[ce]
		if !ok {
			return nil
		}
		names, _, _ := namesDefined(cs)
		if len(names) == 0 {
			return nil
		}
		return names[0]
	} else if r.Op() == ir.ONAME {
		return r.(*ir.Name)
	}
	return nil
}
