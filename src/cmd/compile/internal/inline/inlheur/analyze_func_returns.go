// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inlheur

import (
	"cmd/compile/internal/ir"
	"fmt"
	"go/constant"
	"go/token"
	"os"
)

// resultsAnalyzer stores state information for the process of
// computing flags/properties for the return values of a specific Go
// function, as part of inline heuristics synthesis.
type resultsAnalyzer struct {
	fname           string
	props           []ResultPropBits
	values          []resultVal
	inlineMaxBudget int
	*nameFinder
}

// resultVal captures information about a specific result returned from
// the function we're analyzing; we are interested in cases where
// the func always returns the same constant, or always returns
// the same function, etc. This container stores info on a the specific
// scenarios we're looking for.
type resultVal struct {
	cval    constant.Value
	fn      *ir.Name
	fnClo   bool
	top     bool
	derived bool // see deriveReturnFlagsFromCallee below
}

// addResultsAnalyzer creates a new resultsAnalyzer helper object for
// the function fn, appends it to the analyzers list, and returns the
// new list. If the function in question doesn't have any returns (or
// any interesting returns) then the analyzer list is left as is, and
// the result flags in "fp" are updated accordingly.
func addResultsAnalyzer(fn *ir.Func, analyzers []propAnalyzer, fp *FuncProps, inlineMaxBudget int, nf *nameFinder) []propAnalyzer {
	ra, props := makeResultsAnalyzer(fn, inlineMaxBudget, nf)
	if ra != nil {
		analyzers = append(analyzers, ra)
	} else {
		fp.ResultFlags = props
	}
	return analyzers
}

// makeResultsAnalyzer creates a new helper object to analyze results
// in function fn. If the function doesn't have any interesting
// results, a nil helper is returned along with a set of default
// result flags for the func.
func makeResultsAnalyzer(fn *ir.Func, inlineMaxBudget int, nf *nameFinder) (*resultsAnalyzer, []ResultPropBits) {
	results := fn.Type().Results()
	if len(results) == 0 {
		return nil, nil
	}
	props := make([]ResultPropBits, len(results))
	if fn.Inl == nil {
		return nil, props
	}
	vals := make([]resultVal, len(results))
	interestingToAnalyze := false
	for i := range results {
		rt := results[i].Type
		if !rt.IsScalar() && !rt.HasNil() {
			// existing properties not applicable here (for things
			// like structs, arrays, slices, etc).
			continue
		}
		// set the "top" flag (as in "top element of data flow lattice")
		// meaning "we have no info yet, but we might later on".
		vals[i].top = true
		interestingToAnalyze = true
	}
	if !interestingToAnalyze {
		return nil, props
	}
	ra := &resultsAnalyzer{
		props:           props,
		values:          vals,
		inlineMaxBudget: inlineMaxBudget,
		nameFinder:      nf,
	}
	return ra, nil
}

// setResults transfers the calculated result properties for this
// function to 'funcProps'.
func (ra *resultsAnalyzer) setResults(funcProps *FuncProps) {
	// Promote ResultAlwaysSameFunc to ResultAlwaysSameInlinableFunc
	for i := range ra.values {
		if ra.props[i] == ResultAlwaysSameFunc && !ra.values[i].derived {
			f := ra.values[i].fn.Func
			// HACK: in order to allow for call site score
			// adjustments, we used a relaxed inline budget in
			// determining inlinability. For the check below, however,
			// we want to know is whether the func in question is
			// likely to be inlined, as opposed to whether it might
			// possibly be inlined if all the right score adjustments
			// happened, so do a simple check based on the cost.
			if f.Inl != nil && f.Inl.Cost <= int32(ra.inlineMaxBudget) {
				ra.props[i] = ResultAlwaysSameInlinableFunc
			}
		}
	}
	funcProps.ResultFlags = ra.props
}

func (ra *resultsAnalyzer) pessimize() {
	for i := range ra.props {
		ra.props[i] = ResultNoInfo
	}
}

func (ra *resultsAnalyzer) nodeVisitPre(n ir.Node) {
}

func (ra *resultsAnalyzer) nodeVisitPost(n ir.Node) {
	if len(ra.values) == 0 {
		return
	}
	if n.Op() != ir.ORETURN {
		return
	}
	if debugTrace&debugTraceResults != 0 {
		fmt.Fprintf(os.Stderr, "=+= returns nodevis %v %s\n",
			ir.Line(n), n.Op().String())
	}

	// No support currently for named results, so if we see an empty
	// "return" stmt, be conservative.
	rs := n.(*ir.ReturnStmt)
	if len(rs.Results) != len(ra.values) {
		ra.pessimize()
		return
	}
	for i, r := range rs.Results {
		ra.analyzeResult(i, r)
	}
}

// analyzeResult examines the expression 'n' being returned as the
// 'ii'th argument in some return statement to see whether has
// interesting characteristics (for example, returns a constant), then
// applies a dataflow "meet" operation to combine this result with any
// previous result (for the given return slot) that we've already
// processed.
func (ra *resultsAnalyzer) analyzeResult(ii int, n ir.Node) {
	isAllocMem := ra.isAllocatedMem(n)
	isConcConvItf := ra.isConcreteConvIface(n)
	constVal := ra.constValue(n)
	isConst := (constVal != nil)
	isNil := ra.isNil(n)
	rfunc := ra.funcName(n)
	isFunc := (rfunc != nil)
	isClo := (rfunc != nil && rfunc.Func.OClosure != nil)
	curp := ra.props[ii]
	dprops, isDerivedFromCall := ra.deriveReturnFlagsFromCallee(n)
	newp := ResultNoInfo
	var newcval constant.Value
	var newfunc *ir.Name

	if debugTrace&debugTraceResults != 0 {
		fmt.Fprintf(os.Stderr, "=-= %v: analyzeResult n=%s ismem=%v isconcconv=%v isconst=%v isnil=%v isfunc=%v isclo=%v\n", ir.Line(n), n.Op().String(), isAllocMem, isConcConvItf, isConst, isNil, isFunc, isClo)
	}

	if ra.values[ii].top {
		ra.values[ii].top = false
		// this is the first return we've seen; record
		// whatever properties it has.
		switch {
		case isAllocMem:
			newp = ResultIsAllocatedMem
		case isConcConvItf:
			newp = ResultIsConcreteTypeConvertedToInterface
		case isFunc:
			newp = ResultAlwaysSameFunc
			newfunc = rfunc
		case isConst:
			newp = ResultAlwaysSameConstant
			newcval = constVal
		case isNil:
			newp = ResultAlwaysSameConstant
			newcval = nil
		case isDerivedFromCall:
			newp = dprops
			ra.values[ii].derived = true
		}
	} else {
		if !ra.values[ii].derived {
			// this is not the first return we've seen; apply
			// what amounts of a "meet" operator to combine
			// the properties we see here with what we saw on
			// the previous returns.
			switch curp {
			case ResultIsAllocatedMem:
				if isAllocMem {
					newp = ResultIsAllocatedMem
				}
			case ResultIsConcreteTypeConvertedToInterface:
				if isConcConvItf {
					newp = ResultIsConcreteTypeConvertedToInterface
				}
			case ResultAlwaysSameConstant:
				if isNil && ra.values[ii].cval == nil {
					newp = ResultAlwaysSameConstant
					newcval = nil
				} else if isConst && constant.Compare(constVal, token.EQL, ra.values[ii].cval) {
					newp = ResultAlwaysSameConstant
					newcval = constVal
				}
			case ResultAlwaysSameFunc:
				if isFunc && isSameFuncName(rfunc, ra.values[ii].fn) {
					newp = ResultAlwaysSameFunc
					newfunc = rfunc
				}
			}
		}
	}
	ra.values[ii].fn = newfunc
	ra.values[ii].fnClo = isClo
	ra.values[ii].cval = newcval
	ra.props[ii] = newp

	if debugTrace&debugTraceResults != 0 {
		fmt.Fprintf(os.Stderr, "=-= %v: analyzeResult newp=%s\n",
			ir.Line(n), newp)
	}
}

// deriveReturnFlagsFromCallee tries to set properties for a given
// return result where we're returning call expression; return value
// is a return property value and a boolean indicating whether the
// prop is valid. Examples:
//
//	func foo() int { return bar() }
//	func bar() int { return 42 }
//	func blix() int { return 43 }
//	func two(y int) int {
//	  if y < 0 { return bar() } else { return blix() }
//	}
//
// Since "foo" always returns the result of a call to "bar", we can
// set foo's return property to that of bar. In the case of "two", however,
// even though each return path returns a constant, we don't know
// whether the constants are identical, hence we need to be conservative.
func (ra *resultsAnalyzer) deriveReturnFlagsFromCallee(n ir.Node) (ResultPropBits, bool) {
	if n.Op() != ir.OCALLFUNC {
		return 0, false
	}
	ce := n.(*ir.CallExpr)
	if ce.Fun.Op() != ir.ONAME {
		return 0, false
	}
	called := ir.StaticValue(ce.Fun)
	if called.Op() != ir.ONAME {
		return 0, false
	}
	cname := ra.funcName(called)
	if cname == nil {
		return 0, false
	}
	calleeProps := propsForFunc(cname.Func)
	if calleeProps == nil {
		return 0, false
	}
	if len(calleeProps.ResultFlags) != 1 {
		return 0, false
	}
	return calleeProps.ResultFlags[0], true
}
