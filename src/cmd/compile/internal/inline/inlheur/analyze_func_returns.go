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

// returnsAnalyzer stores state information for the process of
// computing flags/properties for the return values of a specific Go
// function, as part of inline heuristics synthesis.
type returnsAnalyzer struct {
	fname           string
	props           []ResultPropBits
	values          []resultVal
	canInline       func(*ir.Func)
	inlineMaxBudget int32
}

// resultVal captures information about a specific result returned from
// the function we're analyzing; we are interested in cases where
// the func always returns the same constant, or always returns
// the same function, etc. This container stores info on a the specific
// scenarios we're looking for.
type resultVal struct {
	lit     constant.Value
	fn      *ir.Name
	fnClo   bool
	top     bool
	derived bool // see deriveReturnFlagsFromCallee below
}

func makeResultsAnalyzer(fn *ir.Func, canInline func(*ir.Func), inlineMaxBudget int32) *returnsAnalyzer {
	results := fn.Type().Results()
	props := make([]ResultPropBits, len(results))
	vals := make([]resultVal, len(results))
	for i := range results {
		rt := results[i].Type
		if !rt.IsScalar() && !rt.HasNil() {
			// existing properties not applicable here (for things
			// like structs, arrays, slices, etc).
			props[i] = ResultNoInfo
			continue
		}
		// set the "top" flag (as in "top element of data flow lattice")
		// meaning "we have no info yet, but we might later on".
		vals[i].top = true
	}
	return &returnsAnalyzer{
		props:           props,
		values:          vals,
		canInline:       canInline,
		inlineMaxBudget: inlineMaxBudget,
	}
}

// setResults transfers the calculated result properties for this
// function to 'funcProps'.
func (ra *returnsAnalyzer) setResults(funcProps *FuncProps) {
	// Promote ResultAlwaysSameFunc to ResultAlwaysSameInlinableFunc
	for i := range ra.values {
		if ra.props[i] == ResultAlwaysSameFunc && !ra.values[i].derived {
			f := ra.values[i].fn.Func
			// If the function being returns is a closure that hasn't
			// yet been checked by CanInline, invoke it now. NB: this
			// is hacky, it would be better if things were structured
			// so that all closures were visited ahead of time.
			if ra.values[i].fnClo {
				if f != nil && !f.InlinabilityChecked() {
					ra.canInline(f)
				}
			}
			// HACK: in order to allow for call site score
			// adjustments, we used a relaxed inline budget in
			// determining inlinability. Here what we want to know is
			// whether the func in question is likely to be inlined,
			// as opposed to whether it might possibly be inlined if
			// all the right score adjustments happened, so check the
			// cost here as well.
			if f.Inl != nil && f.Inl.Cost <= ra.inlineMaxBudget {
				ra.props[i] = ResultAlwaysSameInlinableFunc
			}
		}
	}
	funcProps.ResultFlags = ra.props
}

func (ra *returnsAnalyzer) pessimize() {
	for i := range ra.props {
		ra.props[i] = ResultNoInfo
	}
}

func (ra *returnsAnalyzer) nodeVisitPre(n ir.Node) {
}

func (ra *returnsAnalyzer) nodeVisitPost(n ir.Node) {
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

// isFuncName returns the *ir.Name for the func or method
// corresponding to node 'n', along with a boolean indicating success,
// and another boolean indicating whether the func is closure.
func isFuncName(n ir.Node) (*ir.Name, bool, bool) {
	sv := ir.StaticValue(n)
	if sv.Op() == ir.ONAME {
		name := sv.(*ir.Name)
		if name.Sym() != nil && name.Class == ir.PFUNC {
			return name, true, false
		}
	}
	if sv.Op() == ir.OCLOSURE {
		cloex := sv.(*ir.ClosureExpr)
		return cloex.Func.Nname, true, true
	}
	if sv.Op() == ir.OMETHEXPR {
		if mn := ir.MethodExprName(sv); mn != nil {
			return mn, true, false
		}
	}
	return nil, false, false
}

// analyzeResult examines the expression 'n' being returned as the
// 'ii'th argument in some return statement to see whether has
// interesting characteristics (for example, returns a constant), then
// applies a dataflow "meet" operation to combine this result with any
// previous result (for the given return slot) that we've already
// processed.
func (ra *returnsAnalyzer) analyzeResult(ii int, n ir.Node) {
	isAllocMem := isAllocatedMem(n)
	isConcConvItf := isConcreteConvIface(n)
	lit, isConst := isLiteral(n)
	rfunc, isFunc, isClo := isFuncName(n)
	curp := ra.props[ii]
	dprops, isDerivedFromCall := deriveReturnFlagsFromCallee(n)
	newp := ResultNoInfo
	var newlit constant.Value
	var newfunc *ir.Name

	if debugTrace&debugTraceResults != 0 {
		fmt.Fprintf(os.Stderr, "=-= %v: analyzeResult n=%s ismem=%v isconcconv=%v isconst=%v isfunc=%v isclo=%v\n", ir.Line(n), n.Op().String(), isAllocMem, isConcConvItf, isConst, isFunc, isClo)
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
			newlit = lit
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
				if isAllocatedMem(n) {
					newp = ResultIsAllocatedMem
				}
			case ResultIsConcreteTypeConvertedToInterface:
				if isConcreteConvIface(n) {
					newp = ResultIsConcreteTypeConvertedToInterface
				}
			case ResultAlwaysSameConstant:
				if isConst && isSameLiteral(lit, ra.values[ii].lit) {
					newp = ResultAlwaysSameConstant
					newlit = lit
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
	ra.values[ii].lit = newlit
	ra.props[ii] = newp

	if debugTrace&debugTraceResults != 0 {
		fmt.Fprintf(os.Stderr, "=-= %v: analyzeResult newp=%s\n",
			ir.Line(n), newp)
	}
}

func isAllocatedMem(n ir.Node) bool {
	sv := ir.StaticValue(n)
	switch sv.Op() {
	case ir.OMAKESLICE, ir.ONEW, ir.OPTRLIT, ir.OSLICELIT:
		return true
	}
	return false
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
func deriveReturnFlagsFromCallee(n ir.Node) (ResultPropBits, bool) {
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
	cname, isFunc, _ := isFuncName(called)
	if !isFunc {
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

func isLiteral(n ir.Node) (constant.Value, bool) {
	sv := ir.StaticValue(n)
	switch sv.Op() {
	case ir.ONIL:
		return nil, true
	case ir.OLITERAL:
		return sv.Val(), true
	}
	return nil, false
}

// isSameLiteral checks to see if 'v1' and 'v2' correspond to the same
// literal value, or if they are both nil.
func isSameLiteral(v1, v2 constant.Value) bool {
	if v1 == nil && v2 == nil {
		return true
	}
	if v1 == nil || v2 == nil {
		return false
	}
	return constant.Compare(v1, token.EQL, v2)
}

func isConcreteConvIface(n ir.Node) bool {
	sv := ir.StaticValue(n)
	if sv.Op() != ir.OCONVIFACE {
		return false
	}
	return !sv.(*ir.ConvExpr).X.Type().IsInterface()
}

func isSameFuncName(v1, v2 *ir.Name) bool {
	// NB: there are a few corner cases where pointer equality
	// doesn't work here, but this should be good enough for
	// our purposes here.
	return v1 == v2
}
