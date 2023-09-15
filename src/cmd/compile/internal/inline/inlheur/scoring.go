// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inlheur

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/pgo"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"fmt"
	"os"
	"sort"
)

// These constants enumerate the set of possible ways/scenarios
// in which we'll adjust the score of a given callsite.
type scoreAdjustTyp uint

// These constants capture the various ways in which the inliner's
// scoring phase can adjust a callsite score based on heuristics. They
// fall broadly into three categories:
//
// 1) adjustments based solely on the callsite context (ex: call
// appears on panic path)
//
// 2) adjustments that take into account specific interesting values
// passed at a call site (ex: passing a constant that could result in
// cprop/deadcode in the caller)
//
// 3) adjustments that take into account values returned from the call
// at a callsite (ex: call always returns the same inlinable function,
// and return value flows unmodified into an indirect call)
//
// For categories 2 and 3 above, each adjustment can have either a
// "must" version and a "may" version (but not both). Here the idea is
// that in the "must" version the value flow is unconditional: if the
// callsite executes, then the condition we're interested in (ex:
// param feeding call) is guaranteed to happen. For the "may" version,
// there may be control flow that could cause the benefit to be
// bypassed.
const (
	// Category 1 adjustments (see above)
	panicPathAdj scoreAdjustTyp = (1 << iota)
	initFuncAdj
	inLoopAdj

	// Category 2 adjustments (see above).
	passConstToIfAdj
	passConstToNestedIfAdj
	passConcreteToItfCallAdj
	passConcreteToNestedItfCallAdj
	passFuncToIndCallAdj
	passFuncToNestedIndCallAdj
	passInlinableFuncToIndCallAdj
	passInlinableFuncToNestedIndCallAdj

	// Category 3 adjustments.
	returnFeedsConstToIfAdj
	returnFeedsFuncToIndCallAdj
	returnFeedsInlinableFuncToIndCallAdj
	returnFeedsConcreteToInterfaceCallAdj
)

// This table records the specific values we use to adjust call
// site scores in a given scenario.
// NOTE: these numbers are chosen very arbitrarily; ideally
// we will go through some sort of turning process to decide
// what value for each one produces the best performance.

var adjValues = map[scoreAdjustTyp]int{
	panicPathAdj:                          40,
	initFuncAdj:                           20,
	inLoopAdj:                             -5,
	passConstToIfAdj:                      -20,
	passConstToNestedIfAdj:                -15,
	passConcreteToItfCallAdj:              -30,
	passConcreteToNestedItfCallAdj:        -25,
	passFuncToIndCallAdj:                  -25,
	passFuncToNestedIndCallAdj:            -20,
	passInlinableFuncToIndCallAdj:         -45,
	passInlinableFuncToNestedIndCallAdj:   -40,
	returnFeedsConstToIfAdj:               -15,
	returnFeedsFuncToIndCallAdj:           -25,
	returnFeedsInlinableFuncToIndCallAdj:  -40,
	returnFeedsConcreteToInterfaceCallAdj: -25,
}

func adjValue(x scoreAdjustTyp) int {
	if val, ok := adjValues[x]; ok {
		return val
	} else {
		panic("internal error unregistered adjustment type")
	}
}

var mayMustAdj = [...]struct{ may, must scoreAdjustTyp }{
	{may: passConstToNestedIfAdj, must: passConstToIfAdj},
	{may: passConcreteToNestedItfCallAdj, must: passConcreteToItfCallAdj},
	{may: passFuncToNestedIndCallAdj, must: passFuncToNestedIndCallAdj},
	{may: passInlinableFuncToNestedIndCallAdj, must: passInlinableFuncToIndCallAdj},
}

func isMay(x scoreAdjustTyp) bool {
	return mayToMust(x) != 0
}

func isMust(x scoreAdjustTyp) bool {
	return mustToMay(x) != 0
}

func mayToMust(x scoreAdjustTyp) scoreAdjustTyp {
	for _, v := range mayMustAdj {
		if x == v.may {
			return v.must
		}
	}
	return 0
}

func mustToMay(x scoreAdjustTyp) scoreAdjustTyp {
	for _, v := range mayMustAdj {
		if x == v.must {
			return v.may
		}
	}
	return 0
}

// computeCallSiteScore takes a given call site whose ir node is 'call' and
// callee function is 'callee' and with previously computed call site
// properties 'csflags', then computes a score for the callsite that
// combines the size cost of the callee with heuristics based on
// previously parameter and function properties.
func computeCallSiteScore(callee *ir.Func, calleeProps *FuncProps, call ir.Node, csflags CSPropBits) (int, scoreAdjustTyp) {
	// Start with the size-based score for the callee.
	score := int(callee.Inl.Cost)
	var tmask scoreAdjustTyp

	if debugTrace&debugTraceScoring != 0 {
		fmt.Fprintf(os.Stderr, "=-= scoring call to %s at %s , initial=%d\n",
			callee.Sym().Name, fmtFullPos(call.Pos()), score)
	}

	// First some score adjustments to discourage inlining in selected cases.
	if csflags&CallSiteOnPanicPath != 0 {
		score, tmask = adjustScore(panicPathAdj, score, tmask)
	}
	if csflags&CallSiteInInitFunc != 0 {
		score, tmask = adjustScore(initFuncAdj, score, tmask)
	}

	// Then adjustments to encourage inlining in selected cases.
	if csflags&CallSiteInLoop != 0 {
		score, tmask = adjustScore(inLoopAdj, score, tmask)
	}

	// Walk through the actual expressions being passed at the call.
	calleeRecvrParms := callee.Type().RecvParams()
	ce := call.(*ir.CallExpr)
	for idx := range ce.Args {
		// ignore blanks
		if calleeRecvrParms[idx].Sym == nil ||
			calleeRecvrParms[idx].Sym.IsBlank() {
			continue
		}
		arg := ce.Args[idx]
		pflag := calleeProps.ParamFlags[idx]
		if debugTrace&debugTraceScoring != 0 {
			fmt.Fprintf(os.Stderr, "=-= arg %d of %d: val %v flags=%s\n",
				idx, len(ce.Args), arg, pflag.String())
		}
		_, islit := isLiteral(arg)
		iscci := isConcreteConvIface(arg)
		fname, isfunc, _ := isFuncName(arg)
		if debugTrace&debugTraceScoring != 0 {
			fmt.Fprintf(os.Stderr, "=-= isLit=%v iscci=%v isfunc=%v for arg %v\n", islit, iscci, isfunc, arg)
		}

		if islit {
			if pflag&ParamMayFeedIfOrSwitch != 0 {
				score, tmask = adjustScore(passConstToNestedIfAdj, score, tmask)
			}
			if pflag&ParamFeedsIfOrSwitch != 0 {
				score, tmask = adjustScore(passConstToIfAdj, score, tmask)
			}
		}

		if iscci {
			// FIXME: ideally here it would be nice to make a
			// distinction between the inlinable case and the
			// non-inlinable case, but this is hard to do. Example:
			//
			//    type I interface { Tiny() int; Giant() }
			//    type Conc struct { x int }
			//    func (c *Conc) Tiny() int { return 42 }
			//    func (c *Conc) Giant() { <huge amounts of code> }
			//
			//    func passConcToItf(c *Conc) {
			//        makesItfMethodCall(c)
			//    }
			//
			// In the code above, function properties will only tell
			// us that 'makesItfMethodCall' invokes a method on its
			// interface parameter, but we don't know whether it calls
			// "Tiny" or "Giant". If we knew if called "Tiny", then in
			// theory in addition to converting the interface call to
			// a direct call, we could also inline (in which case
			// we'd want to decrease the score even more).
			//
			// One thing we could do (not yet implemented) is iterate
			// through all of the methods of "*Conc" that allow it to
			// satisfy I, and if all are inlinable, then exploit that.
			if pflag&ParamMayFeedInterfaceMethodCall != 0 {
				score, tmask = adjustScore(passConcreteToNestedItfCallAdj, score, tmask)
			}
			if pflag&ParamFeedsInterfaceMethodCall != 0 {
				score, tmask = adjustScore(passConcreteToItfCallAdj, score, tmask)
			}
		}

		if isfunc {
			mayadj := passFuncToNestedIndCallAdj
			mustadj := passFuncToIndCallAdj
			if fn := fname.Func; fn != nil && typecheck.HaveInlineBody(fn) {
				mayadj = passInlinableFuncToNestedIndCallAdj
				mustadj = passInlinableFuncToIndCallAdj
			}
			if pflag&ParamMayFeedIndirectCall != 0 {
				score, tmask = adjustScore(mayadj, score, tmask)
			}
			if pflag&ParamFeedsIndirectCall != 0 {
				score, tmask = adjustScore(mustadj, score, tmask)
			}
		}
	}

	return score, tmask
}

func adjustScore(typ scoreAdjustTyp, score int, mask scoreAdjustTyp) (int, scoreAdjustTyp) {

	if isMust(typ) {
		if mask&typ != 0 {
			return score, mask
		}
		may := mustToMay(typ)
		if mask&may != 0 {
			// promote may to must, so undo may
			score -= adjValue(may)
			mask &^= may
		}
	} else if isMay(typ) {
		must := mayToMust(typ)
		if mask&(must|typ) != 0 {
			return score, mask
		}
	}
	if mask&typ == 0 {
		if debugTrace&debugTraceScoring != 0 {
			fmt.Fprintf(os.Stderr, "=-= applying adj %d for %s\n",
				adjValue(typ), typ.String())
		}
		score += adjValue(typ)
		mask |= typ
	}
	return score, mask
}

var resultFlagToPositiveAdj map[ResultPropBits]scoreAdjustTyp
var paramFlagToPositiveAdj map[ParamPropBits]scoreAdjustTyp

func setupFlagToAdjMaps() {
	resultFlagToPositiveAdj = map[ResultPropBits]scoreAdjustTyp{
		ResultIsAllocatedMem:     returnFeedsConcreteToInterfaceCallAdj,
		ResultAlwaysSameFunc:     returnFeedsFuncToIndCallAdj,
		ResultAlwaysSameConstant: returnFeedsConstToIfAdj,
	}
	paramFlagToPositiveAdj = map[ParamPropBits]scoreAdjustTyp{
		ParamMayFeedInterfaceMethodCall: passConcreteToNestedItfCallAdj,
		ParamFeedsInterfaceMethodCall:   passConcreteToItfCallAdj,
		ParamMayFeedIndirectCall:        passInlinableFuncToNestedIndCallAdj,
		ParamFeedsIndirectCall:          passInlinableFuncToIndCallAdj,
	}
}

// largestScoreAdjustment tries to estimate the largest possible
// negative score adjustment that could be applied to a call of the
// function with the specified props. Example:
//
//	func foo() {                  func bar(x int, p *int) int {
//	   ...                          if x < 0 { *p = x }
//	}                               return 99
//	                              }
//
// Function 'foo' above on the left has no interesting properties,
// thus as a result the most we'll adjust any call to is the value for
// "call in loop". If the calculated cost of the function is 150, and
// the in-loop adjustment is 5 (for example), then there is not much
// point treating it as inlinable. On the other hand "bar" has a param
// property (parameter "x" feeds unmodified to an "if" statement") and
// a return property (always returns same constant) meaning that a
// given call _could_ be rescored down as much as -35 points-- thus if
// the size of "bar" is 100 (for example) then there is at least a
// chance that scoring will enable inlining.
func largestScoreAdjustment(fn *ir.Func, props *FuncProps) int {
	if resultFlagToPositiveAdj == nil {
		setupFlagToAdjMaps()
	}
	var tmask scoreAdjustTyp
	score := adjValues[inLoopAdj] // any call can be in a loop
	for _, pf := range props.ParamFlags {
		if adj, ok := paramFlagToPositiveAdj[pf]; ok {
			score, tmask = adjustScore(adj, score, tmask)
		}
	}
	for _, rf := range props.ResultFlags {
		if adj, ok := resultFlagToPositiveAdj[rf]; ok {
			score, tmask = adjustScore(adj, score, tmask)
		}
	}

	if debugTrace&debugTraceScoring != 0 {
		fmt.Fprintf(os.Stderr, "=-= largestScore(%v) is %d\n",
			fn, score)
	}

	return score
}

// DumpInlCallSiteScores is invoked by the inliner if the debug flag
// "-d=dumpinlcallsitescores" is set; it dumps out a human-readable
// summary of all (potentially) inlinable callsites in the package,
// along with info on call site scoring and the adjustments made to a
// given score. Here profile is the PGO profile in use (may be
// nil), budgetCallback is a callback that can be invoked to find out
// the original pre-adjustment hairyness limit for the function, and
// inlineHotMaxBudget is the constant of the same name used in the
// inliner. Sample output lines:
//
// Score  Adjustment  Status  Callee  CallerPos ScoreFlags
// 115    40          DEMOTED cmd/compile/internal/abi.(*ABIParamAssignment).Offset     expand_calls.go:1679:14|6       panicPathAdj
// 76     -5n           PROMOTED runtime.persistentalloc   mcheckmark.go:48:45|3   inLoopAdj
// 201    0           --- PGO  unicode.DecodeRuneInString        utf8.go:312:30|1
// 7      -5          --- PGO  internal/abi.Name.DataChecked     type.go:625:22|0        inLoopAdj
//
// In the dump above, "Score" is the final score calculated for the
// callsite, "Adjustment" is the amount added to or subtracted from
// the original hairyness estimate to form the score. "Status" shows
// whether anything changed with the site -- did the adjustment bump
// it down just below the threshold ("PROMOTED") or instead bump it
// above the threshold ("DEMOTED"); this will be blank ("---") if no
// threshold was crossed as a result of the heuristics. Note that
// "Status" also shows whether PGO was involved. "Callee" is the name
// of the function called, "CallerPos" is the position of the
// callsite, and "ScoreFlags" is a digest of the specific properties
// we used to make adjustments to callsite score via heuristics.
func DumpInlCallSiteScores(profile *pgo.Profile, budgetCallback func(fn *ir.Func, profile *pgo.Profile) (int32, bool)) {

	fmt.Fprintf(os.Stdout, "# scores for package %s\n", types.LocalPkg.Path)

	genstatus := func(cs *CallSite, prof *pgo.Profile) string {
		hairyval := cs.Callee.Inl.Cost
		bud, isPGO := budgetCallback(cs.Callee, prof)
		score := int32(cs.Score)
		st := "---"

		switch {
		case hairyval <= bud && score <= bud:
			// "Normal" inlined case: hairy val sufficiently low that
			// it would have been inlined anyway without heuristics.
		case hairyval > bud && score > bud:
			// "Normal" not inlined case: hairy val sufficiently high
			// and scoring didn't lower it.
		case hairyval > bud && score <= bud:
			// Promoted: we would not have inlined it before, but
			// after score adjustment we decided to inline.
			st = "PROMOTED"
		case hairyval <= bud && score > bud:
			// Demoted: we would have inlined it before, but after
			// score adjustment we decided not to inline.
			st = "DEMOTED"
		}
		if isPGO {
			st += " PGO"
		}
		return st
	}

	if base.Debug.DumpInlCallSiteScores != 0 {
		var sl []*CallSite
		for _, funcInlHeur := range fpmap {
			for _, cs := range funcInlHeur.cstab {
				sl = append(sl, cs)
			}
		}
		sort.Slice(sl, func(i, j int) bool {
			if sl[i].Score != sl[j].Score {
				return sl[i].Score < sl[j].Score
			}
			fni := ir.PkgFuncName(sl[i].Callee)
			fnj := ir.PkgFuncName(sl[j].Callee)
			if fni != fnj {
				return fni < fnj
			}
			ecsi := EncodeCallSiteKey(sl[i])
			ecsj := EncodeCallSiteKey(sl[j])
			return ecsi < ecsj
		})

		mkname := func(fn *ir.Func) string {
			var n string
			if fn == nil || fn.Nname == nil {
				return "<nil>"
			}
			if fn.Sym().Pkg == types.LocalPkg {
				n = "·" + fn.Sym().Name
			} else {
				n = ir.PkgFuncName(fn)
			}
			// don't try to print super-long names
			if len(n) <= 64 {
				return n
			}
			return n[:32] + "..." + n[len(n)-32:len(n)]
		}

		if len(sl) != 0 {
			fmt.Fprintf(os.Stdout, "Score  Adjustment  Status  Callee  CallerPos Flags ScoreFlags\n")
		}
		for _, cs := range sl {
			hairyval := cs.Callee.Inl.Cost
			adj := int32(cs.Score) - hairyval
			fmt.Fprintf(os.Stdout, "%d  %d\t%s\t%s\t%s\t%s\n",
				cs.Score, adj, genstatus(cs, profile),
				mkname(cs.Callee),
				EncodeCallSiteKey(cs),
				cs.ScoreMask.String())
		}
	}
}
