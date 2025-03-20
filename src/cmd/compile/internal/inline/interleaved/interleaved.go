// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package interleaved implements the interleaved devirtualization and
// inlining pass.
package interleaved

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/devirtualize"
	"cmd/compile/internal/inline"
	"cmd/compile/internal/inline/inlheur"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/pgoir"
	"cmd/compile/internal/typecheck"
	"fmt"
)

// DevirtualizeAndInlinePackage interleaves devirtualization and inlining on
// all functions within pkg.
func DevirtualizeAndInlinePackage(pkg *ir.Package, profile *pgoir.Profile) {
	if profile != nil && base.Debug.PGODevirtualize > 0 {
		// TODO(mdempsky): Integrate into DevirtualizeAndInlineFunc below.
		ir.VisitFuncsBottomUp(typecheck.Target.Funcs, func(list []*ir.Func, recursive bool) {
			for _, fn := range list {
				devirtualize.ProfileGuided(fn, profile)
			}
		})
		ir.CurFunc = nil
	}

	if base.Flag.LowerL != 0 {
		inlheur.SetupScoreAdjustments()
	}

	var inlProfile *pgoir.Profile // copy of profile for inlining
	if base.Debug.PGOInline != 0 {
		inlProfile = profile
	}

	// First compute inlinability of all functions in the package.
	inline.CanInlineFuncs(pkg.Funcs, inlProfile)

	inlState := make(map[*ir.Func]*inlClosureState)
	calleeUseCounts := make(map[*ir.Func]int)

	// Pre-process all the functions, adding parentheses around call sites and starting their "inl state".
	for _, fn := range typecheck.Target.Funcs {
		bigCaller := base.Flag.LowerL != 0 && inline.IsBigFunc(fn)
		if bigCaller && base.Flag.LowerM > 1 {
			fmt.Printf("%v: function %v considered 'big'; reducing max cost of inlinees\n", ir.Line(fn), fn)
		}

		s := &inlClosureState{bigCaller: bigCaller, profile: profile, fn: fn, callSites: make(map[*ir.ParenExpr]bool), useCounts: calleeUseCounts}
		s.parenthesize()
		inlState[fn] = s

		// Do a first pass at counting call sites.
		for i := range s.parens {
			s.resolve(i)
		}
	}

	ir.VisitFuncsBottomUp(typecheck.Target.Funcs, func(list []*ir.Func, recursive bool) {

		anyInlineHeuristics := false

		// inline heuristics, placed here because they have static state and that's what seems to work.
		for _, fn := range list {
			if base.Flag.LowerL != 0 {
				if inlheur.Enabled() && !fn.Wrapper() {
					inlheur.ScoreCalls(fn)
					anyInlineHeuristics = true
				}
				if base.Debug.DumpInlFuncProps != "" && !fn.Wrapper() {
					inlheur.DumpFuncProps(fn, base.Debug.DumpInlFuncProps)
				}
			}
		}

		if anyInlineHeuristics {
			defer inlheur.ScoreCallsCleanup()
		}

		// Iterate to a fixed point over all the functions.
		done := false
		for !done {
			done = true
			for _, fn := range list {
				s := inlState[fn]

				ir.WithFunc(fn, func() {
					l1 := len(s.parens)
					l0 := 0

					// Batch iterations so that newly discovered call sites are
					// resolved in a batch before inlining attempts.
					// Do this to avoid discovering new closure calls 1 at a time
					// which might cause first call to be seen as a single (high-budget)
					// call before the second is observed.
					for {
						for i := l0; i < l1; i++ { // can't use "range parens" here
							paren := s.parens[i]
							if new := s.edit(i); new != nil {
								// Update AST and recursively mark nodes.
								paren.X = new
								ir.EditChildren(new, s.mark) // mark may append to parens
								done = false
							}
						}
						l0, l1 = l1, len(s.parens)
						if l0 == l1 {
							break
						}
						for i := l0; i < l1; i++ {
							s.resolve(i)
						}

					}

				}) // WithFunc

			}
		}
	})

	ir.CurFunc = nil

	if base.Flag.LowerL != 0 {
		if base.Debug.DumpInlFuncProps != "" {
			inlheur.DumpFuncProps(nil, base.Debug.DumpInlFuncProps)
		}
		if inlheur.Enabled() {
			inline.PostProcessCallSites(inlProfile)
			inlheur.TearDown()
		}
	}

	// remove parentheses
	for _, fn := range typecheck.Target.Funcs {
		inlState[fn].unparenthesize()
	}

}

// DevirtualizeAndInlineFunc interleaves devirtualization and inlining
// on a single function.
func DevirtualizeAndInlineFunc(fn *ir.Func, profile *pgoir.Profile) {
	ir.WithFunc(fn, func() {
		if base.Flag.LowerL != 0 {
			if inlheur.Enabled() && !fn.Wrapper() {
				inlheur.ScoreCalls(fn)
				defer inlheur.ScoreCallsCleanup()
			}
			if base.Debug.DumpInlFuncProps != "" && !fn.Wrapper() {
				inlheur.DumpFuncProps(fn, base.Debug.DumpInlFuncProps)
			}
		}

		bigCaller := base.Flag.LowerL != 0 && inline.IsBigFunc(fn)
		if bigCaller && base.Flag.LowerM > 1 {
			fmt.Printf("%v: function %v considered 'big'; reducing max cost of inlinees\n", ir.Line(fn), fn)
		}

		s := &inlClosureState{bigCaller: bigCaller, profile: profile, fn: fn, callSites: make(map[*ir.ParenExpr]bool), useCounts: make(map[*ir.Func]int)}
		s.parenthesize()
		s.fixpoint()
		s.unparenthesize()
	})
}

type callSite struct {
	fn         *ir.Func
	whichParen int
}

type inlClosureState struct {
	fn        *ir.Func
	profile   *pgoir.Profile
	callSites map[*ir.ParenExpr]bool // callSites[p] == "p appears in parens" (do not append again)
	resolved  []*ir.Func             // for each call in parens, the resolved target of the call
	useCounts map[*ir.Func]int       // shared among all InlClosureStates
	parens    []*ir.ParenExpr
	bigCaller bool
}

// resolve attempts to resolve a call to a potentially inlineable callee
// and updates use counts on the callees.  Returns the call site count
// for that callee.
func (s *inlClosureState) resolve(i int) (*ir.Func, int) {
	p := s.parens[i]
	if i < len(s.resolved) {
		if callee := s.resolved[i]; callee != nil {
			return callee, s.useCounts[callee]
		}
	}
	n := p.X
	call, ok := n.(*ir.CallExpr)
	if !ok { // previously inlined
		return nil, -1
	}
	devirtualize.StaticCall(call)
	if callee := inline.InlineCallTarget(s.fn, call, s.profile); callee != nil {
		for len(s.resolved) <= i {
			s.resolved = append(s.resolved, nil)
		}
		s.resolved[i] = callee
		c := s.useCounts[callee] + 1
		s.useCounts[callee] = c
		return callee, c
	}
	return nil, 0
}

func (s *inlClosureState) edit(i int) ir.Node {
	n := s.parens[i].X
	call, ok := n.(*ir.CallExpr)
	if !ok {
		return nil
	}
	// This is redundant with earlier calls to
	// resolve, but because things can change it
	// must be re-checked.
	callee, count := s.resolve(i)
	if count <= 0 {
		return nil
	}
	if inlCall := inline.TryInlineCall(s.fn, call, s.bigCaller, s.profile, count == 1 && callee.ClosureParent != nil); inlCall != nil {
		return inlCall
	}
	return nil
}

// Mark inserts parentheses, and is called repeatedly.
// These inserted parentheses mark the call sites where
// inlining will be attempted.
func (s *inlClosureState) mark(n ir.Node) ir.Node {
	// Consider the expression "f(g())". We want to be able to replace
	// "g()" in-place with its inlined representation. But if we first
	// replace "f(...)" with its inlined representation, then "g()" will
	// instead appear somewhere within this new AST.
	//
	// To mitigate this, each matched node n is wrapped in a ParenExpr,
	// so we can reliably replace n in-place by assigning ParenExpr.X.
	// It's safe to use ParenExpr here, because typecheck already
	// removed them all.

	p, _ := n.(*ir.ParenExpr)
	if p != nil && s.callSites[p] {
		return n // already visited n.X before wrapping
	}

	if isTestingBLoop(n) {
		// No inlining nor devirtualization performed on b.Loop body
		if base.Flag.LowerM > 0 {
			fmt.Printf("%v: skip inlining within testing.B.loop for %v\n", ir.Line(n), n)
		}
		// We still want to explore inlining opportunities in other parts of ForStmt.
		nFor, _ := n.(*ir.ForStmt)
		nForInit := nFor.Init()
		for i, x := range nForInit {
			if x != nil {
				nForInit[i] = s.mark(x)
			}
		}
		if nFor.Cond != nil {
			nFor.Cond = s.mark(nFor.Cond)
		}
		if nFor.Post != nil {
			nFor.Post = s.mark(nFor.Post)
		}
		return n
	}

	if p != nil {
		n = p.X // in this case p was copied in from a (marked) inlined function, this is a new unvisited node.
	}

	ok := match(n)

	ir.EditChildren(n, s.mark)

	if ok {
		if p == nil {
			p = ir.NewParenExpr(n.Pos(), n)
			p.SetType(n.Type())
			p.SetTypecheck(n.Typecheck())
			s.callSites[p] = true
		}

		s.parens = append(s.parens, p)
		n = p
	} else if p != nil {
		n = p // didn't change anything, restore n
	}
	return n
}

// parenthesize applies s.mark to all the nodes within
// s.fn to mark calls and simplify rewriting them in place.
func (s *inlClosureState) parenthesize() {
	ir.EditChildren(s.fn, s.mark)
}

func (s *inlClosureState) unparenthesize() {
	if s == nil {
		return
	}
	if len(s.parens) == 0 {
		return // short circuit
	}

	var unparen func(ir.Node) ir.Node
	unparen = func(n ir.Node) ir.Node {
		if paren, ok := n.(*ir.ParenExpr); ok {
			n = paren.X
		}
		ir.EditChildren(n, unparen)
		// special case for tail calls: if the tail call was inlined, transform
		// the tail call to a return stmt if the inlined function was not void,
		// otherwise replace it with the inlined expression followed by a return.
		if tail, ok := n.(*ir.TailCallStmt); ok {
			if inl, done := tail.Call.(*ir.InlinedCallExpr); done {
				if len(inl.ReturnVars) != 0 {
					ret := ir.NewReturnStmt(tail.Pos(), []ir.Node{inl})
					if len(inl.ReturnVars) > 1 {
						typecheck.RewriteMultiValueCall(ret, inl)
					}
					n = ret
				} else {
					ret := ir.NewReturnStmt(tail.Pos(), nil)
					n = ir.NewBlockStmt(tail.Pos(), []ir.Node{inl, ret})
				}
			}
		}
		return n
	}
	ir.EditChildren(s.fn, unparen)
}

// fixpoint repeatedly edits a function until it stabilizes, returning
// whether anything changed in any of the fixpoint iterations.
//
// It applies s.edit(n) to each node n within the parentheses in s.parens.
// If s.edit(n) returns nil, no change is made. Otherwise, the result
// replaces n in fn's body, and fixpoint iterates at least once more.
//
// After an iteration where all edit calls return nil, fixpoint
// returns.
func (s *inlClosureState) fixpoint() bool {
	changed := false
	ir.WithFunc(s.fn, func() {
		done := false
		for !done {
			done = true
			for i := 0; i < len(s.parens); i++ { // can't use "range parens" here
				paren := s.parens[i]
				if new := s.edit(i); new != nil {
					// Update AST and recursively mark nodes.
					paren.X = new
					ir.EditChildren(new, s.mark) // mark may append to parens
					done = false
					changed = true
				}
			}
		}
	})
	return changed
}

func match(n ir.Node) bool {
	switch n.(type) {
	case *ir.CallExpr:
		return true
	}
	return false
}

// isTestingBLoop returns true if it matches the node as a
// testing.(*B).Loop. See issue #61515.
func isTestingBLoop(t ir.Node) bool {
	if t.Op() != ir.OFOR {
		return false
	}
	nFor, ok := t.(*ir.ForStmt)
	if !ok || nFor.Cond == nil || nFor.Cond.Op() != ir.OCALLFUNC {
		return false
	}
	n, ok := nFor.Cond.(*ir.CallExpr)
	if !ok || n.Fun == nil || n.Fun.Op() != ir.OMETHEXPR {
		return false
	}
	name := ir.MethodExprName(n.Fun)
	if name == nil {
		return false
	}
	if fSym := name.Sym(); fSym != nil && name.Class == ir.PFUNC && fSym.Pkg != nil &&
		fSym.Name == "(*B).Loop" && fSym.Pkg.Path == "testing" {
		// Attempting to match a function call to testing.(*B).Loop
		return true
	}
	return false
}
