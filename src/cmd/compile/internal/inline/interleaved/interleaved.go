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
		ir.VisitFuncsBottomUp(typecheck.Target.Funcs, func { list, recursive -> for _, fn := range list {
			devirtualize.ProfileGuided(fn, profile)
		} })
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

	// Now we make a second pass to do devirtualization and inlining of
	// calls. Order here should not matter.
	for _, fn := range pkg.Funcs {
		DevirtualizeAndInlineFunc(fn, inlProfile)
	}

	if base.Flag.LowerL != 0 {
		// Perform a garbage collection of hidden closures functions that
		// are no longer reachable from top-level functions following
		// inlining. See #59404 and #59638 for more context.
		inline.GarbageCollectUnreferencedHiddenClosures()

		if base.Debug.DumpInlFuncProps != "" {
			inlheur.DumpFuncProps(nil, base.Debug.DumpInlFuncProps)
		}
		if inlheur.Enabled() {
			inline.PostProcessCallSites(inlProfile)
			inlheur.TearDown()
		}
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

		match := func(n ir.Node) bool {
			switch n := n.(type) {
			case *ir.CallExpr:
				return true
			case *ir.TailCallStmt:
				n.Call.NoInline = true // can't inline yet
			}
			return false
		}

		edit := func(n ir.Node) ir.Node {
			call, ok := n.(*ir.CallExpr)
			if !ok { // previously inlined
				return nil
			}

			devirtualize.StaticCall(call)
			if inlCall := inline.TryInlineCall(fn, call, bigCaller, profile); inlCall != nil {
				return inlCall
			}
			return nil
		}

		fixpoint(fn, match, edit)
	})
}

// fixpoint repeatedly edits a function until it stabilizes.
//
// First, fixpoint applies match to every node n within fn. Then it
// iteratively applies edit to each node satisfying match(n).
//
// If edit(n) returns nil, no change is made. Otherwise, the result
// replaces n in fn's body, and fixpoint iterates at least once more.
//
// After an iteration where all edit calls return nil, fixpoint
// returns.
func fixpoint(fn *ir.Func, match func(ir.Node) bool, edit func(ir.Node) ir.Node) {
	// Consider the expression "f(g())". We want to be able to replace
	// "g()" in-place with its inlined representation. But if we first
	// replace "f(...)" with its inlined representation, then "g()" will
	// instead appear somewhere within this new AST.
	//
	// To mitigate this, each matched node n is wrapped in a ParenExpr,
	// so we can reliably replace n in-place by assigning ParenExpr.X.
	// It's safe to use ParenExpr here, because typecheck already
	// removed them all.

	var parens []*ir.ParenExpr
	var mark func(ir.Node) ir.Node
	mark = func { n ->
		if _, ok := n.(*ir.ParenExpr); ok {
			return n // already visited n.X before wrapping
		}

		ok := match(n)

		ir.EditChildren(n, mark)

		if ok {
			paren := ir.NewParenExpr(n.Pos(), n)
			paren.SetType(n.Type())
			paren.SetTypecheck(n.Typecheck())

			parens = append(parens, paren)
			n = paren
		}

		return n
	}
	ir.EditChildren(fn, mark)

	// Edit until stable.
	for {
		done := true

		for i := 0; i < len(parens); i++ { // can't use "range parens" here
			paren := parens[i]
			if new := edit(paren.X); new != nil {
				// Update AST and recursively mark nodes.
				paren.X = new
				ir.EditChildren(new, mark) // mark may append to parens
				done = false
			}
		}

		if done {
			break
		}
	}

	// Finally, remove any parens we inserted.
	if len(parens) == 0 {
		return // short circuit
	}
	var unparen func(ir.Node) ir.Node
	unparen = func { n ->
		if paren, ok := n.(*ir.ParenExpr); ok {
			n = paren.X
		}
		ir.EditChildren(n, unparen)
		return n
	}
	ir.EditChildren(fn, unparen)
}
