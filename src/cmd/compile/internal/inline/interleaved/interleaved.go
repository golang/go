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
	"cmd/compile/internal/pgo"
	"cmd/compile/internal/typecheck"
	"fmt"
)

// DevirtualizeAndInlinePackage interleaves devirtualization and inlining on
// all functions within pkg.
func DevirtualizeAndInlinePackage(pkg *ir.Package, profile *pgo.Profile) {
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

	var inlProfile *pgo.Profile // copy of profile for inlining
	if base.Debug.PGOInline != 0 {
		inlProfile = profile
	}
	if inlProfile != nil {
		inline.PGOInlinePrologue(inlProfile, pkg.Funcs)
	}

	ir.VisitFuncsBottomUp(pkg.Funcs, func(funcs []*ir.Func, recursive bool) {
		// We visit functions within an SCC in fairly arbitrary order,
		// so by computing inlinability for all functions in the SCC
		// before performing any inlining, the results are less
		// sensitive to the order within the SCC (see #58905 for an
		// example).

		// First compute inlinability for all functions in the SCC ...
		inline.CanInlineSCC(funcs, recursive, inlProfile)

		// ... then make a second pass to do devirtualization and inlining
		// of calls.
		for _, fn := range funcs {
			DevirtualizeAndInlineFunc(fn, inlProfile)
		}
	})

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
func DevirtualizeAndInlineFunc(fn *ir.Func, profile *pgo.Profile) {
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

		// Walk fn's body and apply devirtualization and inlining.
		var inlCalls []*ir.InlinedCallExpr
		var edit func(ir.Node) ir.Node
		edit = func(n ir.Node) ir.Node {
			switch n := n.(type) {
			case *ir.TailCallStmt:
				n.Call.NoInline = true // can't inline yet
			}

			ir.EditChildren(n, edit)

			if call, ok := n.(*ir.CallExpr); ok {
				devirtualize.StaticCall(call)

				if inlCall := inline.TryInlineCall(fn, call, bigCaller, profile); inlCall != nil {
					inlCalls = append(inlCalls, inlCall)
					n = inlCall
				}
			}

			return n
		}
		ir.EditChildren(fn, edit)

		// If we inlined any calls, we want to recursively visit their
		// bodies for further devirtualization and inlining. However, we
		// need to wait until *after* the original function body has been
		// expanded, or else inlCallee can have false positives (e.g.,
		// #54632).
		for len(inlCalls) > 0 {
			call := inlCalls[0]
			inlCalls = inlCalls[1:]
			ir.EditChildren(call, edit)
		}
	})
}
