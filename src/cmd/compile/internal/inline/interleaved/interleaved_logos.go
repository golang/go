// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package interleaved

import (
	"cmd/compile/internal/devirtualize"
	"cmd/compile/internal/inline"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/pgoir"
)

type inlClosureState struct {
	fn        *ir.Func
	profile   *pgoir.Profile
	callSites map[*ir.ParenExpr]bool
	resolved  []*ir.Func
	useCounts map[*ir.Func]int
	parens    []*ir.ParenExpr
	bigCaller bool
	oracle    ir.ReassignOracle
}

func (s *inlClosureState) initOracle() {
	if len(s.parens) > 0 {
		s.oracle.Init(s.fn)
	}
}

func (s *inlClosureState) inlineCallTargetLogos(call *ir.CallExpr) *ir.Func {
	return inline.InlineCallTarget(s.fn, call, s.profile, &s.oracle)
}

func (s *inlClosureState) tryInlineCallLogos(call *ir.CallExpr, closureCalledOnce bool) *ir.InlinedCallExpr {
	return inline.TryInlineCall(s.fn, call, s.bigCaller, s.profile, closureCalledOnce, &s.oracle)
}

// inlinePassLogos visits every current call site and each batch of sites
// discovered by inlining. Resolve each new batch before editing it so its
// closure uses are counted together. The caller initially resolves existing
// sites and repeats passes in function or SCC order until none changes.
func (s *inlClosureState) inlinePassLogos(state *devirtualize.State) bool {
	changed := false
	s.initOracle()
	for first, end := 0, len(s.parens); ; {
		for i := first; i < end; i++ {
			paren := s.parens[i]
			if origCall, inlinedCall := s.edit(state, i); inlinedCall != nil {
				paren.X = inlinedCall
				ir.EditChildren(inlinedCall, s.mark)
				state.InlinedCall(s.fn, origCall, inlinedCall)
				changed = true
			}
		}
		first, end = end, len(s.parens)
		if first == end {
			return changed
		}
		s.initOracle()
		for i := first; i < end; i++ {
			s.resolve(state, i)
		}
	}
}
