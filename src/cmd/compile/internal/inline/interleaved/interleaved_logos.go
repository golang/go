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

func (s *inlClosureState) fixpointLogos(state *devirtualize.State, changed *bool) bool {
	for i := range s.parens {
		s.resolve(state, i)
	}
	done := false
	for !done {
		done = true
		s.initOracle()
		l1 := len(s.parens)
		l0 := 0
		for {
			for i := l0; i < l1; i++ {
				paren := s.parens[i]
				if origCall, inlinedCall := s.edit(state, i); inlinedCall != nil {
					paren.X = inlinedCall
					ir.EditChildren(inlinedCall, s.mark)
					state.InlinedCall(s.fn, origCall, inlinedCall)
					done = false
					*changed = true
				}
			}
			l0, l1 = l1, len(s.parens)
			if l0 == l1 {
				break
			}
			s.initOracle()
			for i := l0; i < l1; i++ {
				s.resolve(state, i)
			}
		}
	}
	return true
}
