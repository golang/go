// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// An invariant is a count in the report that has reached zero and must stay
// there: the task that drove it to zero is done, so a nonzero value is a
// regression in whatever change caused it, not progress still to make.
// TestInvariants enforces them, naming the items that break each one.
//
// Counts still heading for zero are not invariants yet. When one reaches zero,
// add it here, so that the change that finishes a task also guards it.
type invariant struct {
	name  string   // the figure's label in the report
	tasks string   // SPEC-TRANSITION.md tasks that drove it to zero
	items []string // what breaks it; empty when it holds
}

func (m *metrics) invariants() []invariant {
	return []invariant{
		{"type mismatches with spec (§2)", "fill-enforce", m.typeOK.missed()},
		{"parameter-name mismatches with spec (§2)", "named-results", m.paramNameOK.missed()},
		{"result-name mismatches with spec (§2)", "named-results", m.resultNameOK.missed()},
		{"declarations with unnamed parameters spec names (§2)", "named-results", m.paramsFilled.missed()},
		{"declarations with unnamed results spec names (§2)", "named-results", m.resultsFilled.missed()},
		{"divergent parameter or result types across targets (§2)", "fill-enforce", m.typeSame.missed()},
		{"divergent parameter names across targets (§2)", "fill-enforce, spec-all", m.paramNameSame.missed()},
		{"divergent result names across targets (§2)", "named-results", m.resultNameSame.missed()},
		{"hand-written doc conflicts with spec (§3)", "doc-triage", m.handConflictDecls},
		{"drifted docs in simd mirror, simd_stubs.go vs simd_emulated.go (§3)", "comments-yaml", m.mirror.missed()},
		{"midway/comments.yaml entries spec covers (§4)", "comments-yaml, spec-all", m.commentsSpecCoveredNames},
	}
}
