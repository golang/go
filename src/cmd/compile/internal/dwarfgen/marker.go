// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dwarfgen

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/internal/src"
)

// A ScopeMarker tracks scope nesting and boundaries for later use
// during DWARF generation.
type ScopeMarker struct {
	parents []ir.ScopeID
	marks   []ir.Mark
}

// checkPos validates the given position and returns the current scope.
func (m *ScopeMarker) checkPos(pos src.XPos) ir.ScopeID {
	if !pos.IsKnown() {
		base.Fatalf("unknown scope position")
	}

	if len(m.marks) == 0 {
		return 0
	}

	last := &m.marks[len(m.marks)-1]
	if xposBefore(pos, last.Pos) {
		base.FatalfAt(pos, "non-monotonic scope positions\n\t%v: previous scope position", base.FmtPos(last.Pos))
	}
	return last.Scope
}

// Push records a transition to a new child scope of the current scope.
func (m *ScopeMarker) Push(pos src.XPos) {
	current := m.checkPos(pos)

	m.parents = append(m.parents, current)
	child := ir.ScopeID(len(m.parents))

	m.marks = append(m.marks, ir.Mark{Pos: pos, Scope: child})
}

// Pop records a transition back to the current scope's parent.
func (m *ScopeMarker) Pop(pos src.XPos) {
	current := m.checkPos(pos)

	parent := m.parents[current-1]

	m.marks = append(m.marks, ir.Mark{Pos: pos, Scope: parent})
}

// Unpush removes the current scope, which must be empty.
func (m *ScopeMarker) Unpush() {
	i := len(m.marks) - 1
	current := m.marks[i].Scope

	if current != ir.ScopeID(len(m.parents)) {
		base.FatalfAt(m.marks[i].Pos, "current scope is not empty")
	}

	m.parents = m.parents[:current-1]
	m.marks = m.marks[:i]
}

// WriteTo writes the recorded scope marks to the given function,
// and resets the marker for reuse.
func (m *ScopeMarker) WriteTo(fn *ir.Func) {
	m.compactMarks()

	fn.Parents = make([]ir.ScopeID, len(m.parents))
	copy(fn.Parents, m.parents)
	m.parents = m.parents[:0]

	fn.Marks = make([]ir.Mark, len(m.marks))
	copy(fn.Marks, m.marks)
	m.marks = m.marks[:0]
}

func (m *ScopeMarker) compactMarks() {
	n := 0
	for _, next := range m.marks {
		if n > 0 && next.Pos == m.marks[n-1].Pos {
			m.marks[n-1].Scope = next.Scope
			continue
		}
		m.marks[n] = next
		n++
	}
	m.marks = m.marks[:n]
}
