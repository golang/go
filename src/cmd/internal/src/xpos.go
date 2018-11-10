// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the compressed encoding of source
// positions using a lookup table.

package src

// XPos is a more compact representation of Pos.
type XPos struct {
	index int32
	lico
}

// NoXPos is a valid unknown position.
var NoXPos XPos

// IsKnown reports whether the position p is known.
// XPos.IsKnown() matches Pos.IsKnown() for corresponding
// positions.
func (p XPos) IsKnown() bool {
	return p.index != 0 || p.Line() != 0
}

// Before reports whether the position p comes before q in the source.
// For positions with different bases, ordering is by base index.
func (p XPos) Before(q XPos) bool {
	n, m := p.index, q.index
	return n < m || n == m && p.lico < q.lico
}

// After reports whether the position p comes after q in the source.
// For positions with different bases, ordering is by base index.
func (p XPos) After(q XPos) bool {
	n, m := p.index, q.index
	return n > m || n == m && p.lico > q.lico
}

// A PosTable tracks Pos -> XPos conversions and vice versa.
// Its zero value is a ready-to-use PosTable.
type PosTable struct {
	baseList []*PosBase
	indexMap map[*PosBase]int
}

// XPos returns the corresponding XPos for the given pos,
// adding pos to t if necessary.
func (t *PosTable) XPos(pos Pos) XPos {
	m := t.indexMap
	if m == nil {
		// Create new list and map and populate with nil
		// base so that NoPos always gets index 0.
		t.baseList = append(t.baseList, nil)
		m = map[*PosBase]int{nil: 0}
		t.indexMap = m
	}
	i, ok := m[pos.base]
	if !ok {
		i = len(t.baseList)
		t.baseList = append(t.baseList, pos.base)
		t.indexMap[pos.base] = i
	}
	return XPos{int32(i), pos.lico}
}

// Pos returns the corresponding Pos for the given p.
// If p cannot be translated via t, the function panics.
func (t *PosTable) Pos(p XPos) Pos {
	var base *PosBase
	if p.index != 0 {
		base = t.baseList[p.index]
	}
	return Pos{base, p.lico}
}
