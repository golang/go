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

// SameFile reports whether p and q are positions in the same file.
func (p XPos) SameFile(q XPos) bool {
	return p.index == q.index
}

// SameFileAndLine reports whether p and q are positions on the same line in the same file.
func (p XPos) SameFileAndLine(q XPos) bool {
	return p.index == q.index && p.lico.SameLine(q.lico)
}

// After reports whether the position p comes after q in the source.
// For positions with different bases, ordering is by base index.
func (p XPos) After(q XPos) bool {
	n, m := p.index, q.index
	return n > m || n == m && p.lico > q.lico
}

// WithNotStmt returns the same location to be marked with DWARF is_stmt=0
func (p XPos) WithNotStmt() XPos {
	p.lico = p.lico.withNotStmt()
	return p
}

// WithDefaultStmt returns the same location with undetermined is_stmt
func (p XPos) WithDefaultStmt() XPos {
	p.lico = p.lico.withDefaultStmt()
	return p
}

// WithIsStmt returns the same location to be marked with DWARF is_stmt=1
func (p XPos) WithIsStmt() XPos {
	p.lico = p.lico.withIsStmt()
	return p
}

// WithBogusLine returns a bogus line that won't match any recorded for the source code.
// Its use is to disrupt the statements within an infinite loop so that the debugger
// will not itself loop infinitely waiting for the line number to change.
// gdb chooses not to display the bogus line; delve shows it with a complaint, but the
// alternative behavior is to hang.
func (p XPos) WithBogusLine() XPos {
	if p.index == 0 {
		// See #35652
		panic("Assigning a bogus line to XPos with no file will cause mysterious downstream failures.")
	}
	p.lico = makeBogusLico()
	return p
}

// WithXlogue returns the same location but marked with DWARF function prologue/epilogue
func (p XPos) WithXlogue(x PosXlogue) XPos {
	p.lico = p.lico.withXlogue(x)
	return p
}

// LineNumber returns a string for the line number, "?" if it is not known.
func (p XPos) LineNumber() string {
	if !p.IsKnown() {
		return "?"
	}
	return p.lico.lineNumber()
}

// FileIndex returns a smallish non-negative integer corresponding to the
// file for this source position.  Smallish is relative; it can be thousands
// large, but not millions.
func (p XPos) FileIndex() int32 {
	return p.index
}

func (p XPos) LineNumberHTML() string {
	if !p.IsKnown() {
		return "?"
	}
	return p.lico.lineNumberHTML()
}

// AtColumn1 returns the same location but shifted to column 1.
func (p XPos) AtColumn1() XPos {
	p.lico = p.lico.atColumn1()
	return p
}

// A PosTable tracks Pos -> XPos conversions and vice versa.
// Its zero value is a ready-to-use PosTable.
type PosTable struct {
	baseList []*PosBase
	indexMap map[*PosBase]int
	nameMap  map[string]int // Maps file symbol name to index for debug information.
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
		t.nameMap = make(map[string]int)
	}
	i, ok := m[pos.base]
	if !ok {
		i = len(t.baseList)
		t.baseList = append(t.baseList, pos.base)
		t.indexMap[pos.base] = i
		if _, ok := t.nameMap[pos.base.symFilename]; !ok {
			t.nameMap[pos.base.symFilename] = len(t.nameMap)
		}
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

// FileIndex returns the index of the given filename(symbol) in the PosTable, or -1 if not found.
func (t *PosTable) FileIndex(filename string) int {
	if v, ok := t.nameMap[filename]; ok {
		return v
	}
	return -1
}

// DebugLinesFiles returns the file table for the debug_lines DWARF section.
func (t *PosTable) DebugLinesFileTable() []string {
	// Create a LUT of the global package level file indices. This table is what
	// is written in the debug_lines header, the file[N] will be referenced as
	// N+1 in the debug_lines table.
	fileLUT := make([]string, len(t.nameMap))
	for str, i := range t.nameMap {
		fileLUT[i] = str
	}
	return fileLUT
}
