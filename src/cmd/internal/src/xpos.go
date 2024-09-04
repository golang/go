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

// Compare returns an integer comparing the two positions.
func (p XPos) Compare(q XPos) int {
	//if r := cmp.Compare(p.index, q.index); r != 0 {
	//	return r
	//}
	//return cmp.Compare(p.lico, q.lico)
	if p.index != q.index {
		return int(p.index - q.index)
	}
	return int(p.lico) - int(q.lico)
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
	return XPos{t.baseIndex(pos.base), pos.lico}
}

func (t *PosTable) baseIndex(base *PosBase) int32 {
	if base == nil {
		return 0
	}

	if i, ok := t.indexMap[base]; ok {
		return int32(i)
	}

	if base.fileIndex >= 0 {
		panic("PosBase already registered with a PosTable")
	}

	if t.indexMap == nil {
		t.baseList = append(t.baseList, nil)
		t.indexMap = make(map[*PosBase]int)
		t.nameMap = make(map[string]int)
	}

	i := len(t.baseList)
	t.indexMap[base] = i
	t.baseList = append(t.baseList, base)

	fileIndex, ok := t.nameMap[base.absFilename]
	if !ok {
		fileIndex = len(t.nameMap)
		t.nameMap[base.absFilename] = fileIndex
	}
	base.fileIndex = fileIndex

	return int32(i)
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

// FileTable returns a slice of all files used to build this package.
func (t *PosTable) FileTable() []string {
	// Create a LUT of the global package level file indices. This table is what
	// is written in the debug_lines header, the file[N] will be referenced as
	// N+1 in the debug_lines table.
	fileLUT := make([]string, len(t.nameMap))
	for str, i := range t.nameMap {
		fileLUT[i] = str
	}
	return fileLUT
}
