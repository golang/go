// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the encoding of source positions.

package src

import "strconv"

// A Pos encodes a source position consisting of a (line, column) number pair
// and a position base. A zero Pos is a ready to use "unknown" position (nil
// position base and zero line number).
//
// The (line, column) values refer to a position in a file independent of any
// position base ("absolute" file position).
//
// The position base is used to determine the "relative" position, that is the
// filename and line number relative to the position base. If the base refers
// to the current file, there is no difference between absolute and relative
// positions. If it refers to a //line directive, a relative position is relative
// to that directive. A position base in turn contains the position at which it
// was introduced in the current file.
type Pos struct {
	base *PosBase
	lico
}

// NoPos is a valid unknown position.
var NoPos Pos

// MakePos creates a new Pos value with the given base, and (file-absolute)
// line and column.
func MakePos(base *PosBase, line, col uint) Pos {
	return Pos{base, makeLico(line, col)}
}

// IsKnown reports whether the position p is known.
// A position is known if it either has a non-nil
// position base, or a non-zero line number.
func (p Pos) IsKnown() bool {
	return p.base != nil || p.Line() != 0
}

// Before reports whether the position p comes before q in the source.
// For positions in different files, ordering is by filename.
func (p Pos) Before(q Pos) bool {
	n, m := p.Filename(), q.Filename()
	return n < m || n == m && p.lico < q.lico
}

// After reports whether the position p comes after q in the source.
// For positions in different files, ordering is by filename.
func (p Pos) After(q Pos) bool {
	n, m := p.Filename(), q.Filename()
	return n > m || n == m && p.lico > q.lico
}

// Filename returns the name of the actual file containing this position.
func (p Pos) Filename() string { return p.base.Pos().RelFilename() }

// Base returns the position base.
func (p Pos) Base() *PosBase { return p.base }

// SetBase sets the position base.
func (p *Pos) SetBase(base *PosBase) { p.base = base }

// RelFilename returns the filename recorded with the position's base.
func (p Pos) RelFilename() string { return p.base.Filename() }

// RelLine returns the line number relative to the position's base.
func (p Pos) RelLine() uint {
	b := p.base
	if b.Line() == 0 {
		// base line is unknown => relative line is unknown
		return 0
	}
	return b.Line() + (p.Line() - b.Pos().Line())
}

// RelCol returns the column number relative to the position's base.
func (p Pos) RelCol() uint {
	b := p.base
	if b.Col() == 0 {
		// base column is unknown => relative column is unknown
		// (the current specification for line directives requires
		// this to apply until the next PosBase/line directive,
		// not just until the new newline)
		return 0
	}
	if p.Line() == b.Pos().Line() {
		// p on same line as p's base => column is relative to p's base
		return b.Col() + (p.Col() - b.Pos().Col())
	}
	return p.Col()
}

// AbsFilename() returns the absolute filename recorded with the position's base.
func (p Pos) AbsFilename() string { return p.base.AbsFilename() }

// SymFilename() returns the absolute filename recorded with the position's base,
// prefixed by FileSymPrefix to make it appropriate for use as a linker symbol.
func (p Pos) SymFilename() string { return p.base.SymFilename() }

func (p Pos) String() string {
	return p.Format(true, true)
}

// Format formats a position as "filename:line" or "filename:line:column",
// controlled by the showCol flag and if the column is known (!= 0).
// For positions relative to line directives, the original position is
// shown as well, as in "filename:line[origfile:origline:origcolumn] if
// showOrig is set.
func (p Pos) Format(showCol, showOrig bool) string {
	if !p.IsKnown() {
		return "<unknown line number>"
	}

	if b := p.base; b == b.Pos().base {
		// base is file base (incl. nil)
		return format(p.Filename(), p.Line(), p.Col(), showCol)
	}

	// base is relative
	// Print the column only for the original position since the
	// relative position's column information may be bogus (it's
	// typically generated code and we can't say much about the
	// original source at that point but for the file:line info
	// that's provided via a line directive).
	// TODO(gri) This may not be true if we have an inlining base.
	// We may want to differentiate at some point.
	s := format(p.RelFilename(), p.RelLine(), p.RelCol(), showCol)
	if showOrig {
		s += "[" + format(p.Filename(), p.Line(), p.Col(), showCol) + "]"
	}
	return s
}

// format formats a (filename, line, col) tuple as "filename:line" (showCol
// is false or col == 0) or "filename:line:column" (showCol is true and col != 0).
func format(filename string, line, col uint, showCol bool) string {
	s := filename + ":" + strconv.FormatUint(uint64(line), 10)
	// col == 0 and col == colMax are interpreted as unknown column values
	if showCol && 0 < col && col < colMax {
		s += ":" + strconv.FormatUint(uint64(col), 10)
	}
	return s
}

// ----------------------------------------------------------------------------
// PosBase

// A PosBase encodes a filename and base position.
// Typically, each file and line directive introduce a PosBase.
type PosBase struct {
	pos         Pos    // position at which the relative position is (line, col)
	filename    string // file name used to open source file, for error messages
	absFilename string // absolute file name, for PC-Line tables
	symFilename string // cached symbol file name, to avoid repeated string concatenation
	line, col   uint   // relative line, column number at pos
	inl         int    // inlining index (see cmd/internal/obj/inl.go)
}

// NewFileBase returns a new *PosBase for a file with the given (relative and
// absolute) filenames.
func NewFileBase(filename, absFilename string) *PosBase {
	base := &PosBase{
		filename:    filename,
		absFilename: absFilename,
		symFilename: FileSymPrefix + absFilename,
		line:        1,
		col:         1,
		inl:         -1,
	}
	base.pos = MakePos(base, 1, 1)
	return base
}

// NewLinePragmaBase returns a new *PosBase for a line directive of the form
//      //line filename:line:col
//      /*line filename:line:col*/
// at position pos.
func NewLinePragmaBase(pos Pos, filename, absFilename string, line, col uint) *PosBase {
	return &PosBase{pos, filename, absFilename, FileSymPrefix + absFilename, line, col, -1}
}

// NewInliningBase returns a copy of the old PosBase with the given inlining
// index. If old == nil, the resulting PosBase has no filename.
func NewInliningBase(old *PosBase, inlTreeIndex int) *PosBase {
	if old == nil {
		base := &PosBase{line: 1, col: 1, inl: inlTreeIndex}
		base.pos = MakePos(base, 1, 1)
		return base
	}
	copy := *old
	base := &copy
	base.inl = inlTreeIndex
	if old == old.pos.base {
		base.pos.base = base
	}
	return base
}

var noPos Pos

// Pos returns the position at which base is located.
// If b == nil, the result is the zero position.
func (b *PosBase) Pos() *Pos {
	if b != nil {
		return &b.pos
	}
	return &noPos
}

// Filename returns the filename recorded with the base.
// If b == nil, the result is the empty string.
func (b *PosBase) Filename() string {
	if b != nil {
		return b.filename
	}
	return ""
}

// AbsFilename returns the absolute filename recorded with the base.
// If b == nil, the result is the empty string.
func (b *PosBase) AbsFilename() string {
	if b != nil {
		return b.absFilename
	}
	return ""
}

const FileSymPrefix = "gofile.."

// SymFilename returns the absolute filename recorded with the base,
// prefixed by FileSymPrefix to make it appropriate for use as a linker symbol.
// If b is nil, SymFilename returns FileSymPrefix + "??".
func (b *PosBase) SymFilename() string {
	if b != nil {
		return b.symFilename
	}
	return FileSymPrefix + "??"
}

// Line returns the line number recorded with the base.
// If b == nil, the result is 0.
func (b *PosBase) Line() uint {
	if b != nil {
		return b.line
	}
	return 0
}

// Col returns the column number recorded with the base.
// If b == nil, the result is 0.
func (b *PosBase) Col() uint {
	if b != nil {
		return b.col
	}
	return 0
}

// InliningIndex returns the index into the global inlining
// tree recorded with the base. If b == nil or the base has
// not been inlined, the result is < 0.
func (b *PosBase) InliningIndex() int {
	if b != nil {
		return b.inl
	}
	return -1
}

// ----------------------------------------------------------------------------
// lico

// A lico is a compact encoding of a LIne and COlumn number.
type lico uint32

// Layout constants: 24 bits for line, 8 bits for column.
// (If this is too tight, we can either make lico 64b wide,
// or we can introduce a tiered encoding where we remove column
// information as line numbers grow bigger; similar to what gcc
// does.)
const (
	lineBits, lineMax = 24, 1<<lineBits - 1
	colBits, colMax   = 32 - lineBits, 1<<colBits - 1
)

func makeLico(line, col uint) lico {
	if line > lineMax {
		// cannot represent line, use max. line so we have some information
		line = lineMax
	}
	if col > colMax {
		// cannot represent column, use max. column so we have some information
		col = colMax
	}
	return lico(line<<colBits | col)
}

func (x lico) Line() uint { return uint(x) >> colBits }
func (x lico) Col() uint  { return uint(x) & colMax }
