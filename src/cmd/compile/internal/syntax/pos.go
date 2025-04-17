// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import "fmt"

// PosMax is the largest line or column value that can be represented without loss.
// Incoming values (arguments) larger than PosMax will be set to PosMax.
//
// Keep this consistent with maxLineCol in go/scanner.
const PosMax = 1 << 30

// A Pos represents an absolute (line, col) source position
// with a reference to position base for computing relative
// (to a file, or line directive) position information.
// Pos values are intentionally light-weight so that they
// can be created without too much concern about space use.
type Pos struct {
	base      *PosBase
	line, col uint32
}

// MakePos returns a new Pos for the given PosBase, line and column.
func MakePos(base *PosBase, line, col uint) Pos { return Pos{base, sat32(line), sat32(col)} }

// TODO(gri) IsKnown makes an assumption about linebase < 1.
// Maybe we should check for Base() != nil instead.

func (pos Pos) Pos() Pos       { return pos }
func (pos Pos) IsKnown() bool  { return pos.line > 0 }
func (pos Pos) Base() *PosBase { return pos.base }
func (pos Pos) Line() uint     { return uint(pos.line) }
func (pos Pos) Col() uint      { return uint(pos.col) }

// FileBase returns the PosBase of the file containing pos,
// skipping over intermediate PosBases from //line directives.
// The result is nil if pos doesn't have a file base.
func (pos Pos) FileBase() *PosBase {
	b := pos.base
	for b != nil && b != b.pos.base {
		b = b.pos.base
	}
	// b == nil || b == b.pos.base
	return b
}

func (pos Pos) RelFilename() string { return pos.base.Filename() }

func (pos Pos) RelLine() uint {
	b := pos.base
	if b.Line() == 0 {
		// base line is unknown => relative line is unknown
		return 0
	}
	return b.Line() + (pos.Line() - b.Pos().Line())
}

func (pos Pos) RelCol() uint {
	b := pos.base
	if b.Col() == 0 {
		// base column is unknown => relative column is unknown
		// (the current specification for line directives requires
		// this to apply until the next PosBase/line directive,
		// not just until the new newline)
		return 0
	}
	if pos.Line() == b.Pos().Line() {
		// pos on same line as pos base => column is relative to pos base
		return b.Col() + (pos.Col() - b.Pos().Col())
	}
	return pos.Col()
}

// Cmp compares the positions p and q and returns a result r as follows:
//
//	r <  0: p is before q
//	r == 0: p and q are the same position (but may not be identical)
//	r >  0: p is after q
//
// If p and q are in different files, p is before q if the filename
// of p sorts lexicographically before the filename of q.
func (p Pos) Cmp(q Pos) int {
	pname := p.RelFilename()
	qname := q.RelFilename()
	switch {
	case pname < qname:
		return -1
	case pname > qname:
		return +1
	}

	pline := p.Line()
	qline := q.Line()
	switch {
	case pline < qline:
		return -1
	case pline > qline:
		return +1
	}

	pcol := p.Col()
	qcol := q.Col()
	switch {
	case pcol < qcol:
		return -1
	case pcol > qcol:
		return +1
	}

	return 0
}

func (pos Pos) String() string {
	rel := position_{pos.RelFilename(), pos.RelLine(), pos.RelCol()}
	abs := position_{pos.Base().Pos().RelFilename(), pos.Line(), pos.Col()}
	s := rel.String()
	if rel != abs {
		s += "[" + abs.String() + "]"
	}
	return s
}

// TODO(gri) cleanup: find better name, avoid conflict with position in error_test.go
type position_ struct {
	filename  string
	line, col uint
}

func (p position_) String() string {
	if p.line == 0 {
		if p.filename == "" {
			return "<unknown position>"
		}
		return p.filename
	}
	if p.col == 0 {
		return fmt.Sprintf("%s:%d", p.filename, p.line)
	}
	return fmt.Sprintf("%s:%d:%d", p.filename, p.line, p.col)
}

// A PosBase represents the base for relative position information:
// At position pos, the relative position is filename:line:col.
type PosBase struct {
	pos       Pos
	filename  string
	line, col uint32
	trimmed   bool // whether -trimpath has been applied
}

// NewFileBase returns a new PosBase for the given filename.
// A file PosBase's position is relative to itself, with the
// position being filename:1:1.
func NewFileBase(filename string) *PosBase {
	return NewTrimmedFileBase(filename, false)
}

// NewTrimmedFileBase is like NewFileBase, but allows specifying Trimmed.
func NewTrimmedFileBase(filename string, trimmed bool) *PosBase {
	base := &PosBase{MakePos(nil, linebase, colbase), filename, linebase, colbase, trimmed}
	base.pos.base = base
	return base
}

// NewLineBase returns a new PosBase for a line directive "line filename:line:col"
// relative to pos, which is the position of the character immediately following
// the comment containing the line directive. For a directive in a line comment,
// that position is the beginning of the next line (i.e., the newline character
// belongs to the line comment).
func NewLineBase(pos Pos, filename string, trimmed bool, line, col uint) *PosBase {
	return &PosBase{pos, filename, sat32(line), sat32(col), trimmed}
}

func (base *PosBase) IsFileBase() bool {
	if base == nil {
		return false
	}
	return base.pos.base == base
}

func (base *PosBase) Pos() (_ Pos) {
	if base == nil {
		return
	}
	return base.pos
}

func (base *PosBase) Filename() string {
	if base == nil {
		return ""
	}
	return base.filename
}

func (base *PosBase) Line() uint {
	if base == nil {
		return 0
	}
	return uint(base.line)
}

func (base *PosBase) Col() uint {
	if base == nil {
		return 0
	}
	return uint(base.col)
}

func (base *PosBase) Trimmed() bool {
	if base == nil {
		return false
	}
	return base.trimmed
}

func sat32(x uint) uint32 {
	if x > PosMax {
		return PosMax
	}
	return uint32(x)
}
