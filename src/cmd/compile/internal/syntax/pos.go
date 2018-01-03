// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import "fmt"

// PosMax is the largest line or column value that can be represented without loss.
// Incoming values (arguments) larger than PosMax will be set to PosMax.
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
//           Maybe we should check for Base() != nil instead.

func (pos Pos) IsKnown() bool  { return pos.line > 0 }
func (pos Pos) Base() *PosBase { return pos.base }
func (pos Pos) Line() uint     { return uint(pos.line) }
func (pos Pos) Col() uint      { return uint(pos.col) }

func (pos Pos) RelFilename() string { b := pos.Base(); return b.Filename() }
func (pos Pos) RelLine() uint       { b := pos.Base(); return b.Line() + (pos.Line() - b.Pos().Line()) }
func (pos Pos) RelCol() uint {
	b := pos.Base()
	if pos.Line() == b.Pos().Line() {
		// pos on same line as pos base => column is relative to pos base
		return b.Col() + (pos.Col() - b.Pos().Col())
	}
	return pos.Col()
}

func (pos Pos) String() string {
	s := fmt.Sprintf("%s:%d:%d", pos.RelFilename(), pos.RelLine(), pos.RelCol())
	if bpos := pos.Base().Pos(); bpos.IsKnown() {
		s += fmt.Sprintf("[%s:%d:%d]", bpos.RelFilename(), pos.Line(), pos.Col())
	}
	return s
}

// A PosBase represents the base for relative position information:
// At position pos, the relative position is filename:line:col.
type PosBase struct {
	pos       Pos
	filename  string
	line, col uint32
}

// NewFileBase returns a new PosBase for the given filename.
// The PosBase position is unknown in this case.
func NewFileBase(filename string) *PosBase {
	return &PosBase{filename: filename}
}

// NewLineBase returns a new PosBase for a line directive "line filename:line:col"
// relative to pos, which is the position of the character immediately following
// the comment containing the line directive. For a directive in a line comment,
// that position is the beginning of the next line (i.e., the newline character
// belongs to the line comment).
func NewLineBase(pos Pos, filename string, line, col uint) *PosBase {
	return &PosBase{pos, filename, sat32(line), sat32(col)}
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

func sat32(x uint) uint32 {
	if x > PosMax {
		return PosMax
	}
	return uint32(x)
}
