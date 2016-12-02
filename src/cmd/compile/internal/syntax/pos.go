// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the encoding of source positions.

package syntax

import "strconv"

// A Pos encodes a source position consisting of a (line, column) number pair
// and a position base. A zero Pos is a ready to use "unknown" position (empty
// filename, and unknown line and column number).
//
// The (line, column) values refer to a position in a file independent of any
// position base ("absolute" position). They start at 1, and they are unknown
// if 0.
//
// The position base is used to determine the "relative" position, that is the
// filename and line number relative to the position base. If the base refers
// to the current file, there is no difference between absolute and relative
// positions. If it refers to a //line pragma, a relative position is relative
// to that pragma. A position base in turn contains the position at which it
// was introduced in the current file.
type Pos struct {
	base *PosBase
	lico
}

// MakePos creates a new Pos value with the given base, and (file-absolute)
// line and column.
func MakePos(base *PosBase, line, col uint) Pos {
	return Pos{base, makeLico(line, col)}
}

// Filename returns the name of the actual file containing this position.
func (p Pos) Filename() string { return p.base.Pos().RelFilename() }

// Base returns the position base.
func (p Pos) Base() *PosBase { return p.base }

// RelFilename returns the filename recorded with the position's base.
func (p Pos) RelFilename() string { return p.base.Filename() }

// RelLine returns the line number relative to the positions's base.
func (p Pos) RelLine() uint { b := p.base; return b.Line() + p.Line() - b.Pos().Line() }

func (p Pos) String() string {
	b := p.base

	if b == b.Pos().base {
		// base is file base (incl. nil)
		return posString(b.Filename(), p.Line(), p.Col())
	}

	// base is relative
	return posString(b.Filename(), p.RelLine(), p.Col()) + "[" + b.Pos().String() + "]"
}

// posString formats a (filename, line, col) tuple as a printable position.
func posString(filename string, line, col uint) string {
	s := filename + ":" + strconv.FormatUint(uint64(line), 10)
	if col != 0 {
		s += ":" + strconv.FormatUint(uint64(col), 10)
	}
	return s
}

// ----------------------------------------------------------------------------
// PosBase

// A PosBase encodes a filename and base line number.
// Typically, each file and line pragma introduce a PosBase.
// A nil *PosBase is a ready to use file PosBase for an unnamed
// file with line numbers starting at 1.
type PosBase struct {
	pos      Pos
	filename string
	line     uint
}

// NewFileBase returns a new *PosBase for a file with the given filename.
func NewFileBase(filename string) *PosBase {
	if filename != "" {
		base := &PosBase{filename: filename}
		base.pos = MakePos(base, 0, 0)
		return base
	}
	return nil
}

// NewLinePragmaBase returns a new *PosBase for a line pragma of the form
//      //line filename:line
// at position pos.
func NewLinePragmaBase(pos Pos, filename string, line uint) *PosBase {
	return &PosBase{pos, filename, line - 1}
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

// Line returns the line number recorded with the base.
// If b == nil, the result is 0.
func (b *PosBase) Line() uint {
	if b != nil {
		return b.line
	}
	return 0
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
		// cannot represent column, use 0 to indicate unknown column
		col = 0
	}
	return lico(line<<colBits | col)
}

func (x lico) Line() uint { return uint(x) >> colBits }
func (x lico) Col() uint  { return uint(x) & colMax }
