// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the encoding of source positions.

package syntax

import "fmt"

// A Pos encodes a source position consisting of a (line, column) number pair
// and a position base.
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
func (p *Pos) Filename() string {
	if b := p.base; b != nil {
		return b.pos.RelFilename()
	}
	return ""
}

// Base returns the position base.
func (p *Pos) Base() *PosBase { return p.base }

// RelFilename returns the filename recorded with the position's base.
func (p *Pos) RelFilename() string {
	if b := p.base; b != nil {
		return b.filename
	}
	return ""
}

// RelLine returns the line number relative to the positions's base.
func (p *Pos) RelLine() uint {
	var line0 uint
	if b := p.base; b != nil {
		line0 = b.line - p.base.pos.Line()
	}
	return line0 + p.Line()
}

func (p *Pos) String() string {
	b := p.base

	if b == nil {
		return p.lico.String()
	}

	if b == b.pos.base {
		// base is file base
		return fmt.Sprintf("%s:%s", b.filename, p.lico.String())
	}

	// base is relative
	return fmt.Sprintf("%s:%s[%s]", b.filename, licoString(p.RelLine(), p.Col()), b.pos.String())
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
	base := &PosBase{filename: filename}
	base.pos = MakePos(base, 0, 0)
	return base
}

// NewLinePragmaBase returns a new *PosBase for a line pragma of the form
//      //line filename:line
// at position pos.
func NewLinePragmaBase(pos Pos, filename string, line uint) *PosBase {
	return &PosBase{pos, filename, line - 1}
}

// Pos returns the position at which base is located.
// If b == nil, the result is the empty position.
func (b *PosBase) Pos() Pos {
	if b != nil {
		return b.pos
	}
	return Pos{}
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

// Layout constants: 23 bits for line, 9 bits for column.
// (If this is too tight, we can either make lico 64b wide,
// or we can introduce a tiered encoding where we remove column
// information as line numbers grow bigger; similar to what gcc
// does.)
const (
	lineW, lineM = 23, 1<<lineW - 1
	colW, colM   = 32 - lineW, 1<<colW - 1
)

func makeLico(line, col uint) lico {
	if line > lineM {
		// cannot represent line, use max. line so we have some information
		line = lineM
	}
	if col > colM {
		// cannot represent column, use 0 to indicate unknown column
		col = 0
	}
	return lico(line<<colW | col)
}

func (x lico) Line() uint     { return uint(x) >> colW }
func (x lico) Col() uint      { return uint(x) & colM }
func (x lico) String() string { return licoString(x.Line(), x.Col()) }

func licoString(line, col uint) string {
	if col == 0 {
		return fmt.Sprintf("%d", line)
	}
	return fmt.Sprintf("%d:%d", line, col)
}
