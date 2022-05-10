// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the encoding of source positions.

package src

import (
	"bytes"
	"fmt"
	"io"
)

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

func (p Pos) LineNumber() string {
	if !p.IsKnown() {
		return "?"
	}
	return p.lico.lineNumber()
}

func (p Pos) LineNumberHTML() string {
	if !p.IsKnown() {
		return "?"
	}
	return p.lico.lineNumberHTML()
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
	buf := new(bytes.Buffer)
	p.WriteTo(buf, showCol, showOrig)
	return buf.String()
}

// WriteTo a position to w, formatted as Format does.
func (p Pos) WriteTo(w io.Writer, showCol, showOrig bool) {
	if !p.IsKnown() {
		io.WriteString(w, "<unknown line number>")
		return
	}

	if b := p.base; b == b.Pos().base {
		// base is file base (incl. nil)
		format(w, p.Filename(), p.Line(), p.Col(), showCol)
		return
	}

	// base is relative
	// Print the column only for the original position since the
	// relative position's column information may be bogus (it's
	// typically generated code and we can't say much about the
	// original source at that point but for the file:line info
	// that's provided via a line directive).
	// TODO(gri) This may not be true if we have an inlining base.
	// We may want to differentiate at some point.
	format(w, p.RelFilename(), p.RelLine(), p.RelCol(), showCol)
	if showOrig {
		io.WriteString(w, "[")
		format(w, p.Filename(), p.Line(), p.Col(), showCol)
		io.WriteString(w, "]")
	}
}

// format formats a (filename, line, col) tuple as "filename:line" (showCol
// is false or col == 0) or "filename:line:column" (showCol is true and col != 0).
func format(w io.Writer, filename string, line, col uint, showCol bool) {
	io.WriteString(w, filename)
	io.WriteString(w, ":")
	fmt.Fprint(w, line)
	// col == 0 and col == colMax are interpreted as unknown column values
	if showCol && 0 < col && col < colMax {
		io.WriteString(w, ":")
		fmt.Fprint(w, col)
	}
}

// formatstr wraps format to return a string.
func formatstr(filename string, line, col uint, showCol bool) string {
	buf := new(bytes.Buffer)
	format(buf, filename, line, col, showCol)
	return buf.String()
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

// Layout constants: 20 bits for line, 8 bits for column, 2 for isStmt, 2 for pro/epilogue
// (If this is too tight, we can either make lico 64b wide,
// or we can introduce a tiered encoding where we remove column
// information as line numbers grow bigger; similar to what gcc
// does.)
// The bitfield order is chosen to make IsStmt be the least significant
// part of a position; its use is to communicate statement edges through
// instruction scrambling in code generation, not to impose an order.
// TODO: Prologue and epilogue are perhaps better handled as pseudo-ops for the assembler,
// because they have almost no interaction with other uses of the position.
const (
	lineBits, lineMax     = 20, 1<<lineBits - 2
	bogusLine             = 1 // Used to disrupt infinite loops to prevent debugger looping
	isStmtBits, isStmtMax = 2, 1<<isStmtBits - 1
	xlogueBits, xlogueMax = 2, 1<<xlogueBits - 1
	colBits, colMax       = 32 - lineBits - xlogueBits - isStmtBits, 1<<colBits - 1

	isStmtShift = 0
	isStmtMask  = isStmtMax << isStmtShift
	xlogueShift = isStmtBits + isStmtShift
	xlogueMask  = xlogueMax << xlogueShift
	colShift    = xlogueBits + xlogueShift
	lineShift   = colBits + colShift
)
const (
	// It is expected that the front end or a phase in SSA will usually generate positions tagged with
	// PosDefaultStmt, but note statement boundaries with PosIsStmt.  Simple statements will have a single
	// boundary; for loops with initialization may have one for their entry and one for their back edge
	// (this depends on exactly how the loop is compiled; the intent is to provide a good experience to a
	// user debugging a program; the goal is that a breakpoint set on the loop line fires both on entry
	// and on iteration).  Proper treatment of non-gofmt input with multiple simple statements on a single
	// line is TBD.
	//
	// Optimizing compilation will move instructions around, and some of these will become known-bad as
	// step targets for debugging purposes (examples: register spills and reloads; code generated into
	// the entry block; invariant code hoisted out of loops) but those instructions will still have interesting
	// positions for profiling purposes. To reflect this these positions will be changed to PosNotStmt.
	//
	// When the optimizer removes an instruction marked PosIsStmt; it should attempt to find a nearby
	// instruction with the same line marked PosDefaultStmt to be the new statement boundary.  I.e., the
	// optimizer should make a best-effort to conserve statement boundary positions, and might be enhanced
	// to note when a statement boundary is not conserved.
	//
	// Code cloning, e.g. loop unrolling or loop unswitching, is an exception to the conservation rule
	// because a user running a debugger would expect to see breakpoints active in the copies of the code.
	//
	// In non-optimizing compilation there is still a role for PosNotStmt because of code generation
	// into the entry block.  PosIsStmt statement positions should be conserved.
	//
	// When code generation occurs any remaining default-marked positions are replaced with not-statement
	// positions.
	//
	PosDefaultStmt uint = iota // Default; position is not a statement boundary, but might be if optimization removes the designated statement boundary
	PosIsStmt                  // Position is a statement boundary; if optimization removes the corresponding instruction, it should attempt to find a new instruction to be the boundary.
	PosNotStmt                 // Position should not be a statement boundary, but line should be preserved for profiling and low-level debugging purposes.
)

type PosXlogue uint

const (
	PosDefaultLogue PosXlogue = iota
	PosPrologueEnd
	PosEpilogueBegin
)

func makeLicoRaw(line, col uint) lico {
	return lico(line<<lineShift | col<<colShift)
}

// This is a not-position that will not be elided.
// Depending on the debugger (gdb or delve) it may or may not be displayed.
func makeBogusLico() lico {
	return makeLicoRaw(bogusLine, 0).withIsStmt()
}

func makeLico(line, col uint) lico {
	if line >= lineMax {
		// cannot represent line, use max. line so we have some information
		line = lineMax
		// Drop column information if line number saturates.
		// Ensures line+col is monotonic. See issue 51193.
		col = 0
	}
	if col > colMax {
		// cannot represent column, use max. column so we have some information
		col = colMax
	}
	// default is not-sure-if-statement
	return makeLicoRaw(line, col)
}

func (x lico) Line() uint           { return uint(x) >> lineShift }
func (x lico) SameLine(y lico) bool { return 0 == (x^y)&^lico(1<<lineShift-1) }
func (x lico) Col() uint            { return uint(x) >> colShift & colMax }
func (x lico) IsStmt() uint {
	if x == 0 {
		return PosNotStmt
	}
	return uint(x) >> isStmtShift & isStmtMax
}
func (x lico) Xlogue() PosXlogue {
	return PosXlogue(uint(x) >> xlogueShift & xlogueMax)
}

// withNotStmt returns a lico for the same location, but not a statement
func (x lico) withNotStmt() lico {
	return x.withStmt(PosNotStmt)
}

// withDefaultStmt returns a lico for the same location, with default isStmt
func (x lico) withDefaultStmt() lico {
	return x.withStmt(PosDefaultStmt)
}

// withIsStmt returns a lico for the same location, tagged as definitely a statement
func (x lico) withIsStmt() lico {
	return x.withStmt(PosIsStmt)
}

// withLogue attaches a prologue/epilogue attribute to a lico
func (x lico) withXlogue(xlogue PosXlogue) lico {
	if x == 0 {
		if xlogue == 0 {
			return x
		}
		// Normalize 0 to "not a statement"
		x = lico(PosNotStmt << isStmtShift)
	}
	return lico(uint(x) & ^uint(xlogueMax<<xlogueShift) | (uint(xlogue) << xlogueShift))
}

// withStmt returns a lico for the same location with specified is_stmt attribute
func (x lico) withStmt(stmt uint) lico {
	if x == 0 {
		return lico(0)
	}
	return lico(uint(x) & ^uint(isStmtMax<<isStmtShift) | (stmt << isStmtShift))
}

func (x lico) lineNumber() string {
	return fmt.Sprintf("%d", x.Line())
}

func (x lico) lineNumberHTML() string {
	if x.IsStmt() == PosDefaultStmt {
		return fmt.Sprintf("%d", x.Line())
	}
	style, pfx := "b", "+"
	if x.IsStmt() == PosNotStmt {
		style = "s" // /strike not supported in HTML5
		pfx = ""
	}
	return fmt.Sprintf("<%s>%s%d</%s>", style, pfx, x.Line(), style)
}

func (x lico) atColumn1() lico {
	return makeLico(x.Line(), 1).withIsStmt()
}
