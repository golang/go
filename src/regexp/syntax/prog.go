// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Compiled program.
// May not belong in this package, but convenient for now.

// A Prog is a compiled regular expression program.
type Prog struct {
	Inst   []Inst
	Start  int // index of start instruction
	NumCap int // number of InstCapture insts in re
}

// An InstOp is an instruction opcode.
type InstOp uint8

const (
	InstAlt InstOp = iota
	InstAltMatch
	InstCapture
	InstEmptyWidth
	InstMatch
	InstFail
	InstNop
	InstRune
	InstRune1
	InstRuneAny
	InstRuneAnyNotNL
)

var instOpNames = []string{
	"InstAlt",
	"InstAltMatch",
	"InstCapture",
	"InstEmptyWidth",
	"InstMatch",
	"InstFail",
	"InstNop",
	"InstRune",
	"InstRune1",
	"InstRuneAny",
	"InstRuneAnyNotNL",
}

func (i InstOp) String() string {
	if uint(i) >= uint(len(instOpNames)) {
		return ""
	}
	return instOpNames[i]
}

// An EmptyOp specifies a kind or mixture of zero-width assertions.
type EmptyOp uint8

const (
	EmptyBeginLine EmptyOp = 1 << iota
	EmptyEndLine
	EmptyBeginText
	EmptyEndText
	EmptyWordBoundary
	EmptyNoWordBoundary
)

// EmptyOpContext returns the zero-width assertions
// satisfied at the position between the runes r1 and r2.
// Passing r1 == -1 indicates that the position is
// at the beginning of the text.
// Passing r2 == -1 indicates that the position is
// at the end of the text.
func EmptyOpContext(r1, r2 rune) EmptyOp {
	var op EmptyOp = EmptyNoWordBoundary
	var boundary byte
	switch {
	case IsWordChar(r1):
		boundary = 1
	case r1 == '\n':
		op |= EmptyBeginLine
	case r1 < 0:
		op |= EmptyBeginText | EmptyBeginLine
	}
	switch {
	case IsWordChar(r2):
		boundary ^= 1
	case r2 == '\n':
		op |= EmptyEndLine
	case r2 < 0:
		op |= EmptyEndText | EmptyEndLine
	}
	if boundary != 0 { // IsWordChar(r1) != IsWordChar(r2)
		op ^= (EmptyWordBoundary | EmptyNoWordBoundary)
	}
	return op
}

// IsWordChar reports whether r is considered a “word character”
// during the evaluation of the \b and \B zero-width assertions.
// These assertions are ASCII-only: the word characters are [A-Za-z0-9_].
func IsWordChar(r rune) bool {
	// Test for lowercase letters first, as these occur more
	// frequently than uppercase letters in common cases.
	return 'a' <= r && r <= 'z' || 'A' <= r && r <= 'Z' || '0' <= r && r <= '9' || r == '_'
}

// An Inst is a single instruction in a regular expression program.
type Inst struct {
	Op   InstOp
	Out  uint32 // all but InstMatch, InstFail
	Arg  uint32 // InstAlt, InstAltMatch, InstCapture, InstEmptyWidth
	Rune []rune
}

func (p *Prog) String() string {
	var b strings.Builder
	dumpProg(&b, p)
	return b.String()
}

// skipNop follows any no-op or capturing instructions.
func (p *Prog) skipNop(pc uint32) *Inst {
	i := &p.Inst[pc]
	for i.Op == InstNop || i.Op == InstCapture {
		i = &p.Inst[i.Out]
	}
	return i
}

// op returns i.Op but merges all the Rune special cases into InstRune
func (i *Inst) op() InstOp {
	op := i.Op
	switch op {
	case InstRune1, InstRuneAny, InstRuneAnyNotNL:
		op = InstRune
	}
	return op
}

// Prefix returns a literal string that all matches for the
// regexp must start with. Complete is true if the prefix
// is the entire match.
func (p *Prog) Prefix() (prefix string, complete bool) {
	i := p.skipNop(uint32(p.Start))

	// Avoid allocation of buffer if prefix is empty.
	if i.op() != InstRune || len(i.Rune) != 1 {
		return "", i.Op == InstMatch
	}

	// Have prefix; gather characters.
	var buf strings.Builder
	for i.op() == InstRune && len(i.Rune) == 1 && Flags(i.Arg)&FoldCase == 0 && i.Rune[0] != utf8.RuneError {
		buf.WriteRune(i.Rune[0])
		i = p.skipNop(i.Out)
	}
	return buf.String(), i.Op == InstMatch
}

// StartCond returns the leading empty-width conditions that must
// be true in any match. It returns ^EmptyOp(0) if no matches are possible.
func (p *Prog) StartCond() EmptyOp {
	var flag EmptyOp
	pc := uint32(p.Start)
	i := &p.Inst[pc]
Loop:
	for {
		switch i.Op {
		case InstEmptyWidth:
			flag |= EmptyOp(i.Arg)
		case InstFail:
			return ^EmptyOp(0)
		case InstCapture, InstNop:
			// skip
		default:
			break Loop
		}
		pc = i.Out
		i = &p.Inst[pc]
	}
	return flag
}

const noMatch = -1

// MatchRune reports whether the instruction matches (and consumes) r.
// It should only be called when i.Op == InstRune.
func (i *Inst) MatchRune(r rune) bool {
	return i.MatchRunePos(r) != noMatch
}

// MatchRunePos checks whether the instruction matches (and consumes) r.
// If so, MatchRunePos returns the index of the matching rune pair
// (or, when len(i.Rune) == 1, rune singleton).
// If not, MatchRunePos returns -1.
// MatchRunePos should only be called when i.Op == InstRune.
func (i *Inst) MatchRunePos(r rune) int {
	rune := i.Rune

	switch len(rune) {
	case 0:
		return noMatch

	case 1:
		// Special case: single-rune slice is from literal string, not char class.
		r0 := rune[0]
		if r == r0 {
			return 0
		}
		if Flags(i.Arg)&FoldCase != 0 {
			for r1 := unicode.SimpleFold(r0); r1 != r0; r1 = unicode.SimpleFold(r1) {
				if r == r1 {
					return 0
				}
			}
		}
		return noMatch

	case 2:
		if r >= rune[0] && r <= rune[1] {
			return 0
		}
		return noMatch

	case 4, 6, 8:
		// Linear search for a few pairs.
		// Should handle ASCII well.
		for j := 0; j < len(rune); j += 2 {
			if r < rune[j] {
				return noMatch
			}
			if r <= rune[j+1] {
				return j / 2
			}
		}
		return noMatch
	}

	// Otherwise binary search.
	lo := 0
	hi := len(rune) / 2
	for lo < hi {
		m := int(uint(lo+hi) >> 1)
		if c := rune[2*m]; c <= r {
			if r <= rune[2*m+1] {
				return m
			}
			lo = m + 1
		} else {
			hi = m
		}
	}
	return noMatch
}

// MatchEmptyWidth reports whether the instruction matches
// an empty string between the runes before and after.
// It should only be called when i.Op == InstEmptyWidth.
func (i *Inst) MatchEmptyWidth(before rune, after rune) bool {
	switch EmptyOp(i.Arg) {
	case EmptyBeginLine:
		return before == '\n' || before == -1
	case EmptyEndLine:
		return after == '\n' || after == -1
	case EmptyBeginText:
		return before == -1
	case EmptyEndText:
		return after == -1
	case EmptyWordBoundary:
		return IsWordChar(before) != IsWordChar(after)
	case EmptyNoWordBoundary:
		return IsWordChar(before) == IsWordChar(after)
	}
	panic("unknown empty width arg")
}

func (i *Inst) String() string {
	var b strings.Builder
	dumpInst(&b, i)
	return b.String()
}

func bw(b *strings.Builder, args ...string) {
	for _, s := range args {
		b.WriteString(s)
	}
}

func dumpProg(b *strings.Builder, p *Prog) {
	for j := range p.Inst {
		i := &p.Inst[j]
		pc := strconv.Itoa(j)
		if len(pc) < 3 {
			b.WriteString("   "[len(pc):])
		}
		if j == p.Start {
			pc += "*"
		}
		bw(b, pc, "\t")
		dumpInst(b, i)
		bw(b, "\n")
	}
}

func u32(i uint32) string {
	return strconv.FormatUint(uint64(i), 10)
}

func dumpInst(b *strings.Builder, i *Inst) {
	switch i.Op {
	case InstAlt:
		bw(b, "alt -> ", u32(i.Out), ", ", u32(i.Arg))
	case InstAltMatch:
		bw(b, "altmatch -> ", u32(i.Out), ", ", u32(i.Arg))
	case InstCapture:
		bw(b, "cap ", u32(i.Arg), " -> ", u32(i.Out))
	case InstEmptyWidth:
		bw(b, "empty ", u32(i.Arg), " -> ", u32(i.Out))
	case InstMatch:
		bw(b, "match")
	case InstFail:
		bw(b, "fail")
	case InstNop:
		bw(b, "nop -> ", u32(i.Out))
	case InstRune:
		if i.Rune == nil {
			// shouldn't happen
			bw(b, "rune <nil>")
		}
		bw(b, "rune ", strconv.QuoteToASCII(string(i.Rune)))
		if Flags(i.Arg)&FoldCase != 0 {
			bw(b, "/i")
		}
		bw(b, " -> ", u32(i.Out))
	case InstRune1:
		bw(b, "rune1 ", strconv.QuoteToASCII(string(i.Rune)), " -> ", u32(i.Out))
	case InstRuneAny:
		bw(b, "any -> ", u32(i.Out))
	case InstRuneAnyNotNL:
		bw(b, "anynotnl -> ", u32(i.Out))
	}
}
