package syntax

import (
	"bytes"
	"strconv"
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
)

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

// An Inst is a single instruction in a regular expression program.
type Inst struct {
	Op   InstOp
	Out  uint32 // all but InstMatch, InstFail
	Arg  uint32 // InstAlt, InstAltMatch, InstCapture, InstEmptyWidth
	Rune []int
}

func (p *Prog) String() string {
	var b bytes.Buffer
	dumpProg(&b, p)
	return b.String()
}

// skipNop follows any no-op or capturing instructions
// and returns the resulting pc.
func (p *Prog) skipNop(pc uint32) *Inst {
	i := &p.Inst[pc]
	for i.Op == InstNop || i.Op == InstCapture {
		pc = i.Out
		i = &p.Inst[pc]
	}
	return i
}

// Prefix returns a literal string that all matches for the
// regexp must start with.  Complete is true if the prefix
// is the entire match.
func (p *Prog) Prefix() (prefix string, complete bool) {
	i := p.skipNop(uint32(p.Start))

	// Avoid allocation of buffer if prefix is empty.
	if i.Op != InstRune || len(i.Rune) != 1 {
		return "", i.Op == InstMatch
	}

	// Have prefix; gather characters.
	var buf bytes.Buffer
	for i.Op == InstRune && len(i.Rune) == 1 {
		buf.WriteRune(i.Rune[0])
		i = p.skipNop(i.Out)
	}
	return buf.String(), i.Op == InstMatch
}

// StartCond returns the leading empty-width conditions that must
// be true in any match.  It returns ^EmptyOp(0) if no matches are possible.
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

// MatchRune returns true if the instruction matches (and consumes) r.
// It should only be called when i.Op == InstRune.
func (i *Inst) MatchRune(r int) bool {
	rune := i.Rune

	// Special case: single-rune slice is from literal string, not char class.
	// TODO: Case folding.
	if len(rune) == 1 {
		return r == rune[0]
	}

	// Peek at the first few pairs.
	// Should handle ASCII well.
	for j := 0; j < len(rune) && j <= 8; j += 2 {
		if r < rune[j] {
			return false
		}
		if r <= rune[j+1] {
			return true
		}
	}

	// Otherwise binary search.
	lo := 0
	hi := len(rune) / 2
	for lo < hi {
		m := lo + (hi-lo)/2
		if c := rune[2*m]; c <= r {
			if r <= rune[2*m+1] {
				return true
			}
			lo = m + 1
		} else {
			hi = m
		}
	}
	return false
}

// As per re2's Prog::IsWordChar. Determines whether rune is an ASCII word char.
// Since we act on runes, it would be easy to support Unicode here.
func wordRune(rune int) bool {
	return rune == '_' ||
		('A' <= rune && rune <= 'Z') ||
		('a' <= rune && rune <= 'z') ||
		('0' <= rune && rune <= '9')
}

// MatchEmptyWidth returns true if the instruction matches
// an empty string between the runes before and after.
// It should only be called when i.Op == InstEmptyWidth.
func (i *Inst) MatchEmptyWidth(before int, after int) bool {
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
		return wordRune(before) != wordRune(after)
	case EmptyNoWordBoundary:
		return wordRune(before) == wordRune(after)
	}
	panic("unknown empty width arg")
}

func (i *Inst) String() string {
	var b bytes.Buffer
	dumpInst(&b, i)
	return b.String()
}

func bw(b *bytes.Buffer, args ...string) {
	for _, s := range args {
		b.WriteString(s)
	}
}

func dumpProg(b *bytes.Buffer, p *Prog) {
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
	return strconv.Uitoa64(uint64(i))
}

func dumpInst(b *bytes.Buffer, i *Inst) {
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
		bw(b, "rune ", strconv.QuoteToASCII(string(i.Rune)), " -> ", u32(i.Out))
	}
}
