// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"bytes"
	"sort"
	"strconv"
	"unicode"
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
	InstLast
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
	if i >= InstLast {
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

// IsWordChar reports whether r is consider a ``word character''
// during the evaluation of the \b and \B zero-width assertions.
// These assertions are ASCII-only: the word characters are [A-Za-z0-9_].
func IsWordChar(r rune) bool {
	return 'A' <= r && r <= 'Z' || 'a' <= r && r <= 'z' || '0' <= r && r <= '9' || r == '_'
}

// An Inst is a single instruction in a regular expression program.
type Inst struct {
	Op   InstOp
	Out  uint32 // all but InstMatch, InstFail
	Arg  uint32 // InstAlt, InstAltMatch, InstCapture, InstEmptyWidth
	Rune []rune
	Next []uint32 // If input rune matches
}

func (p *Prog) String() string {
	var b bytes.Buffer
	dumpProg(&b, p)
	return b.String()
}

// skipNop follows any no-op or capturing instructions
// and returns the resulting pc.
func (p *Prog) skipNop(pc uint32) (*Inst, uint32) {
	i := &p.Inst[pc]
	for i.Op == InstNop || i.Op == InstCapture {
		pc = i.Out
		i = &p.Inst[pc]
	}
	return i, pc
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
// regexp must start with.  Complete is true if the prefix
// is the entire match.
func (p *Prog) Prefix() (prefix string, complete bool) {
	i, _ := p.skipNop(uint32(p.Start))

	// Avoid allocation of buffer if prefix is empty.
	if i.op() != InstRune || len(i.Rune) != 1 {
		return "", i.Op == InstMatch
	}

	// Have prefix; gather characters.
	var buf bytes.Buffer
	for i.op() == InstRune && len(i.Rune) == 1 && Flags(i.Arg)&FoldCase == 0 {
		buf.WriteRune(i.Rune[0])
		i, _ = p.skipNop(i.Out)
	}
	return buf.String(), i.Op == InstMatch
}

// OnePassPrefix returns a literal string that all matches for the
// regexp must start with.  Complete is true if the prefix
// is the entire match. Pc is the index of the last rune instruction
// in the string. The OnePassPrefix skips over the mandatory
// EmptyBeginText
func (p *Prog) OnePassPrefix() (prefix string, complete bool, pc uint32) {
	i := &p.Inst[p.Start]
	if i.Op != InstEmptyWidth || (EmptyOp(i.Arg))&EmptyBeginText == 0 {
		return "", i.Op == InstMatch, uint32(p.Start)
	}
	pc = i.Out
	i = &p.Inst[pc]
	for i.Op == InstNop {
		pc = i.Out
		i = &p.Inst[pc]
	}
	// Avoid allocation of buffer if prefix is empty.
	if i.op() != InstRune || len(i.Rune) != 1 {
		return "", i.Op == InstMatch, uint32(p.Start)
	}

	// Have prefix; gather characters.
	var buf bytes.Buffer
	for i.op() == InstRune && len(i.Rune) == 1 && Flags(i.Arg)&FoldCase == 0 {
		buf.WriteRune(i.Rune[0])
		pc, i = i.Out, &p.Inst[i.Out]
	}
	return buf.String(), i.Op == InstEmptyWidth && (EmptyOp(i.Arg))&EmptyBeginText != 0, pc
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

const noMatch = -1

// OnePassNext selects the next actionable state of the prog, based on the input character.
// It should only be called when i.Op == InstAlt or InstAltMatch, and from the one-pass machine.
// One of the alternates may ultimately lead without input to end of line. If the instruction
// is InstAltMatch the path to the InstMatch is in i.Out, the normal node in i.Next.
func (i *Inst) OnePassNext(r rune) uint32 {
	next := i.MatchRunePos(r)
	if next != noMatch {
		return i.Next[next]
	}
	if i.Op == InstAltMatch {
		return i.Out
	}
	return 0
}

// MatchRune returns true if the instruction matches (and consumes) r.
// It should only be called when i.Op == InstRune.
func (i *Inst) MatchRune(r rune) bool {
	return i.MatchRunePos(r) != noMatch
}

// MatchRunePos returns the index of the rune pair if the instruction matches.
// It should only be called when i.Op == InstRune.
func (i *Inst) MatchRunePos(r rune) int {
	rune := i.Rune

	// Special case: single-rune slice is from literal string, not char class.
	if len(rune) == 1 {
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
	}

	// Peek at the first few pairs.
	// Should handle ASCII well.
	for j := 0; j < len(rune) && j <= 8; j += 2 {
		if r < rune[j] {
			return noMatch
		}
		if r <= rune[j+1] {
			return j / 2
		}
	}

	// Otherwise binary search.
	lo := 0
	hi := len(rune) / 2
	for lo < hi {
		m := lo + (hi-lo)/2
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

// As per re2's Prog::IsWordChar. Determines whether rune is an ASCII word char.
// Since we act on runes, it would be easy to support Unicode here.
func wordRune(r rune) bool {
	return r == '_' ||
		('A' <= r && r <= 'Z') ||
		('a' <= r && r <= 'z') ||
		('0' <= r && r <= '9')
}

// MatchEmptyWidth returns true if the instruction matches
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
	return strconv.FormatUint(uint64(i), 10)
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

// Sparse Array implementation is used as a queue.
type queue struct {
	sparse          []uint32
	dense           []uint32
	size, nextIndex uint32
}

func (q *queue) empty() bool {
	return q.nextIndex >= q.size
}

func (q *queue) next() (n uint32) {
	n = q.dense[q.nextIndex]
	q.nextIndex++
	return
}

func (q *queue) clear() {
	q.size = 0
	q.nextIndex = 0
}

func (q *queue) reset() {
	q.nextIndex = 0
}

func (q *queue) contains(u uint32) bool {
	if u >= uint32(len(q.sparse)) {
		return false
	}
	return q.sparse[u] < q.size && q.dense[q.sparse[u]] == u
}

func (q *queue) insert(u uint32) {
	if !q.contains(u) {
		q.insertNew(u)
	}
}

func (q *queue) insertNew(u uint32) {
	if u >= uint32(len(q.sparse)) {
		return
	}
	q.sparse[u] = q.size
	q.dense[q.size] = u
	q.size++
}

func newQueue(size int) (q *queue) {
	return &queue{
		sparse: make([]uint32, size),
		dense:  make([]uint32, size),
	}
}

// mergeRuneSets merges two non-intersecting runesets, and returns the merged result,
// and a NextIp array. The idea is that if a rune matches the OnePassRunes at index
// i, NextIp[i/2] is the target. If the input sets intersect, an empty runeset and a
// NextIp array with the single element mergeFailed is returned.
// The code assumes that both inputs contain ordered and non-intersecting rune pairs.
const mergeFailed = uint32(0xffffffff)

var (
	noRune = []rune{}
	noNext = []uint32{mergeFailed}
)

func mergeRuneSets(leftRunes, rightRunes *[]rune, leftPC, rightPC uint32) ([]rune, []uint32) {
	leftLen := len(*leftRunes)
	rightLen := len(*rightRunes)
	if leftLen&0x1 != 0 || rightLen&0x1 != 0 {
		panic("mergeRuneSets odd length []rune")
	}
	var (
		lx, rx int
	)
	merged := make([]rune, 0)
	next := make([]uint32, 0)
	ok := true
	defer func() {
		if !ok {
			merged = nil
			next = nil
		}
	}()

	ix := -1
	extend := func(newLow *int, newArray *[]rune, pc uint32) bool {
		if ix > 0 && (*newArray)[*newLow] <= merged[ix] {
			return false
		}
		merged = append(merged, (*newArray)[*newLow], (*newArray)[*newLow+1])
		*newLow += 2
		ix += 2
		next = append(next, pc)
		return true
	}

	for lx < leftLen || rx < rightLen {
		switch {
		case rx >= rightLen:
			ok = extend(&lx, leftRunes, leftPC)
		case lx >= leftLen:
			ok = extend(&rx, rightRunes, rightPC)
		case (*rightRunes)[rx] < (*leftRunes)[lx]:
			ok = extend(&rx, rightRunes, rightPC)
		default:
			ok = extend(&lx, leftRunes, leftPC)
		}
		if !ok {
			return noRune, noNext
		}
	}
	return merged, next
}

// cleanupOnePass drops working memory, and restores certain shortcut instructions.
func (prog *Prog) cleanupOnePass(pOriginal *Prog) {
	for ix, instOriginal := range pOriginal.Inst {
		switch instOriginal.Op {
		case InstAlt, InstAltMatch, InstRune:
		case InstCapture, InstEmptyWidth, InstNop, InstMatch, InstFail:
			prog.Inst[ix].Next = nil
		case InstRune1, InstRuneAny, InstRuneAnyNotNL:
			prog.Inst[ix].Next = nil
			prog.Inst[ix] = instOriginal
		}
	}
}

// onePassCopy creates a copy of the original Prog, as we'll be modifying it
func (prog *Prog) onePassCopy() *Prog {
	p := &Prog{
		Inst:   append([]Inst{}[:], prog.Inst...),
		Start:  prog.Start,
		NumCap: prog.NumCap,
	}
	for _, inst := range p.Inst {
		inst.Next = make([]uint32, 0)
	}

	// rewrites one or more common Prog constructs that enable some otherwise
	// non-onepass Progs to be onepass. A:BD (for example) means an InstAlt at
	// ip A, that points to ips B & C.
	// A:BC + B:DA => A:BC + B:CD
	// A:BC + B:DC => A:DC + B:DC
	for pc := range p.Inst {
		switch p.Inst[pc].Op {
		default:
			continue
		case InstAlt, InstAltMatch:
			// A:Bx + B:Ay
			p_A_Other := &p.Inst[pc].Out
			p_A_Alt := &p.Inst[pc].Arg
			// make sure a target is another Alt
			instAlt := p.Inst[*p_A_Alt]
			if !(instAlt.Op == InstAlt || instAlt.Op == InstAltMatch) {
				p_A_Alt, p_A_Other = p_A_Other, p_A_Alt
				instAlt = p.Inst[*p_A_Alt]
				if !(instAlt.Op == InstAlt || instAlt.Op == InstAltMatch) {
					continue
				}
			}
			instOther := p.Inst[*p_A_Other]
			// Analyzing both legs pointing to Alts is for another day
			if instOther.Op == InstAlt || instOther.Op == InstAltMatch {
				// too complicated
				continue
			}
			// simple empty transition loop
			// A:BC + B:DA => A:BC + B:DC
			p_B_Alt := &p.Inst[*p_A_Alt].Out
			p_B_Other := &p.Inst[*p_A_Alt].Arg
			patch := false
			if instAlt.Out == uint32(pc) {
				patch = true
			} else if instAlt.Arg == uint32(pc) {
				patch = true
				p_B_Alt, p_B_Other = p_B_Other, p_B_Alt
			}
			if patch {
				*p_B_Alt = *p_A_Other
			}

			// empty transition to common target
			// A:BC + B:DC => A:DC + B:DC
			if *p_A_Other == *p_B_Alt {
				*p_A_Alt = *p_B_Other
			}
		}
	}
	return p
}

// runeSlice exists to permit sorting the case-folded rune sets.
type runeSlice []rune

func (p runeSlice) Len() int           { return len(p) }
func (p runeSlice) Less(i, j int) bool { return p[i] < p[j] }
func (p runeSlice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

// Sort is a convenience method.
func (p runeSlice) Sort() {
	sort.Sort(p)
}

// makeOnePass creates a onepass Prog, if possible. It is possible if at any alt,
// the match engine can always tell which branch to take. The routine may modify
// p if it is turned into a onepass Prog. If it isn't possible for this to be a
// onepass Prog, the Prog syntax.NotOnePass is returned. makeOnePass is resursive
// to the size of the Prog
func (p *Prog) makeOnePass() *Prog {
	var (
		instQueue    = newQueue(len(p.Inst))
		visitQueue   = newQueue(len(p.Inst))
		build        func(uint32, *queue)
		check        func(uint32, map[uint32]bool) bool
		onePassRunes = make([][]rune, len(p.Inst))
	)
	build = func(pc uint32, q *queue) {
		if q.contains(pc) {
			return
		}
		inst := p.Inst[pc]
		switch inst.Op {
		case InstAlt, InstAltMatch:
			q.insert(inst.Out)
			build(inst.Out, q)
			q.insert(inst.Arg)
		case InstMatch, InstFail:
		default:
			q.insert(inst.Out)
		}
	}

	// check that paths from Alt instructions are unambiguous, and rebuild the new
	// program as a onepass program
	check = func(pc uint32, m map[uint32]bool) (ok bool) {
		ok = true
		inst := &p.Inst[pc]
		if visitQueue.contains(pc) {
			return
		}
		visitQueue.insert(pc)
		switch inst.Op {
		case InstAlt, InstAltMatch:
			ok = check(inst.Out, m) && check(inst.Arg, m)
			// check no-input paths to InstMatch
			matchOut := m[inst.Out]
			matchArg := m[inst.Arg]
			if matchOut && matchArg {
				ok = false
				break
			}
			// Match on empty goes in inst.Out
			if matchArg {
				inst.Out, inst.Arg = inst.Arg, inst.Out
				matchOut, matchArg = matchArg, matchOut
			}
			if matchOut {
				m[pc] = true
				inst.Op = InstAltMatch
			}

			// build a dispatch operator from the two legs of the alt.
			onePassRunes[pc], inst.Next = mergeRuneSets(
				&onePassRunes[inst.Out], &onePassRunes[inst.Arg], inst.Out, inst.Arg)
			if len(inst.Next) > 0 && inst.Next[0] == mergeFailed {
				ok = false
				break
			}
		case InstCapture, InstNop:
			ok = check(inst.Out, m)
			m[pc] = m[inst.Out]
			// pass matching runes back through these no-ops.
			onePassRunes[pc] = append([]rune{}[:], onePassRunes[inst.Out][:]...)
			inst.Next = []uint32{}
			for i := len(onePassRunes[pc]) / 2; i >= 0; i-- {
				inst.Next = append(inst.Next, inst.Out)
			}
		case InstEmptyWidth:
			ok = check(inst.Out, m)
			m[pc] = m[inst.Out]
			onePassRunes[pc] = append([]rune{}[:], onePassRunes[inst.Out][:]...)
			inst.Next = []uint32{}
			for i := len(onePassRunes[pc]) / 2; i >= 0; i-- {
				inst.Next = append(inst.Next, inst.Out)
			}
		case InstMatch, InstFail:
			m[pc] = inst.Op == InstMatch
			break
		case InstRune:
			ok = check(inst.Out, m)
			m[pc] = false
			if len(inst.Next) > 0 {
				break
			}
			if len(inst.Rune) == 0 {
				onePassRunes[pc] = []rune{}[:]
				inst.Next = []uint32{inst.Out}
				break
			}
			runes := make([]rune, 0)
			if len(inst.Rune) == 1 && Flags(inst.Arg)&FoldCase != 0 {
				r0 := inst.Rune[0]
				runes = append(runes, r0, r0)
				for r1 := unicode.SimpleFold(r0); r1 != r0; r1 = unicode.SimpleFold(r1) {
					runes = append(runes, r1, r1)
				}
				sort.Sort(runeSlice(runes))
			} else {
				runes = append(runes, inst.Rune...)
			}
			onePassRunes[pc] = runes
			inst.Next = []uint32{}
			for i := len(onePassRunes[pc]) / 2; i >= 0; i-- {
				inst.Next = append(inst.Next, inst.Out)
			}
			inst.Op = InstRune
		case InstRune1:
			ok = check(inst.Out, m)
			m[pc] = false
			if len(inst.Next) > 0 {
				break
			}
			runes := []rune{}[:]
			// expand case-folded runes
			if Flags(inst.Arg)&FoldCase != 0 {
				r0 := inst.Rune[0]
				runes = append(runes, r0, r0)
				for r1 := unicode.SimpleFold(r0); r1 != r0; r1 = unicode.SimpleFold(r1) {
					runes = append(runes, r1, r1)
				}
				sort.Sort(runeSlice(runes))
			} else {
				runes = append(runes, inst.Rune[0], inst.Rune[0])
			}
			onePassRunes[pc] = runes
			inst.Next = []uint32{}
			for i := len(onePassRunes[pc]) / 2; i >= 0; i-- {
				inst.Next = append(inst.Next, inst.Out)
			}
			inst.Op = InstRune
		case InstRuneAny:
			ok = check(inst.Out, m)
			m[pc] = false
			if len(inst.Next) > 0 {
				break
			}
			onePassRunes[pc] = append([]rune{}[:], anyRune[:]...)
			inst.Next = []uint32{inst.Out}
		case InstRuneAnyNotNL:
			ok = check(inst.Out, m)
			m[pc] = false
			if len(inst.Next) > 0 {
				break
			}
			onePassRunes[pc] = append([]rune{}[:], anyRuneNotNL[:]...)
			inst.Next = []uint32{}
			for i := len(onePassRunes[pc]) / 2; i >= 0; i-- {
				inst.Next = append(inst.Next, inst.Out)
			}
		}
		return
	}

	instQueue.clear()
	instQueue.insert(uint32(p.Start))
	m := make(map[uint32]bool, len(p.Inst))
	for !instQueue.empty() {
		pc := instQueue.next()
		inst := p.Inst[pc]
		visitQueue.clear()
		if !check(uint32(pc), m) {
			p = NotOnePass
			break
		}
		switch inst.Op {
		case InstAlt, InstAltMatch:
			instQueue.insert(inst.Out)
			instQueue.insert(inst.Arg)
		case InstCapture, InstEmptyWidth, InstNop:
			instQueue.insert(inst.Out)
		case InstMatch:
		case InstFail:
		case InstRune, InstRune1, InstRuneAny, InstRuneAnyNotNL:
		default:
		}
	}
	if p != NotOnePass {
		for i, _ := range p.Inst {
			p.Inst[i].Rune = onePassRunes[i][:]
		}
	}
	return p
}

// walk visits each Inst in the prog once, and applies the argument
// function(ip, next), in pre-order.
func (prog *Prog) walk(funcs ...func(ip, next uint32)) {
	var walk1 func(uint32)
	progQueue := newQueue(len(prog.Inst))
	walk1 = func(ip uint32) {
		if progQueue.contains(ip) {
			return
		}
		progQueue.insert(ip)
		inst := prog.Inst[ip]
		switch inst.Op {
		case InstAlt, InstAltMatch:
			for _, f := range funcs {
				f(ip, inst.Out)
				f(ip, inst.Arg)
			}
			walk1(inst.Out)
			walk1(inst.Arg)
		default:
			for _, f := range funcs {
				f(ip, inst.Out)
			}
			walk1(inst.Out)
		}
	}
	walk1(uint32(prog.Start))
}

// find returns the Insts that match the argument predicate function
func (prog *Prog) find(f func(*Prog, int) bool) (matches []uint32) {
	matches = []uint32{}

	for ip := range prog.Inst {
		if f(prog, ip) {
			matches = append(matches, uint32(ip))
		}
	}
	return
}

var NotOnePass *Prog = nil
var debug = false

// CompileOnePass returns a new *Prog suitable for onePass execution if the original Prog
// can be recharacterized as a one-pass regexp program, or syntax.NotOnePass if the
// Prog cannot be converted. For a one pass prog, the fundamental condition that must
// be true is: at any InstAlt, there must be no ambiguity about what branch to  take.
func (prog *Prog) CompileOnePass() (p *Prog) {
	if prog.Start == 0 {
		return NotOnePass
	}
	// onepass regexp is anchored
	if prog.Inst[prog.Start].Op != InstEmptyWidth ||
		EmptyOp(prog.Inst[prog.Start].Arg)&EmptyBeginText != EmptyBeginText {
		return NotOnePass
	}
	// every instruction leading to InstMatch must be EmptyEndText
	for _, inst := range prog.Inst {
		opOut := prog.Inst[inst.Out].Op
		switch inst.Op {
		default:
			if opOut == InstMatch {
				return NotOnePass
			}
		case InstAlt, InstAltMatch:
			if opOut == InstMatch || prog.Inst[inst.Arg].Op == InstMatch {
				return NotOnePass
			}
		case InstEmptyWidth:
			if opOut == InstMatch {
				if EmptyOp(inst.Arg)&EmptyEndText == EmptyEndText {
					continue
				}
				return NotOnePass
			}
		}
	}
	// Creates a slightly optimized copy of the original Prog
	// that cleans up some Prog idioms that block valid onepass programs
	p = prog.onePassCopy()

	// checkAmbiguity on InstAlts, build onepass Prog if possible
	p = p.makeOnePass()

	if p != NotOnePass {
		p.cleanupOnePass(prog)
	}
	return p
}
