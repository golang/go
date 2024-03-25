// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regexp

import (
	"regexp/syntax"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"
)

// "One-pass" regexp execution.
// Some regexps can be analyzed to determine that they never need
// backtracking: they are guaranteed to run in one pass over the string
// without bothering to save all the usual NFA state.
// Detect those and execute them more quickly.

// A onePassProg is a compiled one-pass regular expression program.
// It is the same as syntax.Prog except for the use of onePassInst.
type onePassProg struct {
	Inst   []onePassInst
	Start  int // index of start instruction
	NumCap int // number of InstCapture insts in re
}

// A onePassInst is a single instruction in a one-pass regular expression program.
// It is the same as syntax.Inst except for the new 'Next' field.
type onePassInst struct {
	syntax.Inst
	Next []uint32
}

// onePassPrefix returns a literal string that all matches for the
// regexp must start with. Complete is true if the prefix
// is the entire match. Pc is the index of the last rune instruction
// in the string. The onePassPrefix skips over the mandatory
// EmptyBeginText.
func onePassPrefix(p *syntax.Prog) (prefix string, complete bool, pc uint32) {
	i := &p.Inst[p.Start]
	if i.Op != syntax.InstEmptyWidth || (syntax.EmptyOp(i.Arg))&syntax.EmptyBeginText == 0 {
		return "", i.Op == syntax.InstMatch, uint32(p.Start)
	}
	pc = i.Out
	i = &p.Inst[pc]
	for i.Op == syntax.InstNop {
		pc = i.Out
		i = &p.Inst[pc]
	}
	// Avoid allocation of buffer if prefix is empty.
	if iop(i) != syntax.InstRune || len(i.Rune) != 1 {
		return "", i.Op == syntax.InstMatch, uint32(p.Start)
	}

	// Have prefix; gather characters.
	var buf strings.Builder
	for iop(i) == syntax.InstRune && len(i.Rune) == 1 && syntax.Flags(i.Arg)&syntax.FoldCase == 0 && i.Rune[0] != utf8.RuneError {
		buf.WriteRune(i.Rune[0])
		pc, i = i.Out, &p.Inst[i.Out]
	}
	if i.Op == syntax.InstEmptyWidth &&
		syntax.EmptyOp(i.Arg)&syntax.EmptyEndText != 0 &&
		p.Inst[i.Out].Op == syntax.InstMatch {
		complete = true
	}
	return buf.String(), complete, pc
}

// onePassNext selects the next actionable state of the prog, based on the input character.
// It should only be called when i.Op == InstAlt or InstAltMatch, and from the one-pass machine.
// One of the alternates may ultimately lead without input to end of line. If the instruction
// is InstAltMatch the path to the InstMatch is in i.Out, the normal node in i.Next.
func onePassNext(i *onePassInst, r rune) uint32 {
	next := i.MatchRunePos(r)
	if next >= 0 {
		return i.Next[next]
	}
	if i.Op == syntax.InstAltMatch {
		return i.Out
	}
	return 0
}

func iop(i *syntax.Inst) syntax.InstOp {
	op := i.Op
	switch op {
	case syntax.InstRune1, syntax.InstRuneAny, syntax.InstRuneAnyNotNL:
		op = syntax.InstRune
	}
	return op
}

// Sparse Array implementation is used as a queueOnePass.
type queueOnePass struct {
	sparse          []uint32
	dense           []uint32
	size, nextIndex uint32
}

func (q *queueOnePass) empty() bool {
	return q.nextIndex >= q.size
}

func (q *queueOnePass) next() (n uint32) {
	n = q.dense[q.nextIndex]
	q.nextIndex++
	return
}

func (q *queueOnePass) clear() {
	q.size = 0
	q.nextIndex = 0
}

func (q *queueOnePass) contains(u uint32) bool {
	if u >= uint32(len(q.sparse)) {
		return false
	}
	return q.sparse[u] < q.size && q.dense[q.sparse[u]] == u
}

func (q *queueOnePass) insert(u uint32) {
	if !q.contains(u) {
		q.insertNew(u)
	}
}

func (q *queueOnePass) insertNew(u uint32) {
	if u >= uint32(len(q.sparse)) {
		return
	}
	q.sparse[u] = q.size
	q.dense[q.size] = u
	q.size++
}

func newQueue(size int) (q *queueOnePass) {
	return &queueOnePass{
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
func cleanupOnePass(prog *onePassProg, original *syntax.Prog) {
	for ix, instOriginal := range original.Inst {
		switch instOriginal.Op {
		case syntax.InstAlt, syntax.InstAltMatch, syntax.InstRune:
		case syntax.InstCapture, syntax.InstEmptyWidth, syntax.InstNop, syntax.InstMatch, syntax.InstFail:
			prog.Inst[ix].Next = nil
		case syntax.InstRune1, syntax.InstRuneAny, syntax.InstRuneAnyNotNL:
			prog.Inst[ix].Next = nil
			prog.Inst[ix] = onePassInst{Inst: instOriginal}
		}
	}
}

// onePassCopy creates a copy of the original Prog, as we'll be modifying it.
func onePassCopy(prog *syntax.Prog) *onePassProg {
	p := &onePassProg{
		Start:  prog.Start,
		NumCap: prog.NumCap,
		Inst:   make([]onePassInst, len(prog.Inst)),
	}
	for i, inst := range prog.Inst {
		p.Inst[i] = onePassInst{Inst: inst}
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
		case syntax.InstAlt, syntax.InstAltMatch:
			// A:Bx + B:Ay
			p_A_Other := &p.Inst[pc].Out
			p_A_Alt := &p.Inst[pc].Arg
			// make sure a target is another Alt
			instAlt := p.Inst[*p_A_Alt]
			if !(instAlt.Op == syntax.InstAlt || instAlt.Op == syntax.InstAltMatch) {
				p_A_Alt, p_A_Other = p_A_Other, p_A_Alt
				instAlt = p.Inst[*p_A_Alt]
				if !(instAlt.Op == syntax.InstAlt || instAlt.Op == syntax.InstAltMatch) {
					continue
				}
			}
			instOther := p.Inst[*p_A_Other]
			// Analyzing both legs pointing to Alts is for another day
			if instOther.Op == syntax.InstAlt || instOther.Op == syntax.InstAltMatch {
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

var anyRuneNotNL = []rune{0, '\n' - 1, '\n' + 1, unicode.MaxRune}
var anyRune = []rune{0, unicode.MaxRune}

// makeOnePass creates a onepass Prog, if possible. It is possible if at any alt,
// the match engine can always tell which branch to take. The routine may modify
// p if it is turned into a onepass Prog. If it isn't possible for this to be a
// onepass Prog, the Prog nil is returned. makeOnePass is recursive
// to the size of the Prog.
func makeOnePass(p *onePassProg) *onePassProg {
	// If the machine is very long, it's not worth the time to check if we can use one pass.
	if len(p.Inst) >= 1000 {
		return nil
	}

	var (
		instQueue    = newQueue(len(p.Inst))
		visitQueue   = newQueue(len(p.Inst))
		check        func(uint32, []bool) bool
		onePassRunes = make([][]rune, len(p.Inst))
	)

	// check that paths from Alt instructions are unambiguous, and rebuild the new
	// program as a onepass program
	check = func(pc uint32, m []bool) (ok bool) {
		ok = true
		inst := &p.Inst[pc]
		if visitQueue.contains(pc) {
			return
		}
		visitQueue.insert(pc)
		switch inst.Op {
		case syntax.InstAlt, syntax.InstAltMatch:
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
				inst.Op = syntax.InstAltMatch
			}

			// build a dispatch operator from the two legs of the alt.
			onePassRunes[pc], inst.Next = mergeRuneSets(
				&onePassRunes[inst.Out], &onePassRunes[inst.Arg], inst.Out, inst.Arg)
			if len(inst.Next) > 0 && inst.Next[0] == mergeFailed {
				ok = false
				break
			}
		case syntax.InstCapture, syntax.InstNop:
			ok = check(inst.Out, m)
			m[pc] = m[inst.Out]
			// pass matching runes back through these no-ops.
			onePassRunes[pc] = append([]rune{}, onePassRunes[inst.Out]...)
			inst.Next = make([]uint32, len(onePassRunes[pc])/2+1)
			for i := range inst.Next {
				inst.Next[i] = inst.Out
			}
		case syntax.InstEmptyWidth:
			ok = check(inst.Out, m)
			m[pc] = m[inst.Out]
			onePassRunes[pc] = append([]rune{}, onePassRunes[inst.Out]...)
			inst.Next = make([]uint32, len(onePassRunes[pc])/2+1)
			for i := range inst.Next {
				inst.Next[i] = inst.Out
			}
		case syntax.InstMatch, syntax.InstFail:
			m[pc] = inst.Op == syntax.InstMatch
		case syntax.InstRune:
			m[pc] = false
			if len(inst.Next) > 0 {
				break
			}
			instQueue.insert(inst.Out)
			if len(inst.Rune) == 0 {
				onePassRunes[pc] = []rune{}
				inst.Next = []uint32{inst.Out}
				break
			}
			runes := make([]rune, 0)
			if len(inst.Rune) == 1 && syntax.Flags(inst.Arg)&syntax.FoldCase != 0 {
				r0 := inst.Rune[0]
				runes = append(runes, r0, r0)
				for r1 := unicode.SimpleFold(r0); r1 != r0; r1 = unicode.SimpleFold(r1) {
					runes = append(runes, r1, r1)
				}
				slices.Sort(runes)
			} else {
				runes = append(runes, inst.Rune...)
			}
			onePassRunes[pc] = runes
			inst.Next = make([]uint32, len(onePassRunes[pc])/2+1)
			for i := range inst.Next {
				inst.Next[i] = inst.Out
			}
			inst.Op = syntax.InstRune
		case syntax.InstRune1:
			m[pc] = false
			if len(inst.Next) > 0 {
				break
			}
			instQueue.insert(inst.Out)
			runes := []rune{}
			// expand case-folded runes
			if syntax.Flags(inst.Arg)&syntax.FoldCase != 0 {
				r0 := inst.Rune[0]
				runes = append(runes, r0, r0)
				for r1 := unicode.SimpleFold(r0); r1 != r0; r1 = unicode.SimpleFold(r1) {
					runes = append(runes, r1, r1)
				}
				slices.Sort(runes)
			} else {
				runes = append(runes, inst.Rune[0], inst.Rune[0])
			}
			onePassRunes[pc] = runes
			inst.Next = make([]uint32, len(onePassRunes[pc])/2+1)
			for i := range inst.Next {
				inst.Next[i] = inst.Out
			}
			inst.Op = syntax.InstRune
		case syntax.InstRuneAny:
			m[pc] = false
			if len(inst.Next) > 0 {
				break
			}
			instQueue.insert(inst.Out)
			onePassRunes[pc] = append([]rune{}, anyRune...)
			inst.Next = []uint32{inst.Out}
		case syntax.InstRuneAnyNotNL:
			m[pc] = false
			if len(inst.Next) > 0 {
				break
			}
			instQueue.insert(inst.Out)
			onePassRunes[pc] = append([]rune{}, anyRuneNotNL...)
			inst.Next = make([]uint32, len(onePassRunes[pc])/2+1)
			for i := range inst.Next {
				inst.Next[i] = inst.Out
			}
		}
		return
	}

	instQueue.clear()
	instQueue.insert(uint32(p.Start))
	m := make([]bool, len(p.Inst))
	for !instQueue.empty() {
		visitQueue.clear()
		pc := instQueue.next()
		if !check(pc, m) {
			p = nil
			break
		}
	}
	if p != nil {
		for i := range p.Inst {
			p.Inst[i].Rune = onePassRunes[i]
		}
	}
	return p
}

// compileOnePass returns a new *syntax.Prog suitable for onePass execution if the original Prog
// can be recharacterized as a one-pass regexp program, or syntax.nil if the
// Prog cannot be converted. For a one pass prog, the fundamental condition that must
// be true is: at any InstAlt, there must be no ambiguity about what branch to  take.
func compileOnePass(prog *syntax.Prog) (p *onePassProg) {
	if prog.Start == 0 {
		return nil
	}
	// onepass regexp is anchored
	if prog.Inst[prog.Start].Op != syntax.InstEmptyWidth ||
		syntax.EmptyOp(prog.Inst[prog.Start].Arg)&syntax.EmptyBeginText != syntax.EmptyBeginText {
		return nil
	}
	// every instruction leading to InstMatch must be EmptyEndText
	for _, inst := range prog.Inst {
		opOut := prog.Inst[inst.Out].Op
		switch inst.Op {
		default:
			if opOut == syntax.InstMatch {
				return nil
			}
		case syntax.InstAlt, syntax.InstAltMatch:
			if opOut == syntax.InstMatch || prog.Inst[inst.Arg].Op == syntax.InstMatch {
				return nil
			}
		case syntax.InstEmptyWidth:
			if opOut == syntax.InstMatch {
				if syntax.EmptyOp(inst.Arg)&syntax.EmptyEndText == syntax.EmptyEndText {
					continue
				}
				return nil
			}
		}
	}
	// Creates a slightly optimized copy of the original Prog
	// that cleans up some Prog idioms that block valid onepass programs
	p = onePassCopy(prog)

	// checkAmbiguity on InstAlts, build onepass Prog if possible
	p = makeOnePass(p)

	if p != nil {
		cleanupOnePass(p, prog)
	}
	return p
}
