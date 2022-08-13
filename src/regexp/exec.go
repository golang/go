// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regexp

import (
	"io"
	"regexp/syntax"
	"sync"
)

// A queue is a 'sparse array' holding pending threads of execution.
// See https://research.swtch.com/2008/03/using-uninitialized-memory-for-fun-and.html
type queue struct {
	sparse []uint32
	dense  []entry
}

// An entry is an entry on a queue.
// It holds both the instruction pc and the actual thread.
// Some queue entries are just place holders so that the machine
// knows it has considered that pc. Such entries have t == nil.
type entry struct {
	pc uint32
	t  *thread
}

// A thread is the state of a single path through the machine:
// an instruction and a corresponding capture array.
// See https://swtch.com/~rsc/regexp/regexp2.html
type thread struct {
	inst *syntax.Inst
	cap  []int
}

// A machine holds all the state during an NFA simulation for p.
type machine struct {
	re       *Regexp      // corresponding Regexp
	p        *syntax.Prog // compiled program
	q0, q1   queue        // two queues for runq, nextq
	pool     []*thread    // pool of available threads
	matched  bool         // whether a match was found
	matchcap []int        // capture information for the match

	inputs inputs
}

type inputs struct {
	// cached inputs, to avoid allocation
	bytes  inputBytes
	string inputString
	reader inputReader
}

func (i *inputs) newBytes(b []byte) input {
	i.bytes.str = b
	return &i.bytes
}

func (i *inputs) newString(s string) input {
	i.string.str = s
	return &i.string
}

func (i *inputs) newReader(r io.RuneReader) input {
	i.reader.r = r
	i.reader.atEOT = false
	i.reader.pos = 0
	return &i.reader
}

func (i *inputs) clear() {
	// We need to clear 1 of these.
	// Avoid the expense of clearing the others (pointer write barrier).
	if i.bytes.str != nil {
		i.bytes.str = nil
	} else if i.reader.r != nil {
		i.reader.r = nil
	} else {
		i.string.str = ""
	}
}

func (i *inputs) init(r io.RuneReader, b []byte, s string) (input, int) {
	if r != nil {
		return i.newReader(r), 0
	}
	if b != nil {
		return i.newBytes(b), len(b)
	}
	return i.newString(s), len(s)
}

func (m *machine) init(ncap int) {
	for _, t := range m.pool {
		t.cap = t.cap[:ncap]
	}
	m.matchcap = m.matchcap[:ncap]
}

// alloc allocates a new thread with the given instruction.
// It uses the free pool if possible.
func (m *machine) alloc(i *syntax.Inst) *thread {
	var t *thread
	if n := len(m.pool); n > 0 {
		t = m.pool[n-1]
		m.pool = m.pool[:n-1]
	} else {
		t = new(thread)
		t.cap = make([]int, len(m.matchcap), cap(m.matchcap))
	}
	t.inst = i
	return t
}

// A lazyFlag is a lazily-evaluated syntax.EmptyOp,
// for checking zero-width flags like ^ $ \A \z \B \b.
// It records the pair of relevant runes and does not
// determine the implied flags until absolutely necessary
// (most of the time, that means never).
type lazyFlag uint64

func newLazyFlag(r1, r2 rune) lazyFlag {
	return lazyFlag(uint64(r1)<<32 | uint64(uint32(r2)))
}

func (f lazyFlag) match(op syntax.EmptyOp) bool {
	if op == 0 {
		return true
	}
	r1 := rune(f >> 32)
	if op&syntax.EmptyBeginLine != 0 {
		if r1 != '\n' && r1 >= 0 {
			return false
		}
		op &^= syntax.EmptyBeginLine
	}
	if op&syntax.EmptyBeginText != 0 {
		if r1 >= 0 {
			return false
		}
		op &^= syntax.EmptyBeginText
	}
	if op == 0 {
		return true
	}
	r2 := rune(f)
	if op&syntax.EmptyEndLine != 0 {
		if r2 != '\n' && r2 >= 0 {
			return false
		}
		op &^= syntax.EmptyEndLine
	}
	if op&syntax.EmptyEndText != 0 {
		if r2 >= 0 {
			return false
		}
		op &^= syntax.EmptyEndText
	}
	if op == 0 {
		return true
	}
	if syntax.IsWordChar(r1) != syntax.IsWordChar(r2) {
		op &^= syntax.EmptyWordBoundary
	} else {
		op &^= syntax.EmptyNoWordBoundary
	}
	return op == 0
}

// match runs the machine over the input starting at pos.
// It reports whether a match was found.
// If so, m.matchcap holds the submatch information.
func (m *machine) match(i input, pos int) bool {
	startCond := m.re.cond
	if startCond == ^syntax.EmptyOp(0) { // impossible
		return false
	}
	m.matched = false
	for i := range m.matchcap {
		m.matchcap[i] = -1
	}
	runq, nextq := &m.q0, &m.q1
	r, r1 := endOfText, endOfText
	width, width1 := 0, 0
	r, width = i.step(pos)
	if r != endOfText {
		r1, width1 = i.step(pos + width)
	}
	var flag lazyFlag
	if pos == 0 {
		flag = newLazyFlag(-1, r)
	} else {
		flag = i.context(pos)
	}
	for {
		if len(runq.dense) == 0 {
			if startCond&syntax.EmptyBeginText != 0 && pos != 0 {
				// Anchored match, past beginning of text.
				break
			}
			if m.matched {
				// Have match; finished exploring alternatives.
				break
			}
			if len(m.re.prefix) > 0 && r1 != m.re.prefixRune && i.canCheckPrefix() {
				// Match requires literal prefix; fast search for it.
				advance := i.index(m.re, pos)
				if advance < 0 {
					break
				}
				pos += advance
				r, width = i.step(pos)
				r1, width1 = i.step(pos + width)
			}
		}
		if !m.matched {
			if len(m.matchcap) > 0 {
				m.matchcap[0] = pos
			}
			m.add(runq, uint32(m.p.Start), pos, m.matchcap, &flag, nil)
		}
		flag = newLazyFlag(r, r1)
		m.step(runq, nextq, pos, pos+width, r, &flag)
		if width == 0 {
			break
		}
		if len(m.matchcap) == 0 && m.matched {
			// Found a match and not paying attention
			// to where it is, so any match will do.
			break
		}
		pos += width
		r, width = r1, width1
		if r != endOfText {
			r1, width1 = i.step(pos + width)
		}
		runq, nextq = nextq, runq
	}
	m.clear(nextq)
	return m.matched
}

// clear frees all threads on the thread queue.
func (m *machine) clear(q *queue) {
	for _, d := range q.dense {
		if d.t != nil {
			m.pool = append(m.pool, d.t)
		}
	}
	q.dense = q.dense[:0]
}

// step executes one step of the machine, running each of the threads
// on runq and appending new threads to nextq.
// The step processes the rune c (which may be endOfText),
// which starts at position pos and ends at nextPos.
// nextCond gives the setting for the empty-width flags after c.
func (m *machine) step(runq, nextq *queue, pos, nextPos int, c rune, nextCond *lazyFlag) {
	longest := m.re.longest
	for j := 0; j < len(runq.dense); j++ {
		d := &runq.dense[j]
		t := d.t
		if t == nil {
			continue
		}
		if longest && m.matched && len(t.cap) > 0 && m.matchcap[0] < t.cap[0] {
			m.pool = append(m.pool, t)
			continue
		}
		i := t.inst
		add := false
		switch i.Op {
		default:
			panic("bad inst")

		case syntax.InstMatch:
			if len(t.cap) > 0 && (!longest || !m.matched || m.matchcap[1] < pos) {
				t.cap[1] = pos
				copy(m.matchcap, t.cap)
			}
			if !longest {
				// First-match mode: cut off all lower-priority threads.
				for _, d := range runq.dense[j+1:] {
					if d.t != nil {
						m.pool = append(m.pool, d.t)
					}
				}
				runq.dense = runq.dense[:0]
			}
			m.matched = true

		case syntax.InstRune:
			add = i.MatchRune(c)
		case syntax.InstRune1:
			add = c == i.Rune[0]
		case syntax.InstRuneAny:
			add = true
		case syntax.InstRuneAnyNotNL:
			add = c != '\n'
		}
		if add {
			t = m.add(nextq, i.Out, nextPos, t.cap, nextCond, t)
		}
		if t != nil {
			m.pool = append(m.pool, t)
		}
	}
	runq.dense = runq.dense[:0]
}

// add adds an entry to q for pc, unless the q already has such an entry.
// It also recursively adds an entry for all instructions reachable from pc by following
// empty-width conditions satisfied by cond.  pos gives the current position
// in the input.
func (m *machine) add(q *queue, pc uint32, pos int, cap []int, cond *lazyFlag, t *thread) *thread {
Again:
	if pc == 0 {
		return t
	}
	if j := q.sparse[pc]; j < uint32(len(q.dense)) && q.dense[j].pc == pc {
		return t
	}

	j := len(q.dense)
	q.dense = q.dense[:j+1]
	d := &q.dense[j]
	d.t = nil
	d.pc = pc
	q.sparse[pc] = uint32(j)

	i := &m.p.Inst[pc]
	switch i.Op {
	default:
		panic("unhandled")
	case syntax.InstFail:
		// nothing
	case syntax.InstAlt, syntax.InstAltMatch:
		t = m.add(q, i.Out, pos, cap, cond, t)
		pc = i.Arg
		goto Again
	case syntax.InstEmptyWidth:
		if cond.match(syntax.EmptyOp(i.Arg)) {
			pc = i.Out
			goto Again
		}
	case syntax.InstNop:
		pc = i.Out
		goto Again
	case syntax.InstCapture:
		if int(i.Arg) < len(cap) {
			opos := cap[i.Arg]
			cap[i.Arg] = pos
			m.add(q, i.Out, pos, cap, cond, nil)
			cap[i.Arg] = opos
		} else {
			pc = i.Out
			goto Again
		}
	case syntax.InstMatch, syntax.InstRune, syntax.InstRune1, syntax.InstRuneAny, syntax.InstRuneAnyNotNL:
		if t == nil {
			t = m.alloc(i)
		} else {
			t.inst = i
		}
		if len(cap) > 0 && &t.cap[0] != &cap[0] {
			copy(t.cap, cap)
		}
		d.t = t
		t = nil
	}
	return t
}

type onePassMachine struct {
	inputs   inputs
	matchcap []int
}

var onePassPool sync.Pool

func newOnePassMachine() *onePassMachine {
	m, ok := onePassPool.Get().(*onePassMachine)
	if !ok {
		m = new(onePassMachine)
	}
	return m
}

func freeOnePassMachine(m *onePassMachine) {
	m.inputs.clear()
	onePassPool.Put(m)
}

// doOnePass implements r.doExecute using the one-pass execution engine.
func (re *Regexp) doOnePass(ir io.RuneReader, ib []byte, is string, pos, ncap int, dstCap []int) []int {
	startCond := re.cond
	if startCond == ^syntax.EmptyOp(0) { // impossible
		return nil
	}

	m := newOnePassMachine()
	if cap(m.matchcap) < ncap {
		m.matchcap = make([]int, ncap)
	} else {
		m.matchcap = m.matchcap[:ncap]
	}

	matched := false
	for i := range m.matchcap {
		m.matchcap[i] = -1
	}

	i, _ := m.inputs.init(ir, ib, is)

	r, r1 := endOfText, endOfText
	width, width1 := 0, 0
	r, width = i.step(pos)
	if r != endOfText {
		r1, width1 = i.step(pos + width)
	}
	var flag lazyFlag
	if pos == 0 {
		flag = newLazyFlag(-1, r)
	} else {
		flag = i.context(pos)
	}
	pc := re.onepass.Start
	inst := &re.onepass.Inst[pc]
	// If there is a simple literal prefix, skip over it.
	if pos == 0 && flag.match(syntax.EmptyOp(inst.Arg)) &&
		len(re.prefix) > 0 && i.canCheckPrefix() {
		// Match requires literal prefix; fast search for it.
		if !i.hasPrefix(re) {
			goto Return
		}
		pos += len(re.prefix)
		r, width = i.step(pos)
		r1, width1 = i.step(pos + width)
		flag = i.context(pos)
		pc = int(re.prefixEnd)
	}
	for {
		inst = &re.onepass.Inst[pc]
		pc = int(inst.Out)
		switch inst.Op {
		default:
			panic("bad inst")
		case syntax.InstMatch:
			matched = true
			if len(m.matchcap) > 0 {
				m.matchcap[0] = 0
				m.matchcap[1] = pos
			}
			goto Return
		case syntax.InstRune:
			if !inst.MatchRune(r) {
				goto Return
			}
		case syntax.InstRune1:
			if r != inst.Rune[0] {
				goto Return
			}
		case syntax.InstRuneAny:
			// Nothing
		case syntax.InstRuneAnyNotNL:
			if r == '\n' {
				goto Return
			}
		// peek at the input rune to see which branch of the Alt to take
		case syntax.InstAlt, syntax.InstAltMatch:
			pc = int(onePassNext(inst, r))
			continue
		case syntax.InstFail:
			goto Return
		case syntax.InstNop:
			continue
		case syntax.InstEmptyWidth:
			if !flag.match(syntax.EmptyOp(inst.Arg)) {
				goto Return
			}
			continue
		case syntax.InstCapture:
			if int(inst.Arg) < len(m.matchcap) {
				m.matchcap[inst.Arg] = pos
			}
			continue
		}
		if width == 0 {
			break
		}
		flag = newLazyFlag(r, r1)
		pos += width
		r, width = r1, width1
		if r != endOfText {
			r1, width1 = i.step(pos + width)
		}
	}

Return:
	if !matched {
		freeOnePassMachine(m)
		return nil
	}

	dstCap = append(dstCap, m.matchcap...)
	freeOnePassMachine(m)
	return dstCap
}

// doMatch reports whether either r, b or s match the regexp.
func (re *Regexp) doMatch(r io.RuneReader, b []byte, s string) bool {
	return re.doExecute(r, b, s, 0, 0, nil) != nil
}

// doExecute finds the leftmost match in the input, appends the position
// of its subexpressions to dstCap and returns dstCap.
//
// nil is returned if no matches are found and non-nil if matches are found.
func (re *Regexp) doExecute(r io.RuneReader, b []byte, s string, pos int, ncap int, dstCap []int) []int {
	if dstCap == nil {
		// Make sure 'return dstCap' is non-nil.
		dstCap = arrayNoInts[:0:0]
	}

	if r == nil && len(b)+len(s) < re.minInputLen {
		return nil
	}

	if re.onepass != nil {
		return re.doOnePass(r, b, s, pos, ncap, dstCap)
	}
	if r == nil && len(b)+len(s) < re.maxBitStateLen {
		return re.backtrack(b, s, pos, ncap, dstCap)
	}

	m := re.get()
	i, _ := m.inputs.init(r, b, s)

	m.init(ncap)
	if !m.match(i, pos) {
		re.put(m)
		return nil
	}

	dstCap = append(dstCap, m.matchcap...)
	re.put(m)
	return dstCap
}

// arrayNoInts is returned by doExecute match if nil dstCap is passed
// to it with ncap=0.
var arrayNoInts [0]int
