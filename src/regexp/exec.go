// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regexp

import (
	"io"
	"regexp/syntax"
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
	re             *Regexp      // corresponding Regexp
	p              *syntax.Prog // compiled program
	op             *onePassProg // compiled onepass program, or notOnePass
	maxBitStateLen int          // max length of string to search with bitstate
	b              *bitState    // state for backtracker, allocated lazily
	q0, q1         queue        // two queues for runq, nextq
	pool           []*thread    // pool of available threads
	matched        bool         // whether a match was found
	matchcap       []int        // capture information for the match

	needSaveFork bool
	// index of instrument of forked threads
	fork     []int
	nextfork []int
	// index of instrument of forked threads on last match
	matchfork     []int
	matchnextfork []int

	// cached inputs, to avoid allocation
	inputBytes  inputBytes
	inputString inputString
	inputReader inputReader
}

func (m *machine) newInputBytes(b []byte) input {
	m.inputBytes.str = b
	return &m.inputBytes
}

func (m *machine) newInputString(s string) input {
	m.inputString.str = s
	return &m.inputString
}

func (m *machine) newInputReader(r io.RuneReader) input {
	m.inputReader.r = r
	m.inputReader.atEOT = false
	m.inputReader.pos = 0
	return &m.inputReader
}

// progMachine returns a new machine running the prog p.
func progMachine(p *syntax.Prog, op *onePassProg) *machine {
	m := &machine{p: p, op: op}
	n := len(m.p.Inst)
	m.q0 = queue{make([]uint32, n), make([]entry, 0, n)}
	m.q1 = queue{make([]uint32, n), make([]entry, 0, n)}
	ncap := p.NumCap
	if ncap < 2 {
		ncap = 2
	}
	if op == notOnePass {
		m.maxBitStateLen = maxBitStateLen(p)
	}
	m.matchcap = make([]int, ncap)
	return m
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

// match runs the machine over the input starting at pos.
// It reports whether a match was found.
// If so, m.matchcap holds the submatch information.
func (m *machine) match(i input, pos int, proc []int) bool {
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
	var flag syntax.EmptyOp
	if pos == 0 {
		flag = syntax.EmptyOpContext(-1, r)
	} else {
		flag = i.context(pos)
	}
	var matchproc []bool
	if len(m.p.Fork) > 0 {
		matchproc = make([]bool, len(m.p.Fork))
		for _, n := range proc {
			m.add(runq, uint32(n), pos, m.matchcap, flag, nil, matchproc, true)
		}
	}
	numFork := len(runq.dense)
	for {
		if len(runq.dense)-numFork == 0 {
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
			m.add(runq, uint32(m.p.Start), pos, m.matchcap, flag, nil, matchproc, false)
		}
		if len(m.p.Fork) > 0 {
			for i := range matchproc {
				matchproc[i] = false
			}
			if m.needSaveFork {
				m.fork = append(m.fork[:0], m.nextfork...)
				m.nextfork = m.nextfork[:0]
			}
		}
		flag = syntax.EmptyOpContext(r, r1)
		numFork = m.step(runq, nextq, pos, pos+width, r, flag, numFork, matchproc)
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
func (m *machine) step(runq, nextq *queue, pos, nextPos int, c rune, nextCond syntax.EmptyOp, numFork int, matchproc []bool) int {
	longest := m.re.longest
	var nextNumFork int
	for j := 0; j < len(runq.dense); j++ {
		if numFork == j+1 {
			nextNumFork = len(nextq.dense)
		}

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
				if len(m.p.Fork) > 0 && m.needSaveFork {
					m.matchfork = append(m.matchfork[:0], m.fork...)
					m.matchnextfork = append(m.matchnextfork[:0], m.nextfork...)
				}
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
		case syntax.InstRuneAny, syntax.InstRepeatAny:
			add = true
		case syntax.InstRuneAnyNotNL:
			add = c != '\n'
		}
		if add {
			t = m.add(nextq, i.Out, nextPos, t.cap, nextCond, t, matchproc, j < numFork)
		}
		if t != nil {
			m.pool = append(m.pool, t)
		}
	}
	runq.dense = runq.dense[:0]
	return nextNumFork
}

// add adds an entry to q for pc, unless the q already has such an entry.
// It also recursively adds an entry for all instructions reachable from pc by following
// empty-width conditions satisfied by cond.  pos gives the current position
// in the input.
func (m *machine) add(q *queue, pc uint32, pos int, cap []int, cond syntax.EmptyOp, t *thread, matchproc []bool, forked bool) *thread {
	if pc == 0 {
		return t
	}
	if forked && m.needSaveFork {
		ok := true
		for _, n := range m.nextfork {
			if uint32(n) == pc {
				ok = false
				break
			}
		}
		if ok {
			m.nextfork = append(m.nextfork, int(pc))
		}
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
		t = m.add(q, i.Out, pos, cap, cond, t, matchproc, forked)
		t = m.add(q, i.Arg, pos, cap, cond, t, matchproc, forked)
	case syntax.InstEmptyWidth:
		if syntax.EmptyOp(i.Arg)&^cond == 0 {
			t = m.add(q, i.Out, pos, cap, cond, t, matchproc, forked)
		}
	case syntax.InstNop:
		t = m.add(q, i.Out, pos, cap, cond, t, matchproc, forked)
	case syntax.InstCapture:
		if int(i.Arg) < len(cap) {
			opos := cap[i.Arg]
			cap[i.Arg] = pos
			m.add(q, i.Out, pos, cap, cond, nil, matchproc, forked)
			cap[i.Arg] = opos
		} else {
			t = m.add(q, i.Out, pos, cap, cond, t, matchproc, forked)
		}
	case syntax.InstMatchProc:
		matchproc[i.Arg] = true
	case syntax.InstCheckProc:
		need := i.Arg&1 == 0
		if need == matchproc[i.Arg>>1] {
			m.add(q, i.Out, pos, cap, cond, nil, matchproc, forked)
		}
	case syntax.InstMatch, syntax.InstRune, syntax.InstRune1, syntax.InstRuneAny, syntax.InstRuneAnyNotNL, syntax.InstRepeatAny:
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
		if i.Op == syntax.InstRepeatAny {
			m.add(q, i.Arg, pos, cap, cond, nil, matchproc, forked)
		}
	}
	return t
}

// onepass runs the machine over the input starting at pos.
// It reports whether a match was found.
// If so, m.matchcap holds the submatch information.
// ncap is the number of captures.
func (m *machine) onepass(i input, pos, ncap int) bool {
	startCond := m.re.cond
	if startCond == ^syntax.EmptyOp(0) { // impossible
		return false
	}
	m.matched = false
	m.matchcap = m.matchcap[:ncap]
	for i := range m.matchcap {
		m.matchcap[i] = -1
	}
	r, r1 := endOfText, endOfText
	width, width1 := 0, 0
	r, width = i.step(pos)
	if r != endOfText {
		r1, width1 = i.step(pos + width)
	}
	var flag syntax.EmptyOp
	if pos == 0 {
		flag = syntax.EmptyOpContext(-1, r)
	} else {
		flag = i.context(pos)
	}
	pc := m.op.Start
	inst := m.op.Inst[pc]
	// If there is a simple literal prefix, skip over it.
	if pos == 0 && syntax.EmptyOp(inst.Arg)&^flag == 0 &&
		len(m.re.prefix) > 0 && i.canCheckPrefix() {
		// Match requires literal prefix; fast search for it.
		if !i.hasPrefix(m.re) {
			return m.matched
		}
		pos += len(m.re.prefix)
		r, width = i.step(pos)
		r1, width1 = i.step(pos + width)
		flag = i.context(pos)
		pc = int(m.re.prefixEnd)
	}
	for {
		inst = m.op.Inst[pc]
		pc = int(inst.Out)
		switch inst.Op {
		default:
			panic("bad inst")
		case syntax.InstMatch:
			m.matched = true
			if len(m.matchcap) > 0 {
				m.matchcap[0] = 0
				m.matchcap[1] = pos
			}
			return m.matched
		case syntax.InstRune:
			if !inst.MatchRune(r) {
				return m.matched
			}
		case syntax.InstRune1:
			if r != inst.Rune[0] {
				return m.matched
			}
		case syntax.InstRuneAny:
			// Nothing
		case syntax.InstRuneAnyNotNL:
			if r == '\n' {
				return m.matched
			}
		// peek at the input rune to see which branch of the Alt to take
		case syntax.InstAlt, syntax.InstAltMatch:
			pc = int(onePassNext(&inst, r))
			continue
		case syntax.InstFail:
			return m.matched
		case syntax.InstNop:
			continue
		case syntax.InstEmptyWidth:
			if syntax.EmptyOp(inst.Arg)&^flag != 0 {
				return m.matched
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
		flag = syntax.EmptyOpContext(r, r1)
		pos += width
		r, width = r1, width1
		if r != endOfText {
			r1, width1 = i.step(pos + width)
		}
	}
	return m.matched
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
	m := re.get()
	cap := re.doExecuteWithMachine(r, b, s, pos, ncap, dstCap, m.p.Fork, m)
	re.put(m)
	return cap
}

func (re *Regexp) doExecuteWithMachine(r io.RuneReader, b []byte, s string, pos int, ncap int, dstCap []int, proc []int, m *machine) []int {
	if len(m.p.Fork) > 0 && m.needSaveFork {
		m.fork = m.fork[:0]
		m.nextfork = m.nextfork[:0]
		m.clear(&m.q0)
		m.clear(&m.q1)
	}
	var i input
	var size int
	if r != nil {
		i = m.newInputReader(r)
	} else if b != nil {
		i = m.newInputBytes(b)
		size = len(b)
	} else {
		i = m.newInputString(s)
		size = len(s)
	}
	if m.op != notOnePass {
		if !m.onepass(i, pos, ncap) {
			return nil
		}
	} else if size < m.maxBitStateLen && r == nil {
		if m.b == nil {
			m.b = newBitState(m.p)
		}
		if !m.backtrack(i, pos, size, ncap) {
			return nil
		}
	} else {
		m.init(ncap)
		if !m.match(i, pos, proc) {
			return nil
		}
	}
	dstCap = append(dstCap, m.matchcap...)
	if dstCap == nil {
		// Keep the promise of returning non-nil value on match.
		dstCap = arrayNoInts[:0]
	}
	return dstCap
}

// arrayNoInts is returned by doExecute match if nil dstCap is passed
// to it with ncap=0.
var arrayNoInts [0]int
