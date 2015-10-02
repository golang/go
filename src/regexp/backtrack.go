// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// backtrack is a regular expression search with submatch
// tracking for small regular expressions and texts. It allocates
// a bit vector with (length of input) * (length of prog) bits,
// to make sure it never explores the same (character position, instruction)
// state multiple times. This limits the search to run in time linear in
// the length of the test.
//
// backtrack is a fast replacement for the NFA code on small
// regexps when onepass cannot be used.

package regexp

import "regexp/syntax"

// A job is an entry on the backtracker's job stack. It holds
// the instruction pc and the position in the input.
type job struct {
	pc  uint32
	arg int
	pos int
}

const (
	visitedBits        = 32
	maxBacktrackProg   = 500        // len(prog.Inst) <= max
	maxBacktrackVector = 256 * 1024 // bit vector size <= max (bits)
)

// bitState holds state for the backtracker.
type bitState struct {
	prog *syntax.Prog

	end     int
	cap     []int
	input   input
	jobs    []job
	visited []uint32
}

var notBacktrack *bitState = nil

// maxBitStateLen returns the maximum length of a string to search with
// the backtracker using prog.
func maxBitStateLen(prog *syntax.Prog) int {
	if !shouldBacktrack(prog) {
		return 0
	}
	return maxBacktrackVector / len(prog.Inst)
}

// newBitState returns a new bitState for the given prog,
// or notBacktrack if the size of the prog exceeds the maximum size that
// the backtracker will be run for.
func newBitState(prog *syntax.Prog) *bitState {
	if !shouldBacktrack(prog) {
		return notBacktrack
	}
	return &bitState{
		prog: prog,
	}
}

// shouldBacktrack reports whether the program is too
// long for the backtracker to run.
func shouldBacktrack(prog *syntax.Prog) bool {
	return len(prog.Inst) <= maxBacktrackProg
}

// reset resets the state of the backtracker.
// end is the end position in the input.
// ncap is the number of captures.
func (b *bitState) reset(end int, ncap int) {
	b.end = end

	if cap(b.jobs) == 0 {
		b.jobs = make([]job, 0, 256)
	} else {
		b.jobs = b.jobs[:0]
	}

	visitedSize := (len(b.prog.Inst)*(end+1) + visitedBits - 1) / visitedBits
	if cap(b.visited) < visitedSize {
		b.visited = make([]uint32, visitedSize, maxBacktrackVector/visitedBits)
	} else {
		b.visited = b.visited[:visitedSize]
		for i := range b.visited {
			b.visited[i] = 0
		}
	}

	if cap(b.cap) < ncap {
		b.cap = make([]int, ncap)
	} else {
		b.cap = b.cap[:ncap]
	}
	for i := range b.cap {
		b.cap[i] = -1
	}
}

// shouldVisit reports whether the combination of (pc, pos) has not
// been visited yet.
func (b *bitState) shouldVisit(pc uint32, pos int) bool {
	n := uint(int(pc)*(b.end+1) + pos)
	if b.visited[n/visitedBits]&(1<<(n&(visitedBits-1))) != 0 {
		return false
	}
	b.visited[n/visitedBits] |= 1 << (n & (visitedBits - 1))
	return true
}

// push pushes (pc, pos, arg) onto the job stack if it should be
// visited.
func (b *bitState) push(pc uint32, pos int, arg int) {
	if b.prog.Inst[pc].Op == syntax.InstFail {
		return
	}

	// Only check shouldVisit when arg == 0.
	// When arg > 0, we are continuing a previous visit.
	if arg == 0 && !b.shouldVisit(pc, pos) {
		return
	}

	b.jobs = append(b.jobs, job{pc: pc, arg: arg, pos: pos})
}

// tryBacktrack runs a backtracking search starting at pos.
func (m *machine) tryBacktrack(b *bitState, i input, pc uint32, pos int) bool {
	longest := m.re.longest
	m.matched = false

	b.push(pc, pos, 0)
	for len(b.jobs) > 0 {
		l := len(b.jobs) - 1
		// Pop job off the stack.
		pc := b.jobs[l].pc
		pos := b.jobs[l].pos
		arg := b.jobs[l].arg
		b.jobs = b.jobs[:l]

		// Optimization: rather than push and pop,
		// code that is going to Push and continue
		// the loop simply updates ip, p, and arg
		// and jumps to CheckAndLoop.  We have to
		// do the ShouldVisit check that Push
		// would have, but we avoid the stack
		// manipulation.
		goto Skip
	CheckAndLoop:
		if !b.shouldVisit(pc, pos) {
			continue
		}
	Skip:

		inst := b.prog.Inst[pc]

		switch inst.Op {
		default:
			panic("bad inst")
		case syntax.InstFail:
			panic("unexpected InstFail")
		case syntax.InstAlt:
			// Cannot just
			//   b.push(inst.Out, pos, 0)
			//   b.push(inst.Arg, pos, 0)
			// If during the processing of inst.Out, we encounter
			// inst.Arg via another path, we want to process it then.
			// Pushing it here will inhibit that. Instead, re-push
			// inst with arg==1 as a reminder to push inst.Arg out
			// later.
			switch arg {
			case 0:
				b.push(pc, pos, 1)
				pc = inst.Out
				goto CheckAndLoop
			case 1:
				// Finished inst.Out; try inst.Arg.
				arg = 0
				pc = inst.Arg
				goto CheckAndLoop
			}
			panic("bad arg in InstAlt")

		case syntax.InstAltMatch:
			// One opcode consumes runes; the other leads to match.
			switch b.prog.Inst[inst.Out].Op {
			case syntax.InstRune, syntax.InstRune1, syntax.InstRuneAny, syntax.InstRuneAnyNotNL:
				// inst.Arg is the match.
				b.push(inst.Arg, pos, 0)
				pc = inst.Arg
				pos = b.end
				goto CheckAndLoop
			}
			// inst.Out is the match - non-greedy
			b.push(inst.Out, b.end, 0)
			pc = inst.Out
			goto CheckAndLoop

		case syntax.InstRune:
			r, width := i.step(pos)
			if !inst.MatchRune(r) {
				continue
			}
			pos += width
			pc = inst.Out
			goto CheckAndLoop

		case syntax.InstRune1:
			r, width := i.step(pos)
			if r != inst.Rune[0] {
				continue
			}
			pos += width
			pc = inst.Out
			goto CheckAndLoop

		case syntax.InstRuneAnyNotNL:
			r, width := i.step(pos)
			if r == '\n' || r == endOfText {
				continue
			}
			pos += width
			pc = inst.Out
			goto CheckAndLoop

		case syntax.InstRuneAny:
			r, width := i.step(pos)
			if r == endOfText {
				continue
			}
			pos += width
			pc = inst.Out
			goto CheckAndLoop

		case syntax.InstCapture:
			switch arg {
			case 0:
				if 0 <= inst.Arg && inst.Arg < uint32(len(b.cap)) {
					// Capture pos to register, but save old value.
					b.push(pc, b.cap[inst.Arg], 1) // come back when we're done.
					b.cap[inst.Arg] = pos
				}
				pc = inst.Out
				goto CheckAndLoop
			case 1:
				// Finished inst.Out; restore the old value.
				b.cap[inst.Arg] = pos
				continue

			}
			panic("bad arg in InstCapture")
			continue

		case syntax.InstEmptyWidth:
			if syntax.EmptyOp(inst.Arg)&^i.context(pos) != 0 {
				continue
			}
			pc = inst.Out
			goto CheckAndLoop

		case syntax.InstNop:
			pc = inst.Out
			goto CheckAndLoop

		case syntax.InstMatch:
			// We found a match. If the caller doesn't care
			// where the match is, no point going further.
			if len(b.cap) == 0 {
				m.matched = true
				return m.matched
			}

			// Record best match so far.
			// Only need to check end point, because this entire
			// call is only considering one start position.
			if len(b.cap) > 1 {
				b.cap[1] = pos
			}
			if !m.matched || (longest && pos > 0 && pos > m.matchcap[1]) {
				copy(m.matchcap, b.cap)
			}
			m.matched = true

			// If going for first match, we're done.
			if !longest {
				return m.matched
			}

			// If we used the entire text, no longer match is possible.
			if pos == b.end {
				return m.matched
			}

			// Otherwise, continue on in hope of a longer match.
			continue
		}
		panic("unreachable")
	}

	return m.matched
}

// backtrack runs a backtracking search of prog on the input starting at pos.
func (m *machine) backtrack(i input, pos int, end int, ncap int) bool {
	if !i.canCheckPrefix() {
		panic("backtrack called for a RuneReader")
	}

	startCond := m.re.cond
	if startCond == ^syntax.EmptyOp(0) { // impossible
		return false
	}
	if startCond&syntax.EmptyBeginText != 0 && pos != 0 {
		// Anchored match, past beginning of text.
		return false
	}

	b := m.b
	b.reset(end, ncap)

	m.matchcap = m.matchcap[:ncap]
	for i := range m.matchcap {
		m.matchcap[i] = -1
	}

	// Anchored search must start at the beginning of the input
	if startCond&syntax.EmptyBeginText != 0 {
		if len(b.cap) > 0 {
			b.cap[0] = pos
		}
		return m.tryBacktrack(b, i, uint32(m.p.Start), pos)
	}

	// Unanchored search, starting from each possible text position.
	// Notice that we have to try the empty string at the end of
	// the text, so the loop condition is pos <= end, not pos < end.
	// This looks like it's quadratic in the size of the text,
	// but we are not clearing visited between calls to TrySearch,
	// so no work is duplicated and it ends up still being linear.
	width := -1
	for ; pos <= end && width != 0; pos += width {
		if len(m.re.prefix) > 0 {
			// Match requires literal prefix; fast search for it.
			advance := i.index(m.re, pos)
			if advance < 0 {
				return false
			}
			pos += advance
		}

		if len(b.cap) > 0 {
			b.cap[0] = pos
		}
		if m.tryBacktrack(b, i, uint32(m.p.Start), pos) {
			// Match must be leftmost; done.
			return true
		}
		_, width = i.step(pos)
	}
	return false
}
