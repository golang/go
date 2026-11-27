// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regexp

import (
	"bytes"
	"regexp/syntax"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"unicode/utf8"
	"unsafe"
	"weak"
)

// The lazy DFA runs over the same rune-level programs as the NFA.
// Each state is the ordered list of threads left after consuming a rune,
// built on first use and cached with its transitions.
// The forward pass finds the end of the leftmost-first match.
// A pass over the reversed program, anchored at that end, then finds its start.

// dfaMaxStates bounds the memory of a state cache.
// A search that needs more states falls back to the NFA.
const dfaMaxStates = 2048

// dfaMaxOther bounds the transitions a state caches on non-ASCII runes.
// Past it, they are recomputed on each use.
const dfaMaxOther = 256

// dfaBackoff is the number of searches left to the other engines after the DFA gives up.
const dfaBackoff = 64

const (
	dfaMatch     uint8 = 1 << iota // a match ends just before the rune that led to this state
	dfaMatchSeen                   // a match was found, so no new thread starts
	dfaDead                        // no thread is left and none can start
	dfaIdle                        // no thread is left, but one starts at the next position
)

type dfaState struct {
	next  [utf8.RuneSelf]*dfaState
	other map[rune]*dfaState // transitions on non-ASCII runes and on end of text
	insts []uint32
	prev  rune // the class of the previous rune, as passed to syntax.EmptyOpContext
	flags uint8
}

// A dfa is one direction of the search.
type dfa struct {
	prog    *syntax.Prog
	forward bool           // unanchored leftmost-first if true, anchored leftmost-longest otherwise
	cond    syntax.EmptyOp // empty-width conditions required at start of match
}

type dfaProg struct {
	prog weak.Pointer[syntax.Prog] // tells a live entry from one whose key address was reused
	rev  *syntax.Prog
	pool sync.Pool    // of *dfaMachine
	skip atomic.Int32 // searches left to the other engines, see dfaBackoff
}

// dfaProgs maps the address of a forward *syntax.Prog to its *dfaProg.
// It lives outside Regexp so that compiled regexps stay comparable with reflect.DeepEqual after use.
// Entries hold no strong reference to their key, so that a cleanup can remove them.
var dfaProgs sync.Map

type dfaMachine struct {
	fwd, rev dfaCache
}

type dfaCache struct {
	states  map[string]*dfaState
	starts  [4]*dfaState // start states, by context of the rune before the start
	visited []bool
	stack   []uint32
	leaves  []uint32
	kernel  []uint32
	key     []byte
}

func (re *Regexp) dfaProg() *dfaProg {
	k := uintptr(unsafe.Pointer(re.prog))
	if p, ok := dfaProgs.Load(k); ok && p.(*dfaProg).prog.Value() == re.prog {
		return p.(*dfaProg)
	}
	// Longest is false, so re was compiled from Perl syntax.
	syn, err := syntax.Parse(re.expr, syntax.Perl)
	if err != nil {
		panic("regexp: recompiling " + re.expr + ": " + err.Error())
	}
	rev, err := syntax.Compile(reverseRegexp(syn.Simplify()))
	if err != nil {
		panic("regexp: recompiling " + re.expr + ": " + err.Error())
	}
	p := &dfaProg{prog: weak.Make(re.prog), rev: rev}
	dfaProgs.Store(k, p)
	runtime.AddCleanup(re.prog, func(p *dfaProg) { dfaProgs.CompareAndDelete(k, p) }, p)
	return p
}

func (p *dfaProg) get(fwd *syntax.Prog) *dfaMachine {
	if m, ok := p.pool.Get().(*dfaMachine); ok {
		return m
	}
	return &dfaMachine{
		fwd: dfaCache{states: make(map[string]*dfaState), visited: make([]bool, len(fwd.Inst))},
		rev: dfaCache{states: make(map[string]*dfaState), visited: make([]bool, len(p.rev.Inst))},
	}
}

// reverseRegexp returns a copy of re matching the reversed strings.
func reverseRegexp(re *syntax.Regexp) *syntax.Regexp {
	r := *re
	switch r.Op {
	case syntax.OpLiteral:
		r.Rune = slices.Clone(re.Rune)
		slices.Reverse(r.Rune)
	case syntax.OpBeginLine:
		r.Op = syntax.OpEndLine
	case syntax.OpEndLine:
		r.Op = syntax.OpBeginLine
	case syntax.OpBeginText:
		r.Op = syntax.OpEndText
	case syntax.OpEndText:
		r.Op = syntax.OpBeginText
	}
	if len(re.Sub) == 0 {
		return &r
	}
	r.Sub = make([]*syntax.Regexp, len(re.Sub))
	for i, sub := range re.Sub {
		r.Sub[i] = reverseRegexp(sub)
	}
	if r.Op == syntax.OpConcat {
		slices.Reverse(r.Sub)
	}
	return &r
}

// dfaContext reduces r to the only properties syntax.EmptyOpContext looks at.
func dfaContext(r rune) rune {
	switch {
	case r < 0, r == '\n':
		return r
	case syntax.IsWordChar(r):
		return 'a'
	}
	return ' '
}

func dfaContextIndex(ctx rune) int {
	switch ctx {
	case -1:
		return 0
	case '\n':
		return 1
	case 'a':
		return 2
	}
	return 3
}

// start returns the state before the first rune, given the rune r before it.
func (c *dfaCache) start(d *dfa, r rune) *dfaState {
	ctx := dfaContext(r)
	if st := c.starts[dfaContextIndex(ctx)]; st != nil {
		return st
	}
	var st *dfaState
	if d.forward {
		st = c.intern(nil, ctx, dfaIdle)
	} else {
		st = c.intern([]uint32{uint32(d.prog.Start)}, ctx, 0)
	}
	c.starts[dfaContextIndex(ctx)] = st
	return st
}

func (c *dfaCache) reset() {
	clear(c.states)
	c.starts = [4]*dfaState{}
}

func (c *dfaCache) intern(insts []uint32, prev rune, flags uint8) *dfaState {
	c.key = append(c.key[:0], byte(prev), flags)
	for _, pc := range insts {
		c.key = append(c.key, byte(pc), byte(pc>>8), byte(pc>>16), byte(pc>>24))
	}
	if st, ok := c.states[string(c.key)]; ok {
		return st
	}
	if len(c.states) >= dfaMaxStates {
		return nil
	}
	st := &dfaState{insts: slices.Clone(insts), prev: prev, flags: flags}
	c.states[string(c.key)] = st
	return st
}

// closure appends to c.leaves, in priority order, the rune and match
// instructions reachable from pc when the empty-width conditions in flags hold.
func (c *dfaCache) closure(prog *syntax.Prog, pc uint32, flags syntax.EmptyOp) {
	c.stack = append(c.stack[:0], pc)
	for len(c.stack) > 0 {
		pc := c.stack[len(c.stack)-1]
		c.stack = c.stack[:len(c.stack)-1]
		if c.visited[pc] {
			continue
		}
		c.visited[pc] = true
		i := &prog.Inst[pc]
		switch i.Op {
		case syntax.InstAlt, syntax.InstAltMatch:
			c.stack = append(c.stack, i.Arg, i.Out)
		case syntax.InstCapture, syntax.InstNop:
			c.stack = append(c.stack, i.Out)
		case syntax.InstEmptyWidth:
			if syntax.EmptyOp(i.Arg)&^flags == 0 {
				c.stack = append(c.stack, i.Out)
			}
		case syntax.InstMatch, syntax.InstRune, syntax.InstRune1, syntax.InstRuneAny, syntax.InstRuneAnyNotNL:
			c.leaves = append(c.leaves, pc)
		}
	}
}

// step returns the state reached from st on r, where r < 0 means end of text.
// It returns nil when the cache is full.
func (c *dfaCache) step(d *dfa, st *dfaState, r rune) *dfaState {
	flags := syntax.EmptyOpContext(st.prev, r)
	c.leaves = c.leaves[:0]
	for _, pc := range st.insts {
		c.closure(d.prog, pc, flags)
	}
	if d.forward && st.flags&dfaMatchSeen == 0 && (d.cond&syntax.EmptyBeginText == 0 || st.prev < 0) {
		c.closure(d.prog, uint32(d.prog.Start), flags)
	}
	clear(c.visited)

	var f uint8
	c.kernel = c.kernel[:0]
	for _, pc := range c.leaves {
		i := &d.prog.Inst[pc]
		if i.Op == syntax.InstMatch {
			f |= dfaMatch
			if d.forward {
				// Leftmost-first: lower-priority threads are cut.
				break
			}
			continue
		}
		if r >= 0 && i.MatchRune(r) && !slices.Contains(c.kernel, i.Out) {
			c.kernel = append(c.kernel, i.Out)
		}
	}
	if d.forward && (f&dfaMatch != 0 || st.flags&dfaMatchSeen != 0) {
		f |= dfaMatchSeen
	}
	if len(c.kernel) == 0 {
		if d.forward && f&dfaMatchSeen == 0 && d.cond&syntax.EmptyBeginText == 0 {
			f |= dfaIdle
		} else {
			f |= dfaDead
		}
	}
	return c.intern(c.kernel, dfaContext(r), f)
}

func (c *dfaCache) next(d *dfa, st *dfaState, r rune) *dfaState {
	if 0 <= r && r < utf8.RuneSelf {
		if nx := st.next[r]; nx != nil {
			return nx
		}
		nx := c.step(d, st, r)
		st.next[r] = nx
		return nx
	}
	if nx, ok := st.other[r]; ok {
		return nx
	}
	nx := c.step(d, st, r)
	if nx != nil && len(st.other) < dfaMaxOther {
		if st.other == nil {
			st.other = make(map[rune]*dfaState)
		}
		st.other[r] = nx
	}
	return nx
}

func dfaRuneAt[T []byte | string](s T, i int) (rune, int) {
	if i >= len(s) {
		return -1, 0
	}
	if c := s[i]; c < utf8.RuneSelf {
		return rune(c), 1
	}
	var buf [utf8.UTFMax]byte
	return utf8.DecodeRune(buf[:copy(buf[:], s[i:])])
}

func dfaRuneBefore[T []byte | string](s T, i int) (rune, int) {
	if i <= 0 {
		return -1, 0
	}
	if c := s[i-1]; c < utf8.RuneSelf {
		return rune(c), 1
	}
	var buf [utf8.UTFMax]byte
	return utf8.DecodeLastRune(buf[:copy(buf[:], s[max(0, i-utf8.UTFMax):i])])
}

// dfaForward returns the end of the leftmost-first match starting at or after pos, or -1.
// With earliest, it returns as soon as a match is known to exist.
// index, if not nil, returns the next position at or after i where a match can start, or -1.
// ok is false when the state cache filled up.
func dfaForward[T []byte | string](c *dfaCache, d *dfa, s T, pos int, earliest bool, index func(i int) int) (end int, ok bool) {
	end = -1
	r, _ := dfaRuneBefore(s, pos)
	st := c.start(d, r)
	if st == nil {
		return -1, false
	}
	for i := pos; ; {
		if st.flags&dfaIdle != 0 && index != nil {
			j := index(i)
			if j < 0 {
				return end, true
			}
			if j > i {
				i = j
				r, _ := dfaRuneBefore(s, i)
				if st = c.start(d, r); st == nil {
					return -1, false
				}
			}
		}
		if i >= len(s) {
			nx := c.next(d, st, -1)
			if nx == nil {
				return -1, false
			}
			if nx.flags&dfaMatch != 0 {
				end = i
			}
			return end, true
		}
		w := 1
		var nx *dfaState
		if b := s[i]; b < utf8.RuneSelf {
			if nx = st.next[b]; nx == nil {
				nx = c.next(d, st, rune(b))
			}
		} else {
			var r rune
			r, w = dfaRuneAt(s, i)
			nx = c.next(d, st, r)
		}
		if nx == nil {
			return -1, false
		}
		st = nx
		if st.flags&(dfaMatch|dfaDead) != 0 {
			if st.flags&dfaMatch != 0 {
				end = i
				if earliest {
					return end, true
				}
			}
			if st.flags&dfaDead != 0 {
				return end, true
			}
		}
		i += w
	}
}

// dfaReverse returns the start of the longest match of the reversed program
// that ends at end and starts at or after pos, or -1.
func dfaReverse[T []byte | string](c *dfaCache, d *dfa, s T, pos, end int) (start int, ok bool) {
	start = -1
	r, _ := dfaRuneAt(s, end)
	st := c.start(d, r)
	if st == nil {
		return -1, false
	}
	for i := end; ; {
		// Decoding stops at pos, as the forward pass decoded from there.
		r, w := dfaRuneBefore(s[pos:], i-pos)
		if i <= pos {
			r, _ = dfaRuneBefore(s, i)
		}
		nx := c.next(d, st, r)
		if nx == nil {
			return -1, false
		}
		if nx.flags&dfaMatch != 0 {
			start = i
		}
		if i <= pos || nx.flags&dfaDead != 0 {
			return start, true
		}
		st = nx
		i -= w
	}
}

func dfaSearch[T []byte | string](m *dfaMachine, fwd, rev *dfa, s T, pos int, earliest bool, index func(i int) int) (start, end int, ok bool) {
	if len(m.fwd.states) >= dfaMaxStates {
		m.fwd.reset()
	}
	if len(m.rev.states) >= dfaMaxStates {
		m.rev.reset()
	}
	end, ok = dfaForward(&m.fwd, fwd, s, pos, earliest, index)
	if !ok || end < 0 || earliest {
		return -1, end, ok
	}
	start, ok = dfaReverse(&m.rev, rev, s, pos, end)
	if start < 0 {
		// The forward pass found a match, so the reverse one must too.
		ok = false
	}
	return start, end, ok
}

// dfaMinPrefixInput is the input length below which a regexp with a literal prefix
// is left to the other engines, which jump to that prefix with a substring search.
const dfaMinPrefixInput = 1024

// useDFA reports whether the DFA should run on n bytes of input before the other engines.
func (re *Regexp) useDFA(n, ncap int) bool {
	switch {
	case re.longest, re.prefixComplete:
		return false
	case ncap > 2:
		// The other engines still run from the start of the match to find the submatches.
		return re.onepass == nil && n >= re.maxBitStateLen
	case re.prefix != "":
		// A onepass regexp is anchored, so the onepass engine rejects a missing prefix at once.
		return re.onepass == nil && n >= dfaMinPrefixInput
	}
	return true
}

// doDFA runs the DFA for find. When done is false, the other engines
// must run from pos, either because the DFA gave up or to find the submatches.
func (re *Regexp) doDFA(b []byte, s string, pos, ncap int, dstCap []int) (_ []int, _ int, done bool) {
	start, end, ok := re.dfaBounds(b, s, pos, ncap == 0)
	switch {
	case !ok:
		return dstCap, pos, false
	case end < 0:
		return nil, pos, true
	case ncap == 0:
		return dstCap, pos, true
	case ncap == 2:
		return append(dstCap, start, end), pos, true
	}
	return dstCap, start, false
}

// dfaBounds reports the bounds of the leftmost-first match at or after pos.
// With earliest, start is not computed.
// ok is false when the DFA gave up and the caller must use another engine.
func (re *Regexp) dfaBounds(b []byte, s string, pos int, earliest bool) (start, end int, ok bool) {
	if re.cond&syntax.EmptyBeginText != 0 && pos != 0 {
		return -1, -1, true
	}
	p := re.dfaProg()
	if p.skip.Load() > 0 {
		p.skip.Add(-1)
		return -1, -1, false
	}
	fwd := dfa{prog: re.prog, forward: true, cond: re.cond}
	rev := dfa{prog: p.rev}
	m := p.get(re.prog)
	usePrefix := re.prefix != "" && re.onepass == nil
	if b != nil {
		var index func(int) int
		if usePrefix {
			index = func(i int) int {
				if j := bytes.Index(b[i:], re.prefixBytes); j >= 0 {
					return i + j
				}
				return -1
			}
		}
		start, end, ok = dfaSearch(m, &fwd, &rev, b, pos, earliest, index)
	} else {
		var index func(int) int
		if usePrefix {
			index = func(i int) int {
				if j := strings.Index(s[i:], re.prefix); j >= 0 {
					return i + j
				}
				return -1
			}
		}
		start, end, ok = dfaSearch(m, &fwd, &rev, s, pos, earliest, index)
	}
	p.pool.Put(m)
	if !ok {
		p.skip.Store(dfaBackoff)
	}
	return start, end, ok
}
