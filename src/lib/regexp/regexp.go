// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package regexp implements a simple regular expression library.
//
// The syntax of the regular expressions accepted is:
//
//	regexp:
//		concatenation { '|' concatenation }
//	concatenation:
//		{ closure }
//	closure:
//		term [ '*' | '+' | '?' ]
//	term:
//		'^'
//		'$'
//		'.'
//		character
//		'[' [ '^' ] character-ranges ']'
//		'(' regexp ')'
//
package regexp

import (
	"os";
	"utf8";
	"vector";
)

var debug = false;

// Error codes returned by failures to parse an expression.
var (
	ErrInternal = os.NewError("internal error");
	ErrUnmatchedLpar = os.NewError("unmatched '('");
	ErrUnmatchedRpar = os.NewError("unmatched ')'");
	ErrUnmatchedLbkt = os.NewError("unmatched '['");
	ErrUnmatchedRbkt = os.NewError("unmatched ']'");
	ErrBadRange = os.NewError("bad range in character class");
	ErrExtraneousBackslash = os.NewError("extraneous backslash");
	ErrBadClosure = os.NewError("repeated closure (**, ++, etc.)");
	ErrBareClosure = os.NewError("closure applies to nothing");
	ErrBadBackslash = os.NewError("illegal backslash escape");
)

// An instruction executed by the NFA
type instr interface {
	kind()	int;	// the type of this instruction: _CHAR, _ANY, etc.
	next()	instr;	// the instruction to execute after this one
	setNext(i instr);
	index()	int;
	setIndex(i int);
	print();
}

// Fields and methods common to all instructions
type common struct {
	_next	instr;
	_index	int;
}

func (c *common) next() instr { return c._next }
func (c *common) setNext(i instr) { c._next = i }
func (c *common) index() int { return c._index }
func (c *common) setIndex(i int) { c._index = i }

// The representation of a compiled regular expression.
// The public interface is entirely through methods.
type Regexp struct {
	expr	string;	// the original expression
	ch	chan<- *Regexp;	// reply channel when we're done
	error	*os.Error;	// compile- or run-time error; nil if OK
	inst	*vector.Vector;
	start	instr;
	nbra	int;	// number of brackets in expression, for subexpressions
}

const (
	_START	// beginning of program
		= iota;
	_END;		// end of program: success
	_BOT;		// '^' beginning of text
	_EOT;		// '$' end of text
	_CHAR;	// 'a' regular character
	_CHARCLASS;	// [a-z] character class
	_ANY;		// '.' any character
	_BRA;		// '(' parenthesized expression
	_EBRA;	// ')'; end of '(' parenthesized expression
	_ALT;		// '|' alternation
	_NOP;		// do nothing; makes it easy to link without patching
)

// --- START start of program
type _Start struct {
	common
}

func (start *_Start) kind() int { return _START }
func (start *_Start) print() { print("start") }

// --- END end of program
type _End struct {
	common
}

func (end *_End) kind() int { return _END }
func (end *_End) print() { print("end") }

// --- BOT beginning of text
type _Bot struct {
	common
}

func (bot *_Bot) kind() int { return _BOT }
func (bot *_Bot) print() { print("bot") }

// --- EOT end of text
type _Eot struct {
	common
}

func (eot *_Eot) kind() int { return _EOT }
func (eot *_Eot) print() { print("eot") }

// --- CHAR a regular character
type _Char struct {
	common;
	char	int;
}

func (char *_Char) kind() int { return _CHAR }
func (char *_Char) print() { print("char ", string(char.char)) }

func newChar(char int) *_Char {
	c := new(_Char);
	c.char = char;
	return c;
}

// --- CHARCLASS [a-z]

type _CharClass struct {
	common;
	char	int;
	negate	bool;	// is character class negated? ([^a-z])
	// vector of int, stored pairwise: [a-z] is (a,z); x is (x,x):
	ranges	*vector.IntVector;
}

func (cclass *_CharClass) kind() int { return _CHARCLASS }

func (cclass *_CharClass) print() {
	print("charclass");
	if cclass.negate {
		print(" (negated)");
	}
	for i := 0; i < cclass.ranges.Len(); i += 2 {
		l := cclass.ranges.At(i);
		r := cclass.ranges.At(i+1);
		if l == r {
			print(" [", string(l), "]");
		} else {
			print(" [", string(l), "-", string(r), "]");
		}
	}
}

func (cclass *_CharClass) addRange(a, b int) {
	// range is a through b inclusive
	cclass.ranges.Push(a);
	cclass.ranges.Push(b);
}

func (cclass *_CharClass) matches(c int) bool {
	for i := 0; i < cclass.ranges.Len(); i = i+2 {
		min := cclass.ranges.At(i);
		max := cclass.ranges.At(i+1);
		if min <= c && c <= max {
			return !cclass.negate
		}
	}
	return cclass.negate
}

func newCharClass() *_CharClass {
	c := new(_CharClass);
	c.ranges = vector.NewIntVector(0);
	return c;
}

// --- ANY any character
type _Any struct {
	common
}

func (any *_Any) kind() int { return _ANY }
func (any *_Any) print() { print("any") }

// --- BRA parenthesized expression
type _Bra struct {
	common;
	n	int;	// subexpression number
}

func (bra *_Bra) kind() int { return _BRA }
func (bra *_Bra) print() { print("bra", bra.n); }

// --- EBRA end of parenthesized expression
type _Ebra struct {
	common;
	n	int;	// subexpression number
}

func (ebra *_Ebra) kind() int { return _EBRA }
func (ebra *_Ebra) print() { print("ebra ", ebra.n); }

// --- ALT alternation
type _Alt struct {
	common;
	left	instr;	// other branch
}

func (alt *_Alt) kind() int { return _ALT }
func (alt *_Alt) print() { print("alt(", alt.left.index(), ")"); }

// --- NOP no operation
type _Nop struct {
	common
}

func (nop *_Nop) kind() int { return _NOP }
func (nop *_Nop) print() { print("nop") }

// report error and exit compiling/executing goroutine
func (re *Regexp) setError(err *os.Error) {
	re.error = err;
	re.ch <- re;
	sys.Goexit();
}

func (re *Regexp) add(i instr) instr {
	i.setIndex(re.inst.Len());
	re.inst.Push(i);
	return i;
}

type parser struct {
	re	*Regexp;
	nlpar	int;	// number of unclosed lpars
	pos	int;
	ch	int;
}

const endOfFile = -1

func (p *parser) c() int {
	return p.ch;
}

func (p *parser) nextc() int {
	if p.pos >= len(p.re.expr) {
		p.ch = endOfFile
	} else {
		c, w := utf8.DecodeRuneInString(p.re.expr, p.pos);
		p.ch = c;
		p.pos += w;
	}
	return p.ch;
}

func newParser(re *Regexp) *parser {
	p := new(parser);
	p.re = re;
	p.nextc();	// load p.ch
	return p;
}

func (p *parser) regexp() (start, end instr)

var iNULL instr

func special(c int) bool {
	s := `\.+*?()|[]`;
	for i := 0; i < len(s); i++ {
		if c == int(s[i]) {
			return true
		}
	}
	return false
}

func specialcclass(c int) bool {
	s := `\-[]`;
	for i := 0; i < len(s); i++ {
		if c == int(s[i]) {
			return true
		}
	}
	return false
}

func (p *parser) charClass() instr {
	cc := newCharClass();
	p.re.add(cc);
	if p.c() == '^' {
		cc.negate = true;
		p.nextc();
	}
	left := -1;
	for {
		switch c := p.c(); c {
		case ']', endOfFile:
			if left >= 0 {
				p.re.setError(ErrBadRange);
			}
			return cc;
		case '-':	// do this before backslash processing
			p.re.setError(ErrBadRange);
		case '\\':
			c = p.nextc();
			switch {
			case c == endOfFile:
				p.re.setError(ErrExtraneousBackslash);
			case c == 'n':
				c = '\n';
			case specialcclass(c):
				// c is as delivered
			default:
				p.re.setError(ErrBadBackslash);
			}
			fallthrough;
		default:
			p.nextc();
			switch {
			case left < 0:	// first of pair
				if p.c() == '-' {	// range
					p.nextc();
					left = c;
				} else {	// single char
					cc.addRange(c, c);
				}
			case left <= c:	// second of pair
				cc.addRange(left, c);
				left = -1;
			default:
				p.re.setError(ErrBadRange);
			}
		}
	}
	return iNULL
}

func (p *parser) term() (start, end instr) {
	switch c := p.c(); c {
	case '|', endOfFile:
		return iNULL, iNULL;
	case '*', '+':
		p.re.setError(ErrBareClosure);
	case ')':
		if p.nlpar == 0 {
			p.re.setError(ErrUnmatchedRpar);
		}
		return iNULL, iNULL;
	case ']':
		p.re.setError(ErrUnmatchedRbkt);
	case '^':
		p.nextc();
		start = p.re.add(new(_Bot));
		return start, start;
	case '$':
		p.nextc();
		start = p.re.add(new(_Eot));
		return start, start;
	case '.':
		p.nextc();
		start = p.re.add(new(_Any));
		return start, start;
	case '[':
		p.nextc();
		start = p.charClass();
		if p.c() != ']' {
			p.re.setError(ErrUnmatchedLbkt);
		}
		p.nextc();
		return start, start;
	case '(':
		p.nextc();
		p.nlpar++;
		p.re.nbra++;	// increment first so first subexpr is \1
		nbra := p.re.nbra;
		start, end = p.regexp();
		if p.c() != ')' {
			p.re.setError(ErrUnmatchedLpar);
		}
		p.nlpar--;
		p.nextc();
		bra := new(_Bra);
		p.re.add(bra);
		ebra := new(_Ebra);
		p.re.add(ebra);
		bra.n = nbra;
		ebra.n = nbra;
		if start == iNULL {
			if end == iNULL {
				p.re.setError(ErrInternal)
			}
			start = ebra
		} else {
			end.setNext(ebra);
		}
		bra.setNext(start);
		return bra, ebra;
	case '\\':
		c = p.nextc();
		switch {
		case c == endOfFile:
			p.re.setError(ErrExtraneousBackslash);
		case c == 'n':
			c = '\n';
		case special(c):
			// c is as delivered
		default:
			p.re.setError(ErrBadBackslash);
		}
		fallthrough;
	default:
		p.nextc();
		start = newChar(c);
		p.re.add(start);
		return start, start
	}
	panic("unreachable");
}

func (p *parser) closure() (start, end instr) {
	start, end = p.term();
	if start == iNULL {
		return
	}
	switch p.c() {
	case '*':
		// (start,end)*:
		alt := new(_Alt);
		p.re.add(alt);
		end.setNext(alt);	// after end, do alt
		alt.left = start;	// alternate brach: return to start
		start = alt;	// alt becomes new (start, end)
		end = alt;
	case '+':
		// (start,end)+:
		alt := new(_Alt);
		p.re.add(alt);
		end.setNext(alt);	// after end, do alt
		alt.left = start;	// alternate brach: return to start
		end = alt;	// start is unchanged; end is alt
	case '?':
		// (start,end)?:
		alt := new(_Alt);
		p.re.add(alt);
		nop := new(_Nop);
		p.re.add(nop);
		alt.left = start;	// alternate branch is start
		alt.setNext(nop);	// follow on to nop
		end.setNext(nop);	// after end, go to nop
		start = alt;	// start is now alt
		end = nop;	// end is nop pointed to by both branches
	default:
		return
	}
	switch p.nextc() {
	case '*', '+', '?':
		p.re.setError(ErrBadClosure);
	}
	return
}

func (p *parser) concatenation() (start, end instr) {
	start, end = iNULL, iNULL;
	for {
		nstart, nend := p.closure();
		switch {
		case nstart == iNULL:	// end of this concatenation
			if start == iNULL {	// this is the empty string
				nop := p.re.add(new(_Nop));
				return nop, nop;
			}
			return;
		case start == iNULL:	// this is first element of concatenation
			start, end = nstart, nend;
		default:
			end.setNext(nstart);
			end = nend;
		}
	}
	panic("unreachable");
}

func (p *parser) regexp() (start, end instr) {
	start, end = p.concatenation();
	for {
		switch p.c() {
		default:
			return;
		case '|':
			p.nextc();
			nstart, nend := p.concatenation();
			alt := new(_Alt);
			p.re.add(alt);
			alt.left = start;
			alt.setNext(nstart);
			nop := new(_Nop);
			p.re.add(nop);
			end.setNext(nop);
			nend.setNext(nop);
			start, end = alt, nop;
		}
	}
	panic("unreachable");
}

func unNop(i instr) instr {
	for i.kind() == _NOP {
		i = i.next()
	}
	return i
}

func (re *Regexp) eliminateNops() {
	for i := 0; i < re.inst.Len(); i++ {
		inst := re.inst.At(i).(instr);
		if inst.kind() == _END {
			continue
		}
		inst.setNext(unNop(inst.next()));
		if inst.kind() == _ALT {
			alt := inst.(*_Alt);
			alt.left = unNop(alt.left);
		}
	}
}

func (re *Regexp) dump() {
	for i := 0; i < re.inst.Len(); i++ {
		inst := re.inst.At(i).(instr);
		print(inst.index(), ": ");
		inst.print();
		if inst.kind() != _END {
			print(" -> ", inst.next().index())
		}
		print("\n");
	}
}

func (re *Regexp) doParse() {
	p := newParser(re);
	start := new(_Start);
	re.add(start);
	s, e := p.regexp();
	start.setNext(s);
	re.start = start;
	e.setNext(re.add(new(_End)));

	if debug {
		re.dump();
		println();
	}

	re.eliminateNops();

	if debug {
		re.dump();
		println();
	}
}


func compiler(str string, ch chan *Regexp) {
	re := new(Regexp);
	re.expr = str;
	re.inst = vector.New(0);
	re.ch = ch;
	re.doParse();
	ch <- re;
}

// Compile parses a regular expression and returns, if successful, a Regexp
// object that can be used to match against text.
func Compile(str string) (regexp *Regexp, error *os.Error) {
	// Compile in a separate goroutine and wait for the result.
	ch := make(chan *Regexp);
	go compiler(str, ch);
	re := <-ch;
	return re, re.error
}

type state struct {
	inst	instr;	// next instruction to execute
	match	[]int;	// pairs of bracketing submatches. 0th is start,end
}

// Append new state to to-do list.  Leftmost-longest wins so avoid
// adding a state that's already active.
func addState(s []state, inst instr, match []int) []state {
	index := inst.index();
	l := len(s);
	pos := match[0];
	// TODO: Once the state is a vector and we can do insert, have inputs always
	// go in order correctly and this "earlier" test is never necessary,
	for i := 0; i < l; i++ {
		if s[i].inst.index() == index && // same instruction
		   s[i].match[0] < pos {	// earlier match already going; lefmost wins
		   	return s
		 }
	}
	if l == cap(s) {
		s1 := make([]state, 2*l)[0:l];
		for i := 0; i < l; i++ {
			s1[i] = s[i];
		}
		s = s1;
	}
	s = s[0:l+1];
	s[l].inst = inst;
	s[l].match = match;
	return s;
}

func (re *Regexp) doExecute(str string, pos int) []int {
	var s [2][]state;	// TODO: use a vector when state values (not ptrs) can be vector elements
	s[0] = make([]state, 10)[0:0];
	s[1] = make([]state, 10)[0:0];
	in, out := 0, 1;
	var final state;
	found := false;
	for pos <= len(str) {
		if !found {
			// prime the pump if we haven't seen a match yet
			match := make([]int, 2*(re.nbra+1));
			for i := 0; i < len(match); i++ {
				match[i] = -1;	// no match seen; catches cases like "a(b)?c" on "ac"
			}
			match[0]  = pos;
			s[out] = addState(s[out], re.start.next(), match);
		}
		in, out = out, in;	// old out state is new in state
		s[out] = s[out][0:0];	// clear out state
		if len(s[in]) == 0 {
			// machine has completed
			break;
		}
		charwidth := 1;
		c := endOfFile;
		if pos < len(str) {
			c, charwidth = utf8.DecodeRuneInString(str, pos);
		}
		for i := 0; i < len(s[in]); i++ {
			st := s[in][i];
			switch s[in][i].inst.kind() {
			case _BOT:
				if pos == 0 {
					s[in] = addState(s[in], st.inst.next(), st.match)
				}
			case _EOT:
				if pos == len(str) {
					s[in] = addState(s[in], st.inst.next(), st.match)
				}
			case _CHAR:
				if c == st.inst.(*_Char).char {
					s[out] = addState(s[out], st.inst.next(), st.match)
				}
			case _CHARCLASS:
				if st.inst.(*_CharClass).matches(c) {
					s[out] = addState(s[out], st.inst.next(), st.match)
				}
			case _ANY:
				if c != endOfFile {
					s[out] = addState(s[out], st.inst.next(), st.match)
				}
			case _BRA:
				n := st.inst.(*_Bra).n;
				st.match[2*n] = pos;
				s[in] = addState(s[in], st.inst.next(), st.match);
			case _EBRA:
				n := st.inst.(*_Ebra).n;
				st.match[2*n+1] = pos;
				s[in] = addState(s[in], st.inst.next(), st.match);
			case _ALT:
				s[in] = addState(s[in], st.inst.(*_Alt).left, st.match);
				// give other branch a copy of this match vector
				s1 := make([]int, 2*(re.nbra+1));
				for i := 0; i < len(s1); i++ {
					s1[i] = st.match[i]
				}
				s[in] = addState(s[in], st.inst.next(), s1);
			case _END:
				// choose leftmost longest
				if !found ||	// first
				   st.match[0] < final.match[0] ||	// leftmost
				   (st.match[0] == final.match[0] && pos > final.match[1])  {	// longest
					final = st;
					final.match[1] = pos;
				}
				found = true;
			default:
				st.inst.print();
				panic("unknown instruction in execute");
			}
		}
		pos += charwidth;
	}
	return final.match;
}


// Execute matches the Regexp against the string s.
// The return value is an array of integers, in pairs, identifying the positions of
// substrings matched by the expression.
//    s[a[0]:a[1]] is the substring matched by the entire expression.
//    s[a[2*i]:a[2*i+1]] for i > 0 is the substring matched by the ith parenthesized subexpression.
// A negative value means the subexpression did not match any element of the string.
// An empty array means "no match".
func (re *Regexp) Execute(s string) (a []int) {
	return re.doExecute(s, 0)
}


// Match returns whether the Regexp matches the string s.
// The return value is a boolean: true for match, false for no match.
func (re *Regexp) Match(s string) bool {
	return len(re.doExecute(s, 0)) > 0
}


// MatchStrings matches the Regexp against the string s.
// The return value is an array of strings matched by the expression.
//    a[0] is the substring matched by the entire expression.
//    a[i] for i > 0 is the substring matched by the ith parenthesized subexpression.
// An empty array means ``no match''.
func (re *Regexp) MatchStrings(s string) (a []string) {
	r := re.doExecute(s, 0);
	if r == nil {
		return nil
	}
	a = make([]string, len(r)/2);
	for i := 0; i < len(r); i += 2 {
		if r[i] != -1 {	// -1 means no match for this subexpression
			a[i/2] = s[r[i] : r[i+1]]
		}
	}
	return
}

// Match checks whether a textual regular expression
// matches a substring.  More complicated queries need
// to use Compile and the full Regexp interface.
func Match(pattern string, s string) (matched bool, error *os.Error) {
	re, err := Compile(pattern);
	if err != nil {
		return false, err
	}
	return re.Match(s), nil
}
