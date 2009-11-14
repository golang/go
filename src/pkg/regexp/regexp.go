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
	"bytes";
	"container/vector";
	"io";
	"os";
	"utf8";
)

var debug = false

// Error codes returned by failures to parse an expression.
var (
	ErrInternal		= os.NewError("internal error");
	ErrUnmatchedLpar	= os.NewError("unmatched '('");
	ErrUnmatchedRpar	= os.NewError("unmatched ')'");
	ErrUnmatchedLbkt	= os.NewError("unmatched '['");
	ErrUnmatchedRbkt	= os.NewError("unmatched ']'");
	ErrBadRange		= os.NewError("bad range in character class");
	ErrExtraneousBackslash	= os.NewError("extraneous backslash");
	ErrBadClosure		= os.NewError("repeated closure (**, ++, etc.)");
	ErrBareClosure		= os.NewError("closure applies to nothing");
	ErrBadBackslash		= os.NewError("illegal backslash escape");
)

// An instruction executed by the NFA
type instr interface {
	kind() int;	// the type of this instruction: _CHAR, _ANY, etc.
	next() instr;	// the instruction to execute after this one
	setNext(i instr);
	index() int;
	setIndex(i int);
	print();
}

// Fields and methods common to all instructions
type common struct {
	_next	instr;
	_index	int;
}

func (c *common) next() instr		{ return c._next }
func (c *common) setNext(i instr)	{ c._next = i }
func (c *common) index() int		{ return c._index }
func (c *common) setIndex(i int)	{ c._index = i }

// Regexp is the representation of a compiled regular expression.
// The public interface is entirely through methods.
type Regexp struct {
	expr	string;	// the original expression
	inst	*vector.Vector;
	start	instr;
	nbra	int;	// number of brackets in expression, for subexpressions
}

const (
	_START		= iota;	// beginning of program
	_END;		// end of program: success
	_BOT;		// '^' beginning of text
	_EOT;		// '$' end of text
	_CHAR;		// 'a' regular character
	_CHARCLASS;	// [a-z] character class
	_ANY;		// '.' any character including newline
	_NOTNL;		// [^\n] special case: any character but newline
	_BRA;		// '(' parenthesized expression
	_EBRA;		// ')'; end of '(' parenthesized expression
	_ALT;		// '|' alternation
	_NOP;		// do nothing; makes it easy to link without patching
)

// --- START start of program
type _Start struct {
	common;
}

func (start *_Start) kind() int	{ return _START }
func (start *_Start) print()	{ print("start") }

// --- END end of program
type _End struct {
	common;
}

func (end *_End) kind() int	{ return _END }
func (end *_End) print()	{ print("end") }

// --- BOT beginning of text
type _Bot struct {
	common;
}

func (bot *_Bot) kind() int	{ return _BOT }
func (bot *_Bot) print()	{ print("bot") }

// --- EOT end of text
type _Eot struct {
	common;
}

func (eot *_Eot) kind() int	{ return _EOT }
func (eot *_Eot) print()	{ print("eot") }

// --- CHAR a regular character
type _Char struct {
	common;
	char	int;
}

func (char *_Char) kind() int	{ return _CHAR }
func (char *_Char) print()	{ print("char ", string(char.char)) }

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

func (cclass *_CharClass) kind() int	{ return _CHARCLASS }

func (cclass *_CharClass) print() {
	print("charclass");
	if cclass.negate {
		print(" (negated)")
	}
	for i := 0; i < cclass.ranges.Len(); i += 2 {
		l := cclass.ranges.At(i);
		r := cclass.ranges.At(i + 1);
		if l == r {
			print(" [", string(l), "]")
		} else {
			print(" [", string(l), "-", string(r), "]")
		}
	}
}

func (cclass *_CharClass) addRange(a, b int) {
	// range is a through b inclusive
	cclass.ranges.Push(a);
	cclass.ranges.Push(b);
}

func (cclass *_CharClass) matches(c int) bool {
	for i := 0; i < cclass.ranges.Len(); i = i + 2 {
		min := cclass.ranges.At(i);
		max := cclass.ranges.At(i + 1);
		if min <= c && c <= max {
			return !cclass.negate
		}
	}
	return cclass.negate;
}

func newCharClass() *_CharClass {
	c := new(_CharClass);
	c.ranges = vector.NewIntVector(0);
	return c;
}

// --- ANY any character
type _Any struct {
	common;
}

func (any *_Any) kind() int	{ return _ANY }
func (any *_Any) print()	{ print("any") }

// --- NOTNL any character but newline
type _NotNl struct {
	common;
}

func (notnl *_NotNl) kind() int	{ return _NOTNL }
func (notnl *_NotNl) print()	{ print("notnl") }

// --- BRA parenthesized expression
type _Bra struct {
	common;
	n	int;	// subexpression number
}

func (bra *_Bra) kind() int	{ return _BRA }
func (bra *_Bra) print()	{ print("bra", bra.n) }

// --- EBRA end of parenthesized expression
type _Ebra struct {
	common;
	n	int;	// subexpression number
}

func (ebra *_Ebra) kind() int	{ return _EBRA }
func (ebra *_Ebra) print()	{ print("ebra ", ebra.n) }

// --- ALT alternation
type _Alt struct {
	common;
	left	instr;	// other branch
}

func (alt *_Alt) kind() int	{ return _ALT }
func (alt *_Alt) print()	{ print("alt(", alt.left.index(), ")") }

// --- NOP no operation
type _Nop struct {
	common;
}

func (nop *_Nop) kind() int	{ return _NOP }
func (nop *_Nop) print()	{ print("nop") }

func (re *Regexp) add(i instr) instr {
	i.setIndex(re.inst.Len());
	re.inst.Push(i);
	return i;
}

type parser struct {
	re	*Regexp;
	error	os.Error;
	nlpar	int;	// number of unclosed lpars
	pos	int;
	ch	int;
}

const endOfFile = -1

func (p *parser) c() int	{ return p.ch }

func (p *parser) nextc() int {
	if p.pos >= len(p.re.expr) {
		p.ch = endOfFile
	} else {
		c, w := utf8.DecodeRuneInString(p.re.expr[p.pos:len(p.re.expr)]);
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

func special(c int) bool {
	s := `\.+*?()|[]^$`;
	for i := 0; i < len(s); i++ {
		if c == int(s[i]) {
			return true
		}
	}
	return false;
}

func specialcclass(c int) bool {
	s := `\-[]`;
	for i := 0; i < len(s); i++ {
		if c == int(s[i]) {
			return true
		}
	}
	return false;
}

func (p *parser) charClass() instr {
	cc := newCharClass();
	if p.c() == '^' {
		cc.negate = true;
		p.nextc();
	}
	left := -1;
	for {
		switch c := p.c(); c {
		case ']', endOfFile:
			if left >= 0 {
				p.error = ErrBadRange;
				return nil;
			}
			// Is it [^\n]?
			if cc.negate && cc.ranges.Len() == 2 &&
				cc.ranges.At(0) == '\n' && cc.ranges.At(1) == '\n' {
				nl := new(_NotNl);
				p.re.add(nl);
				return nl;
			}
			p.re.add(cc);
			return cc;
		case '-':	// do this before backslash processing
			p.error = ErrBadRange;
			return nil;
		case '\\':
			c = p.nextc();
			switch {
			case c == endOfFile:
				p.error = ErrExtraneousBackslash;
				return nil;
			case c == 'n':
				c = '\n'
			case specialcclass(c):
				// c is as delivered
			default:
				p.error = ErrBadBackslash;
				return nil;
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
					cc.addRange(c, c)
				}
			case left <= c:	// second of pair
				cc.addRange(left, c);
				left = -1;
			default:
				p.error = ErrBadRange;
				return nil;
			}
		}
	}
	return nil;
}

func (p *parser) term() (start, end instr) {
	// term() is the leaf of the recursion, so it's sufficient to pick off the
	// error state here for early exit.
	// The other functions (closure(), concatenation() etc.) assume
	// it's safe to recur to here.
	if p.error != nil {
		return
	}
	switch c := p.c(); c {
	case '|', endOfFile:
		return nil, nil
	case '*', '+':
		p.error = ErrBareClosure;
		return;
	case ')':
		if p.nlpar == 0 {
			p.error = ErrUnmatchedRpar;
			return;
		}
		return nil, nil;
	case ']':
		p.error = ErrUnmatchedRbkt;
		return;
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
		if p.error != nil {
			return
		}
		if p.c() != ']' {
			p.error = ErrUnmatchedLbkt;
			return;
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
			p.error = ErrUnmatchedLpar;
			return;
		}
		p.nlpar--;
		p.nextc();
		bra := new(_Bra);
		p.re.add(bra);
		ebra := new(_Ebra);
		p.re.add(ebra);
		bra.n = nbra;
		ebra.n = nbra;
		if start == nil {
			if end == nil {
				p.error = ErrInternal;
				return;
			}
			start = ebra;
		} else {
			end.setNext(ebra)
		}
		bra.setNext(start);
		return bra, ebra;
	case '\\':
		c = p.nextc();
		switch {
		case c == endOfFile:
			p.error = ErrExtraneousBackslash;
			return;
		case c == 'n':
			c = '\n'
		case special(c):
			// c is as delivered
		default:
			p.error = ErrBadBackslash;
			return;
		}
		fallthrough;
	default:
		p.nextc();
		start = newChar(c);
		p.re.add(start);
		return start, start;
	}
	panic("unreachable");
}

func (p *parser) closure() (start, end instr) {
	start, end = p.term();
	if start == nil || p.error != nil {
		return
	}
	switch p.c() {
	case '*':
		// (start,end)*:
		alt := new(_Alt);
		p.re.add(alt);
		end.setNext(alt);	// after end, do alt
		alt.left = start;	// alternate brach: return to start
		start = alt;		// alt becomes new (start, end)
		end = alt;
	case '+':
		// (start,end)+:
		alt := new(_Alt);
		p.re.add(alt);
		end.setNext(alt);	// after end, do alt
		alt.left = start;	// alternate brach: return to start
		end = alt;		// start is unchanged; end is alt
	case '?':
		// (start,end)?:
		alt := new(_Alt);
		p.re.add(alt);
		nop := new(_Nop);
		p.re.add(nop);
		alt.left = start;	// alternate branch is start
		alt.setNext(nop);	// follow on to nop
		end.setNext(nop);	// after end, go to nop
		start = alt;		// start is now alt
		end = nop;		// end is nop pointed to by both branches
	default:
		return
	}
	switch p.nextc() {
	case '*', '+', '?':
		p.error = ErrBadClosure
	}
	return;
}

func (p *parser) concatenation() (start, end instr) {
	for {
		nstart, nend := p.closure();
		if p.error != nil {
			return
		}
		switch {
		case nstart == nil:	// end of this concatenation
			if start == nil {	// this is the empty string
				nop := p.re.add(new(_Nop));
				return nop, nop;
			}
			return;
		case start == nil:	// this is first element of concatenation
			start, end = nstart, nend
		default:
			end.setNext(nstart);
			end = nend;
		}
	}
	panic("unreachable");
}

func (p *parser) regexp() (start, end instr) {
	start, end = p.concatenation();
	if p.error != nil {
		return
	}
	for {
		switch p.c() {
		default:
			return
		case '|':
			p.nextc();
			nstart, nend := p.concatenation();
			if p.error != nil {
				return
			}
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
	return i;
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

func (re *Regexp) doParse() os.Error {
	p := newParser(re);
	start := new(_Start);
	re.add(start);
	s, e := p.regexp();
	if p.error != nil {
		return p.error
	}
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
	return p.error;
}

// Compile parses a regular expression and returns, if successful, a Regexp
// object that can be used to match against text.
func Compile(str string) (regexp *Regexp, error os.Error) {
	regexp = new(Regexp);
	regexp.expr = str;
	regexp.inst = vector.New(0);
	error = regexp.doParse();
	return;
}

// MustCompile is like Compile but panics if the expression cannot be parsed.
// It simplifies safe initialization of global variables holding compiled regular
// expressions.
func MustCompile(str string) *Regexp {
	regexp, error := Compile(str);
	if error != nil {
		panicln(`regexp: compiling "`, str, `": `, error.String())
	}
	return regexp;
}

type state struct {
	inst	instr;	// next instruction to execute
	match	[]int;	// pairs of bracketing submatches. 0th is start,end
}

// Append new state to to-do list.  Leftmost-longest wins so avoid
// adding a state that's already active.
func (re *Regexp) addState(s []state, inst instr, match []int, pos, end int) []state {
	switch inst.kind() {
	case _BOT:
		if pos == 0 {
			s = re.addState(s, inst.next(), match, pos, end)
		}
		return s;
	case _EOT:
		if pos == end {
			s = re.addState(s, inst.next(), match, pos, end)
		}
		return s;
	case _BRA:
		n := inst.(*_Bra).n;
		match[2*n] = pos;
		s = re.addState(s, inst.next(), match, pos, end);
		return s;
	case _EBRA:
		n := inst.(*_Ebra).n;
		match[2*n+1] = pos;
		s = re.addState(s, inst.next(), match, pos, end);
		return s;
	}
	index := inst.index();
	l := len(s);
	begin := match[0];
	// TODO: Once the state is a vector and we can do insert, have inputs always
	// go in order correctly and this "earlier" test is never necessary,
	for i := 0; i < l; i++ {
		if s[i].inst.index() == index &&	// same instruction
			s[i].match[0] <= begin {	// earlier match already going; lefmost wins
			return s
		}
	}
	if l == cap(s) {
		s1 := make([]state, 2*l)[0:l];
		for i := 0; i < l; i++ {
			s1[i] = s[i]
		}
		s = s1;
	}
	s = s[0 : l+1];
	s[l].inst = inst;
	s[l].match = match;
	if inst.kind() == _ALT {
		s1 := make([]int, 2*(re.nbra+1));
		for i := 0; i < len(s1); i++ {
			s1[i] = match[i]
		}
		s = re.addState(s, inst.(*_Alt).left, s1, pos, end);
		// give other branch a copy of this match vector
		s1 = make([]int, 2*(re.nbra+1));
		for i := 0; i < len(s1); i++ {
			s1[i] = match[i]
		}
		s = re.addState(s, inst.next(), s1, pos, end);
	}
	return s;
}

// Accepts either string or bytes - the logic is identical either way.
// If bytes == nil, scan str.
func (re *Regexp) doExecute(str string, bytes []byte, pos int) []int {
	var s [2][]state;	// TODO: use a vector when state values (not ptrs) can be vector elements
	s[0] = make([]state, 10)[0:0];
	s[1] = make([]state, 10)[0:0];
	in, out := 0, 1;
	var final state;
	found := false;
	end := len(str);
	if bytes != nil {
		end = len(bytes)
	}
	for pos <= end {
		if !found {
			// prime the pump if we haven't seen a match yet
			match := make([]int, 2*(re.nbra+1));
			for i := 0; i < len(match); i++ {
				match[i] = -1	// no match seen; catches cases like "a(b)?c" on "ac"
			}
			match[0] = pos;
			s[out] = re.addState(s[out], re.start.next(), match, pos, end);
		}
		in, out = out, in;	// old out state is new in state
		s[out] = s[out][0:0];	// clear out state
		if found && len(s[in]) == 0 {
			// machine has completed
			break
		}
		charwidth := 1;
		c := endOfFile;
		if pos < end {
			if bytes == nil {
				c, charwidth = utf8.DecodeRuneInString(str[pos:end])
			} else {
				c, charwidth = utf8.DecodeRune(bytes[pos:end])
			}
		}
		pos += charwidth;
		for i := 0; i < len(s[in]); i++ {
			st := s[in][i];
			switch s[in][i].inst.kind() {
			case _BOT:
			case _EOT:
			case _CHAR:
				if c == st.inst.(*_Char).char {
					s[out] = re.addState(s[out], st.inst.next(), st.match, pos, end)
				}
			case _CHARCLASS:
				if st.inst.(*_CharClass).matches(c) {
					s[out] = re.addState(s[out], st.inst.next(), st.match, pos, end)
				}
			case _ANY:
				if c != endOfFile {
					s[out] = re.addState(s[out], st.inst.next(), st.match, pos, end)
				}
			case _NOTNL:
				if c != endOfFile && c != '\n' {
					s[out] = re.addState(s[out], st.inst.next(), st.match, pos, end)
				}
			case _BRA:
			case _EBRA:
			case _ALT:
			case _END:
				// choose leftmost longest
				if !found ||	// first
					st.match[0] < final.match[0] ||	// leftmost
					(st.match[0] == final.match[0] && pos-charwidth > final.match[1]) {	// longest
					final = st;
					final.match[1] = pos - charwidth;
				}
				found = true;
			default:
				st.inst.print();
				panic("unknown instruction in execute");
			}
		}
	}
	return final.match;
}


// ExecuteString matches the Regexp against the string s.
// The return value is an array of integers, in pairs, identifying the positions of
// substrings matched by the expression.
//    s[a[0]:a[1]] is the substring matched by the entire expression.
//    s[a[2*i]:a[2*i+1]] for i > 0 is the substring matched by the ith parenthesized subexpression.
// A negative value means the subexpression did not match any element of the string.
// An empty array means "no match".
func (re *Regexp) ExecuteString(s string) (a []int) {
	return re.doExecute(s, nil, 0)
}


// Execute matches the Regexp against the byte slice b.
// The return value is an array of integers, in pairs, identifying the positions of
// subslices matched by the expression.
//    b[a[0]:a[1]] is the subslice matched by the entire expression.
//    b[a[2*i]:a[2*i+1]] for i > 0 is the subslice matched by the ith parenthesized subexpression.
// A negative value means the subexpression did not match any element of the slice.
// An empty array means "no match".
func (re *Regexp) Execute(b []byte) (a []int)	{ return re.doExecute("", b, 0) }


// MatchString returns whether the Regexp matches the string s.
// The return value is a boolean: true for match, false for no match.
func (re *Regexp) MatchString(s string) bool	{ return len(re.doExecute(s, nil, 0)) > 0 }


// Match returns whether the Regexp matches the byte slice b.
// The return value is a boolean: true for match, false for no match.
func (re *Regexp) Match(b []byte) bool	{ return len(re.doExecute("", b, 0)) > 0 }


// MatchStrings matches the Regexp against the string s.
// The return value is an array of strings matched by the expression.
//    a[0] is the substring matched by the entire expression.
//    a[i] for i > 0 is the substring matched by the ith parenthesized subexpression.
// An empty array means ``no match''.
func (re *Regexp) MatchStrings(s string) (a []string) {
	r := re.doExecute(s, nil, 0);
	if r == nil {
		return nil
	}
	a = make([]string, len(r)/2);
	for i := 0; i < len(r); i += 2 {
		if r[i] != -1 {	// -1 means no match for this subexpression
			a[i/2] = s[r[i]:r[i+1]]
		}
	}
	return;
}

// MatchSlices matches the Regexp against the byte slice b.
// The return value is an array of subslices matched by the expression.
//    a[0] is the subslice matched by the entire expression.
//    a[i] for i > 0 is the subslice matched by the ith parenthesized subexpression.
// An empty array means ``no match''.
func (re *Regexp) MatchSlices(b []byte) (a [][]byte) {
	r := re.doExecute("", b, 0);
	if r == nil {
		return nil
	}
	a = make([][]byte, len(r)/2);
	for i := 0; i < len(r); i += 2 {
		if r[i] != -1 {	// -1 means no match for this subexpression
			a[i/2] = b[r[i]:r[i+1]]
		}
	}
	return;
}

// MatchString checks whether a textual regular expression
// matches a string.  More complicated queries need
// to use Compile and the full Regexp interface.
func MatchString(pattern string, s string) (matched bool, error os.Error) {
	re, err := Compile(pattern);
	if err != nil {
		return false, err
	}
	return re.MatchString(s), nil;
}

// Match checks whether a textual regular expression
// matches a byte slice.  More complicated queries need
// to use Compile and the full Regexp interface.
func Match(pattern string, b []byte) (matched bool, error os.Error) {
	re, err := Compile(pattern);
	if err != nil {
		return false, err
	}
	return re.Match(b), nil;
}

// ReplaceAllString returns a copy of src in which all matches for the Regexp
// have been replaced by repl.  No support is provided for expressions
// (e.g. \1 or $1) in the replacement string.
func (re *Regexp) ReplaceAllString(src, repl string) string {
	lastMatchEnd := 0;	// end position of the most recent match
	searchPos := 0;		// position where we next look for a match
	buf := new(bytes.Buffer);
	for searchPos <= len(src) {
		a := re.doExecute(src, nil, searchPos);
		if len(a) == 0 {
			break	// no more matches
		}

		// Copy the unmatched characters before this match.
		io.WriteString(buf, src[lastMatchEnd:a[0]]);

		// Now insert a copy of the replacement string, but not for a
		// match of the empty string immediately after another match.
		// (Otherwise, we get double replacement for patterns that
		// match both empty and nonempty strings.)
		if a[1] > lastMatchEnd || a[0] == 0 {
			io.WriteString(buf, repl)
		}
		lastMatchEnd = a[1];

		// Advance past this match; always advance at least one character.
		_, width := utf8.DecodeRuneInString(src[searchPos:len(src)]);
		if searchPos+width > a[1] {
			searchPos += width
		} else if searchPos+1 > a[1] {
			// This clause is only needed at the end of the input
			// string.  In that case, DecodeRuneInString returns width=0.
			searchPos++
		} else {
			searchPos = a[1]
		}
	}

	// Copy the unmatched characters after the last match.
	io.WriteString(buf, src[lastMatchEnd:len(src)]);

	return buf.String();
}

// ReplaceAll returns a copy of src in which all matches for the Regexp
// have been replaced by repl.  No support is provided for expressions
// (e.g. \1 or $1) in the replacement text.
func (re *Regexp) ReplaceAll(src, repl []byte) []byte {
	lastMatchEnd := 0;	// end position of the most recent match
	searchPos := 0;		// position where we next look for a match
	buf := new(bytes.Buffer);
	for searchPos <= len(src) {
		a := re.doExecute("", src, searchPos);
		if len(a) == 0 {
			break	// no more matches
		}

		// Copy the unmatched characters before this match.
		buf.Write(src[lastMatchEnd:a[0]]);

		// Now insert a copy of the replacement string, but not for a
		// match of the empty string immediately after another match.
		// (Otherwise, we get double replacement for patterns that
		// match both empty and nonempty strings.)
		if a[1] > lastMatchEnd || a[0] == 0 {
			buf.Write(repl)
		}
		lastMatchEnd = a[1];

		// Advance past this match; always advance at least one character.
		_, width := utf8.DecodeRune(src[searchPos:len(src)]);
		if searchPos+width > a[1] {
			searchPos += width
		} else if searchPos+1 > a[1] {
			// This clause is only needed at the end of the input
			// string.  In that case, DecodeRuneInString returns width=0.
			searchPos++
		} else {
			searchPos = a[1]
		}
	}

	// Copy the unmatched characters after the last match.
	buf.Write(src[lastMatchEnd:len(src)]);

	return buf.Bytes();
}

// QuoteMeta returns a string that quotes all regular expression metacharacters
// inside the argument text; the returned string is a regular expression matching
// the literal text.  For example, QuoteMeta(`[foo]`) returns `\[foo\]`.
func QuoteMeta(s string) string {
	b := make([]byte, 2*len(s));

	// A byte loop is correct because all metacharacters are ASCII.
	j := 0;
	for i := 0; i < len(s); i++ {
		if special(int(s[i])) {
			b[j] = '\\';
			j++;
		}
		b[j] = s[i];
		j++;
	}
	return string(b[0:j]);
}

// Find matches in slice b if b is non-nil, otherwise find matches in string s.
func (re *Regexp) allMatches(s string, b []byte, n int, deliver func(int, int)) {
	var end int;
	if b == nil {
		end = len(s)
	} else {
		end = len(b)
	}

	for pos, i, prevMatchEnd := 0, 0, -1; i < n && pos <= end; {
		matches := re.doExecute(s, b, pos);
		if len(matches) == 0 {
			break
		}

		accept := true;
		if matches[1] == pos {
			// We've found an empty match.
			if matches[0] == prevMatchEnd {
				// We don't allow an empty match right
				// after a previous match, so ignore it.
				accept = false
			}
			var width int;
			if b == nil {
				_, width = utf8.DecodeRuneInString(s[pos:end])
			} else {
				_, width = utf8.DecodeRune(b[pos:end])
			}
			if width > 0 {
				pos += width
			} else {
				pos = end + 1
			}
		} else {
			pos = matches[1]
		}
		prevMatchEnd = matches[1];

		if accept {
			deliver(matches[0], matches[1]);
			i++;
		}
	}
}

// AllMatches slices the byte slice b into substrings that are successive
// matches of the Regexp within b. If n > 0, the function returns at most n
// matches. Text that does not match the expression will be skipped. Empty
// matches abutting a preceding match are ignored. The function returns a slice
// containing the matching substrings.
func (re *Regexp) AllMatches(b []byte, n int) [][]byte {
	if n <= 0 {
		n = len(b) + 1
	}
	result := make([][]byte, n);
	i := 0;
	re.allMatches("", b, n, func(start, end int) {
		result[i] = b[start:end];
		i++;
	});
	return result[0:i];
}

// AllMatchesString slices the string s into substrings that are successive
// matches of the Regexp within s. If n > 0, the function returns at most n
// matches. Text that does not match the expression will be skipped. Empty
// matches abutting a preceding match are ignored. The function returns a slice
// containing the matching substrings.
func (re *Regexp) AllMatchesString(s string, n int) []string {
	if n <= 0 {
		n = len(s) + 1
	}
	result := make([]string, n);
	i := 0;
	re.allMatches(s, nil, n, func(start, end int) {
		result[i] = s[start:end];
		i++;
	});
	return result[0:i];
}

// AllMatchesIter slices the byte slice b into substrings that are successive
// matches of the Regexp within b. If n > 0, the function returns at most n
// matches. Text that does not match the expression will be skipped. Empty
// matches abutting a preceding match are ignored. The function returns a
// channel that iterates over the matching substrings.
func (re *Regexp) AllMatchesIter(b []byte, n int) <-chan []byte {
	if n <= 0 {
		n = len(b) + 1
	}
	c := make(chan []byte, 10);
	go func() {
		re.allMatches("", b, n, func(start, end int) { c <- b[start:end] });
		close(c);
	}();
	return c;
}

// AllMatchesStringIter slices the string s into substrings that are successive
// matches of the Regexp within s. If n > 0, the function returns at most n
// matches. Text that does not match the expression will be skipped. Empty
// matches abutting a preceding match are ignored. The function returns a
// channel that iterates over the matching substrings.
func (re *Regexp) AllMatchesStringIter(s string, n int) <-chan string {
	if n <= 0 {
		n = len(s) + 1
	}
	c := make(chan string, 10);
	go func() {
		re.allMatches(s, nil, n, func(start, end int) { c <- s[start:end] });
		close(c);
	}();
	return c;
}
