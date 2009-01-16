// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Regular expression library.

package regexp

import (
	"os";
	"array";
)

var debug = false;


export var ErrInternal = os.NewError("internal error");
export var ErrUnmatchedLpar = os.NewError("unmatched '('");
export var ErrUnmatchedRpar = os.NewError("unmatched ')'");
export var ErrUnmatchedLbkt = os.NewError("unmatched '['");
export var ErrUnmatchedRbkt = os.NewError("unmatched ']'");
export var ErrBadRange = os.NewError("bad range in character class");
export var ErrExtraneousBackslash = os.NewError("extraneous backslash");
export var ErrBadClosure = os.NewError("repeated closure (**, ++, etc.)");
export var ErrBareClosure = os.NewError("closure applies to nothing");
export var ErrBadBackslash = os.NewError("illegal backslash escape");

// An instruction executed by the NFA
type instr interface {
	Type()	int;	// the type of this instruction: cCHAR, cANY, etc.
	Next()	instr;	// the instruction to execute after this one
	SetNext(i instr);
	Index()	int;
	SetIndex(i int);
	Print();
}

// Fields and methods common to all instructions
type iCommon struct {
	next	instr;
	index	int;
}

func (c *iCommon) Next() instr { return c.next }
func (c *iCommon) SetNext(i instr) { c.next = i }
func (c *iCommon) Index() int { return c.index }
func (c *iCommon) SetIndex(i int) { c.index = i }

type regExp struct {
	expr	string;	// the original expression
	ch	chan<- *regExp;	// reply channel when we're done
	error	*os.Error;	// compile- or run-time error; nil if OK
	inst	*array.Array;
	start	instr;
	nbra	int;	// number of brackets in expression, for subexpressions
}

const (
	cSTART	// beginning of program
		= iota;
	cEND;		// end of program: success
	cBOT;		// '^' beginning of text
	cEOT;		// '$' end of text
	cCHAR;	// 'a' regular character
	cCHARCLASS;	// [a-z] character class
	cANY;		// '.' any character
	cBRA;		// '(' parenthesized expression
	cEBRA;	// ')'; end of '(' parenthesized expression
	cALT;		// '|' alternation
	cNOP;		// do nothing; makes it easy to link without patching
)

// --- START start of program
type iStart struct {
	iCommon
}

func (start *iStart) Type() int { return cSTART }
func (start *iStart) Print() { print("start") }

// --- END end of program
type iEnd struct {
	iCommon
}

func (end *iEnd) Type() int { return cEND }
func (end *iEnd) Print() { print("end") }

// --- BOT beginning of text
type iBot struct {
	iCommon
}

func (bot *iBot) Type() int { return cBOT }
func (bot *iBot) Print() { print("bot") }

// --- EOT end of text
type iEot struct {
	iCommon
}

func (eot *iEot) Type() int { return cEOT }
func (eot *iEot) Print() { print("eot") }

// --- CHAR a regular character
type iChar struct {
	iCommon;
	char	int;
}

func (char *iChar) Type() int { return cCHAR }
func (char *iChar) Print() { print("char ", string(char.char)) }

func newChar(char int) *iChar {
	c := new(iChar);
	c.char = char;
	return c;
}

// --- CHARCLASS [a-z]

type iCharClass struct {
	iCommon;
	char	int;
	negate	bool;	// is character class negated? ([^a-z])
	// array of int, stored pairwise: [a-z] is (a,z); x is (x,x):
	ranges	*array.IntArray;
}

func (cclass *iCharClass) Type() int { return cCHARCLASS }

func (cclass *iCharClass) Print() {
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

func (cclass *iCharClass) AddRange(a, b int) {
	// range is a through b inclusive
	cclass.ranges.Push(a);
	cclass.ranges.Push(b);
}

func (cclass *iCharClass) Matches(c int) bool {
	for i := 0; i < cclass.ranges.Len(); i = i+2 {
		min := cclass.ranges.At(i);
		max := cclass.ranges.At(i+1);
		if min <= c && c <= max {
			return !cclass.negate
		}
	}
	return cclass.negate
}

func newCharClass() *iCharClass {
	c := new(iCharClass);
	c.ranges = array.NewIntArray(0);
	return c;
}

// --- ANY any character
type iAny struct {
	iCommon
}

func (any *iAny) Type() int { return cANY }
func (any *iAny) Print() { print("any") }

// --- BRA parenthesized expression
type iBra struct {
	iCommon;
	n	int;	// subexpression number
}

func (bra *iBra) Type() int { return cBRA }
func (bra *iBra) Print() { print("bra", bra.n); }

// --- EBRA end of parenthesized expression
type iEbra struct {
	iCommon;
	n	int;	// subexpression number
}

func (ebra *iEbra) Type() int { return cEBRA }
func (ebra *iEbra) Print() { print("ebra ", ebra.n); }

// --- ALT alternation
type iAlt struct {
	iCommon;
	left	instr;	// other branch
}

func (alt *iAlt) Type() int { return cALT }
func (alt *iAlt) Print() { print("alt(", alt.left.Index(), ")"); }

// --- NOP no operation
type iNop struct {
	iCommon
}

func (nop *iNop) Type() int { return cNOP }
func (nop *iNop) Print() { print("nop") }

// report error and exit compiling/executing goroutine
func (re *regExp) Error(err *os.Error) {
	re.error = err;
	re.ch <- re;
	sys.goexit();
}

func (re *regExp) Add(i instr) instr {
	i.SetIndex(re.inst.Len());
	re.inst.Push(i);
	return i;
}

type parser struct {
	re	*regExp;
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
		c, w := sys.stringtorune(p.re.expr, p.pos);
		p.ch = c;
		p.pos += w;
	}
	return p.ch;
}

func newParser(re *regExp) *parser {
	p := new(parser);
	p.re = re;
	p.nextc();	// load p.ch
	return p;
}

/*

Grammar:
	regexp:
		concatenation { '|' concatenation }
	concatenation:
		{ closure }
	closure:
		term [ '*' | '+' | '?' ]
	term:
		'^'
		'$'
		'.'
		character
		'[' [ '^' ] character-ranges ']'
		'(' regexp ')'

*/

func (p *parser) Regexp() (start, end instr)

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

func (p *parser) CharClass() instr {
	cc := newCharClass();
	p.re.Add(cc);
	if p.c() == '^' {
		cc.negate = true;
		p.nextc();
	}
	left := -1;
	for {
		switch c := p.c(); c {
		case ']', endOfFile:
			if left >= 0 {
				p.re.Error(ErrBadRange);
			}
			return cc;
		case '-':	// do this before backslash processing
			p.re.Error(ErrBadRange);
		case '\\':
			c = p.nextc();
			switch {
			case c == endOfFile:
				p.re.Error(ErrExtraneousBackslash);
			case c == 'n':
				c = '\n';
			case specialcclass(c):
				// c is as delivered
			default:
				p.re.Error(ErrBadBackslash);
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
					cc.AddRange(c, c);
				}
			case left <= c:	// second of pair
				cc.AddRange(left, c);
				left = -1;
			default:
				p.re.Error(ErrBadRange);
			}
		}
	}
	return iNULL
}

func (p *parser) Term() (start, end instr) {
	switch c := p.c(); c {
	case '|', endOfFile:
		return iNULL, iNULL;
	case '*', '+':
		p.re.Error(ErrBareClosure);
	case ')':
		if p.nlpar == 0 {
			p.re.Error(ErrUnmatchedRpar);
		}
		return iNULL, iNULL;
	case ']':
		p.re.Error(ErrUnmatchedRbkt);
	case '^':
		p.nextc();
		start = p.re.Add(new(iBot));
		return start, start;
	case '$':
		p.nextc();
		start = p.re.Add(new(iEot));
		return start, start;
	case '.':
		p.nextc();
		start = p.re.Add(new(iAny));
		return start, start;
	case '[':
		p.nextc();
		start = p.CharClass();
		if p.c() != ']' {
			p.re.Error(ErrUnmatchedLbkt);
		}
		p.nextc();
		return start, start;
	case '(':
		p.nextc();
		p.nlpar++;
		p.re.nbra++;	// increment first so first subexpr is \1
		nbra := p.re.nbra;
		start, end = p.Regexp();
		if p.c() != ')' {
			p.re.Error(ErrUnmatchedLpar);
		}
		p.nlpar--;
		p.nextc();
		bra := new(iBra);
		p.re.Add(bra);
		ebra := new(iEbra);
		p.re.Add(ebra);
		bra.n = nbra;
		ebra.n = nbra;
		if start == iNULL {
			if end == iNULL { p.re.Error(ErrInternal) }
			start = ebra
		} else {
			end.SetNext(ebra);
		}
		bra.SetNext(start);
		return bra, ebra;
	case '\\':
		c = p.nextc();
		switch {
		case c == endOfFile:
			p.re.Error(ErrExtraneousBackslash);
		case c == 'n':
			c = '\n';
		case special(c):
			// c is as delivered
		default:
			p.re.Error(ErrBadBackslash);
		}
		fallthrough;
	default:
		p.nextc();
		start = newChar(c);
		p.re.Add(start);
		return start, start
	}
	panic("unreachable");
}

func (p *parser) Closure() (start, end instr) {
	start, end = p.Term();
	if start == iNULL {
		return
	}
	switch p.c() {
	case '*':
		// (start,end)*:
		alt := new(iAlt);
		p.re.Add(alt);
		end.SetNext(alt);	// after end, do alt
		alt.left = start;	// alternate brach: return to start
		start = alt;	// alt becomes new (start, end)
		end = alt;
	case '+':
		// (start,end)+:
		alt := new(iAlt);
		p.re.Add(alt);
		end.SetNext(alt);	// after end, do alt
		alt.left = start;	// alternate brach: return to start
		end = alt;	// start is unchanged; end is alt
	case '?':
		// (start,end)?:
		alt := new(iAlt);
		p.re.Add(alt);
		nop := new(iNop);
		p.re.Add(nop);
		alt.left = start;	// alternate branch is start
		alt.next = nop;	// follow on to nop
		end.SetNext(nop);	// after end, go to nop
		start = alt;	// start is now alt
		end = nop;	// end is nop pointed to by both branches
	default:
		return
	}
	switch p.nextc() {
	case '*', '+', '?':
		p.re.Error(ErrBadClosure);
	}
	return
}

func (p *parser) Concatenation() (start, end instr) {
	start, end = iNULL, iNULL;
	for {
		nstart, nend := p.Closure();
		switch {
		case nstart == iNULL:	// end of this concatenation
			if start == iNULL {	// this is the empty string
				nop := p.re.Add(new(iNop));
				return nop, nop;
			}
			return;
		case start == iNULL:	// this is first element of concatenation
			start, end = nstart, nend;
		default:
			end.SetNext(nstart);
			end = nend;
		}
	}
	panic("unreachable");
}

func (p *parser) Regexp() (start, end instr) {
	start, end = p.Concatenation();
	for {
		switch p.c() {
		default:
			return;
		case '|':
			p.nextc();
			nstart, nend := p.Concatenation();
			alt := new(iAlt);
			p.re.Add(alt);
			alt.left = start;
			alt.next = nstart;
			nop := new(iNop);
			p.re.Add(nop);
			end.SetNext(nop);
			nend.SetNext(nop);
			start, end = alt, nop;
		}
	}
	panic("unreachable");
}

func UnNop(i instr) instr {
	for i.Type() == cNOP {
		i = i.Next()
	}
	return i
}

func (re *regExp) EliminateNops() {
	for i := 0; i < re.inst.Len(); i++ {
		inst := re.inst.At(i).(instr);
		if inst.Type() == cEND {
			continue
		}
		inst.SetNext(UnNop(inst.Next()));
		if inst.Type() == cALT {
			alt := inst.(*iAlt);
			alt.left = UnNop(alt.left);
		}
	}
}

func (re *regExp) Dump() {
	for i := 0; i < re.inst.Len(); i++ {
		inst := re.inst.At(i).(instr);
		print(inst.Index(), ": ");
		inst.Print();
		if inst.Type() != cEND {
			print(" -> ", inst.Next().Index())
		}
		print("\n");
	}
}

func (re *regExp) DoParse() {
	p := newParser(re);
	start := new(iStart);
	re.Add(start);
	s, e := p.Regexp();
	start.next = s;
	re.start = start;
	e.SetNext(re.Add(new(iEnd)));

	if debug {
		re.Dump();
		println();
	}

	re.EliminateNops();

	if debug {
		re.Dump();
		println();
	}
}


func Compiler(str string, ch chan *regExp) {
	re := new(regExp);
	re.expr = str;
	re.inst = array.New(0);
	re.ch = ch;
	re.DoParse();
	ch <- re;
}

// Public interface has only execute functionality
export type Regexp interface {
	Execute(s string) []int;
	Match(s string) bool;
	MatchStrings(s string) []string;
}

// Compile in separate goroutine; wait for result
export func Compile(str string) (regexp Regexp, error *os.Error) {
	ch := make(chan *regExp);
	go Compiler(str, ch);
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
	index := inst.Index();
	l := len(s);
	pos := match[0];
	// TODO: Once the state is a vector and we can do insert, have inputs always
	// go in order correctly and this "earlier" test is never necessary,
	for i := 0; i < l; i++ {
		if s[i].inst.Index() == index && // same instruction
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

func (re *regExp) DoExecute(str string, pos int) []int {
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
			s[out] = addState(s[out], re.start.Next(), match);
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
			c, charwidth = sys.stringtorune(str, pos);
		}
		for i := 0; i < len(s[in]); i++ {
			st := s[in][i];
			switch s[in][i].inst.Type() {
			case cBOT:
				if pos == 0 {
					s[in] = addState(s[in], st.inst.Next(), st.match)
				}
			case cEOT:
				if pos == len(str) {
					s[in] = addState(s[in], st.inst.Next(), st.match)
				}
			case cCHAR:
				if c == st.inst.(*iChar).char {
					s[out] = addState(s[out], st.inst.Next(), st.match)
				}
			case cCHARCLASS:
				if st.inst.(*iCharClass).Matches(c) {
					s[out] = addState(s[out], st.inst.Next(), st.match)
				}
			case cANY:
				if c != endOfFile {
					s[out] = addState(s[out], st.inst.Next(), st.match)
				}
			case cBRA:
				n := st.inst.(*iBra).n;
				st.match[2*n] = pos;
				s[in] = addState(s[in], st.inst.Next(), st.match);
			case cEBRA:
				n := st.inst.(*iEbra).n;
				st.match[2*n+1] = pos;
				s[in] = addState(s[in], st.inst.Next(), st.match);
			case cALT:
				s[in] = addState(s[in], st.inst.(*iAlt).left, st.match);
				// give other branch a copy of this match vector
				s1 := make([]int, 2*(re.nbra+1));
				for i := 0; i < len(s1); i++ {
					s1[i] = st.match[i]
				}
				s[in] = addState(s[in], st.inst.Next(), s1);
			case cEND:
				// choose leftmost longest
				if !found ||	// first
				   st.match[0] < final.match[0] ||	// leftmost
				   (st.match[0] == final.match[0] && pos > final.match[1])  {	// longest
					final = st;
					final.match[1] = pos;
				}
				found = true;
			default:
				st.inst.Print();
				panic("unknown instruction in execute");
			}
		}
		pos += charwidth;
	}
	return final.match;
}


func (re *regExp) Execute(s string) []int {
	return re.DoExecute(s, 0)
}


func (re *regExp) Match(s string) bool {
	return len(re.DoExecute(s, 0)) > 0
}


func (re *regExp) MatchStrings(s string) []string {
	r := re.DoExecute(s, 0);
	if r == nil {
		return nil
	}
	a := make([]string, len(r)/2);
	for i := 0; i < len(r); i += 2 {
		a[i/2] = s[r[i] : r[i+1]]
	}
	return a
}

// Exported function for simple boolean check.  Anything more fancy
// needs a call to Compile.
export func Match(pattern string, s string) (matched bool, error *os.Error) {
	re, err := Compile(pattern);
	if err != nil {
		return false, err
	}
	return re.Match(s), nil
}
