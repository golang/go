// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Regular expression library.

package regexp

import os "os"

export var ErrUnimplemented = os.NewError("unimplemented");
export var ErrInternal = os.NewError("internal error");
export var ErrUnmatchedLpar = os.NewError("unmatched '('");
export var ErrUnmatchedRpar = os.NewError("unmatched ')'");
export var ErrExtraneousBackslash = os.NewError("extraneous backslash");
export var ErrEmpty = os.NewError("empty subexpression or alternation");
export var ErrBadClosure = os.NewError("repeated closure (**, ++, etc.)");
export var ErrBareClosure = os.NewError("closure applies to nothing");
export var ErrBadBackslash = os.NewError("illegal backslash escape");

// An instruction executed by the NFA
type Inst interface {
	Type()	int;	// the type of this instruction: CHAR, ANY, etc.
	Next()	int;	// the index of the instruction to execute after this one
	SetNext(i int);
	Print(ind string);
}

type RE struct {
	expr	string;	// the original expression
	ch	*chan<- *RE;	// reply channel when we're done
	error *os.Error;	// compile- or run-time error; nil if OK
	ninst	int;
	inst *[]Inst;
	start	Inst;
}

const (
	START	// beginning of program: marker to start
		= iota;
	END;		// end of program: success
	BOT;		// '^' beginning of text
	EOT;		// '$' end of text
	CHAR;	// 'a' regular character
	ANY;		// '.' any character
	BRA;		// '(' parenthesized expression
	EBRA;	// ')'; end of '(' parenthesized expression
	ALT;		// '|' alternation
	NOP;		// do nothing; makes it easy to link without patching
)

// --- START start of program
type Start struct {
	this	int;
	next	int;
}

func (start *Start) Type() int { return START }
func (start *Start) Next() int { return start.next }
func (start *Start) SetNext(i int) { start.next = i }
func (start *Start) Print(ind string) { print(ind, "start") }

// --- END end of program
type End struct {
	this	int;
	next	int;
}

func (end *End) Type() int { return END }
func (end *End) Next() int { return end.next }
func (end *End) SetNext(i int) { end.next = i }
func (end *End) Print(ind string) { print(ind, "end") }

// --- BOT beginning of text
type Bot struct {
	this	int;
	next	int;
}

func (bot *Bot) Type() int { return BOT }
func (bot *Bot) Next() int { return bot.next }
func (bot *Bot) SetNext(i int) { bot.next = i }
func (bot *Bot) Print(ind string) { print(ind, "bot") }

// --- EOT end of text
type Eot struct {
	this	int;
	next	int;
}

func (eot *Eot) Type() int { return EOT }
func (eot *Eot) Next() int { return eot.next }
func (eot *Eot) SetNext(i int) { eot.next = i }
func (eot *Eot) Print(ind string) { print(ind, "eot") }

// --- CHAR a regular character
type Char struct {
	this	int;
	next	int;
	char	int;
	set	bool;
}

func (char *Char) Type() int { return CHAR }
func (char *Char) Next() int { return char.next }
func (char *Char) SetNext(i int) { char.next = i }
func (char *Char) Print(ind string) { print(ind, "char ", string(char.char)) }

func NewChar(char int) *Char {
	c := new(Char);
	c.char = char;
	return c;
}

// --- ANY any character
type Any struct {
	this	int;
	next	int;
}

func (any *Any) Type() int { return ANY }
func (any *Any) Next() int { return any.next }
func (any *Any) SetNext(i int) { any.next = i }
func (any *Any) Print(ind string) { print(ind, "any") }

// --- BRA parenthesized expression
type Bra struct {
	this	int;
	next	int;
}

func (bra *Bra) Type() int { return BRA }
func (bra *Bra) Next() int { return bra.next }
func (bra *Bra) SetNext(i int) { bra.next = i }
func (bra *Bra) Print(ind string) { print(ind , "bra"); }

// --- EBRA end of parenthesized expression
type Ebra struct {
	this	int;
	next	int;
	n	int;	// subexpression number
}

func (ebra *Ebra) Type() int { return BRA }
func (ebra *Ebra) Next() int { return ebra.next }
func (ebra *Ebra) SetNext(i int) { ebra.next = i }
func (ebra *Ebra) Print(ind string) { print(ind , "ebra ", ebra.n); }

// --- ALT alternation
type Alt struct {
	this	int;
	next	int;
	left	int;	// other branch
}

func (alt *Alt) Type() int { return ALT }
func (alt *Alt) Next() int { return alt.next }
func (alt *Alt) SetNext(i int) { alt.next = i }
func (alt *Alt) Print(ind string) { print(ind , "alt(", alt.left, ")"); }

// --- NOP no operation
type Nop struct {
	this	int;
	next	int;
}

func (nop *Nop) Type() int { return NOP }
func (nop *Nop) Next() int { return nop.next }
func (nop *Nop) SetNext(i int) { nop.next = i }
func (nop *Nop) Print(ind string) { print(ind, "nop") }


func (re *RE) AddInst(inst Inst) int {
	if re.ninst >= cap(re.inst) {
		panic("write the code to grow inst")
	}
	re.inst[re.ninst] = inst;
	i := re.ninst;
	re.ninst++;
	inst.SetNext(re.ninst);
	return i;
}

// report error and exit compiling/executing goroutine
func (re *RE) Error(err *os.Error) {
	re.error = err;
	re.ch <- re;
	sys.goexit();
}

type Parser struct {
	re	*RE;
	nbra	int;
	pos	int;
	ch	int;
}

const EOF = -1

func (p *Parser) c() int {
	return p.ch;
}

func (p *Parser) nextc() int {
	if p.pos >= len(p.re.expr) {
		p.ch = EOF
	} else {
		c, w := sys.stringtorune(p.re.expr, p.pos);	// TODO: stringotorune shoudl take a string*
		p.ch = c;
		p.pos += w;
	}
	return p.ch;
}

func NewParser(re *RE) *Parser {
	parser := new(Parser);
	parser.re = re;
	parser.nextc();	// load p.ch
	return parser;
}

/*

Grammar:
	regexp:
		concatenation { '|' concatenation }
	concatenation:
		{ closure }
	closure:
		term { '*' | '+' | '?' }
	term:
		'.'
		character
		characterclass
		'(' regexp ')'

*/

func (p *Parser) Regexp() (start, end int)

const NULL = -1

func special(c int) bool {
	s := `\.+*?()|[-]`;
	for i := 0; i < len(s); i++ {
		if c == int(s[i]) {
			return true
		}
	}
	return false
}

func (p *Parser) Term() (start, end int) {
	switch c := p.c(); c {
	case '|', EOF:
		return NULL, NULL;
	case '*', '+', '|':
		p.re.Error(ErrBareClosure);
	case ')':
		p.re.Error(ErrUnmatchedRpar);
	case '.':
		p.nextc();
		start = p.re.AddInst(new(Any));
		return start, start;
	case '(':
		p.nextc();
		start, end = p.Regexp();
		if p.c() != ')' {
			p.re.Error(ErrUnmatchedLpar);
		}
		p.nextc();
		p.nbra++;
		bra := new(Bra);
		brai := p.re.AddInst(bra);
		ebra := new(Ebra);
		ebrai := p.re.AddInst(ebra);
		ebra.n = p.nbra;
		if start == NULL {
			if end != NULL { p.re.Error(ErrInternal) }
			start = ebrai
		} else {
			p.re.inst[end].SetNext(ebrai);
		}
		bra.SetNext(start);
		return brai, ebrai;
	case '\\':
		c = p.nextc();
		switch {
		case c == EOF:
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
		start = p.re.AddInst(NewChar(c));
		return start, start
	}
	panic("unreachable");
}

func (p *Parser) Closure() (start, end int) {
	start, end = p.Term();
	if start == NULL {
		return start, end
	}
	switch p.c() {
	case '*':
		// (start,end)*:
		alt := new(Alt);
		alti := p.re.AddInst(alt);
		p.re.inst[end].SetNext(alti);	// after end, do alt
		alt.left = start;	// alternate brach: return to start
		start = alti;	// alt becomes new (start, end)
		end = alti;
	case '+':
		// (start,end)+:
		alt := new(Alt);
		alti := p.re.AddInst(alt);
		p.re.inst[end].SetNext(alti);	// after end, do alt
		alt.left = start;	// alternate brach: return to start
		end = alti;	// start is unchanged; end is alt
	case '?':
		// (start,end)?:
		alt := new(Alt);
		alti := p.re.AddInst(alt);
		nop := new(Nop);
		nopi := p.re.AddInst(nop);
		alt.left = start;	// alternate branch is start
		alt.next = nopi;	// follow on to nop
		p.re.inst[end].SetNext(nopi);	// after end, go to nop
		start = alti;	// start is now alt
		end = nopi;	// end is nop pointed to by both branches
	default:
		return start, end;
	}
	switch p.nextc() {
	case '*', '+', '?':
		p.re.Error(ErrBadClosure);
	}
	return start, end;
}

func (p *Parser) Concatenation() (start, end int) {
	start, end = NULL, NULL;
	for {
		nstart, nend := p.Closure();
		switch {
		case nstart == NULL:	// end of this concatenation
			return start, end;
		case start == NULL:	// this is first element of concatenation
			start, end = nstart, nend;
		default:
			p.re.inst[end].SetNext(nstart);
			end = nend;
		}
	}
	panic("unreachable");
}

func (p *Parser) Regexp() (start, end int) {
	start, end = p.Concatenation();
	if start == NULL {
		return NULL, NULL
	}
	for {
		switch p.c() {
		default:
			return start, end;
		case '|':
			p.nextc();
			nstart, nend := p.Concatenation();
			// xyz|(nothing) is xyz or nop
			if nstart == NULL {
				nopi := p.re.AddInst(new(Nop));
				nstart, nend = nopi, nopi;
			}
			alt := new(Alt);
			alti := p.re.AddInst(alt);
			alt.left = start;
			alt.next = nstart;
			nop := new(Nop);
			nopi := p.re.AddInst(nop);
			p.re.inst[end].SetNext(nopi);
			p.re.inst[nend].SetNext(nopi);
			start, end = alti, nopi;
		}
	}
	panic("unreachable");
}

func (re *RE) UnNop(i int) int {
	for re.inst[i].Type() == NOP {
		i = re.inst[i].Next()
	}
	return i
}

func (re *RE) EliminateNops() {
	for i := 0; i < re.ninst - 1; i++ {	// last one is END
		inst := re.inst[i];
		inst.SetNext(re.UnNop(inst.Next()));
		if inst.Type() == ALT {
			alt := inst.(*Alt);
			alt.left = re.UnNop(alt.left)
		}
	}
}

func (re *RE) DoParse() {
	parser := NewParser(re);
	start := new(Start);
	starti := re.AddInst(start);
	s, e := parser.Regexp();
	if s == NULL {
		if e != NULL { re.Error(ErrInternal) }
		e = starti;
	}
	start.next = s;
	re.inst[e].SetNext(re.AddInst(new(End)));

	for i := 0; i < re.ninst; i++ {
		inst := re.inst[i];
		print(i, ":\t");
		inst.Print("\t");
		print(" -> ", inst.Next(), "\n");
	}
	println();

	re.EliminateNops();

	for i := 0; i < re.ninst; i++ {
		inst := re.inst[i];
		print(i, ":\t");
		inst.Print("\t");
		print(" -> ", inst.Next(), "\n");
	}
	println();

	re.Error(ErrUnimplemented);
}

func Compiler(str string, ch *chan *RE) {
	re := new(RE);
	re.inst = new([]Inst, 100);
	re.expr = str;
	re.ch = ch;
	re.DoParse();
	ch <- re;
}

// Public interface has only execute functionality (not yet implemented)
export type Regexp interface {
	// Execute() bool
}

// compile in separate goroutine; wait for result
export func Compile(str string) (regexp Regexp, error *os.Error) {
	ch := new(chan *RE);
	go Compiler(str, ch);
	re := <-ch;
	return re, re.error
}
