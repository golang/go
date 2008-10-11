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
	Next()	Inst;	// the instruction to execute after this one
	SetNext(i Inst);
	Print(ind string);
}

type RE struct {
	expr	string;	// the original expression
	ch	*chan<- *RE;	// reply channel when we're done
	error *os.Error;	// compile- or run-time error; nil if OK
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
	next	Inst;
}

func (start *Start) Type() int { return START }
func (start *Start) Next() Inst { return start.next }
func (start *Start) SetNext(i Inst) { start.next = i }
func (start *Start) Print(ind string) { print(ind, "start") }

// --- END end of program
type End struct {
	next	Inst;
}

func (end *End) Type() int { return END }
func (end *End) Next() Inst { return end.next }
func (end *End) SetNext(i Inst) { end.next = i }
func (end *End) Print(ind string) { print(ind, "end") }

// --- BOT beginning of text
type Bot struct {
	next	Inst;
}

func (bot *Bot) Type() int { return BOT }
func (bot *Bot) Next() Inst { return bot.next }
func (bot *Bot) SetNext(i Inst) { bot.next = i }
func (bot *Bot) Print(ind string) { print(ind, "bot") }

// --- EOT end of text
type Eot struct {
	next	Inst;
}

func (eot *Eot) Type() int { return EOT }
func (eot *Eot) Next() Inst { return eot.next }
func (eot *Eot) SetNext(i Inst) { eot.next = i }
func (eot *Eot) Print(ind string) { print(ind, "eot") }

// --- CHAR a regular character
type Char struct {
	next	Inst;
	char	int;
	set	bool;
}

func (char *Char) Type() int { return CHAR }
func (char *Char) Next() Inst { return char.next }
func (char *Char) SetNext(i Inst) { char.next = i }
func (char *Char) Print(ind string) { print(ind, "char ", string(char.char)) }

func NewChar(char int) *Char {
	c := new(Char);
	c.char = char;
	return c;
}

// --- ANY any character
type Any struct {
	next	Inst;
}

func (any *Any) Type() int { return ANY }
func (any *Any) Next() Inst { return any.next }
func (any *Any) SetNext(i Inst) { any.next = i }
func (any *Any) Print(ind string) { print(ind, "any") }

// --- BRA parenthesized expression
type Bra struct {
	next	Inst;
	n	int;	// subexpression number
}

func (bra *Bra) Type() int { return BRA }
func (bra *Bra) Next() Inst { return bra.next }
func (bra *Bra) SetNext(i Inst) { bra.next = i }
func (bra *Bra) Print(ind string) { print(ind , "bra"); }

// --- EBRA end of parenthesized expression
type Ebra struct {
	next	Inst;
	n	int;	// subexpression number
}

func (ebra *Ebra) Type() int { return BRA }
func (ebra *Ebra) Next() Inst { return ebra.next }
func (ebra *Ebra) SetNext(i Inst) { ebra.next = i }
func (ebra *Ebra) Print(ind string) { print(ind , "ebra ", ebra.n); }

// --- ALT alternation
type Alt struct {
	next	Inst;
	left	Inst;	// other branch
}

func (alt *Alt) Type() int { return ALT }
func (alt *Alt) Next() Inst { return alt.next }
func (alt *Alt) SetNext(i Inst) { alt.next = i }
func (alt *Alt) Print(ind string) { print(ind , "alt(", alt.left, ")"); }

// --- NOP no operation
type Nop struct {
	next	Inst;
}

func (nop *Nop) Type() int { return NOP }
func (nop *Nop) Next() Inst { return nop.next }
func (nop *Nop) SetNext(i Inst) { nop.next = i }
func (nop *Nop) Print(ind string) { print(ind, "nop") }

// report error and exit compiling/executing goroutine
func (re *RE) Error(err *os.Error) {
	re.error = err;
	re.ch <- re;
	sys.goexit();
}

type Parser struct {
	re	*RE;
	nbra	int;	// number of brackets in expression, for subexpressions
	nlpar	int;	// number of unclosed lpars
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

func (p *Parser) Regexp() (start, end Inst)

var NULL Inst
type BUGinter interface{}

// same as i == NULL.  TODO: remove when 6g lets me do i == NULL
func isNULL(i Inst) bool {
	return sys.BUG_intereq(i.(BUGinter), NULL.(BUGinter))
}

// same as i == j.  TODO: remove when 6g lets me do i == j
func isEQ(i,j Inst) bool {
	return sys.BUG_intereq(i.(BUGinter), j.(BUGinter))
}

func special(c int) bool {
	s := `\.+*?()|[-]`;
	for i := 0; i < len(s); i++ {
		if c == int(s[i]) {
			return true
		}
	}
	return false
}

func (p *Parser) Term() (start, end Inst) {
	switch c := p.c(); c {
	case '|', EOF:
		return NULL, NULL;
	case '*', '+', '|':
		p.re.Error(ErrBareClosure);
	case ')':
		if p.nlpar == 0 {
			p.re.Error(ErrUnmatchedRpar);
		}
		return NULL, NULL;
	case '.':
		p.nextc();
		start = new(Any);
		return start, start;
	case '(':
		p.nextc();
		p.nlpar++;
		start, end = p.Regexp();
		if p.c() != ')' {
			p.re.Error(ErrUnmatchedLpar);
		}
		p.nlpar--;
		p.nextc();
		p.nbra++;
		bra := new(Bra);
		ebra := new(Ebra);
		bra.n = p.nbra;
		ebra.n = p.nbra;
		if isNULL(start) {
			if !isNULL(end) { p.re.Error(ErrInternal) }
			start = ebra
		} else {
			end.SetNext(ebra);
		}
		bra.SetNext(start);
		return bra, ebra;
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
		start = NewChar(c);
		return start, start
	}
	panic("unreachable");
}

func (p *Parser) Closure() (start, end Inst) {
	start, end = p.Term();
	if isNULL(start) {
		return start, end
	}
	switch p.c() {
	case '*':
		// (start,end)*:
		alt := new(Alt);
		end.SetNext(alt);	// after end, do alt
		alt.left = start;	// alternate brach: return to start
		start = alt;	// alt becomes new (start, end)
		end = alt;
	case '+':
		// (start,end)+:
		alt := new(Alt);
		end.SetNext(alt);	// after end, do alt
		alt.left = start;	// alternate brach: return to start
		end = alt;	// start is unchanged; end is alt
	case '?':
		// (start,end)?:
		alt := new(Alt);
		nop := new(Nop);
		alt.left = start;	// alternate branch is start
		alt.next = nop;	// follow on to nop
		end.SetNext(nop);	// after end, go to nop
		start = alt;	// start is now alt
		end = nop;	// end is nop pointed to by both branches
	default:
		return start, end;
	}
	switch p.nextc() {
	case '*', '+', '?':
		p.re.Error(ErrBadClosure);
	}
	return start, end;
}

func (p *Parser) Concatenation() (start, end Inst) {
	start, end = NULL, NULL;
	for {
		switch p.c() {
		case '|', ')', EOF:
			if isNULL(start) {	// this is the empty string
				nop := new(Nop);
				return nop, nop;
			}
			return start, end;
		}
		nstart, nend := p.Closure();
		switch {
		case isNULL(nstart):	// end of this concatenation
			return start, end;
		case isNULL(start):	// this is first element of concatenation
			start, end = nstart, nend;
		default:
			end.SetNext(nstart);
			end = nend;
		}
	}
	panic("unreachable");
}

func (p *Parser) Regexp() (start, end Inst) {
	start, end = p.Concatenation();
	if isNULL(start) {
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
			if isNULL(nstart) {
				nop := new(Nop);
				nstart, nend = nop, nop;
			}
			alt := new(Alt);
			alt.left = start;
			alt.next = nstart;
			nop := new(Nop);
			end.SetNext(nop);
			nend.SetNext(nop);
			start, end = alt, nop;
		}
	}
	panic("unreachable");
}

func UnNop(i Inst) Inst {
	for i.Type() == NOP {
		i = i.Next()
	}
	return i
}

func (re *RE) EliminateNops(start Inst) {
	for i := start; i.Type() != END; i = i.Next() {	// last one is END
		i.SetNext(UnNop(i.Next()));
		if i.Type() == ALT {
			alt := i.(*Alt);
			alt.left = UnNop(alt.left);
			re.EliminateNops(alt.left);
		}
	}
}

// use a 'done' array to know where we've already printed.
// the output is not pretty but it is serviceable.
func (re *RE) Dump(ind string, inst Inst, done *[]Inst) {
	// see if we've been here, and mark it
	for i := 0; i < len(done); i++ {
		if isEQ(inst, done[i]) {
			print(ind, inst, ": -> ", inst.Next(), "...\n");
			return;
		}
	}
	slot := len(done);
	done= done[0:slot+1];
	done[slot] = inst;

	if isNULL(inst) {
		println("NULL");
		return;
	}
	if inst.Type() == END { print(inst, ": END\n"); return }
	print(ind, inst, ": ");
	inst.Print("");
	print(" -> ", inst.Next(), "\n");
	switch inst.Type() {
	case END:
		return;
	case ALT:
		re.Dump(ind + "\t", inst.(*Alt).left, done);
	}
	re.Dump(ind, inst.Next(), done);
}

func (re *RE) DumpAll() {
	re.Dump("", re.start, new([]Inst, 1000)[0:0]);
}

func (re *RE) DoParse() {
	parser := NewParser(re);
	start := new(Start);
	s, e := parser.Regexp();
	if isNULL(s) {
		if !isNULL(e) { re.Error(ErrInternal) }
		e = start;
	}
	start.next = s;
	re.start = start;
	e.SetNext(new(End));

	re.DumpAll();
	println();

	re.EliminateNops(re.start);

	re.DumpAll();
	println();

	re.Error(ErrUnimplemented);
}

func Compiler(str string, ch *chan *RE) {
	re := new(RE);
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
