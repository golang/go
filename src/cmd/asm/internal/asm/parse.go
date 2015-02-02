// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package asm implements the parser and instruction generator for the assembler.
// TODO: Split apart?
package asm

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"text/scanner"

	"cmd/asm/internal/addr"
	"cmd/asm/internal/arch"
	"cmd/asm/internal/lex"
	"cmd/internal/obj"
)

type Parser struct {
	lex           lex.TokenReader
	lineNum       int   // Line number in source file.
	histLineNum   int   // Cumulative line number across source files.
	errorLine     int   // (Cumulative) line number of last error.
	errorCount    int   // Number of errors.
	pc            int64 // virtual PC; count of Progs; doesn't advance for GLOBL or DATA.
	input         []lex.Token
	inputPos      int
	pendingLabels []string // Labels to attach to next instruction.
	labels        map[string]*obj.Prog
	toPatch       []Patch
	addr          []addr.Addr
	arch          *arch.Arch
	linkCtxt      *obj.Link
	firstProg     *obj.Prog
	lastProg      *obj.Prog
	dataAddr      map[string]int64 // Most recent address for DATA for this symbol.
}

type Patch struct {
	prog  *obj.Prog
	label string
}

func NewParser(ctxt *obj.Link, ar *arch.Arch, lexer lex.TokenReader) *Parser {
	return &Parser{
		linkCtxt: ctxt,
		arch:     ar,
		lex:      lexer,
		labels:   make(map[string]*obj.Prog),
		dataAddr: make(map[string]int64),
	}
}

func (p *Parser) errorf(format string, args ...interface{}) {
	if p.histLineNum == p.errorLine {
		// Only one error per line.
		return
	}
	p.errorLine = p.histLineNum
	// Put file and line information on head of message.
	format = "%s:%d: " + format + "\n"
	args = append([]interface{}{p.lex.File(), p.lineNum}, args...)
	fmt.Fprintf(os.Stderr, format, args...)
	p.errorCount++
	if p.errorCount > 10 {
		log.Fatal("too many errors")
	}
}

func (p *Parser) Parse() (*obj.Prog, bool) {
	for p.line() {
	}
	if p.errorCount > 0 {
		return nil, false
	}
	p.patch()
	return p.firstProg, true
}

// WORD [ arg {, arg} ] (';' | '\n')
func (p *Parser) line() bool {
	// Skip newlines.
	var tok lex.ScanToken
	for {
		tok = p.lex.Next()
		// We save the line number here so error messages from this instruction
		// are labeled with this line. Otherwise we complain after we've absorbed
		// the terminating newline and the line numbers are off by one in errors.
		p.lineNum = p.lex.Line()
		p.histLineNum = lex.HistLine()
		switch tok {
		case '\n', ';':
			continue
		case scanner.EOF:
			return false
		}
		break
	}
	// First item must be an identifier.
	if tok != scanner.Ident {
		p.errorf("expected identifier, found %q", p.lex.Text())
		return false // Might as well stop now.
	}
	word := p.lex.Text()
	operands := make([][]lex.Token, 0, 3)
	// Zero or more comma-separated operands, one per loop.
	for tok != '\n' && tok != ';' {
		// Process one operand.
		items := make([]lex.Token, 0, 3)
		for {
			tok = p.lex.Next()
			if tok == ':' && len(operands) == 0 && len(items) == 0 { // First token.
				p.pendingLabels = append(p.pendingLabels, word)
				return true
			}
			if tok == scanner.EOF {
				p.errorf("unexpected EOF")
				return false
			}
			if tok == '\n' || tok == ';' || tok == ',' {
				break
			}
			items = append(items, lex.Make(tok, p.lex.Text()))
		}
		if len(items) > 0 {
			operands = append(operands, items)
		} else if len(operands) > 0 || tok == ',' {
			// Had a comma with nothing after.
			p.errorf("missing operand")
		}
	}
	i := p.arch.Pseudos[word]
	if i != 0 {
		p.pseudo(i, word, operands)
		return true
	}
	i = p.arch.Instructions[word]
	if i != 0 {
		p.instruction(i, word, operands)
		return true
	}
	p.errorf("unrecognized instruction %s", word)
	return true
}

func (p *Parser) instruction(op int, word string, operands [][]lex.Token) {
	p.addr = p.addr[0:0]
	for _, op := range operands {
		p.addr = append(p.addr, p.address(op))
	}
	// Is it a jump? TODO
	if word[0] == 'J' || word == "CALL" {
		p.asmJump(op, p.addr)
		return
	}
	p.asmInstruction(op, p.addr)
}

func (p *Parser) pseudo(op int, word string, operands [][]lex.Token) {
	switch op {
	case p.arch.ATEXT:
		p.asmText(word, operands)
	case p.arch.ADATA:
		p.asmData(word, operands)
	case p.arch.AGLOBL:
		p.asmGlobl(word, operands)
	case p.arch.APCDATA:
		p.asmPCData(word, operands)
	case p.arch.AFUNCDATA:
		p.asmFuncData(word, operands)
	default:
		p.errorf("unimplemented: %s", word)
	}
}

func (p *Parser) start(operand []lex.Token) {
	p.input = operand
	p.inputPos = 0
}

// address parses the operand into a link address structure.
func (p *Parser) address(operand []lex.Token) addr.Addr {
	p.start(operand)
	addr := addr.Addr{}
	p.operand(&addr)
	return addr
}

// parse (R). The opening paren is known to be there.
// The return value states whether it was a scaled mode.
func (p *Parser) parenRegister(a *addr.Addr) bool {
	p.next()
	tok := p.next()
	if tok.ScanToken != scanner.Ident {
		p.errorf("expected register, got %s", tok)
	}
	r, present := p.arch.Registers[tok.String()]
	if !present {
		p.errorf("expected register, found %s", tok.String())
	}
	a.IsIndirect = true
	scaled := p.peek() == '*'
	if scaled {
		// (R*2)
		p.next()
		tok := p.get(scanner.Int)
		a.Scale = p.scale(tok.String())
		a.Index = int16(r) // TODO: r should have type int16 but is uint8.
	} else {
		if a.HasRegister {
			p.errorf("multiple indirections")
		}
		a.HasRegister = true
		a.Register = int16(r)
	}
	p.expect(')')
	p.next()
	return scaled
}

// scale converts a decimal string into a valid scale factor.
func (p *Parser) scale(s string) int8 {
	switch s {
	case "1", "2", "4", "8":
		return int8(s[0] - '0')
	}
	p.errorf("bad scale: %s", s)
	return 0
}

// parse (R) or (R)(R*scale). The opening paren is known to be there.
func (p *Parser) addressMode(a *addr.Addr) {
	scaled := p.parenRegister(a)
	if !scaled && p.peek() == '(' {
		p.parenRegister(a)
	}
}

// operand parses a general operand and stores the result in *a.
func (p *Parser) operand(a *addr.Addr) bool {
	if len(p.input) == 0 {
		p.errorf("empty operand: cannot happen")
		return false
	}
	switch p.peek() {
	case '$':
		p.next()
		switch p.peek() {
		case scanner.Ident:
			a.IsImmediateAddress = true
			p.operand(a) // TODO
		case scanner.String:
			a.IsImmediateConstant = true
			a.HasString = true
			a.String = p.atos(p.next().String())
		case scanner.Int, scanner.Float, '+', '-', '~', '(':
			a.IsImmediateConstant = true
			if p.have(scanner.Float) {
				a.HasFloat = true
				a.Float = p.floatExpr()
			} else {
				a.HasOffset = true
				a.Offset = int64(p.expr())
			}
		default:
			p.errorf("illegal %s in immediate operand", p.next().String())
		}
	case '*':
		p.next()
		tok := p.next()
		r, present := p.arch.Registers[tok.String()]
		if !present {
			p.errorf("expected register; got %s", tok.String())
		}
		a.HasRegister = true
		a.Register = int16(r)
	case '(':
		p.next()
		if p.peek() == scanner.Ident {
			p.back()
			p.addressMode(a)
			break
		}
		p.back()
		fallthrough
	case '+', '-', '~', scanner.Int, scanner.Float:
		if p.have(scanner.Float) {
			a.HasFloat = true
			a.Float = p.floatExpr()
		} else {
			a.HasOffset = true
			a.Offset = int64(p.expr())
		}
		if p.peek() != scanner.EOF {
			p.expect('(')
			p.addressMode(a)
		}
	case scanner.Ident:
		tok := p.next()
		// Either R or (most general) ident<>+4(SB)(R*scale).
		if r, present := p.arch.Registers[tok.String()]; present {
			a.HasRegister = true
			a.Register = int16(r)
			// Possibly register pair: DX:AX.
			if p.peek() == ':' {
				p.next()
				tok = p.get(scanner.Ident)
				a.HasRegister2 = true
				a.Register2 = int16(p.arch.Registers[tok.String()])
			}
			break
		}
		// Weirdness with statics: Might now have "<>".
		if p.peek() == '<' {
			p.next()
			p.get('>')
			a.IsStatic = true
		}
		if p.peek() == '+' || p.peek() == '-' {
			a.HasOffset = true
			a.Offset = int64(p.expr())
		}
		a.Symbol = tok.String()
		if p.peek() == scanner.EOF {
			break
		}
		// Expect (SB) or (FP)
		p.expect('(')
		p.parenRegister(a)
		if a.Register != arch.RSB && a.Register != arch.RFP && a.Register != arch.RSP {
			p.errorf("expected SB, FP, or SP offset for %s", tok)
		}
		// Possibly have scaled register (CX*8).
		if p.peek() != scanner.EOF {
			p.expect('(')
			p.addressMode(a)
		}
	default:
		p.errorf("unexpected %s in operand", p.next())
	}
	p.expect(scanner.EOF)
	return true
}

// Note: There are two changes in the expression handling here
// compared to the old yacc/C implemenatations. Neither has
// much practical consequence because the expressions we
// see in assembly code are simple, but for the record:
//
// 1) Evaluation uses uint64; the old one used int64.
// 2) Precedence uses Go rules not C rules.

// expr = term | term ('+' | '-' | '|' | '^') term.
func (p *Parser) expr() uint64 {
	value := p.term()
	for {
		switch p.peek() {
		case '+':
			p.next()
			value += p.term()
		case '-':
			p.next()
			value -= p.term()
		case '|':
			p.next()
			value |= p.term()
		case '^':
			p.next()
			value ^= p.term()
		default:
			return value
		}
	}
}

// floatExpr = fconst | '-' floatExpr | '+' floatExpr | '(' floatExpr ')'
func (p *Parser) floatExpr() float64 {
	tok := p.next()
	switch tok.ScanToken {
	case '(':
		v := p.floatExpr()
		if p.next().ScanToken != ')' {
			p.errorf("missing closing paren")
		}
		return v
	case '+':
		return +p.floatExpr()
	case '-':
		return -p.floatExpr()
	case scanner.Float:
		return p.atof(tok.String())
	}
	p.errorf("unexpected %s evaluating float expression", tok)
	return 0
}

// term = factor | factor ('*' | '/' | '%' | '>>' | '<<' | '&') factor
func (p *Parser) term() uint64 {
	value := p.factor()
	for {
		switch p.peek() {
		case '*':
			p.next()
			value *= p.factor()
		case '/':
			p.next()
			if value&(1<<63) != 0 {
				p.errorf("divide with high bit set")
			}
			value /= p.factor()
		case '%':
			p.next()
			value %= p.factor()
		case lex.LSH:
			p.next()
			shift := p.factor()
			if int64(shift) < 0 {
				p.errorf("negative left shift %d", shift)
			}
			return value << shift
		case lex.RSH:
			p.next()
			shift := p.term()
			if shift < 0 {
				p.errorf("negative right shift %d", shift)
			}
			if shift > 0 && value&(1<<63) != 0 {
				p.errorf("right shift with high bit set")
			}
			value >>= uint(shift)
		case '&':
			p.next()
			value &= p.factor()
		default:
			return value
		}
	}
}

// factor = const | '+' factor | '-' factor | '~' factor | '(' expr ')'
func (p *Parser) factor() uint64 {
	tok := p.next()
	switch tok.ScanToken {
	case scanner.Int:
		return p.atoi(tok.String())
	case '+':
		return +p.factor()
	case '-':
		return -p.factor()
	case '~':
		return ^p.factor()
	case '(':
		v := p.expr()
		if p.next().ScanToken != ')' {
			p.errorf("missing closing paren")
		}
		return v
	}
	p.errorf("unexpected %s evaluating expression", tok)
	return 0
}

// positiveAtoi returns an int64 that must be >= 0.
func (p *Parser) positiveAtoi(str string) int64 {
	value, err := strconv.ParseInt(str, 0, 64)
	if err != nil {
		p.errorf("%s", err)
	}
	if value < 0 {
		p.errorf("%s overflows int64", str)
	}
	return value
}

func (p *Parser) atoi(str string) uint64 {
	value, err := strconv.ParseUint(str, 0, 64)
	if err != nil {
		p.errorf("%s", err)
	}
	return value
}

func (p *Parser) atof(str string) float64 {
	value, err := strconv.ParseFloat(str, 64)
	if err != nil {
		p.errorf("%s", err)
	}
	return value
}

func (p *Parser) atos(str string) string {
	value, err := strconv.Unquote(str)
	if err != nil {
		p.errorf("%s", err)
	}
	return value
}

// EOF represents the end of input.
var EOF = lex.Make(scanner.EOF, "EOF")

func (p *Parser) next() lex.Token {
	if !p.more() {
		return EOF
	}
	tok := p.input[p.inputPos]
	p.inputPos++
	return tok
}

func (p *Parser) back() {
	p.inputPos--
}

func (p *Parser) peek() lex.ScanToken {
	if p.more() {
		return p.input[p.inputPos].ScanToken
	}
	return scanner.EOF
}

func (p *Parser) more() bool {
	return p.inputPos < len(p.input)
}

// get verifies that the next item has the expected type and returns it.
func (p *Parser) get(expected lex.ScanToken) lex.Token {
	p.expect(expected)
	return p.next()
}

// expect verifies that the next item has the expected type. It does not consume it.
func (p *Parser) expect(expected lex.ScanToken) {
	if p.peek() != expected {
		p.errorf("expected %s, found %s", expected, p.next())
	}
}

// have reports whether the remaining tokens contain the specified token.
func (p *Parser) have(token lex.ScanToken) bool {
	for i := p.inputPos; i < len(p.input); i++ {
		if p.input[i].ScanToken == token {
			return true
		}
	}
	return false
}
