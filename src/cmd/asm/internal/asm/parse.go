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

	"cmd/asm/internal/arch"
	"cmd/asm/internal/lex"
	"cmd/internal/obj"
)

type Parser struct {
	lex           lex.TokenReader
	lineNum       int   // Line number in source file.
	histLineNum   int32 // Cumulative line number across source files.
	errorLine     int32 // (Cumulative) line number of last error.
	errorCount    int   // Number of errors.
	pc            int64 // virtual PC; count of Progs; doesn't advance for GLOBL or DATA.
	input         []lex.Token
	inputPos      int
	pendingLabels []string // Labels to attach to next instruction.
	labels        map[string]*obj.Prog
	toPatch       []Patch
	addr          []obj.Addr
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
	i := arch.Pseudos[word]
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
	isJump := word[0] == 'J' || word == "CALL" // TODO: do this better
	for _, op := range operands {
		addr := p.address(op)
		if !isJump && addr.Reg < 0 { // Jumps refer to PC, a pseudo.
			p.errorf("illegal use of pseudo-register")
		}
		p.addr = append(p.addr, addr)
	}
	if isJump {
		p.asmJump(op, p.addr)
		return
	}
	p.asmInstruction(op, p.addr)
}

func (p *Parser) pseudo(op int, word string, operands [][]lex.Token) {
	switch op {
	case obj.ATEXT:
		p.asmText(word, operands)
	case obj.ADATA:
		p.asmData(word, operands)
	case obj.AGLOBL:
		p.asmGlobl(word, operands)
	case obj.APCDATA:
		p.asmPCData(word, operands)
	case obj.AFUNCDATA:
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
func (p *Parser) address(operand []lex.Token) obj.Addr {
	p.start(operand)
	addr := obj.Addr{}
	p.operand(&addr)
	return addr
}

// parseScale converts a decimal string into a valid scale factor.
func (p *Parser) parseScale(s string) int8 {
	switch s {
	case "1", "2", "4", "8":
		return int8(s[0] - '0')
	}
	p.errorf("bad scale: %s", s)
	return 0
}

// operand parses a general operand and stores the result in *a.
func (p *Parser) operand(a *obj.Addr) bool {
	if len(p.input) == 0 {
		p.errorf("empty operand: cannot happen")
		return false
	}
	// General address (with a few exceptions) looks like
	//	$sym±offset(symkind)(reg)(index*scale)
	// Every piece is optional, so we scan left to right and what
	// we discover tells us where we are.
	var prefix rune
	switch tok := p.peek(); tok {
	case '$', '*':
		prefix = rune(tok)
		p.next()
	}
	switch p.peek() {
	case scanner.Ident:
		tok := p.next()
		if r1, r2, scale, ok := p.register(tok.String(), prefix); ok {
			if scale != 0 {
				p.errorf("expected simple register reference")
			}
			a.Type = obj.TYPE_REG
			a.Reg = r1
			if r2 != 0 {
				// Form is R1:R2. It is on RHS and the second register
				// needs to go into the LHS. This is a horrible hack. TODO.
				a.Class = int8(r2)
			}
			break // Nothing can follow.
		}
		p.symbolReference(a, tok.String(), prefix)
	case scanner.Int, scanner.Float, scanner.String, '+', '-', '~', '(':
		if p.have(scanner.Float) {
			if prefix != '$' {
				p.errorf("floating-point constant must be an immediate")
			}
			a.Type = obj.TYPE_FCONST
			a.U.Dval = p.floatExpr()
			break
		}
		if p.have(scanner.String) {
			if prefix != '$' {
				p.errorf("string constant must be an immediate")
			}
			str, err := strconv.Unquote(p.get(scanner.String).String())
			if err != nil {
				p.errorf("string parse error: %s", err)
			}
			a.Type = obj.TYPE_SCONST
			a.U.Sval = str
			break
		}
		// Might be parenthesized arithmetic expression or (possibly scaled) register indirect.
		// Peek into the input to discriminate.
		if p.peek() == '(' && len(p.input[p.inputPos:]) >= 3 && p.input[p.inputPos+1].ScanToken == scanner.Ident {
			// Register indirect (the identifier must be a register). The offset will be zero.
		} else {
			// Integer offset before register.
			a.Offset = int64(p.expr())
		}
		if p.peek() != '(' {
			// Just an integer.
			switch prefix {
			case '$':
				a.Type = obj.TYPE_CONST
			case '*':
				a.Type = obj.TYPE_INDIR // Can appear but is illegal, will be rejected by the linker.
			default:
				a.Type = obj.TYPE_MEM
			}
			break // Nothing can follow.
		}
		p.next()
		tok := p.next()
		r1, r2, scale, ok := p.register(tok.String(), 0)
		if !ok {
			p.errorf("indirect through non-register %s", tok)
		}
		if r2 != 0 {
			p.errorf("indirect through register pair")
		}
		a.Type = obj.TYPE_MEM
		if prefix == '$' {
			a.Type = obj.TYPE_ADDR
		}
		a.Reg = r1
		a.Scale = scale
		p.get(')')
		if scale == 0 && p.peek() == '(' {
			p.next()
			tok := p.next()
			r1, r2, scale, ok = p.register(tok.String(), 0)
			if !ok {
				p.errorf("indirect through non-register %s", tok)
			}
			if r2 != 0 {
				p.errorf("unimplemented two-register form")
			}
			a.Index = r1
			a.Scale = scale
			p.get(')')
		}
	}
	p.expect(scanner.EOF)
	return true
}

// register parses a register reference where there is no symbol present (as in 4(R0) not sym(SB)).
func (p *Parser) register(name string, prefix rune) (r1, r2 int16, scale int8, ok bool) {
	// R1 or R1:R2 or R1*scale.
	var present bool
	r1, present = p.arch.Registers[name]
	if !present {
		return
	}
	if prefix != 0 {
		p.errorf("prefix %c not allowed for register: $%s", prefix, name)
	}
	if p.peek() == ':' {
		// 2nd register.
		p.next()
		name := p.next().String()
		r2, present = p.arch.Registers[name]
		if !present {
			p.errorf("%s not a register", name)
		}
	}
	if p.peek() == '*' {
		// Scale
		p.next()
		scale = p.parseScale(p.next().String())
	}
	// TODO: Shifted register for ARM
	return r1, r2, scale, true
}

// symbolReference parses a symbol that is known not to be a register.
func (p *Parser) symbolReference(a *obj.Addr, name string, prefix rune) {
	// Identifier is a name.
	switch prefix {
	case 0:
		a.Type = obj.TYPE_MEM
	case '$':
		a.Type = obj.TYPE_ADDR
	case '*':
		a.Type = obj.TYPE_INDIR
	}
	// Weirdness with statics: Might now have "<>".
	isStatic := 0 // TODO: Really a boolean, but Linklookup wants a "version" integer.
	if p.peek() == '<' {
		isStatic = 1
		p.next()
		p.get('>')
	}
	if p.peek() == '+' || p.peek() == '-' {
		a.Offset = int64(p.expr())
	}
	a.Sym = obj.Linklookup(p.linkCtxt, name, isStatic)
	if p.peek() == scanner.EOF {
		return
	}
	// Expect (SB) or (FP) or (SP).
	p.get('(')
	reg := p.get(scanner.Ident).String()
	switch reg {
	case "FP":
		a.Name = obj.NAME_PARAM
	case "SB":
		a.Name = obj.NAME_EXTERN
		if isStatic != 0 {
			a.Name = obj.NAME_STATIC
		}
	case "SP":
		a.Name = obj.NAME_AUTO // The pseudo-stack.
	default:
		p.errorf("expected SB, FP, or SP offset for %s", name)
	}
	a.Reg = 0 // There is no register here; these are pseudo-registers.
	p.get(')')
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
