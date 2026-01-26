// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package asm implements the parser and instruction generator for the assembler.
// TODO: Split apart?
package asm

import (
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
	"text/scanner"
	"unicode/utf8"

	"cmd/asm/internal/arch"
	"cmd/asm/internal/flags"
	"cmd/asm/internal/lex"
	"cmd/internal/obj"
	"cmd/internal/obj/arm64"
	"cmd/internal/obj/riscv"
	"cmd/internal/obj/x86"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"cmd/internal/sys"
)

type Parser struct {
	lex           lex.TokenReader
	lineNum       int   // Line number in source file.
	errorLine     int   // Line number of last error.
	errorCount    int   // Number of errors.
	sawCode       bool  // saw code in this file (as opposed to comments and blank lines)
	pc            int64 // virtual PC; count of Progs; doesn't advance for GLOBL or DATA.
	input         []lex.Token
	inputPos      int
	pendingLabels []string // Labels to attach to next instruction.
	labels        map[string]*obj.Prog
	toPatch       []Patch
	addr          []obj.Addr
	arch          *arch.Arch
	ctxt          *obj.Link
	firstProg     *obj.Prog
	lastProg      *obj.Prog
	dataAddr      map[string]int64 // Most recent address for DATA for this symbol.
	isJump        bool             // Instruction being assembled is a jump.
	allowABI      bool             // Whether ABI selectors are allowed.
	pkgPrefix     string           // Prefix to add to local symbols.
	errorWriter   io.Writer
}

type Patch struct {
	addr  *obj.Addr
	label string
}

func NewParser(ctxt *obj.Link, ar *arch.Arch, lexer lex.TokenReader) *Parser {
	pkgPrefix := obj.UnlinkablePkg
	if ctxt != nil {
		pkgPrefix = objabi.PathToPrefix(ctxt.Pkgpath)
	}
	return &Parser{
		ctxt:        ctxt,
		arch:        ar,
		lex:         lexer,
		labels:      make(map[string]*obj.Prog),
		dataAddr:    make(map[string]int64),
		errorWriter: os.Stderr,
		allowABI:    ctxt != nil && objabi.LookupPkgSpecial(ctxt.Pkgpath).AllowAsmABI,
		pkgPrefix:   pkgPrefix,
	}
}

// panicOnError is enabled when testing to abort execution on the first error
// and turn it into a recoverable panic.
var panicOnError bool

func (p *Parser) errorf(format string, args ...any) {
	if panicOnError {
		panic(fmt.Errorf(format, args...))
	}
	if p.lineNum == p.errorLine {
		// Only one error per line.
		return
	}
	p.errorLine = p.lineNum
	if p.lex != nil {
		// Put file and line information on head of message.
		format = "%s:%d: " + format + "\n"
		args = append([]any{p.lex.File(), p.lineNum}, args...)
	}
	fmt.Fprintf(p.errorWriter, format, args...)
	p.errorCount++
	if p.errorCount > 10 && !*flags.AllErrors {
		log.Fatal("too many errors")
	}
}

func (p *Parser) pos() src.XPos {
	return p.ctxt.PosTable.XPos(src.MakePos(p.lex.Base(), uint(p.lineNum), 0))
}

func (p *Parser) Parse() (*obj.Prog, bool) {
	scratch := make([][]lex.Token, 0, 3)
	for {
		word, cond, operands, ok := p.line(scratch)
		if !ok {
			break
		}
		scratch = operands

		if p.pseudo(word, operands) {
			continue
		}
		i, present := p.arch.Instructions[word]
		if present {
			p.instruction(i, word, cond, operands)
			continue
		}
		p.errorf("unrecognized instruction %q", word)
	}
	if p.errorCount > 0 {
		return nil, false
	}
	p.patch()
	return p.firstProg, true
}

// ParseSymABIs parses p's assembly code to find text symbol
// definitions and references and writes a symabis file to w.
func (p *Parser) ParseSymABIs(w io.Writer) bool {
	operands := make([][]lex.Token, 0, 3)
	for {
		word, _, operands1, ok := p.line(operands)
		if !ok {
			break
		}
		operands = operands1

		p.symDefRef(w, word, operands)
	}
	return p.errorCount == 0
}

// nextToken returns the next non-build-comment token from the lexer.
// It reports misplaced //go:build comments but otherwise discards them.
func (p *Parser) nextToken() lex.ScanToken {
	for {
		tok := p.lex.Next()
		if tok == lex.BuildComment {
			if p.sawCode {
				p.errorf("misplaced //go:build comment")
			}
			continue
		}
		if tok != '\n' {
			p.sawCode = true
		}
		if tok == '#' {
			// A leftover wisp of a #include/#define/etc,
			// to let us know that p.sawCode should be true now.
			// Otherwise ignored.
			continue
		}
		return tok
	}
}

// line consumes a single assembly line from p.lex of the form
//
//	{label:} WORD[.cond] [ arg {, arg} ] (';' | '\n')
//
// It adds any labels to p.pendingLabels and returns the word, cond,
// operand list, and true. If there is an error or EOF, it returns
// ok=false.
//
// line may reuse the memory from scratch.
func (p *Parser) line(scratch [][]lex.Token) (word, cond string, operands [][]lex.Token, ok bool) {
next:
	// Skip newlines.
	var tok lex.ScanToken
	for {
		tok = p.nextToken()
		// We save the line number here so error messages from this instruction
		// are labeled with this line. Otherwise we complain after we've absorbed
		// the terminating newline and the line numbers are off by one in errors.
		p.lineNum = p.lex.Line()
		switch tok {
		case '\n', ';':
			continue
		case scanner.EOF:
			return "", "", nil, false
		}
		break
	}
	// First item must be an identifier.
	if tok != scanner.Ident {
		p.errorf("expected identifier, found %q", p.lex.Text())
		return "", "", nil, false // Might as well stop now.
	}
	word, cond = p.lex.Text(), ""
	operands = scratch[:0]
	// Zero or more comma-separated operands, one per loop.
	nesting := 0
	colon := -1
	for tok != '\n' && tok != ';' {
		// Process one operand.
		var items []lex.Token
		if cap(operands) > len(operands) {
			// Reuse scratch items slice.
			items = operands[:cap(operands)][len(operands)][:0]
		} else {
			items = make([]lex.Token, 0, 3)
		}
		for {
			tok = p.nextToken()
			if len(operands) == 0 && len(items) == 0 {
				if p.arch.InFamily(sys.ARM, sys.ARM64, sys.AMD64, sys.I386, sys.Loong64, sys.RISCV64) && tok == '.' {
					// Suffixes: ARM conditionals, Loong64 vector instructions, RISCV rounding mode or x86 modifiers.
					tok = p.nextToken()
					str := p.lex.Text()
					if tok != scanner.Ident {
						p.errorf("instruction suffix expected identifier, found %s", str)
					}
					cond = cond + "." + str
					continue
				}
				if tok == ':' {
					// Labels.
					p.pendingLabels = append(p.pendingLabels, word)
					goto next
				}
			}
			if tok == scanner.EOF {
				p.errorf("unexpected EOF")
				return "", "", nil, false
			}
			// Split operands on comma. Also, the old syntax on x86 for a "register pair"
			// was AX:DX, for which the new syntax is DX, AX. Note the reordering.
			if tok == '\n' || tok == ';' || (nesting == 0 && (tok == ',' || tok == ':')) {
				if tok == ':' {
					// Remember this location so we can swap the operands below.
					if colon >= 0 {
						p.errorf("invalid ':' in operand")
						return word, cond, operands, true
					}
					colon = len(operands)
				}
				break
			}
			if tok == '(' || tok == '[' {
				nesting++
			}
			if tok == ')' || tok == ']' {
				nesting--
			}
			items = append(items, lex.Make(tok, p.lex.Text()))
		}
		if len(items) > 0 {
			operands = append(operands, items)
			if colon >= 0 && len(operands) == colon+2 {
				// AX:DX becomes DX, AX.
				operands[colon], operands[colon+1] = operands[colon+1], operands[colon]
				colon = -1
			}
		} else if len(operands) > 0 || tok == ',' || colon >= 0 {
			// Had a separator with nothing after.
			p.errorf("missing operand")
		}
	}
	return word, cond, operands, true
}

func (p *Parser) instruction(op obj.As, word, cond string, operands [][]lex.Token) {
	p.addr = p.addr[0:0]
	p.isJump = p.arch.IsJump(word)
	for _, op := range operands {
		addr := p.address(op)
		if !p.isJump && addr.Reg < 0 { // Jumps refer to PC, a pseudo.
			p.errorf("illegal use of pseudo-register in %s", word)
		}
		p.addr = append(p.addr, addr)
	}
	if p.isJump {
		p.asmJump(op, cond, p.addr)
		return
	}
	p.asmInstruction(op, cond, p.addr)
}

func (p *Parser) pseudo(word string, operands [][]lex.Token) bool {
	switch word {
	case "DATA":
		p.asmData(operands)
	case "FUNCDATA":
		p.asmFuncData(operands)
	case "GLOBL":
		p.asmGlobl(operands)
	case "PCDATA":
		p.asmPCData(operands)
	case "PCALIGN":
		p.asmPCAlign(operands)
	case "TEXT":
		p.asmText(operands)
	default:
		return false
	}
	return true
}

// symDefRef scans a line for potential text symbol definitions and
// references and writes symabis information to w.
//
// The symabis format is documented at
// cmd/compile/internal/ssagen.ReadSymABIs.
func (p *Parser) symDefRef(w io.Writer, word string, operands [][]lex.Token) {
	switch word {
	case "TEXT":
		// Defines text symbol in operands[0].
		if len(operands) > 0 {
			p.start(operands[0])
			if name, abi, ok := p.funcAddress(); ok {
				fmt.Fprintf(w, "def %s %s\n", name, abi)
			}
		}
		return
	case "GLOBL", "PCDATA":
		// No text definitions or symbol references.
	case "DATA", "FUNCDATA":
		// For DATA, operands[0] is defined symbol.
		// For FUNCDATA, operands[0] is an immediate constant.
		// Remaining operands may have references.
		if len(operands) < 2 {
			return
		}
		operands = operands[1:]
	}
	// Search for symbol references.
	for _, op := range operands {
		p.start(op)
		if name, abi, ok := p.funcAddress(); ok {
			fmt.Fprintf(w, "ref %s %s\n", name, abi)
		}
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
func (p *Parser) operand(a *obj.Addr) {
	//fmt.Printf("Operand: %v\n", p.input)
	if len(p.input) == 0 {
		p.errorf("empty operand: cannot happen")
		return
	}
	// General address (with a few exceptions) looks like
	//	$sym±offset(SB)(reg)(index*scale)
	// Exceptions are:
	//
	//	R1
	//	offset
	//	$offset
	// Every piece is optional, so we scan left to right and what
	// we discover tells us where we are.

	// Prefix: $.
	var prefix rune
	switch tok := p.peek(); tok {
	case '$', '*':
		prefix = rune(tok)
		p.next()
	}

	// Symbol: sym±offset(SB)
	tok := p.next()
	name := tok.String()
	if tok.ScanToken == scanner.Ident && !p.atStartOfRegister(name) {
		// See if this is an architecture specific special operand.
		switch p.arch.Family {
		case sys.ARM64:
			if opd := arch.ARM64SpecialOperand(name); opd != arm64.SPOP_END {
				a.Type = obj.TYPE_SPECIAL
				a.Offset = int64(opd)
			}
		case sys.RISCV64:
			if opd := arch.RISCV64SpecialOperand(name); opd != riscv.SPOP_END {
				a.Type = obj.TYPE_SPECIAL
				a.Offset = int64(opd)
			}
		}

		if a.Type != obj.TYPE_SPECIAL {
			// We have a symbol. Parse $sym±offset(symkind)
			p.symbolReference(a, p.qualifySymbol(name), prefix)
		}
		// fmt.Printf("SYM %s\n", obj.Dconv(&emptyProg, 0, a))
		if p.peek() == scanner.EOF {
			return
		}
	}

	// Special register list syntax for arm: [R1,R3-R7]
	if tok.ScanToken == '[' {
		if prefix != 0 {
			p.errorf("illegal use of register list")
		}
		p.registerList(a)
		p.expectOperandEnd()
		return
	}

	// Register: R1
	if tok.ScanToken == scanner.Ident && p.atStartOfRegister(name) {
		if p.atRegisterShift() {
			// ARM shifted register such as R1<<R2 or R1>>2.
			a.Type = obj.TYPE_SHIFT
			a.Offset = p.registerShift(tok.String(), prefix)
			if p.peek() == '(' {
				// Can only be a literal register here.
				p.next()
				tok := p.next()
				name := tok.String()
				if !p.atStartOfRegister(name) {
					p.errorf("expected register; found %s", name)
				}
				a.Reg, _ = p.registerReference(name)
				p.get(')')
			}
		} else if p.atRegisterExtension() {
			a.Type = obj.TYPE_REG
			p.registerExtension(a, tok.String(), prefix)
			p.expectOperandEnd()
			return
		} else if r1, r2, scale, ok := p.register(tok.String(), prefix); ok {
			if scale != 0 {
				p.errorf("expected simple register reference")
			}
			a.Type = obj.TYPE_REG
			a.Reg = r1
			if r2 != 0 {
				// Form is R1:R2. It is on RHS and the second register
				// needs to go into the LHS.
				panic("cannot happen (Addr.Reg2)")
			}
		}
		// fmt.Printf("REG %s\n", obj.Dconv(&emptyProg, 0, a))
		p.expectOperandEnd()
		return
	}

	// Constant.
	haveConstant := false
	switch tok.ScanToken {
	case scanner.Int, scanner.Float, scanner.String, scanner.Char, '+', '-', '~':
		haveConstant = true
	case '(':
		// Could be parenthesized expression or (R). Must be something, though.
		tok := p.next()
		if tok.ScanToken == scanner.EOF {
			p.errorf("missing right parenthesis")
			return
		}
		rname := tok.String()
		p.back()
		haveConstant = !p.atStartOfRegister(rname)
		if !haveConstant {
			p.back() // Put back the '('.
		}
	}
	if haveConstant {
		p.back()
		if p.have(scanner.Float) {
			if prefix != '$' {
				p.errorf("floating-point constant must be an immediate")
			}
			a.Type = obj.TYPE_FCONST
			a.Val = p.floatExpr()
			// fmt.Printf("FCONST %s\n", obj.Dconv(&emptyProg, 0, a))
			p.expectOperandEnd()
			return
		}
		if p.have(scanner.String) {
			if prefix != '$' {
				p.errorf("string constant must be an immediate")
				return
			}
			str, err := strconv.Unquote(p.get(scanner.String).String())
			if err != nil {
				p.errorf("string parse error: %s", err)
			}
			a.Type = obj.TYPE_SCONST
			a.Val = str
			// fmt.Printf("SCONST %s\n", obj.Dconv(&emptyProg, 0, a))
			p.expectOperandEnd()
			return
		}
		a.Offset = int64(p.expr())
		if p.peek() != '(' {
			switch prefix {
			case '$':
				a.Type = obj.TYPE_CONST
			case '*':
				a.Type = obj.TYPE_INDIR // Can appear but is illegal, will be rejected by the linker.
			default:
				a.Type = obj.TYPE_MEM
			}
			// fmt.Printf("CONST %d %s\n", a.Offset, obj.Dconv(&emptyProg, 0, a))
			p.expectOperandEnd()
			return
		}
		// fmt.Printf("offset %d \n", a.Offset)
	}

	// Register indirection: (reg) or (index*scale). We are on the opening paren.
	p.registerIndirect(a, prefix)
	// fmt.Printf("DONE %s\n", p.arch.Dconv(&emptyProg, 0, a))

	p.expectOperandEnd()
	return
}

// atStartOfRegister reports whether the parser is at the start of a register definition.
func (p *Parser) atStartOfRegister(name string) bool {
	// Simple register: R10.
	_, present := p.arch.Register[name]
	if present {
		return true
	}
	// Parenthesized register: R(10).
	return p.arch.RegisterPrefix[name] && p.peek() == '('
}

// atRegisterShift reports whether we are at the start of an ARM shifted register.
// We have consumed the register or R prefix.
func (p *Parser) atRegisterShift() bool {
	// ARM only.
	if !p.arch.InFamily(sys.ARM, sys.ARM64) {
		return false
	}
	// R1<<...
	if lex.IsRegisterShift(p.peek()) {
		return true
	}
	// R(1)<<...   Ugly check. TODO: Rethink how we handle ARM register shifts to be
	// less special.
	if p.peek() != '(' || len(p.input)-p.inputPos < 4 {
		return false
	}
	return p.at('(', scanner.Int, ')') && lex.IsRegisterShift(p.input[p.inputPos+3].ScanToken)
}

// atRegisterExtension reports whether we are at the start of an ARM64 extended register.
// We have consumed the register or R prefix.
func (p *Parser) atRegisterExtension() bool {
	switch p.arch.Family {
	case sys.ARM64, sys.Loong64:
		// R1.xxx
		return p.peek() == '.'
	default:
		return false
	}
}

// registerReference parses a register given either the name, R10, or a parenthesized form, SPR(10).
func (p *Parser) registerReference(name string) (int16, bool) {
	r, present := p.arch.Register[name]
	if present {
		return r, true
	}
	if !p.arch.RegisterPrefix[name] {
		p.errorf("expected register; found %s", name)
		return 0, false
	}
	p.get('(')
	tok := p.get(scanner.Int)
	num, err := strconv.ParseInt(tok.String(), 10, 16)
	p.get(')')
	if err != nil {
		p.errorf("parsing register list: %s", err)
		return 0, false
	}
	r, ok := p.arch.RegisterNumber(name, int16(num))
	if !ok {
		p.errorf("illegal register %s(%d)", name, r)
		return 0, false
	}
	return r, true
}

// register parses a full register reference where there is no symbol present (as in 4(R0) or R(10) but not sym(SB))
// including forms involving multiple registers such as R1:R2.
func (p *Parser) register(name string, prefix rune) (r1, r2 int16, scale int8, ok bool) {
	// R1 or R(1) R1:R2 R1,R2 R1+R2, or R1*scale.
	r1, ok = p.registerReference(name)
	if !ok {
		return
	}
	if prefix != 0 && prefix != '*' { // *AX is OK.
		p.errorf("prefix %c not allowed for register: %c%s", prefix, prefix, name)
	}
	c := p.peek()
	if c == ':' || c == ',' || c == '+' {
		// 2nd register; syntax (R1+R2) etc. No two architectures agree.
		// Check the architectures match the syntax.
		switch p.next().ScanToken {
		case ',':
			if !p.arch.InFamily(sys.ARM, sys.ARM64) {
				p.errorf("(register,register) not supported on this architecture")
				return
			}
		case '+':
			if p.arch.Family != sys.PPC64 {
				p.errorf("(register+register) not supported on this architecture")
				return
			}
		}
		name := p.next().String()
		r2, ok = p.registerReference(name)
		if !ok {
			return
		}
	}
	if p.peek() == '*' {
		// Scale
		p.next()
		scale = p.parseScale(p.next().String())
	}
	return r1, r2, scale, true
}

// registerShift parses an ARM/ARM64 shifted register reference and returns the encoded representation.
// There is known to be a register (current token) and a shift operator (peeked token).
func (p *Parser) registerShift(name string, prefix rune) int64 {
	if prefix != 0 {
		p.errorf("prefix %c not allowed for shifted register: $%s", prefix, name)
	}
	// R1 op R2 or r1 op constant.
	// op is:
	//	"<<" == 0
	//	">>" == 1
	//	"->" == 2
	//	"@>" == 3
	r1, ok := p.registerReference(name)
	if !ok {
		return 0
	}
	var op int16
	switch p.next().ScanToken {
	case lex.LSH:
		op = 0
	case lex.RSH:
		op = 1
	case lex.ARR:
		op = 2
	case lex.ROT:
		// following instructions on ARM64 support rotate right
		// AND, ANDS, TST, BIC, BICS, EON, EOR, ORR, MVN, ORN
		op = 3
	}
	tok := p.next()
	str := tok.String()
	var count int16
	switch tok.ScanToken {
	case scanner.Ident:
		if p.arch.Family == sys.ARM64 {
			p.errorf("rhs of shift must be integer: %s", str)
		} else {
			r2, ok := p.registerReference(str)
			if !ok {
				p.errorf("rhs of shift must be register or integer: %s", str)
			}
			count = (r2&15)<<8 | 1<<4
		}
	case scanner.Int, '(':
		p.back()
		x := int64(p.expr())
		if p.arch.Family == sys.ARM64 {
			if x >= 64 {
				p.errorf("register shift count too large: %s", str)
			}
			count = int16((x & 63) << 10)
		} else {
			if x >= 32 {
				p.errorf("register shift count too large: %s", str)
			}
			count = int16((x & 31) << 7)
		}
	default:
		p.errorf("unexpected %s in register shift", tok.String())
	}
	if p.arch.Family == sys.ARM64 {
		off, err := arch.ARM64RegisterShift(r1, op, count)
		if err != nil {
			p.errorf("%v", err)
		}
		return off
	} else {
		return int64((r1 & 15) | op<<5 | count)
	}
}

// registerExtension parses a register with extension or arrangement.
// There is known to be a register (current token) and an extension operator (peeked token).
func (p *Parser) registerExtension(a *obj.Addr, name string, prefix rune) {
	if prefix != 0 {
		p.errorf("prefix %c not allowed for shifted register: $%s", prefix, name)
	}

	reg, ok := p.registerReference(name)
	if !ok {
		p.errorf("unexpected %s in register extension", name)
		return
	}

	isIndex := false
	num := int16(0)
	isAmount := true // Amount is zero by default
	ext := ""
	if p.peek() == lex.LSH {
		// (Rn)(Rm<<2), the shifted offset register.
		ext = "LSL"
	} else {
		// (Rn)(Rm.UXTW<1), the extended offset register.
		// Rm.UXTW<<3, the extended register.
		p.get('.')
		tok := p.next()
		ext = tok.String()
	}
	if p.peek() == lex.LSH {
		// parses left shift amount applied after extension: <<Amount
		p.get(lex.LSH)
		tok := p.get(scanner.Int)
		amount, err := strconv.ParseInt(tok.String(), 10, 16)
		if err != nil {
			p.errorf("parsing left shift amount: %s", err)
		}
		num = int16(amount)
	} else if p.peek() == '[' {
		// parses an element: [Index]
		p.get('[')
		tok := p.get(scanner.Int)
		index, err := strconv.ParseInt(tok.String(), 10, 16)
		p.get(']')
		if err != nil {
			p.errorf("parsing element index: %s", err)
		}
		isIndex = true
		isAmount = false
		num = int16(index)
	}

	switch p.arch.Family {
	case sys.ARM64:
		err := arm64.ARM64RegisterExtension(a, ext, reg, num, isAmount, isIndex)
		if err != nil {
			p.errorf("%v", err)
		}
	case sys.Loong64:
		err := arch.Loong64RegisterExtension(a, ext, reg, num, isAmount, isIndex)
		if err != nil {
			p.errorf("%v", err)
		}
	default:
		p.errorf("register extension not supported on this architecture")
	}
}

// qualifySymbol returns name as a package-qualified symbol name. If
// name starts with a period, qualifySymbol prepends the package
// prefix. Otherwise it returns name unchanged.
func (p *Parser) qualifySymbol(name string) string {
	if strings.HasPrefix(name, ".") {
		name = p.pkgPrefix + name
	}
	return name
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

	// Parse optional <> (indicates a static symbol) or
	// <ABIxxx> (selecting text symbol with specific ABI).
	doIssueError := true
	isStatic, abi := p.symRefAttrs(name, doIssueError)

	if p.peek() == '+' || p.peek() == '-' {
		a.Offset = int64(p.expr())
	}
	if isStatic {
		a.Sym = p.ctxt.LookupStatic(name)
	} else {
		a.Sym = p.ctxt.LookupABI(name, abi)
	}
	if p.peek() == scanner.EOF {
		if prefix == 0 && p.isJump {
			// Symbols without prefix or suffix are jump labels.
			return
		}
		p.errorf("illegal or missing addressing mode for symbol %s", name)
		return
	}
	// Expect (SB), (FP), (PC), or (SP)
	p.get('(')
	reg := p.get(scanner.Ident).String()
	p.get(')')
	p.setPseudoRegister(a, reg, isStatic, prefix)
}

// setPseudoRegister sets the NAME field of addr for a pseudo-register reference such as (SB).
func (p *Parser) setPseudoRegister(addr *obj.Addr, reg string, isStatic bool, prefix rune) {
	if addr.Reg != 0 {
		p.errorf("internal error: reg %s already set in pseudo", reg)
	}
	switch reg {
	case "FP":
		addr.Name = obj.NAME_PARAM
	case "PC":
		if prefix != 0 {
			p.errorf("illegal addressing mode for PC")
		}
		addr.Type = obj.TYPE_BRANCH // We set the type and leave NAME untouched. See asmJump.
	case "SB":
		addr.Name = obj.NAME_EXTERN
		if isStatic {
			addr.Name = obj.NAME_STATIC
		}
	case "SP":
		addr.Name = obj.NAME_AUTO // The pseudo-stack.
	default:
		p.errorf("expected pseudo-register; found %s", reg)
	}
	if prefix == '$' {
		addr.Type = obj.TYPE_ADDR
	}
}

// symRefAttrs parses an optional function symbol attribute clause for
// the function symbol 'name', logging an error for a malformed
// attribute clause if 'issueError' is true. The return value is a
// (boolean, ABI) pair indicating that the named symbol is either
// static or a particular ABI specification.
//
// The expected form of the attribute clause is:
//
// empty,           yielding (false, obj.ABI0)
// "<>",            yielding (true,  obj.ABI0)
// "<ABI0>"         yielding (false, obj.ABI0)
// "<ABIInternal>"  yielding (false, obj.ABIInternal)
//
// Anything else beginning with "<" logs an error if issueError is
// true, otherwise returns (false, obj.ABI0).
func (p *Parser) symRefAttrs(name string, issueError bool) (bool, obj.ABI) {
	abi := obj.ABI0
	isStatic := false
	if p.peek() != '<' {
		return isStatic, abi
	}
	p.next()
	tok := p.peek()
	if tok == '>' {
		isStatic = true
	} else if tok == scanner.Ident {
		abistr := p.get(scanner.Ident).String()
		if !p.allowABI {
			if issueError {
				p.errorf("ABI selector only permitted when compiling runtime, reference was to %q", name)
			}
		} else {
			theabi, valid := obj.ParseABI(abistr)
			if !valid {
				if issueError {
					p.errorf("malformed ABI selector %q in reference to %q",
						abistr, name)
				}
			} else {
				abi = theabi
			}
		}
	}
	p.get('>')
	return isStatic, abi
}

// funcAddress parses an external function address. This is a
// constrained form of the operand syntax that's always SB-based,
// non-static, and has at most a simple integer offset:
//
//	[$|*]sym[<abi>][+Int](SB)
func (p *Parser) funcAddress() (string, obj.ABI, bool) {
	switch p.peek() {
	case '$', '*':
		// Skip prefix.
		p.next()
	}

	tok := p.next()
	name := tok.String()
	if tok.ScanToken != scanner.Ident || p.atStartOfRegister(name) {
		return "", obj.ABI0, false
	}
	name = p.qualifySymbol(name)
	// Parse optional <> (indicates a static symbol) or
	// <ABIxxx> (selecting text symbol with specific ABI).
	noErrMsg := false
	isStatic, abi := p.symRefAttrs(name, noErrMsg)
	if isStatic {
		return "", obj.ABI0, false // This function rejects static symbols.
	}
	tok = p.next()
	if tok.ScanToken == '+' {
		if p.next().ScanToken != scanner.Int {
			return "", obj.ABI0, false
		}
		tok = p.next()
	}
	if tok.ScanToken != '(' {
		return "", obj.ABI0, false
	}
	if reg := p.next(); reg.ScanToken != scanner.Ident || reg.String() != "SB" {
		return "", obj.ABI0, false
	}
	if p.next().ScanToken != ')' || p.peek() != scanner.EOF {
		return "", obj.ABI0, false
	}
	return name, abi, true
}

// registerIndirect parses the general form of a register indirection.
// It can be (R1), (R2*scale), (R1)(R2*scale), (R1)(R2.SXTX<<3) or (R1)(R2<<3)
// where R1 may be a simple register or register pair R:R or (R, R) or (R+R).
// Or it might be a pseudo-indirection like (FP).
// We are sitting on the opening parenthesis.
func (p *Parser) registerIndirect(a *obj.Addr, prefix rune) {
	p.get('(')
	tok := p.next()
	name := tok.String()
	r1, r2, scale, ok := p.register(name, 0)
	if !ok {
		p.errorf("indirect through non-register %s", tok)
	}
	p.get(')')
	a.Type = obj.TYPE_MEM
	if r1 < 0 {
		// Pseudo-register reference.
		if r2 != 0 {
			p.errorf("cannot use pseudo-register in pair")
			return
		}
		// For SB, SP, and FP, there must be a name here. 0(FP) is not legal.
		if name != "PC" && a.Name == obj.NAME_NONE {
			p.errorf("cannot reference %s without a symbol", name)
		}
		p.setPseudoRegister(a, name, false, prefix)
		return
	}
	a.Reg = r1
	if r2 != 0 {
		// TODO: Consistency in the encoding would be nice here.
		if p.arch.InFamily(sys.ARM, sys.ARM64) {
			// Special form
			// ARM: destination register pair (R1, R2).
			// ARM64: register pair (R1, R2) for LDP/STP.
			if prefix != 0 || scale != 0 {
				p.errorf("illegal address mode for register pair")
				return
			}
			a.Type = obj.TYPE_REGREG
			a.Offset = int64(r2)
			// Nothing may follow
			return
		}
		if p.arch.Family == sys.PPC64 {
			// Special form for PPC64: (R1+R2); alias for (R1)(R2).
			if prefix != 0 || scale != 0 {
				p.errorf("illegal address mode for register+register")
				return
			}
			a.Type = obj.TYPE_MEM
			a.Scale = 0
			a.Index = r2
			// Nothing may follow.
			return
		}
	}
	if r2 != 0 {
		p.errorf("indirect through register pair")
	}
	if prefix == '$' {
		a.Type = obj.TYPE_ADDR
	}
	if r1 == arch.RPC && prefix != 0 {
		p.errorf("illegal addressing mode for PC")
	}
	if scale == 0 && p.peek() == '(' {
		// General form (R)(R*scale).
		p.next()
		tok := p.next()
		if p.atRegisterExtension() {
			p.registerExtension(a, tok.String(), prefix)
		} else if p.atRegisterShift() {
			// (R1)(R2<<3)
			p.registerExtension(a, tok.String(), prefix)
		} else {
			r1, r2, scale, ok = p.register(tok.String(), 0)
			if !ok {
				p.errorf("indirect through non-register %s", tok)
			}
			if r2 != 0 {
				p.errorf("unimplemented two-register form")
			}
			a.Index = r1
			if scale != 0 && scale != 1 && (p.arch.Family == sys.ARM64 ||
				p.arch.Family == sys.PPC64) {
				// Support (R1)(R2) (no scaling) and (R1)(R2*1).
				p.errorf("%s doesn't support scaled register format", p.arch.Name)
			} else {
				a.Scale = int16(scale)
			}
		}
		p.get(')')
	} else if scale != 0 {
		if p.arch.Family == sys.ARM64 {
			p.errorf("arm64 doesn't support scaled register format")
		}
		// First (R) was missing, all we have is (R*scale).
		a.Reg = 0
		a.Index = r1
		a.Scale = int16(scale)
	}
}

// registerList parses an ARM or ARM64 register list expression, a list of
// registers in []. There may be comma-separated ranges or individual
// registers, as in [R1,R3-R5] or [V1.S4, V2.S4, V3.S4, V4.S4].
// For ARM, only R0 through R15 may appear.
// For ARM64, V0 through V31 with arrangement may appear.
//
// For 386/AMD64 register list specifies 4VNNIW-style multi-source operand.
// For range of 4 elements, Intel manual uses "+3" notation, for example:
//
//	VP4DPWSSDS zmm1{k1}{z}, zmm2+3, m128
//
// Given asm line:
//
//	VP4DPWSSDS Z5, [Z10-Z13], (AX)
//
// zmm2 is Z10, and Z13 is the only valid value for it (Z10+3).
// Only simple ranges are accepted, like [Z0-Z3].
//
// The opening bracket has been consumed.
func (p *Parser) registerList(a *obj.Addr) {
	if p.arch.InFamily(sys.I386, sys.AMD64) {
		p.registerListX86(a)
	} else {
		p.registerListARM(a)
	}
}

func (p *Parser) registerListARM(a *obj.Addr) {
	// One range per loop.
	var maxReg int
	var bits uint16
	var arrangement int64
	switch p.arch.Family {
	case sys.ARM:
		maxReg = 16
	case sys.ARM64:
		maxReg = 32
	default:
		p.errorf("unexpected register list")
	}
	firstReg := -1
	nextReg := -1
	regCnt := 0
ListLoop:
	for {
		tok := p.next()
		switch tok.ScanToken {
		case ']':
			break ListLoop
		case scanner.EOF:
			p.errorf("missing ']' in register list")
			return
		}
		switch p.arch.Family {
		case sys.ARM64:
			// Vn.T
			name := tok.String()
			r, ok := p.registerReference(name)
			if !ok {
				p.errorf("invalid register: %s", name)
			}
			reg := r - p.arch.Register["V0"]
			p.get('.')
			tok := p.next()
			ext := tok.String()
			curArrangement, err := arch.ARM64RegisterArrangement(reg, name, ext)
			if err != nil {
				p.errorf("%v", err)
			}
			if firstReg == -1 {
				// only record the first register and arrangement
				firstReg = int(reg)
				nextReg = firstReg
				arrangement = curArrangement
			} else if curArrangement != arrangement {
				p.errorf("inconsistent arrangement in ARM64 register list")
			} else if nextReg != int(reg) {
				p.errorf("incontiguous register in ARM64 register list: %s", name)
			}
			regCnt++
			nextReg = (nextReg + 1) % 32
		case sys.ARM:
			// Parse the upper and lower bounds.
			lo := p.registerNumber(tok.String())
			hi := lo
			if p.peek() == '-' {
				p.next()
				hi = p.registerNumber(p.next().String())
			}
			if hi < lo {
				lo, hi = hi, lo
			}
			// Check there are no duplicates in the register list.
			for i := 0; lo <= hi && i < maxReg; i++ {
				if bits&(1<<lo) != 0 {
					p.errorf("register R%d already in list", lo)
				}
				bits |= 1 << lo
				lo++
			}
		default:
			p.errorf("unexpected register list")
		}
		if p.peek() != ']' {
			p.get(',')
		}
	}
	a.Type = obj.TYPE_REGLIST
	switch p.arch.Family {
	case sys.ARM:
		a.Offset = int64(bits)
	case sys.ARM64:
		offset, err := arch.ARM64RegisterListOffset(firstReg, regCnt, arrangement)
		if err != nil {
			p.errorf("%v", err)
		}
		a.Offset = offset
	default:
		p.errorf("register list not supported on this architecture")
	}
}

func (p *Parser) registerListX86(a *obj.Addr) {
	// Accept only [RegA-RegB] syntax.
	// Don't use p.get() to provide better error messages.

	loName := p.next().String()
	lo, ok := p.arch.Register[loName]
	if !ok {
		if loName == "EOF" {
			p.errorf("register list: expected ']', found EOF")
		} else {
			p.errorf("register list: bad low register in `[%s`", loName)
		}
		return
	}
	if tok := p.next().ScanToken; tok != '-' {
		p.errorf("register list: expected '-' after `[%s`, found %s", loName, tok)
		return
	}
	hiName := p.next().String()
	hi, ok := p.arch.Register[hiName]
	if !ok {
		p.errorf("register list: bad high register in `[%s-%s`", loName, hiName)
		return
	}
	if tok := p.next().ScanToken; tok != ']' {
		p.errorf("register list: expected ']' after `[%s-%s`, found %s", loName, hiName, tok)
	}

	a.Type = obj.TYPE_REGLIST
	a.Reg = lo
	a.Offset = x86.EncodeRegisterRange(lo, hi)
}

// registerNumber is ARM-specific. It returns the number of the specified register.
func (p *Parser) registerNumber(name string) uint16 {
	if p.arch.Family == sys.ARM && name == "g" {
		return 10
	}
	if name[0] != 'R' {
		p.errorf("expected g or R0 through R15; found %s", name)
		return 0
	}
	r, ok := p.registerReference(name)
	if !ok {
		return 0
	}
	reg := r - p.arch.Register["R0"]
	if reg < 0 {
		// Could happen for an architecture having other registers prefixed by R
		p.errorf("expected g or R0 through R15; found %s", name)
		return 0
	}
	return uint16(reg)
}

// Note: There are two changes in the expression handling here
// compared to the old yacc/C implementations. Neither has
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
			if int64(value) < 0 {
				p.errorf("divide of value with high bit set")
			}
			divisor := p.factor()
			if divisor == 0 {
				p.errorf("division by zero")
			} else {
				value /= divisor
			}
		case '%':
			p.next()
			divisor := p.factor()
			if int64(value) < 0 {
				p.errorf("modulo of value with high bit set")
			}
			if divisor == 0 {
				p.errorf("modulo by zero")
			} else {
				value %= divisor
			}
		case lex.LSH:
			p.next()
			shift := p.factor()
			if int64(shift) < 0 {
				p.errorf("negative left shift count")
			}
			return value << shift
		case lex.RSH:
			p.next()
			shift := p.term()
			if int64(shift) < 0 {
				p.errorf("negative right shift count")
			}
			if int64(value) < 0 {
				p.errorf("right shift of value with high bit set")
			}
			value >>= shift
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
	case scanner.Char:
		str, err := strconv.Unquote(tok.String())
		if err != nil {
			p.errorf("%s", err)
		}
		r, w := utf8.DecodeRuneInString(str)
		if w == 1 && r == utf8.RuneError {
			p.errorf("illegal UTF-8 encoding for character constant")
		}
		return uint64(r)
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
	if p.inputPos == 0 {
		p.errorf("internal error: backing up before BOL")
	} else {
		p.inputPos--
	}
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
	p.expect(expected, expected.String())
	return p.next()
}

// expectOperandEnd verifies that the parsing state is properly at the end of an operand.
func (p *Parser) expectOperandEnd() {
	p.expect(scanner.EOF, "end of operand")
}

// expect verifies that the next item has the expected type. It does not consume it.
func (p *Parser) expect(expectedToken lex.ScanToken, expectedMessage string) {
	if p.peek() != expectedToken {
		p.errorf("expected %s, found %s", expectedMessage, p.next())
	}
}

// have reports whether the remaining tokens (including the current one) contain the specified token.
func (p *Parser) have(token lex.ScanToken) bool {
	for i := p.inputPos; i < len(p.input); i++ {
		if p.input[i].ScanToken == token {
			return true
		}
	}
	return false
}

// at reports whether the next tokens are as requested.
func (p *Parser) at(next ...lex.ScanToken) bool {
	if len(p.input)-p.inputPos < len(next) {
		return false
	}
	for i, r := range next {
		if p.input[p.inputPos+i].ScanToken != r {
			return false
		}
	}
	return true
}
