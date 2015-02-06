// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asm

import (
	"fmt"
	"text/scanner"

	"cmd/asm/internal/arch"
	"cmd/asm/internal/flags"
	"cmd/asm/internal/lex"
	"cmd/internal/obj"
)

// TODO: configure the architecture

// append adds the Prog to the end of the program-thus-far.
// If doLabel is set, it also defines the labels collect for this Prog.
func (p *Parser) append(prog *obj.Prog, doLabel bool) {
	if p.firstProg == nil {
		p.firstProg = prog
	} else {
		p.lastProg.Link = prog
	}
	p.lastProg = prog
	if doLabel {
		p.pc++
		for _, label := range p.pendingLabels {
			if p.labels[label] != nil {
				p.errorf("label %q multiply defined", label)
			}
			p.labels[label] = prog
		}
		p.pendingLabels = p.pendingLabels[0:0]
	}
	prog.Pc = int64(p.pc)
	if *flags.Debug {
		fmt.Println(p.histLineNum, prog)
	}
}

func (p *Parser) validatePseudoSymbol(pseudo string, addr *obj.Addr, offsetOk bool) {
	if addr.Name != obj.NAME_EXTERN && addr.Name != obj.NAME_STATIC || addr.Scale != 0 || addr.Reg != 0 {
		p.errorf("%s symbol %q must be a symbol(SB)", pseudo, addr.Sym.Name)
	}
	if !offsetOk && addr.Offset != 0 {
		p.errorf("%s symbol %q must not be offset from SB", pseudo, addr.Sym.Name)
	}
}

func (p *Parser) evalInteger(pseudo string, operands []lex.Token) int64 {
	addr := p.address(operands)
	if addr.Type != obj.TYPE_MEM || addr.Name != 0 || addr.Reg != 0 || addr.Index != 0 {
		p.errorf("%s: text flag must be an integer constant")
	}
	return addr.Offset
}

// asmText assembles a TEXT pseudo-op.
// TEXT runtime·sigtramp(SB),4,$0-0
func (p *Parser) asmText(word string, operands [][]lex.Token) {
	if len(operands) != 2 && len(operands) != 3 {
		p.errorf("expect two or three operands for TEXT")
	}

	// Labels are function scoped. Patch existing labels and
	// create a new label space for this TEXT.
	p.patch()
	p.labels = make(map[string]*obj.Prog)

	// Operand 0 is the symbol name in the form foo(SB).
	// That means symbol plus indirect on SB and no offset.
	nameAddr := p.address(operands[0])
	p.validatePseudoSymbol("TEXT", &nameAddr, false)
	name := nameAddr.Sym.Name
	next := 1

	// Next operand is the optional text flag, a literal integer.
	var flag = int64(0)
	if len(operands) == 3 {
		flag = p.evalInteger("TEXT", operands[1])
		next++
	}

	// Next operand is the frame and arg size.
	// Bizarre syntax: $frameSize-argSize is two words, not subtraction.
	// Both frameSize and argSize must be simple integers; only frameSize
	// can be negative.
	// The "-argSize" may be missing; if so, set it to obj.ArgsSizeUnknown.
	// Parse left to right.
	op := operands[next]
	if len(op) < 2 || op[0].ScanToken != '$' {
		p.errorf("TEXT %s: frame size must be an immediate constant", name)
	}
	op = op[1:]
	negative := false
	if op[0].ScanToken == '-' {
		negative = true
		op = op[1:]
	}
	if len(op) == 0 || op[0].ScanToken != scanner.Int {
		p.errorf("TEXT %s: frame size must be an immediate constant", name)
	}
	frameSize := p.positiveAtoi(op[0].String())
	if negative {
		frameSize = -frameSize
	}
	op = op[1:]
	argSize := int64(obj.ArgsSizeUnknown)
	if len(op) > 0 {
		// There is an argument size. It must be a minus sign followed by a non-negative integer literal.
		if len(op) != 2 || op[0].ScanToken != '-' || op[1].ScanToken != scanner.Int {
			p.errorf("TEXT %s: argument size must be of form -integer", name)
		}
		argSize = p.positiveAtoi(op[1].String())
	}
	prog := &obj.Prog{
		Ctxt:   p.linkCtxt,
		As:     obj.ATEXT,
		Lineno: p.histLineNum,
		From:   nameAddr,
		From3: obj.Addr{
			Offset: flag,
		},
		To: obj.Addr{
			Type:   obj.TYPE_TEXTSIZE,
			Offset: frameSize,
			// Argsize set below.
		},
	}
	prog.To.U.Argsize = int32(argSize)

	p.append(prog, true)
}

// asmData assembles a DATA pseudo-op.
// DATA masks<>+0x00(SB)/4, $0x00000000
func (p *Parser) asmData(word string, operands [][]lex.Token) {
	if len(operands) != 2 {
		p.errorf("expect two operands for DATA")
	}

	// Operand 0 has the general form foo<>+0x04(SB)/4.
	op := operands[0]
	n := len(op)
	if n < 3 || op[n-2].ScanToken != '/' || op[n-1].ScanToken != scanner.Int {
		p.errorf("expect /size for DATA argument")
	}
	scale := p.parseScale(op[n-1].String())
	op = op[:n-2]
	nameAddr := p.address(op)
	p.validatePseudoSymbol("DATA", &nameAddr, true)
	name := nameAddr.Sym.Name

	// Operand 1 is an immediate constant or address.
	valueAddr := p.address(operands[1])
	switch valueAddr.Type {
	case obj.TYPE_CONST, obj.TYPE_FCONST, obj.TYPE_SCONST, obj.TYPE_ADDR:
		// OK
	default:
		p.errorf("DATA value must be an immediate constant or address")
	}

	// The addresses must not overlap. Easiest test: require monotonicity.
	if lastAddr, ok := p.dataAddr[name]; ok && nameAddr.Offset < lastAddr {
		p.errorf("overlapping DATA entry for %s", name)
	}
	p.dataAddr[name] = nameAddr.Offset + int64(scale)

	prog := &obj.Prog{
		Ctxt:   p.linkCtxt,
		As:     obj.ADATA,
		Lineno: p.histLineNum,
		From:   nameAddr,
		From3: obj.Addr{
			Offset: int64(scale),
		},
		To: valueAddr,
	}

	p.append(prog, false)
}

// asmGlobl assembles a GLOBL pseudo-op.
// GLOBL shifts<>(SB),8,$256
// GLOBL shifts<>(SB),$256
func (p *Parser) asmGlobl(word string, operands [][]lex.Token) {
	if len(operands) != 2 && len(operands) != 3 {
		p.errorf("expect two or three operands for GLOBL")
	}

	// Operand 0 has the general form foo<>+0x04(SB).
	nameAddr := p.address(operands[0])
	p.validatePseudoSymbol("GLOBL", &nameAddr, false)
	name := nameAddr.Sym.Name
	next := 1

	// Next operand is the optional flag, a literal integer.
	var flag = int64(0)
	if len(operands) == 3 {
		flag = p.evalInteger("GLOBL", operands[1])
		next++
	}

	// Final operand is an immediate constant.
	op := operands[next]
	if len(op) < 2 || op[0].ScanToken != '$' || op[1].ScanToken != scanner.Int {
		p.errorf("GLOBL %s: size must be an immediate constant", name)
	}
	size := p.positiveAtoi(op[1].String())

	// log.Printf("GLOBL %s %d, $%d", name, flag, size)
	prog := &obj.Prog{
		Ctxt:   p.linkCtxt,
		As:     obj.AGLOBL,
		Lineno: p.histLineNum,
		From:   nameAddr,
		From3: obj.Addr{
			Offset: flag,
		},
		To: obj.Addr{
			Type:   obj.TYPE_CONST,
			Index:  0,
			Offset: size,
		},
	}
	p.append(prog, false)
}

// asmPCData assembles a PCDATA pseudo-op.
// PCDATA $2, $705
func (p *Parser) asmPCData(word string, operands [][]lex.Token) {
	if len(operands) != 2 {
		p.errorf("expect two operands for PCDATA")
	}

	// Operand 0 must be an immediate constant.
	key := p.address(operands[0])
	if key.Type != obj.TYPE_CONST {
		p.errorf("PCDATA key must be an immediate constant")
	}

	// Operand 1 must be an immediate constant.
	value := p.address(operands[1])
	if value.Type != obj.TYPE_CONST {
		p.errorf("PCDATA value must be an immediate constant")
	}

	// log.Printf("PCDATA $%d, $%d", key.Offset, value.Offset)
	prog := &obj.Prog{
		Ctxt:   p.linkCtxt,
		As:     obj.APCDATA,
		Lineno: p.histLineNum,
		From:   key,
		To:     value,
	}
	p.append(prog, true)
}

// asmFuncData assembles a FUNCDATA pseudo-op.
// FUNCDATA $1, funcdata<>+4(SB)
func (p *Parser) asmFuncData(word string, operands [][]lex.Token) {
	if len(operands) != 2 {
		p.errorf("expect two operands for FUNCDATA")
	}

	// Operand 0 must be an immediate constant.
	valueAddr := p.address(operands[0])
	if valueAddr.Type != obj.TYPE_CONST {
		p.errorf("FUNCDATA value0 must be an immediate constant")
	}

	// Operand 1 is a symbol name in the form foo(SB).
	nameAddr := p.address(operands[1])
	p.validatePseudoSymbol("FUNCDATA", &nameAddr, true)

	prog := &obj.Prog{
		Ctxt:   p.linkCtxt,
		As:     obj.AFUNCDATA,
		Lineno: p.histLineNum,
		From:   valueAddr,
		To:     nameAddr,
	}
	p.append(prog, true)
}

// asmJump assembles a jump instruction.
// JMP	R1
// JMP	exit
// JMP	3(PC)
func (p *Parser) asmJump(op int, a []obj.Addr) {
	var target *obj.Addr
	switch len(a) {
	case 1:
		target = &a[0]
	default:
		p.errorf("wrong number of arguments to jump instruction")
	}
	prog := &obj.Prog{
		Ctxt:   p.linkCtxt,
		Lineno: p.histLineNum,
		As:     int16(op),
	}
	switch {
	case target.Type == obj.TYPE_REG:
		// JMP R1
		prog.To = *target
	case target.Type == obj.TYPE_MEM && (target.Name == obj.NAME_EXTERN || target.Name == obj.NAME_STATIC):
		// JMP main·morestack(SB)
		isStatic := 0
		if target.Name == obj.NAME_STATIC {
			isStatic = 1
		}
		prog.To = obj.Addr{
			Type:   obj.TYPE_BRANCH,
			Sym:    obj.Linklookup(p.linkCtxt, target.Sym.Name, isStatic),
			Index:  0,
			Offset: target.Offset,
		}
	case target.Type == obj.TYPE_MEM && target.Reg == 0 && target.Offset == 0:
		// JMP exit
		targetProg := p.labels[target.Sym.Name]
		if targetProg == nil {
			p.toPatch = append(p.toPatch, Patch{prog, target.Sym.Name})
		} else {
			p.branch(prog, targetProg)
		}
	case target.Type == obj.TYPE_MEM && target.Name == obj.NAME_NONE:
		// JMP 4(PC)
		if target.Reg == arch.RPC {
			prog.To = obj.Addr{
				Type:   obj.TYPE_BRANCH,
				Offset: p.pc + 1 + target.Offset, // +1 because p.pc is incremented in link, below.
			}
		} else {
			prog.To = *target
		}
	default:
		p.errorf("cannot assemble jump %+v", target)
	}
	p.append(prog, true)
}

func (p *Parser) patch() {
	for _, patch := range p.toPatch {
		targetProg := p.labels[patch.label]
		if targetProg == nil {
			p.errorf("undefined label %s", patch.label)
		} else {
			p.branch(patch.prog, targetProg)
		}
	}
	p.toPatch = p.toPatch[:0]
}

func (p *Parser) branch(jmp, target *obj.Prog) {
	jmp.To = obj.Addr{
		Type:  obj.TYPE_BRANCH,
		Index: 0,
	}
	jmp.To.U.Branch = target
}

// asmInstruction assembles an instruction.
// MOVW R9, (R10)
func (p *Parser) asmInstruction(op int, a []obj.Addr) {
	prog := &obj.Prog{
		Ctxt:   p.linkCtxt,
		Lineno: p.histLineNum,
		As:     int16(op),
	}
	switch len(a) {
	case 0:
		// Nothing to do.
	case 1:
		if p.arch.UnaryDestination[op] {
			// prog.From is no address.
			prog.To = a[0]
		} else {
			prog.From = a[0]
			// prog.To is no address.
		}
	case 2:
		prog.From = a[0]
		prog.To = a[1]
		// DX:AX as a register pair can only appear on the RHS.
		// Bizarrely, to obj it's specified by setting index on the LHS.
		// TODO: can we fix this?
		if a[1].Class != 0 {
			if a[0].Class != 0 {
				p.errorf("register pair must be on LHS")
			}
			prog.From.Index = int16(a[1].Class)
			prog.To.Class = 0
		}
	case 3:
		// CMPSD etc.; third operand is imm8, stored in offset, or a register.
		prog.From = a[0]
		prog.To = a[1]
		switch a[2].Type {
		case obj.TYPE_MEM:
			prog.To.Offset = a[2].Offset
		case obj.TYPE_REG:
			// Strange reodering.
			prog.To = a[2]
			prog.From = a[1]
			if a[0].Type != obj.TYPE_CONST {
				p.errorf("expected immediate constant for 1st operand")
			}
			prog.To.Offset = a[0].Offset
		default:
			p.errorf("expected offset or register for 3rd operand")
		}

	default:
		p.errorf("can't handle instruction with %d operands", len(a))
	}
	p.append(prog, true)
}
