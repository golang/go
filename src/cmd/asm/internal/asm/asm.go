// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asm

import (
	"fmt"
	"strings"
	"text/scanner"

	"cmd/asm/internal/addr"
	"cmd/asm/internal/arch"
	"cmd/asm/internal/flags"
	"cmd/asm/internal/lex"
	"cmd/internal/obj"
)

// TODO: This package has many numeric conversions that should be unnecessary.

// symbolType returns the extern/static etc. type appropriate for the symbol.
func (p *Parser) symbolType(a *addr.Addr) int {
	switch a.Register {
	case arch.RFP:
		return p.arch.D_PARAM
	case arch.RSP:
		return p.arch.D_AUTO
	case arch.RSB:
		// See comment in addrToAddr.
		if a.IsImmediateAddress {
			return p.arch.D_ADDR
		}
		if a.IsStatic {
			return p.arch.D_STATIC
		}
		return p.arch.D_EXTERN
	}
	p.errorf("invalid register for symbol %s", a.Symbol)
	return 0
}

// staticVersion reports whether the data's Symbol has <>, as in data<>.
// It returns 1 for static, 0 for non-static, because that's what obj wants.
func staticVersion(a *addr.Addr) int {
	if a.Symbol != "" && a.IsStatic {
		return 1
	}
	return 0
}

// TODO: configure the architecture

// TODO: This is hacky and irregular. When obj settles down, rewrite for simplicity.
func (p *Parser) addrToAddr(a *addr.Addr) obj.Addr {
	out := p.arch.NoAddr
	if a.Has(addr.Symbol) {
		// How to encode the symbols:
		// syntax = Typ,Index
		// $a(SB) = ADDR,EXTERN
		// $a<>(SB) = ADDR,STATIC
		// a(SB) = EXTERN,NONE
		// a<>(SB) = STATIC,NONE
		// The call to symbolType does the first column; we need to fix up Index here.
		out.Type = int16(p.symbolType(a))
		if a.IsImmediateAddress {
			// Index field says whether it's a static.
			switch a.Register {
			case arch.RSB:
				if a.IsStatic {
					out.Index = uint8(p.arch.D_STATIC)
				} else {
					out.Index = uint8(p.arch.D_EXTERN)
				}
			default:
				p.errorf("can't handle immediate address of %s not (SB)\n", a.Symbol)
			}
		}
		out.Sym = obj.Linklookup(p.linkCtxt, a.Symbol, staticVersion(a))
	} else if a.Has(addr.Register) {
		// TODO: SP is tricky, and this isn't good enough.
		// SP = D_SP
		// 4(SP) = 4(D_SP)
		// x+4(SP) = D_AUTO with sym=x TODO
		out.Type = a.Register
		if a.Register == arch.RSP {
			out.Type = int16(p.arch.SP)
		}
		if a.IsIndirect {
			out.Type += int16(p.arch.D_INDIR)
		}
		// a.Register2 handled in the instruction method; it's bizarre.
	}
	if a.Has(addr.Index) {
		out.Index = uint8(a.Index) // TODO: out.Index == p.NoArch.Index should be same type as Register.
	}
	if a.Has(addr.Scale) {
		out.Scale = a.Scale
	}
	if a.Has(addr.Offset) {
		out.Offset = a.Offset
		if a.Is(addr.Offset) {
			// RHS of MOVL $0xf1, 0xf1  // crash
			out.Type = int16(p.arch.D_INDIR + p.arch.D_NONE)
		} else if a.IsImmediateConstant && out.Type == int16(p.arch.D_NONE) {
			out.Type = int16(p.arch.D_CONST)
		}
	}
	if a.Has(addr.Float) {
		out.U.Dval = a.Float
		out.Type = int16(p.arch.D_FCONST)
	}
	if a.Has(addr.String) {
		out.U.Sval = a.String
		out.Type = int16(p.arch.D_SCONST)
	}
	// TODO from https://go-review.googlesource.com/#/c/3196/ {
	// There's a general rule underlying this special case and the one at line 91 (RHS OF MOVL $0xf1).
	//	Unless there's a $, it's an indirect.
	// 4(R1)(R2*8)
	// 4(R1)
	// 4(R2*8)
	// 4
	// (R1)(R2*8)
	// (R1)
	// (R2*8)
	// There should be a more general approach that doesn't just pick off cases.
	// }
	if a.IsIndirect && !a.Has(addr.Register) && a.Has(addr.Index) {
		// LHS of LEAQ	0(BX*8), CX
		out.Type = int16(p.arch.D_INDIR + p.arch.D_NONE)
	}
	return out
}

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
	if !nameAddr.Is(addr.Symbol|addr.Register|addr.Indirect) || nameAddr.Register != arch.RSB {
		p.errorf("TEXT symbol %q must be an offset from SB", nameAddr.Symbol)
	}
	name := nameAddr.Symbol
	next := 1

	// Next operand is the optional text flag, a literal integer.
	flag := int8(0)
	if len(operands) == 3 {
		flagAddr := p.address(operands[next])
		if !flagAddr.Is(addr.Offset) {
			p.errorf("TEXT flag for %s must be an integer", name)
		}
		flag = int8(flagAddr.Offset)
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
		As:     int16(p.arch.ATEXT),
		Lineno: int32(p.histLineNum),
		From: obj.Addr{
			Type:  int16(p.symbolType(&nameAddr)),
			Index: uint8(p.arch.D_NONE),
			Sym:   obj.Linklookup(p.linkCtxt, name, staticVersion(&nameAddr)),
			Scale: flag,
		},
		To: obj.Addr{
			Index: uint8(p.arch.D_NONE),
		},
	}

	// Encoding of frameSize and argSize depends on architecture.
	switch p.arch.Thechar {
	case '6':
		prog.To.Type = int16(p.arch.D_CONST)
		prog.To.Offset = (argSize << 32) | frameSize
	case '8':
		prog.To.Type = int16(p.arch.D_CONST2)
		prog.To.Offset = frameSize
		prog.To.Offset2 = int32(argSize)
	default:
		p.errorf("internal error: can't encode TEXT $arg-frame")
	}

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
	scale := p.scale(op[n-1].String())
	op = op[:n-2]
	nameAddr := p.address(op)
	ok := nameAddr.Is(addr.Symbol|addr.Register|addr.Indirect) || nameAddr.Is(addr.Symbol|addr.Register|addr.Indirect|addr.Offset)
	if !ok || nameAddr.Register != arch.RSB {
		p.errorf("DATA symbol %q must be an offset from SB", nameAddr.Symbol)
	}
	name := strings.Replace(nameAddr.Symbol, "·", ".", 1)

	// Operand 1 is an immediate constant or address.
	valueAddr := p.address(operands[1])
	if !valueAddr.IsImmediateConstant && !valueAddr.IsImmediateAddress {
		p.errorf("DATA value must be an immediate constant or address")
	}

	// The addresses must not overlap. Easiest test: require monotonicity.
	if lastAddr, ok := p.dataAddr[name]; ok && nameAddr.Offset < lastAddr {
		p.errorf("overlapping DATA entry for %s", nameAddr.Symbol)
	}
	p.dataAddr[name] = nameAddr.Offset + int64(scale)

	prog := &obj.Prog{
		Ctxt:   p.linkCtxt,
		As:     int16(p.arch.ADATA),
		Lineno: int32(p.histLineNum),
		From: obj.Addr{
			Type:   int16(p.symbolType(&nameAddr)),
			Index:  uint8(p.arch.D_NONE),
			Sym:    obj.Linklookup(p.linkCtxt, name, staticVersion(&nameAddr)),
			Offset: nameAddr.Offset,
			Scale:  scale,
		},
		To: p.addrToAddr(&valueAddr),
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
	ok := nameAddr.Is(addr.Symbol|addr.Register|addr.Indirect) || nameAddr.Is(addr.Symbol|addr.Register|addr.Indirect|addr.Offset)
	if !ok || nameAddr.Register != arch.RSB {
		p.errorf("GLOBL symbol %q must be an offset from SB", nameAddr.Symbol)
	}
	name := strings.Replace(nameAddr.Symbol, "·", ".", 1)

	// If three operands, middle operand is a scale.
	scale := int8(0)
	op := operands[1]
	if len(operands) == 3 {
		scaleAddr := p.address(op)
		if !scaleAddr.Is(addr.Offset) {
			p.errorf("GLOBL scale must be a constant")
		}
		scale = int8(scaleAddr.Offset)
		op = operands[2]
	}

	// Final operand is an immediate constant.
	sizeAddr := p.address(op)
	if !sizeAddr.Is(addr.ImmediateConstant | addr.Offset) {
		p.errorf("GLOBL size must be an immediate constant")
	}
	size := sizeAddr.Offset

	// log.Printf("GLOBL %s %d, $%d", name, scale, size)
	prog := &obj.Prog{
		Ctxt:   p.linkCtxt,
		As:     int16(p.arch.AGLOBL),
		Lineno: int32(p.histLineNum),
		From: obj.Addr{
			Type:   int16(p.symbolType(&nameAddr)),
			Index:  uint8(p.arch.D_NONE),
			Sym:    obj.Linklookup(p.linkCtxt, name, staticVersion(&nameAddr)),
			Offset: nameAddr.Offset,
			Scale:  scale,
		},
		To: obj.Addr{
			Type:   int16(p.arch.D_CONST),
			Index:  uint8(p.arch.D_NONE),
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
	addr0 := p.address(operands[0])
	if !addr0.Is(addr.ImmediateConstant | addr.Offset) {
		p.errorf("PCDATA value must be an immediate constant")
	}
	value0 := addr0.Offset

	// Operand 1 must be an immediate constant.
	addr1 := p.address(operands[1])
	if !addr1.Is(addr.ImmediateConstant | addr.Offset) {
		p.errorf("PCDATA value must be an immediate constant")
	}
	value1 := addr1.Offset

	// log.Printf("PCDATA $%d, $%d", value0, value1)
	prog := &obj.Prog{
		Ctxt:   p.linkCtxt,
		As:     int16(p.arch.APCDATA),
		Lineno: int32(p.histLineNum),
		From: obj.Addr{
			Type:   int16(p.arch.D_CONST),
			Index:  uint8(p.arch.D_NONE),
			Offset: value0,
		},
		To: obj.Addr{
			Type:   int16(p.arch.D_CONST),
			Index:  uint8(p.arch.D_NONE),
			Offset: value1,
		},
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
	if !valueAddr.Is(addr.ImmediateConstant | addr.Offset) {
		p.errorf("FUNCDATA value must be an immediate constant")
	}
	value0 := valueAddr.Offset

	// Operand 1 is a symbol name in the form foo(SB).
	// That means symbol plus indirect on SB and no offset.
	nameAddr := p.address(operands[1])
	ok := nameAddr.Is(addr.Symbol|addr.Register|addr.Indirect) || nameAddr.Is(addr.Symbol|addr.Register|addr.Indirect|addr.Offset)
	if !ok || nameAddr.Register != arch.RSB {
		p.errorf("FUNCDATA symbol %q must be an offset from SB", nameAddr.Symbol)
	}
	name := strings.Replace(nameAddr.Symbol, "·", ".", 1)
	value1 := nameAddr.Offset

	// log.Printf("FUNCDATA $%d, %d", value0, value1)
	prog := &obj.Prog{
		Ctxt:   p.linkCtxt,
		As:     int16(p.arch.AFUNCDATA),
		Lineno: int32(p.histLineNum),
		From: obj.Addr{
			Type:   int16(p.arch.D_CONST),
			Index:  uint8(p.arch.D_NONE),
			Offset: value0,
		},
		To: obj.Addr{
			Type:   int16(p.symbolType(&nameAddr)),
			Index:  uint8(p.arch.D_NONE),
			Sym:    obj.Linklookup(p.linkCtxt, name, staticVersion(&nameAddr)),
			Offset: value1,
		},
	}
	p.append(prog, true)
}

// asmJump assembles a jump instruction.
// JMP	R1
// JMP	exit
// JMP	3(PC)
func (p *Parser) asmJump(op int, a []addr.Addr) {
	var target *addr.Addr
	switch len(a) {
	default:
		p.errorf("jump must have one or two addresses")
	case 1:
		target = &a[0]
	case 2:
		if !a[0].Is(0) {
			p.errorf("two-address jump must have empty first address")
		}
		target = &a[1]
	}
	prog := &obj.Prog{
		Ctxt:   p.linkCtxt,
		Lineno: int32(p.histLineNum),
		As:     int16(op),
		From:   p.arch.NoAddr,
	}
	switch {
	case target.Is(addr.Register):
		// JMP R1
		prog.To = p.addrToAddr(target)
	case target.Is(addr.Symbol):
		// JMP exit
		targetProg := p.labels[target.Symbol]
		if targetProg == nil {
			p.toPatch = append(p.toPatch, Patch{prog, target.Symbol})
		} else {
			p.branch(prog, targetProg)
		}
	case target.Is(addr.Register | addr.Indirect), target.Is(addr.Register | addr.Indirect | addr.Offset):
		// JMP 4(AX)
		if target.Register == arch.RPC {
			prog.To = obj.Addr{
				Type:   int16(p.arch.D_BRANCH),
				Index:  uint8(p.arch.D_NONE),
				Offset: p.pc + 1 + target.Offset, // +1 because p.pc is incremented in link, below.
			}
		} else {
			prog.To = p.addrToAddr(target)
		}
	case target.Is(addr.Symbol | addr.Indirect | addr.Register):
		// JMP main·morestack(SB)
		if target.Register != arch.RSB {
			p.errorf("jmp to symbol must be SB-relative")
		}
		prog.To = obj.Addr{
			Type:   int16(p.arch.D_BRANCH),
			Sym:    obj.Linklookup(p.linkCtxt, target.Symbol, staticVersion(target)),
			Index:  uint8(p.arch.D_NONE),
			Offset: target.Offset,
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
		Type:  int16(p.arch.D_BRANCH),
		Index: uint8(p.arch.D_NONE),
	}
	jmp.To.U.Branch = target
}

// asmInstruction assembles an instruction.
// MOVW R9, (R10)
func (p *Parser) asmInstruction(op int, a []addr.Addr) {
	prog := &obj.Prog{
		Ctxt:   p.linkCtxt,
		Lineno: int32(p.histLineNum),
		As:     int16(op),
	}
	switch len(a) {
	case 0:
		prog.From = p.arch.NoAddr
		prog.To = p.arch.NoAddr
	case 1:
		if p.arch.UnaryDestination[op] {
			prog.From = p.arch.NoAddr
			prog.To = p.addrToAddr(&a[0])
		} else {
			prog.From = p.addrToAddr(&a[0])
			prog.To = p.arch.NoAddr
		}
	case 2:
		prog.From = p.addrToAddr(&a[0])
		prog.To = p.addrToAddr(&a[1])
		// DX:AX as a register pair can only appear on the RHS.
		// Bizarrely, to obj it's specified by setting index on the LHS.
		// TODO: can we fix this?
		if a[1].Has(addr.Register2) {
			if int(prog.From.Index) != p.arch.D_NONE {
				p.errorf("register pair operand on RHS must have register on LHS")
			}
			prog.From.Index = uint8(a[1].Register2)
		}
	case 3:
		// CMPSD etc.; third operand is imm8, stored in offset, or a register.
		prog.From = p.addrToAddr(&a[0])
		prog.To = p.addrToAddr(&a[1])
		switch {
		case a[2].Is(addr.Offset):
			prog.To.Offset = a[2].Offset
		case a[2].Is(addr.Register):
			// Strange reodering.
			prog.To = p.addrToAddr(&a[2])
			prog.From = p.addrToAddr(&a[1])
			if !a[0].IsImmediateConstant {
				p.errorf("expected $value for 1st operand")
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
