// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asm

import (
	"bytes"
	"fmt"
	"strconv"
	"text/scanner"

	"cmd/asm/internal/arch"
	"cmd/asm/internal/flags"
	"cmd/asm/internal/lex"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
	"cmd/internal/objabi"
	"cmd/internal/sys"
)

// TODO: configure the architecture

var testOut *bytes.Buffer // Gathers output when testing.

// append adds the Prog to the end of the program-thus-far.
// If doLabel is set, it also defines the labels collect for this Prog.
func (p *Parser) append(prog *obj.Prog, cond string, doLabel bool) {
	if cond != "" {
		switch p.arch.Family {
		case sys.ARM:
			if !arch.ARMConditionCodes(prog, cond) {
				p.errorf("unrecognized condition code .%q", cond)
				return
			}

		case sys.ARM64:
			if !arch.ARM64Suffix(prog, cond) {
				p.errorf("unrecognized suffix .%q", cond)
				return
			}

		case sys.AMD64, sys.I386:
			if err := x86.ParseSuffix(prog, cond); err != nil {
				p.errorf("%v", err)
				return
			}

		default:
			p.errorf("unrecognized suffix .%q", cond)
			return
		}
	}
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
				return
			}
			p.labels[label] = prog
		}
		p.pendingLabels = p.pendingLabels[0:0]
	}
	prog.Pc = p.pc
	if *flags.Debug {
		fmt.Println(p.lineNum, prog)
	}
	if testOut != nil {
		fmt.Fprintln(testOut, prog)
	}
}

// validSymbol checks that addr represents a valid name for a pseudo-op.
func (p *Parser) validSymbol(pseudo string, addr *obj.Addr, offsetOk bool) bool {
	if addr.Sym == nil || addr.Name != obj.NAME_EXTERN && addr.Name != obj.NAME_STATIC || addr.Scale != 0 || addr.Reg != 0 {
		p.errorf("%s symbol %q must be a symbol(SB)", pseudo, symbolName(addr))
		return false
	}
	if !offsetOk && addr.Offset != 0 {
		p.errorf("%s symbol %q must not be offset from SB", pseudo, symbolName(addr))
		return false
	}
	return true
}

// evalInteger evaluates an integer constant for a pseudo-op.
func (p *Parser) evalInteger(pseudo string, operands []lex.Token) int64 {
	addr := p.address(operands)
	return p.getConstantPseudo(pseudo, &addr)
}

// validImmediate checks that addr represents an immediate constant.
func (p *Parser) validImmediate(pseudo string, addr *obj.Addr) bool {
	if addr.Type != obj.TYPE_CONST || addr.Name != 0 || addr.Reg != 0 || addr.Index != 0 {
		p.errorf("%s: expected immediate constant; found %s", pseudo, obj.Dconv(&emptyProg, addr))
		return false
	}
	return true
}

// asmText assembles a TEXT pseudo-op.
// TEXT runtime·sigtramp(SB),4,$0-0
func (p *Parser) asmText(operands [][]lex.Token) {
	if len(operands) != 2 && len(operands) != 3 {
		p.errorf("expect two or three operands for TEXT")
		return
	}

	// Labels are function scoped. Patch existing labels and
	// create a new label space for this TEXT.
	p.patch()
	p.labels = make(map[string]*obj.Prog)

	// Operand 0 is the symbol name in the form foo(SB).
	// That means symbol plus indirect on SB and no offset.
	nameAddr := p.address(operands[0])
	if !p.validSymbol("TEXT", &nameAddr, false) {
		return
	}
	name := symbolName(&nameAddr)
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
	// The "-argSize" may be missing; if so, set it to objabi.ArgsSizeUnknown.
	// Parse left to right.
	op := operands[next]
	if len(op) < 2 || op[0].ScanToken != '$' {
		p.errorf("TEXT %s: frame size must be an immediate constant", name)
		return
	}
	op = op[1:]
	negative := false
	if op[0].ScanToken == '-' {
		negative = true
		op = op[1:]
	}
	if len(op) == 0 || op[0].ScanToken != scanner.Int {
		p.errorf("TEXT %s: frame size must be an immediate constant", name)
		return
	}
	frameSize := p.positiveAtoi(op[0].String())
	if negative {
		frameSize = -frameSize
	}
	op = op[1:]
	argSize := int64(objabi.ArgsSizeUnknown)
	if len(op) > 0 {
		// There is an argument size. It must be a minus sign followed by a non-negative integer literal.
		if len(op) != 2 || op[0].ScanToken != '-' || op[1].ScanToken != scanner.Int {
			p.errorf("TEXT %s: argument size must be of form -integer", name)
			return
		}
		argSize = p.positiveAtoi(op[1].String())
	}
	p.ctxt.InitTextSym(nameAddr.Sym, int(flag))
	prog := &obj.Prog{
		Ctxt: p.ctxt,
		As:   obj.ATEXT,
		Pos:  p.pos(),
		From: nameAddr,
		To: obj.Addr{
			Type:   obj.TYPE_TEXTSIZE,
			Offset: frameSize,
			// Argsize set below.
		},
	}
	nameAddr.Sym.Func.Text = prog
	prog.To.Val = int32(argSize)
	p.append(prog, "", true)
}

// asmData assembles a DATA pseudo-op.
// DATA masks<>+0x00(SB)/4, $0x00000000
func (p *Parser) asmData(operands [][]lex.Token) {
	if len(operands) != 2 {
		p.errorf("expect two operands for DATA")
		return
	}

	// Operand 0 has the general form foo<>+0x04(SB)/4.
	op := operands[0]
	n := len(op)
	if n < 3 || op[n-2].ScanToken != '/' || op[n-1].ScanToken != scanner.Int {
		p.errorf("expect /size for DATA argument")
		return
	}
	szop := op[n-1].String()
	sz, err := strconv.Atoi(szop)
	if err != nil {
		p.errorf("bad size for DATA argument: %q", szop)
	}
	op = op[:n-2]
	nameAddr := p.address(op)
	if !p.validSymbol("DATA", &nameAddr, true) {
		return
	}
	name := symbolName(&nameAddr)

	// Operand 1 is an immediate constant or address.
	valueAddr := p.address(operands[1])
	switch valueAddr.Type {
	case obj.TYPE_CONST, obj.TYPE_FCONST, obj.TYPE_SCONST, obj.TYPE_ADDR:
		// OK
	default:
		p.errorf("DATA value must be an immediate constant or address")
		return
	}

	// The addresses must not overlap. Easiest test: require monotonicity.
	if lastAddr, ok := p.dataAddr[name]; ok && nameAddr.Offset < lastAddr {
		p.errorf("overlapping DATA entry for %s", name)
		return
	}
	p.dataAddr[name] = nameAddr.Offset + int64(sz)

	switch valueAddr.Type {
	case obj.TYPE_CONST:
		switch sz {
		case 1, 2, 4, 8:
			nameAddr.Sym.WriteInt(p.ctxt, nameAddr.Offset, int(sz), valueAddr.Offset)
		default:
			p.errorf("bad int size for DATA argument: %d", sz)
		}
	case obj.TYPE_FCONST:
		switch sz {
		case 4:
			nameAddr.Sym.WriteFloat32(p.ctxt, nameAddr.Offset, float32(valueAddr.Val.(float64)))
		case 8:
			nameAddr.Sym.WriteFloat64(p.ctxt, nameAddr.Offset, valueAddr.Val.(float64))
		default:
			p.errorf("bad float size for DATA argument: %d", sz)
		}
	case obj.TYPE_SCONST:
		nameAddr.Sym.WriteString(p.ctxt, nameAddr.Offset, int(sz), valueAddr.Val.(string))
	case obj.TYPE_ADDR:
		if sz == p.arch.PtrSize {
			nameAddr.Sym.WriteAddr(p.ctxt, nameAddr.Offset, int(sz), valueAddr.Sym, valueAddr.Offset)
		} else {
			p.errorf("bad addr size for DATA argument: %d", sz)
		}
	}
}

// asmGlobl assembles a GLOBL pseudo-op.
// GLOBL shifts<>(SB),8,$256
// GLOBL shifts<>(SB),$256
func (p *Parser) asmGlobl(operands [][]lex.Token) {
	if len(operands) != 2 && len(operands) != 3 {
		p.errorf("expect two or three operands for GLOBL")
		return
	}

	// Operand 0 has the general form foo<>+0x04(SB).
	nameAddr := p.address(operands[0])
	if !p.validSymbol("GLOBL", &nameAddr, false) {
		return
	}
	next := 1

	// Next operand is the optional flag, a literal integer.
	var flag = int64(0)
	if len(operands) == 3 {
		flag = p.evalInteger("GLOBL", operands[1])
		next++
	}

	// Final operand is an immediate constant.
	addr := p.address(operands[next])
	if !p.validImmediate("GLOBL", &addr) {
		return
	}

	// log.Printf("GLOBL %s %d, $%d", name, flag, size)
	p.ctxt.Globl(nameAddr.Sym, addr.Offset, int(flag))
}

// asmPCData assembles a PCDATA pseudo-op.
// PCDATA $2, $705
func (p *Parser) asmPCData(operands [][]lex.Token) {
	if len(operands) != 2 {
		p.errorf("expect two operands for PCDATA")
		return
	}

	// Operand 0 must be an immediate constant.
	key := p.address(operands[0])
	if !p.validImmediate("PCDATA", &key) {
		return
	}

	// Operand 1 must be an immediate constant.
	value := p.address(operands[1])
	if !p.validImmediate("PCDATA", &value) {
		return
	}

	// log.Printf("PCDATA $%d, $%d", key.Offset, value.Offset)
	prog := &obj.Prog{
		Ctxt: p.ctxt,
		As:   obj.APCDATA,
		Pos:  p.pos(),
		From: key,
		To:   value,
	}
	p.append(prog, "", true)
}

// asmPCAlign assembles a PCALIGN pseudo-op.
// PCALIGN $16
func (p *Parser) asmPCAlign(operands [][]lex.Token) {
	if len(operands) != 1 {
		p.errorf("expect one operand for PCALIGN")
		return
	}

	// Operand 0 must be an immediate constant.
	key := p.address(operands[0])
	if !p.validImmediate("PCALIGN", &key) {
		return
	}

	prog := &obj.Prog{
		Ctxt: p.ctxt,
		As:   obj.APCALIGN,
		From: key,
	}
	p.append(prog, "", true)
}

// asmFuncData assembles a FUNCDATA pseudo-op.
// FUNCDATA $1, funcdata<>+4(SB)
func (p *Parser) asmFuncData(operands [][]lex.Token) {
	if len(operands) != 2 {
		p.errorf("expect two operands for FUNCDATA")
		return
	}

	// Operand 0 must be an immediate constant.
	valueAddr := p.address(operands[0])
	if !p.validImmediate("FUNCDATA", &valueAddr) {
		return
	}

	// Operand 1 is a symbol name in the form foo(SB).
	nameAddr := p.address(operands[1])
	if !p.validSymbol("FUNCDATA", &nameAddr, true) {
		return
	}

	prog := &obj.Prog{
		Ctxt: p.ctxt,
		As:   obj.AFUNCDATA,
		Pos:  p.pos(),
		From: valueAddr,
		To:   nameAddr,
	}
	p.append(prog, "", true)
}

// asmJump assembles a jump instruction.
// JMP	R1
// JMP	exit
// JMP	3(PC)
func (p *Parser) asmJump(op obj.As, cond string, a []obj.Addr) {
	var target *obj.Addr
	prog := &obj.Prog{
		Ctxt: p.ctxt,
		Pos:  p.pos(),
		As:   op,
	}
	switch len(a) {
	case 0:
		if p.arch.Family == sys.Wasm {
			target = &obj.Addr{Type: obj.TYPE_NONE}
			break
		}
		p.errorf("wrong number of arguments to %s instruction", op)
		return
	case 1:
		target = &a[0]
	case 2:
		// Special 2-operand jumps.
		target = &a[1]
		prog.From = a[0]
	case 3:
		if p.arch.Family == sys.PPC64 {
			// Special 3-operand jumps.
			// First two must be constants; a[1] is a register number.
			target = &a[2]
			prog.From = obj.Addr{
				Type:   obj.TYPE_CONST,
				Offset: p.getConstant(prog, op, &a[0]),
			}
			reg := int16(p.getConstant(prog, op, &a[1]))
			reg, ok := p.arch.RegisterNumber("R", reg)
			if !ok {
				p.errorf("bad register number %d", reg)
				return
			}
			prog.Reg = reg
			break
		}
		if p.arch.Family == sys.MIPS || p.arch.Family == sys.MIPS64 {
			// 3-operand jumps.
			// First two must be registers
			target = &a[2]
			prog.From = a[0]
			prog.Reg = p.getRegister(prog, op, &a[1])
			break
		}
		if p.arch.Family == sys.S390X {
			// 3-operand jumps.
			target = &a[2]
			prog.From = a[0]
			if a[1].Reg != 0 {
				// Compare two registers and jump.
				prog.Reg = p.getRegister(prog, op, &a[1])
			} else {
				// Compare register with immediate and jump.
				prog.SetFrom3(a[1])
			}
			break
		}
		if p.arch.Family == sys.ARM64 {
			// Special 3-operand jumps.
			// a[0] must be immediate constant; a[1] is a register.
			if a[0].Type != obj.TYPE_CONST {
				p.errorf("%s: expected immediate constant; found %s", op, obj.Dconv(prog, &a[0]))
				return
			}
			prog.From = a[0]
			prog.Reg = p.getRegister(prog, op, &a[1])
			target = &a[2]
			break
		}

		fallthrough
	default:
		p.errorf("wrong number of arguments to %s instruction", op)
		return
	}
	switch {
	case target.Type == obj.TYPE_BRANCH:
		// JMP 4(PC)
		prog.To = obj.Addr{
			Type:   obj.TYPE_BRANCH,
			Offset: p.pc + 1 + target.Offset, // +1 because p.pc is incremented in append, below.
		}
	case target.Type == obj.TYPE_REG:
		// JMP R1
		prog.To = *target
	case target.Type == obj.TYPE_MEM && (target.Name == obj.NAME_EXTERN || target.Name == obj.NAME_STATIC):
		// JMP main·morestack(SB)
		prog.To = *target
	case target.Type == obj.TYPE_INDIR && (target.Name == obj.NAME_EXTERN || target.Name == obj.NAME_STATIC):
		// JMP *main·morestack(SB)
		prog.To = *target
		prog.To.Type = obj.TYPE_INDIR
	case target.Type == obj.TYPE_MEM && target.Reg == 0 && target.Offset == 0:
		// JMP exit
		if target.Sym == nil {
			// Parse error left name unset.
			return
		}
		targetProg := p.labels[target.Sym.Name]
		if targetProg == nil {
			p.toPatch = append(p.toPatch, Patch{prog, target.Sym.Name})
		} else {
			p.branch(prog, targetProg)
		}
	case target.Type == obj.TYPE_MEM && target.Name == obj.NAME_NONE:
		// JMP 4(R0)
		prog.To = *target
		// On the ppc64, 9a encodes BR (CTR) as BR CTR. We do the same.
		if p.arch.Family == sys.PPC64 && target.Offset == 0 {
			prog.To.Type = obj.TYPE_REG
		}
	case target.Type == obj.TYPE_CONST:
		// JMP $4
		prog.To = a[0]
	case target.Type == obj.TYPE_NONE:
		// JMP
	default:
		p.errorf("cannot assemble jump %+v", target)
		return
	}

	p.append(prog, cond, true)
}

func (p *Parser) patch() {
	for _, patch := range p.toPatch {
		targetProg := p.labels[patch.label]
		if targetProg == nil {
			p.errorf("undefined label %s", patch.label)
			return
		}
		p.branch(patch.prog, targetProg)
	}
	p.toPatch = p.toPatch[:0]
}

func (p *Parser) branch(jmp, target *obj.Prog) {
	jmp.To = obj.Addr{
		Type:  obj.TYPE_BRANCH,
		Index: 0,
	}
	jmp.To.Val = target
}

// asmInstruction assembles an instruction.
// MOVW R9, (R10)
func (p *Parser) asmInstruction(op obj.As, cond string, a []obj.Addr) {
	// fmt.Printf("%s %+v\n", op, a)
	prog := &obj.Prog{
		Ctxt: p.ctxt,
		Pos:  p.pos(),
		As:   op,
	}
	switch len(a) {
	case 0:
		// Nothing to do.
	case 1:
		if p.arch.UnaryDst[op] || op == obj.ARET || op == obj.AGETCALLERPC {
			// prog.From is no address.
			prog.To = a[0]
		} else {
			prog.From = a[0]
			// prog.To is no address.
		}
		if p.arch.Family == sys.PPC64 && arch.IsPPC64NEG(op) {
			// NEG: From and To are both a[0].
			prog.To = a[0]
			prog.From = a[0]
			break
		}
	case 2:
		if p.arch.Family == sys.ARM {
			if arch.IsARMCMP(op) {
				prog.From = a[0]
				prog.Reg = p.getRegister(prog, op, &a[1])
				break
			}
			// Strange special cases.
			if arch.IsARMFloatCmp(op) {
				prog.From = a[0]
				prog.Reg = p.getRegister(prog, op, &a[1])
				break
			}
		} else if p.arch.Family == sys.ARM64 && arch.IsARM64CMP(op) {
			prog.From = a[0]
			prog.Reg = p.getRegister(prog, op, &a[1])
			break
		} else if p.arch.Family == sys.MIPS || p.arch.Family == sys.MIPS64 {
			if arch.IsMIPSCMP(op) || arch.IsMIPSMUL(op) {
				prog.From = a[0]
				prog.Reg = p.getRegister(prog, op, &a[1])
				break
			}
		}
		prog.From = a[0]
		prog.To = a[1]
	case 3:
		switch p.arch.Family {
		case sys.MIPS, sys.MIPS64:
			prog.From = a[0]
			prog.Reg = p.getRegister(prog, op, &a[1])
			prog.To = a[2]
		case sys.ARM:
			// Special cases.
			if arch.IsARMSTREX(op) {
				/*
					STREX x, (y), z
						from=(y) reg=x to=z
				*/
				prog.From = a[1]
				prog.Reg = p.getRegister(prog, op, &a[0])
				prog.To = a[2]
				break
			}
			if arch.IsARMBFX(op) {
				// a[0] and a[1] must be constants, a[2] must be a register
				prog.From = a[0]
				prog.SetFrom3(a[1])
				prog.To = a[2]
				break
			}
			// Otherwise the 2nd operand (a[1]) must be a register.
			prog.From = a[0]
			prog.Reg = p.getRegister(prog, op, &a[1])
			prog.To = a[2]
		case sys.AMD64:
			prog.From = a[0]
			prog.SetFrom3(a[1])
			prog.To = a[2]
		case sys.ARM64:
			// ARM64 instructions with one input and two outputs.
			if arch.IsARM64STLXR(op) {
				prog.From = a[0]
				prog.To = a[1]
				if a[2].Type != obj.TYPE_REG {
					p.errorf("invalid addressing modes for third operand to %s instruction, must be register", op)
					return
				}
				prog.RegTo2 = a[2].Reg
				break
			}
			if arch.IsARM64TBL(op) {
				prog.From = a[0]
				if a[1].Type != obj.TYPE_REGLIST {
					p.errorf("%s: expected list; found %s", op, obj.Dconv(prog, &a[1]))
				}
				prog.SetFrom3(a[1])
				prog.To = a[2]
				break
			}
			prog.From = a[0]
			prog.Reg = p.getRegister(prog, op, &a[1])
			prog.To = a[2]
		case sys.I386:
			prog.From = a[0]
			prog.SetFrom3(a[1])
			prog.To = a[2]
		case sys.PPC64:
			if arch.IsPPC64CMP(op) {
				// CMPW etc.; third argument is a CR register that goes into prog.Reg.
				prog.From = a[0]
				prog.Reg = p.getRegister(prog, op, &a[2])
				prog.To = a[1]
				break
			}
			// Arithmetic. Choices are:
			// reg reg reg
			// imm reg reg
			// reg imm reg
			// If the immediate is the middle argument, use From3.
			switch a[1].Type {
			case obj.TYPE_REG:
				prog.From = a[0]
				prog.Reg = p.getRegister(prog, op, &a[1])
				prog.To = a[2]
			case obj.TYPE_CONST:
				prog.From = a[0]
				prog.SetFrom3(a[1])
				prog.To = a[2]
			default:
				p.errorf("invalid addressing modes for %s instruction", op)
				return
			}
		case sys.S390X:
			prog.From = a[0]
			if a[1].Type == obj.TYPE_REG {
				prog.Reg = p.getRegister(prog, op, &a[1])
			} else {
				prog.SetFrom3(a[1])
			}
			prog.To = a[2]
		default:
			p.errorf("TODO: implement three-operand instructions for this architecture")
			return
		}
	case 4:
		if p.arch.Family == sys.ARM {
			if arch.IsARMBFX(op) {
				// a[0] and a[1] must be constants, a[2] and a[3] must be registers
				prog.From = a[0]
				prog.SetFrom3(a[1])
				prog.Reg = p.getRegister(prog, op, &a[2])
				prog.To = a[3]
				break
			}
			if arch.IsARMMULA(op) {
				// All must be registers.
				p.getRegister(prog, op, &a[0])
				r1 := p.getRegister(prog, op, &a[1])
				r2 := p.getRegister(prog, op, &a[2])
				p.getRegister(prog, op, &a[3])
				prog.From = a[0]
				prog.To = a[3]
				prog.To.Type = obj.TYPE_REGREG2
				prog.To.Offset = int64(r2)
				prog.Reg = r1
				break
			}
		}
		if p.arch.Family == sys.AMD64 {
			prog.From = a[0]
			prog.RestArgs = []obj.Addr{a[1], a[2]}
			prog.To = a[3]
			break
		}
		if p.arch.Family == sys.ARM64 {
			prog.From = a[0]
			prog.Reg = p.getRegister(prog, op, &a[1])
			prog.SetFrom3(a[2])
			prog.To = a[3]
			break
		}
		if p.arch.Family == sys.PPC64 {
			if arch.IsPPC64RLD(op) {
				prog.From = a[0]
				prog.Reg = p.getRegister(prog, op, &a[1])
				prog.SetFrom3(a[2])
				prog.To = a[3]
				break
			} else if arch.IsPPC64ISEL(op) {
				// ISEL BC,RB,RA,RT becomes isel rt,ra,rb,bc
				prog.SetFrom3(a[2])                       // ra
				prog.From = a[0]                          // bc
				prog.Reg = p.getRegister(prog, op, &a[1]) // rb
				prog.To = a[3]                            // rt
				break
			}
			// Else, it is a VA-form instruction
			// reg reg reg reg
			// imm reg reg reg
			// Or a VX-form instruction
			// imm imm reg reg
			if a[1].Type == obj.TYPE_REG {
				prog.From = a[0]
				prog.Reg = p.getRegister(prog, op, &a[1])
				prog.SetFrom3(a[2])
				prog.To = a[3]
				break
			} else if a[1].Type == obj.TYPE_CONST {
				prog.From = a[0]
				prog.Reg = p.getRegister(prog, op, &a[2])
				prog.SetFrom3(a[1])
				prog.To = a[3]
				break
			} else {
				p.errorf("invalid addressing modes for %s instruction", op)
				return
			}
		}
		if p.arch.Family == sys.S390X {
			if a[1].Type != obj.TYPE_REG {
				p.errorf("second operand must be a register in %s instruction", op)
				return
			}
			prog.From = a[0]
			prog.Reg = p.getRegister(prog, op, &a[1])
			prog.SetFrom3(a[2])
			prog.To = a[3]
			break
		}
		p.errorf("can't handle %s instruction with 4 operands", op)
		return
	case 5:
		if p.arch.Family == sys.PPC64 && arch.IsPPC64RLD(op) {
			// Always reg, reg, con, con, reg.  (con, con is a 'mask').
			prog.From = a[0]
			prog.Reg = p.getRegister(prog, op, &a[1])
			mask1 := p.getConstant(prog, op, &a[2])
			mask2 := p.getConstant(prog, op, &a[3])
			var mask uint32
			if mask1 < mask2 {
				mask = (^uint32(0) >> uint(mask1)) & (^uint32(0) << uint(31-mask2))
			} else {
				mask = (^uint32(0) >> uint(mask2+1)) & (^uint32(0) << uint(31-(mask1-1)))
			}
			prog.SetFrom3(obj.Addr{
				Type:   obj.TYPE_CONST,
				Offset: int64(mask),
			})
			prog.To = a[4]
			break
		}
		if p.arch.Family == sys.AMD64 {
			prog.From = a[0]
			prog.RestArgs = []obj.Addr{a[1], a[2], a[3]}
			prog.To = a[4]
			break
		}
		if p.arch.Family == sys.S390X {
			prog.From = a[0]
			prog.RestArgs = []obj.Addr{a[1], a[2], a[3]}
			prog.To = a[4]
			break
		}
		p.errorf("can't handle %s instruction with 5 operands", op)
		return
	case 6:
		if p.arch.Family == sys.ARM && arch.IsARMMRC(op) {
			// Strange special case: MCR, MRC.
			prog.To.Type = obj.TYPE_CONST
			x0 := p.getConstant(prog, op, &a[0])
			x1 := p.getConstant(prog, op, &a[1])
			x2 := int64(p.getRegister(prog, op, &a[2]))
			x3 := int64(p.getRegister(prog, op, &a[3]))
			x4 := int64(p.getRegister(prog, op, &a[4]))
			x5 := p.getConstant(prog, op, &a[5])
			// Cond is handled specially for this instruction.
			offset, MRC, ok := arch.ARMMRCOffset(op, cond, x0, x1, x2, x3, x4, x5)
			if !ok {
				p.errorf("unrecognized condition code .%q", cond)
			}
			prog.To.Offset = offset
			cond = ""
			prog.As = MRC // Both instructions are coded as MRC.
			break
		}
		fallthrough
	default:
		p.errorf("can't handle %s instruction with %d operands", op, len(a))
		return
	}

	p.append(prog, cond, true)
}

// newAddr returns a new(Addr) initialized to x.
func newAddr(x obj.Addr) *obj.Addr {
	p := new(obj.Addr)
	*p = x
	return p
}

// symbolName returns the symbol name, or an error string if none if available.
func symbolName(addr *obj.Addr) string {
	if addr.Sym != nil {
		return addr.Sym.Name
	}
	return "<erroneous symbol>"
}

var emptyProg obj.Prog

// getConstantPseudo checks that addr represents a plain constant and returns its value.
func (p *Parser) getConstantPseudo(pseudo string, addr *obj.Addr) int64 {
	if addr.Type != obj.TYPE_MEM || addr.Name != 0 || addr.Reg != 0 || addr.Index != 0 {
		p.errorf("%s: expected integer constant; found %s", pseudo, obj.Dconv(&emptyProg, addr))
	}
	return addr.Offset
}

// getConstant checks that addr represents a plain constant and returns its value.
func (p *Parser) getConstant(prog *obj.Prog, op obj.As, addr *obj.Addr) int64 {
	if addr.Type != obj.TYPE_MEM || addr.Name != 0 || addr.Reg != 0 || addr.Index != 0 {
		p.errorf("%s: expected integer constant; found %s", op, obj.Dconv(prog, addr))
	}
	return addr.Offset
}

// getImmediate checks that addr represents an immediate constant and returns its value.
func (p *Parser) getImmediate(prog *obj.Prog, op obj.As, addr *obj.Addr) int64 {
	if addr.Type != obj.TYPE_CONST || addr.Name != 0 || addr.Reg != 0 || addr.Index != 0 {
		p.errorf("%s: expected immediate constant; found %s", op, obj.Dconv(prog, addr))
	}
	return addr.Offset
}

// getRegister checks that addr represents a register and returns its value.
func (p *Parser) getRegister(prog *obj.Prog, op obj.As, addr *obj.Addr) int16 {
	if addr.Type != obj.TYPE_REG || addr.Offset != 0 || addr.Name != 0 || addr.Index != 0 {
		p.errorf("%s: expected register; found %s", op, obj.Dconv(prog, addr))
	}
	return addr.Reg
}
