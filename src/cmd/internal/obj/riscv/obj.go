// Copyright © 2015 The Go Authors.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package riscv

import (
	"cmd/internal/obj"
	"cmd/internal/sys"
	"fmt"
)

// TODO(jsing): Populate.
var RISCV64DWARFRegisters = map[int16]int16{}

func buildop(ctxt *obj.Link) {}

// progedit is called individually for each *obj.Prog. It normalizes instruction
// formats and eliminates as many pseudo-instructions as possible.
func progedit(ctxt *obj.Link, p *obj.Prog, newprog obj.ProgAlloc) {

	// Expand binary instructions to ternary ones.
	if p.Reg == 0 {
		switch p.As {
		case AADDI, ASLTI, ASLTIU, AANDI, AORI, AXORI, ASLLI, ASRLI, ASRAI,
			AADD, AAND, AOR, AXOR, ASLL, ASRL, ASUB, ASRA:
			p.Reg = p.To.Reg
		}
	}

	// Rewrite instructions with constant operands to refer to the immediate
	// form of the instruction.
	if p.From.Type == obj.TYPE_CONST {
		switch p.As {
		case AADD:
			p.As = AADDI
		case ASLT:
			p.As = ASLTI
		case ASLTU:
			p.As = ASLTIU
		case AAND:
			p.As = AANDI
		case AOR:
			p.As = AORI
		case AXOR:
			p.As = AXORI
		case ASLL:
			p.As = ASLLI
		case ASRL:
			p.As = ASRLI
		case ASRA:
			p.As = ASRAI
		}
	}
}

// setPCs sets the Pc field in all instructions reachable from p.
// It uses pc as the initial value.
func setPCs(p *obj.Prog, pc int64) {
	for ; p != nil; p = p.Link {
		p.Pc = pc
		pc += int64(encodingForProg(p).length)
	}
}

func preprocess(ctxt *obj.Link, cursym *obj.LSym, newprog obj.ProgAlloc) {
	if cursym.Func.Text == nil || cursym.Func.Text.Link == nil {
		return
	}

	text := cursym.Func.Text
	if text.As != obj.ATEXT {
		ctxt.Diag("preprocess: found symbol that does not start with TEXT directive")
		return
	}

	stacksize := text.To.Offset
	if stacksize == -8 {
		// Historical way to mark NOFRAME.
		text.From.Sym.Set(obj.AttrNoFrame, true)
		stacksize = 0
	}
	if stacksize < 0 {
		ctxt.Diag("negative frame size %d - did you mean NOFRAME?", stacksize)
	}
	if text.From.Sym.NoFrame() {
		if stacksize != 0 {
			ctxt.Diag("NOFRAME functions must have a frame size of 0, not %d", stacksize)
		}
	}

	cursym.Func.Args = text.To.Val.(int32)
	cursym.Func.Locals = int32(stacksize)

	// TODO(jsing): Implement.

	setPCs(cursym.Func.Text, 0)

	// Validate all instructions - this provides nice error messages.
	for p := cursym.Func.Text; p != nil; p = p.Link {
		encodingForProg(p).validate(p)
	}
}

func regVal(r, min, max int16) uint32 {
	if r < min || r > max {
		panic(fmt.Sprintf("register out of range, want %d < %d < %d", min, r, max))
	}
	return uint32(r - min)
}

// regI returns an integer register.
func regI(r int16) uint32 {
	return regVal(r, REG_X0, REG_X31)
}

// regAddr extracts a register from an Addr.
func regAddr(a obj.Addr, min, max int16) uint32 {
	if a.Type != obj.TYPE_REG {
		panic(fmt.Sprintf("ill typed: %+v", a))
	}
	return regVal(a.Reg, min, max)
}

// regIAddr extracts the integer register from an Addr.
func regIAddr(a obj.Addr) uint32 {
	return regAddr(a, REG_X0, REG_X31)
}

// immFits reports whether immediate value x fits in nbits bits as a
// signed integer.
func immFits(x int64, nbits uint) bool {
	nbits--
	var min int64 = -1 << nbits
	var max int64 = 1<<nbits - 1
	return min <= x && x <= max
}

// immI extracts the integer literal of the specified size from an Addr.
func immI(a obj.Addr, nbits uint) uint32 {
	if a.Type != obj.TYPE_CONST {
		panic(fmt.Sprintf("ill typed: %+v", a))
	}
	if !immFits(a.Offset, nbits) {
		panic(fmt.Sprintf("immediate %d in %v cannot fit in %d bits", a.Offset, a, nbits))
	}
	return uint32(a.Offset)
}

func wantImm(p *obj.Prog, pos string, a obj.Addr, nbits uint) {
	if a.Type != obj.TYPE_CONST {
		p.Ctxt.Diag("%v\texpected immediate in %s position but got %s", p, pos, obj.Dconv(p, &a))
		return
	}
	if !immFits(a.Offset, nbits) {
		p.Ctxt.Diag("%v\timmediate in %s position cannot be larger than %d bits but got %d", p, pos, nbits, a.Offset)
	}
}

func wantReg(p *obj.Prog, pos string, descr string, r, min, max int16) {
	if r < min || r > max {
		p.Ctxt.Diag("%v\texpected %s register in %s position but got non-%s register %s", p, descr, pos, descr, regName(int(r)))
	}
}

// wantIntReg checks that r is an integer register.
func wantIntReg(p *obj.Prog, pos string, r int16) {
	wantReg(p, pos, "integer", r, REG_X0, REG_X31)
}

func wantRegAddr(p *obj.Prog, pos string, a *obj.Addr, descr string, min int16, max int16) {
	if a == nil {
		p.Ctxt.Diag("%v\texpected register in %s position but got nothing", p, pos)
		return
	}
	if a.Type != obj.TYPE_REG {
		p.Ctxt.Diag("%v\texpected register in %s position but got %s", p, pos, obj.Dconv(p, a))
		return
	}
	if a.Reg < min || a.Reg > max {
		p.Ctxt.Diag("%v\texpected %s register in %s position but got non-%s register %s", p, descr, pos, descr, obj.Dconv(p, a))
	}
}

// wantIntRegAddr checks that a contains an integer register.
func wantIntRegAddr(p *obj.Prog, pos string, a *obj.Addr) {
	wantRegAddr(p, pos, a, "integer", REG_X0, REG_X31)
}

func validateRIII(p *obj.Prog) {
	wantIntRegAddr(p, "from", &p.From)
	wantIntReg(p, "reg", p.Reg)
	wantIntRegAddr(p, "to", &p.To)
}

func validateII(p *obj.Prog) {
	wantImm(p, "from", p.From, 12)
	wantIntReg(p, "reg", p.Reg)
	wantIntRegAddr(p, "to", &p.To)
}

func validateRaw(p *obj.Prog) {
	// Treat the raw value specially as a 32-bit unsigned integer.
	// Nobody wants to enter negative machine code.
	a := p.From
	if a.Type != obj.TYPE_CONST {
		p.Ctxt.Diag("%v\texpected immediate in raw position but got %s", p, obj.Dconv(p, &a))
		return
	}
	if a.Offset < 0 || 1<<32 <= a.Offset {
		p.Ctxt.Diag("%v\timmediate in raw position cannot be larger than 32 bits but got %d", p, a.Offset)
	}
}

// encodeR encodes an R-type RISC-V instruction.
func encodeR(p *obj.Prog, rs1 uint32, rs2 uint32, rd uint32) uint32 {
	ins := encode(p.As)
	if ins == nil {
		panic("encodeR: could not encode instruction")
	}
	if ins.rs2 != 0 && rs2 != 0 {
		panic("encodeR: instruction uses rs2, but rs2 was nonzero")
	}

	// Use Scond for the floating-point rounding mode override.
	// TODO(sorear): Is there a more appropriate way to handle opcode extension bits like this?
	return ins.funct7<<25 | ins.rs2<<20 | rs2<<20 | rs1<<15 | ins.funct3<<12 | uint32(p.Scond)<<12 | rd<<7 | ins.opcode
}

func encodeRIII(p *obj.Prog) uint32 {
	return encodeR(p, regI(p.Reg), regIAddr(p.From), regIAddr(p.To))
}

// encodeI encodes an I-type RISC-V instruction.
func encodeI(p *obj.Prog, rd uint32) uint32 {
	imm := immI(p.From, 12)
	rs1 := regI(p.Reg)
	ins := encode(p.As)
	if ins == nil {
		panic("encodeI: could not encode instruction")
	}
	imm |= uint32(ins.csr)
	return imm<<20 | rs1<<15 | ins.funct3<<12 | rd<<7 | ins.opcode
}

func encodeII(p *obj.Prog) uint32 {
	return encodeI(p, regIAddr(p.To))
}

// encodeRaw encodes a raw instruction value.
func encodeRaw(p *obj.Prog) uint32 {
	// Treat the raw value specially as a 32-bit unsigned integer.
	// Nobody wants to enter negative machine code.
	a := p.From
	if a.Type != obj.TYPE_CONST {
		panic(fmt.Sprintf("ill typed: %+v", a))
	}
	if a.Offset < 0 || 1<<32 <= a.Offset {
		panic(fmt.Sprintf("immediate %d in %v cannot fit in 32 bits", a.Offset, a))
	}
	return uint32(a.Offset)
}

type encoding struct {
	encode   func(*obj.Prog) uint32 // encode returns the machine code for an *obj.Prog
	validate func(*obj.Prog)        // validate validates an *obj.Prog, calling ctxt.Diag for any issues
	length   int                    // length of encoded instruction; 0 for pseudo-ops, 4 otherwise
}

var (
	// Encodings have the following naming convention:
	//
	//  1. the instruction encoding (R/I/S/SB/U/UJ), in lowercase
	//  2. zero or more register operand identifiers (I = integer
	//     register, F = float register), in uppercase
	//  3. the word "Encoding"
	//
	// For example, rIIIEncoding indicates an R-type instruction with two
	// integer register inputs and an integer register output; sFEncoding
	// indicates an S-type instruction with rs2 being a float register.

	rIIIEncoding = encoding{encode: encodeRIII, validate: validateRIII, length: 4}

	iIEncoding = encoding{encode: encodeII, validate: validateII, length: 4}

	// rawEncoding encodes a raw instruction byte sequence.
	rawEncoding = encoding{encode: encodeRaw, validate: validateRaw, length: 4}

	// pseudoOpEncoding panics if encoding is attempted, but does no validation.
	pseudoOpEncoding = encoding{encode: nil, validate: func(*obj.Prog) {}, length: 0}

	// badEncoding is used when an invalid op is encountered.
	// An error has already been generated, so let anything else through.
	badEncoding = encoding{encode: func(*obj.Prog) uint32 { return 0 }, validate: func(*obj.Prog) {}, length: 0}
)

// encodingForAs contains the encoding for a RISC-V instruction.
// Instructions are masked with obj.AMask to keep indices small.
var encodingForAs = [ALAST & obj.AMask]encoding{
	// TODO(jsing): Implement remaining instructions.

	// Unprivileged ISA

	// 2.4: Integer Computational Instructions
	AADDI & obj.AMask:  iIEncoding,
	ASLTI & obj.AMask:  iIEncoding,
	ASLTIU & obj.AMask: iIEncoding,
	AANDI & obj.AMask:  iIEncoding,
	AORI & obj.AMask:   iIEncoding,
	AXORI & obj.AMask:  iIEncoding,
	ASLLI & obj.AMask:  iIEncoding,
	ASRLI & obj.AMask:  iIEncoding,
	ASRAI & obj.AMask:  iIEncoding,
	AADD & obj.AMask:   rIIIEncoding,
	ASLT & obj.AMask:   rIIIEncoding,
	ASLTU & obj.AMask:  rIIIEncoding,
	AAND & obj.AMask:   rIIIEncoding,
	AOR & obj.AMask:    rIIIEncoding,
	AXOR & obj.AMask:   rIIIEncoding,
	ASLL & obj.AMask:   rIIIEncoding,
	ASRL & obj.AMask:   rIIIEncoding,
	ASUB & obj.AMask:   rIIIEncoding,
	ASRA & obj.AMask:   rIIIEncoding,

	// Escape hatch
	AWORD & obj.AMask: rawEncoding,

	// Pseudo-operations
	obj.AFUNCDATA: pseudoOpEncoding,
	obj.APCDATA:   pseudoOpEncoding,
	obj.ATEXT:     pseudoOpEncoding,
	obj.ANOP:      pseudoOpEncoding,
}

// encodingForProg returns the encoding (encode+validate funcs) for an *obj.Prog.
func encodingForProg(p *obj.Prog) encoding {
	if base := p.As &^ obj.AMask; base != obj.ABaseRISCV && base != 0 {
		p.Ctxt.Diag("encodingForProg: not a RISC-V instruction %s", p.As)
		return badEncoding
	}
	as := p.As & obj.AMask
	if int(as) >= len(encodingForAs) {
		p.Ctxt.Diag("encodingForProg: bad RISC-V instruction %s", p.As)
		return badEncoding
	}
	enc := encodingForAs[as]
	if enc.validate == nil {
		p.Ctxt.Diag("encodingForProg: no encoding for instruction %s", p.As)
		return badEncoding
	}
	return enc
}

// assemble emits machine code.
// It is called at the very end of the assembly process.
func assemble(ctxt *obj.Link, cursym *obj.LSym, newprog obj.ProgAlloc) {
	var symcode []uint32
	for p := cursym.Func.Text; p != nil; p = p.Link {
		enc := encodingForProg(p)
		if enc.length > 0 {
			symcode = append(symcode, enc.encode(p))
		}
	}
	cursym.Size = int64(4 * len(symcode))

	cursym.Grow(cursym.Size)
	for p, i := cursym.P, 0; i < len(symcode); p, i = p[4:], i+1 {
		ctxt.Arch.ByteOrder.PutUint32(p, symcode[i])
	}
}

var LinkRISCV64 = obj.LinkArch{
	Arch:           sys.ArchRISCV64,
	Init:           buildop,
	Preprocess:     preprocess,
	Assemble:       assemble,
	Progedit:       progedit,
	UnaryDst:       unaryDst,
	DWARFRegisters: RISCV64DWARFRegisters,
}
