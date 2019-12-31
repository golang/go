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
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"fmt"
)

// TODO(jsing): Populate.
var RISCV64DWARFRegisters = map[int16]int16{}

func buildop(ctxt *obj.Link) {}

// jalrToSym replaces p with a set of Progs needed to jump to the Sym in p.
// lr is the link register to use for the JALR.
// p must be a CALL, JMP or RET.
func jalrToSym(ctxt *obj.Link, p *obj.Prog, newprog obj.ProgAlloc, lr int16) *obj.Prog {
	if p.As != obj.ACALL && p.As != obj.AJMP && p.As != obj.ARET {
		ctxt.Diag("unexpected Prog in jalrToSym: %v", p)
		return p
	}

	// TODO(jsing): Consider using a single JAL instruction and teaching
	// the linker to provide trampolines for the case where the destination
	// offset is too large. This would potentially reduce instructions for
	// the common case, but would require three instructions to go via the
	// trampoline.

	to := p.To

	// This offset isn't really encoded with either instruction. It will be
	// extracted for a relocation later.
	p.As = AAUIPC
	p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: to.Offset, Sym: to.Sym}
	p.Reg = 0
	p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_TMP}
	p.Mark |= NEED_PCREL_ITYPE_RELOC
	p = obj.Appendp(p, newprog)

	// Leave Sym only for the CALL reloc in assemble.
	p.As = AJALR
	p.From.Type = obj.TYPE_REG
	p.From.Reg = lr
	p.From.Sym = to.Sym
	p.Reg = 0
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_TMP
	lowerJALR(p)

	return p
}

// lowerJALR normalizes a JALR instruction.
func lowerJALR(p *obj.Prog) {
	if p.As != AJALR {
		panic("lowerJALR: not a JALR")
	}

	// JALR gets parsed like JAL - the linkage pointer goes in From,
	// and the target is in To. However, we need to assemble it as an
	// I-type instruction, so place the linkage pointer in To, the
	// target register in Reg, and the offset in From.
	p.Reg = p.To.Reg
	p.From, p.To = p.To, p.From
	p.From.Type, p.From.Reg = obj.TYPE_CONST, obj.REG_NONE
}

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

	switch p.As {
	case ALW, ALWU, ALH, ALHU, ALB, ALBU, ALD, AFLW, AFLD:
		switch p.From.Type {
		case obj.TYPE_MEM:
			// Convert loads from memory/addresses to ternary form.
			p.Reg = p.From.Reg
			p.From.Type, p.From.Reg = obj.TYPE_CONST, obj.REG_NONE
		default:
			p.Ctxt.Diag("%v\tmemory required for source", p)
		}

	case ASW, ASH, ASB, ASD, AFSW, AFSD:
		switch p.To.Type {
		case obj.TYPE_MEM:
			// Convert stores to memory/addresses to ternary form.
			p.Reg = p.From.Reg
			p.From.Type, p.From.Offset, p.From.Reg = obj.TYPE_CONST, p.To.Offset, obj.REG_NONE
			p.To.Type, p.To.Offset = obj.TYPE_REG, 0
		default:
			p.Ctxt.Diag("%v\tmemory required for destination", p)
		}

	case obj.AJMP:
		// Turn JMP into JAL ZERO or JALR ZERO.
		// p.From is actually an _output_ for this instruction.
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_ZERO

		switch p.To.Type {
		case obj.TYPE_BRANCH:
			p.As = AJAL
		case obj.TYPE_MEM:
			switch p.To.Name {
			case obj.NAME_NONE:
				p.As = AJALR
				lowerJALR(p)
			case obj.NAME_EXTERN:
				// Handled in preprocess.
			default:
				ctxt.Diag("progedit: unsupported name %d for %v", p.To.Name, p)
			}
		default:
			panic(fmt.Sprintf("unhandled type %+v", p.To.Type))
		}

	case obj.ACALL:
		switch p.To.Type {
		case obj.TYPE_MEM:
			// Handled in preprocess.
		case obj.TYPE_REG:
			p.As = AJALR
			p.From.Type = obj.TYPE_REG
			p.From.Reg = REG_LR
			lowerJALR(p)
		default:
			ctxt.Diag("unknown destination type %+v in CALL: %v", p.To.Type, p)
		}

	case AJALR:
		lowerJALR(p)

	case obj.AUNDEF, AECALL, AEBREAK, ASCALL, ASBREAK, ARDCYCLE, ARDTIME, ARDINSTRET:
		switch p.As {
		case obj.AUNDEF:
			p.As = AEBREAK
		case ASCALL:
			// SCALL is the old name for ECALL.
			p.As = AECALL
		case ASBREAK:
			// SBREAK is the old name for EBREAK.
			p.As = AEBREAK
		}

		ins := encode(p.As)
		if ins == nil {
			panic("progedit: tried to rewrite nonexistent instruction")
		}

		// The CSR isn't exactly an offset, but it winds up in the
		// immediate area of the encoded instruction, so record it in
		// the Offset field.
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = ins.csr
		p.Reg = REG_ZERO
		if p.To.Type == obj.TYPE_NONE {
			p.To.Type, p.To.Reg = obj.TYPE_REG, REG_ZERO
		}

	case AFSQRTS, AFSQRTD:
		// These instructions expect a zero (i.e. float register 0)
		// to be the second input operand.
		p.Reg = p.From.Reg
		p.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_F0}

	case AFCVTWS, AFCVTLS, AFCVTWUS, AFCVTLUS, AFCVTWD, AFCVTLD, AFCVTWUD, AFCVTLUD:
		// Set the rounding mode in funct3 to round to zero.
		p.Scond = 1

	case ASEQZ:
		// SEQZ rs, rd -> SLTIU $1, rs, rd
		p.As = ASLTIU
		p.Reg = p.From.Reg
		p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: 1}

	case ASNEZ:
		// SNEZ rs, rd -> SLTU rs, x0, rd
		p.As = ASLTU
		p.Reg = REG_ZERO

	case AFNEGS:
		// FNEGS rs, rd -> FSGNJNS rs, rs, rd
		p.As = AFSGNJNS
		p.Reg = p.From.Reg

	case AFNEGD:
		// FNEGD rs, rd -> FSGNJND rs, rs, rd
		p.As = AFSGNJND
		p.Reg = p.From.Reg
	}
}

// addrToReg extracts the register from an Addr, handling special Addr.Names.
func addrToReg(a obj.Addr) int16 {
	switch a.Name {
	case obj.NAME_PARAM, obj.NAME_AUTO:
		return REG_SP
	}
	return a.Reg
}

// movToLoad converts a MOV mnemonic into the corresponding load instruction.
func movToLoad(mnemonic obj.As) obj.As {
	switch mnemonic {
	case AMOV:
		return ALD
	case AMOVB:
		return ALB
	case AMOVH:
		return ALH
	case AMOVW:
		return ALW
	case AMOVBU:
		return ALBU
	case AMOVHU:
		return ALHU
	case AMOVWU:
		return ALWU
	case AMOVF:
		return AFLW
	case AMOVD:
		return AFLD
	default:
		panic(fmt.Sprintf("%+v is not a MOV", mnemonic))
	}
}

// movToStore converts a MOV mnemonic into the corresponding store instruction.
func movToStore(mnemonic obj.As) obj.As {
	switch mnemonic {
	case AMOV:
		return ASD
	case AMOVB:
		return ASB
	case AMOVH:
		return ASH
	case AMOVW:
		return ASW
	case AMOVF:
		return AFSW
	case AMOVD:
		return AFSD
	default:
		panic(fmt.Sprintf("%+v is not a MOV", mnemonic))
	}
}

// rewriteMOV rewrites MOV pseudo-instructions.
func rewriteMOV(ctxt *obj.Link, newprog obj.ProgAlloc, p *obj.Prog) {
	switch p.As {
	case AMOV, AMOVB, AMOVH, AMOVW, AMOVBU, AMOVHU, AMOVWU, AMOVF, AMOVD:
	default:
		panic(fmt.Sprintf("%+v is not a MOV pseudo-instruction", p.As))
	}

	switch p.From.Type {
	case obj.TYPE_MEM: // MOV c(Rs), Rd -> L $c, Rs, Rd
		switch p.From.Name {
		case obj.NAME_AUTO, obj.NAME_PARAM, obj.NAME_NONE:
			if p.To.Type != obj.TYPE_REG {
				ctxt.Diag("unsupported load at %v", p)
			}
			p.As = movToLoad(p.As)
			p.Reg = addrToReg(p.From)
			p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: p.From.Offset}

		case obj.NAME_EXTERN, obj.NAME_STATIC:
			// AUIPC $off_hi, R
			// L $off_lo, R
			as := p.As
			to := p.To

			// The offset is not really encoded with either instruction.
			// It will be extracted later for a relocation.
			p.As = AAUIPC
			p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: p.From.Offset, Sym: p.From.Sym}
			p.Reg = 0
			p.To = obj.Addr{Type: obj.TYPE_REG, Reg: to.Reg}
			p.Mark |= NEED_PCREL_ITYPE_RELOC
			p = obj.Appendp(p, newprog)

			p.As = movToLoad(as)
			p.From = obj.Addr{Type: obj.TYPE_CONST}
			p.Reg = to.Reg
			p.To = to

		default:
			ctxt.Diag("unsupported name %d for %v", p.From.Name, p)
		}

	case obj.TYPE_REG:
		switch p.To.Type {
		case obj.TYPE_REG:
			switch p.As {
			case AMOV: // MOV Ra, Rb -> ADDI $0, Ra, Rb
				p.As = AADDI
				p.Reg = p.From.Reg
				p.From = obj.Addr{Type: obj.TYPE_CONST}

			case AMOVF: // MOVF Ra, Rb -> FSGNJS Ra, Ra, Rb
				p.As = AFSGNJS
				p.Reg = p.From.Reg

			case AMOVD: // MOVD Ra, Rb -> FSGNJD Ra, Ra, Rb
				p.As = AFSGNJD
				p.Reg = p.From.Reg

			default:
				ctxt.Diag("unsupported register-register move at %v", p)
			}

		case obj.TYPE_MEM: // MOV Rs, c(Rd) -> S $c, Rs, Rd
			switch p.As {
			case AMOVBU, AMOVHU, AMOVWU:
				ctxt.Diag("unsupported unsigned store at %v", p)
			}
			switch p.To.Name {
			case obj.NAME_AUTO, obj.NAME_PARAM, obj.NAME_NONE:
				// The destination address goes in p.From and p.To here,
				// with the offset in p.From and the register in p.To.
				// The source register goes in Reg.
				p.As = movToStore(p.As)
				p.Reg = p.From.Reg
				p.From = p.To
				p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: p.From.Offset}
				p.To = obj.Addr{Type: obj.TYPE_REG, Reg: addrToReg(p.To)}

			case obj.NAME_EXTERN:
				// AUIPC $off_hi, TMP
				// S $off_lo, TMP, R
				as := p.As
				from := p.From

				// The offset is not really encoded with either instruction.
				// It will be extracted later for a relocation.
				p.As = AAUIPC
				p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: p.To.Offset, Sym: p.To.Sym}
				p.Reg = 0
				p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_TMP}
				p.Mark |= NEED_PCREL_STYPE_RELOC
				p = obj.Appendp(p, newprog)

				p.As = movToStore(as)
				p.From = obj.Addr{Type: obj.TYPE_CONST}
				p.Reg = from.Reg
				p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_TMP}

			default:
				ctxt.Diag("unsupported name %d for %v", p.From.Name, p)
			}

		default:
			ctxt.Diag("unsupported MOV at %v", p)
		}

	case obj.TYPE_CONST:
		// MOV $c, R
		// If c is small enough, convert to:
		//   ADD $c, ZERO, R
		// If not, convert to:
		//   LUI top20bits(c), R
		//   ADD bottom12bits(c), R, R
		if p.As != AMOV {
			ctxt.Diag("unsupported constant load at %v", p)
		}
		off := p.From.Offset
		to := p.To

		low, high, err := Split32BitImmediate(off)
		if err != nil {
			ctxt.Diag("%v: constant %d too large: %v", p, off, err)
		}

		// LUI is only necessary if the offset doesn't fit in 12-bits.
		needLUI := high != 0
		if needLUI {
			p.As = ALUI
			p.To = to
			// Pass top 20 bits to LUI.
			p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: high}
			p = obj.Appendp(p, newprog)
		}
		p.As = AADDIW
		p.To = to
		p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: low}
		p.Reg = REG_ZERO
		if needLUI {
			p.Reg = to.Reg
		}

	case obj.TYPE_ADDR: // MOV $sym+off(SP/SB), R
		if p.To.Type != obj.TYPE_REG || p.As != AMOV {
			ctxt.Diag("unsupported addr MOV at %v", p)
		}
		switch p.From.Name {
		case obj.NAME_EXTERN, obj.NAME_STATIC:
			// AUIPC $off_hi, R
			// ADDI $off_lo, R
			to := p.To

			// The offset is not really encoded with either instruction.
			// It will be extracted later for a relocation.
			p.As = AAUIPC
			p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: p.From.Offset, Sym: p.From.Sym}
			p.Reg = 0
			p.To = to
			p.Mark |= NEED_PCREL_ITYPE_RELOC
			p = obj.Appendp(p, newprog)

			p.As = AADDI
			p.From = obj.Addr{Type: obj.TYPE_CONST}
			p.Reg = to.Reg
			p.To = to

		case obj.NAME_PARAM, obj.NAME_AUTO:
			p.As = AADDI
			p.Reg = REG_SP
			p.From.Type = obj.TYPE_CONST

		case obj.NAME_NONE:
			p.As = AADDI
			p.Reg = p.From.Reg
			p.From.Type = obj.TYPE_CONST
			p.From.Reg = 0

		default:
			ctxt.Diag("bad addr MOV from name %v at %v", p.From.Name, p)
		}

	default:
		ctxt.Diag("unsupported MOV at %v", p)
	}
}

// invertBranch inverts the condition of a conditional branch.
func invertBranch(i obj.As) obj.As {
	switch i {
	case ABEQ:
		return ABNE
	case ABNE:
		return ABEQ
	case ABLT:
		return ABGE
	case ABGE:
		return ABLT
	case ABLTU:
		return ABGEU
	case ABGEU:
		return ABLTU
	default:
		panic("invertBranch: not a branch")
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

// stackOffset updates Addr offsets based on the current stack size.
//
// The stack looks like:
// -------------------
// |                 |
// |      PARAMs     |
// |                 |
// |                 |
// -------------------
// |    Parent RA    |   SP on function entry
// -------------------
// |                 |
// |                 |
// |       AUTOs     |
// |                 |
// |                 |
// -------------------
// |        RA       |   SP during function execution
// -------------------
//
// FixedFrameSize makes other packages aware of the space allocated for RA.
//
// A nicer version of this diagram can be found on slide 21 of the presentation
// attached to:
//
//   https://golang.org/issue/16922#issuecomment-243748180
//
func stackOffset(a *obj.Addr, stacksize int64) {
	switch a.Name {
	case obj.NAME_AUTO:
		// Adjust to the top of AUTOs.
		a.Offset += stacksize
	case obj.NAME_PARAM:
		// Adjust to the bottom of PARAMs.
		a.Offset += stacksize + 8
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

	// Update stack-based offsets.
	for p := cursym.Func.Text; p != nil; p = p.Link {
		stackOffset(&p.From, stacksize)
		stackOffset(&p.To, stacksize)
	}

	// Additional instruction rewriting. Any rewrites that change the number
	// of instructions must occur here (before jump target resolution).
	for p := cursym.Func.Text; p != nil; p = p.Link {
		switch p.As {
		case obj.AGETCALLERPC:
			if cursym.Leaf() {
				// MOV LR, Rd
				p.As = AMOV
				p.From.Type = obj.TYPE_REG
				p.From.Reg = REG_LR
			} else {
				// MOV (RSP), Rd
				p.As = AMOV
				p.From.Type = obj.TYPE_MEM
				p.From.Reg = REG_SP
			}

		case obj.ACALL:
			switch p.To.Type {
			case obj.TYPE_MEM:
				jalrToSym(ctxt, p, newprog, REG_LR)
			}

		case obj.AJMP:
			switch p.To.Type {
			case obj.TYPE_MEM:
				switch p.To.Name {
				case obj.NAME_EXTERN:
					// JMP to symbol.
					jalrToSym(ctxt, p, newprog, REG_ZERO)
				}
			}

		// Replace FNE[SD] with FEQ[SD] and NOT.
		case AFNES:
			if p.To.Type != obj.TYPE_REG {
				ctxt.Diag("progedit: FNES needs an integer register output")
			}
			dst := p.To.Reg
			p.As = AFEQS
			p = obj.Appendp(p, newprog)

			p.As = AXORI // [bit] xor 1 = not [bit]
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = 1
			p.Reg = dst
			p.To.Type = obj.TYPE_REG
			p.To.Reg = dst

		case AFNED:
			if p.To.Type != obj.TYPE_REG {
				ctxt.Diag("progedit: FNED needs an integer register output")
			}
			dst := p.To.Reg
			p.As = AFEQD
			p = obj.Appendp(p, newprog)

			p.As = AXORI // [bit] xor 1 = not [bit]
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = 1
			p.Reg = dst
			p.To.Type = obj.TYPE_REG
			p.To.Reg = dst
		}
	}

	// Rewrite MOV pseudo-instructions. This cannot be done in
	// progedit, as SP offsets need to be applied before we split
	// up some of the Addrs.
	for p := cursym.Func.Text; p != nil; p = p.Link {
		switch p.As {
		case AMOV, AMOVB, AMOVH, AMOVW, AMOVBU, AMOVHU, AMOVWU, AMOVF, AMOVD:
			rewriteMOV(ctxt, newprog, p)
		}
	}

	// Split immediates larger than 12-bits.
	for p := cursym.Func.Text; p != nil; p = p.Link {
		switch p.As {
		// <opi> $imm, REG, TO
		case AADDI, AANDI, AORI, AXORI:
			// LUI $high, TMP
			// ADDI $low, TMP, TMP
			// <op> TMP, REG, TO
			q := *p
			low, high, err := Split32BitImmediate(p.From.Offset)
			if err != nil {
				ctxt.Diag("%v: constant %d too large", p, p.From.Offset, err)
			}
			if high == 0 {
				break // no need to split
			}

			p.As = ALUI
			p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: high}
			p.Reg = 0
			p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_TMP}
			p.Spadj = 0 // needed if TO is SP
			p = obj.Appendp(p, newprog)

			p.As = AADDIW
			p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: low}
			p.Reg = REG_TMP
			p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_TMP}
			p = obj.Appendp(p, newprog)

			switch q.As {
			case AADDI:
				p.As = AADD
			case AANDI:
				p.As = AAND
			case AORI:
				p.As = AOR
			case AXORI:
				p.As = AXOR
			default:
				ctxt.Diag("progedit: unsupported inst %v for splitting", q)
			}
			p.Spadj = q.Spadj
			p.To = q.To
			p.Reg = q.Reg
			p.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_TMP}

		// <load> $imm, REG, TO (load $imm+(REG), TO)
		// <store> $imm, REG, TO (store $imm+(TO), REG)
		case ALD, ALB, ALH, ALW, ALBU, ALHU, ALWU,
			ASD, ASB, ASH, ASW:
			// LUI $high, TMP
			// ADDI $low, TMP, TMP
			q := *p
			low, high, err := Split32BitImmediate(p.From.Offset)
			if err != nil {
				ctxt.Diag("%v: constant %d too large", p, p.From.Offset)
			}
			if high == 0 {
				break // no need to split
			}

			switch q.As {
			case ALD, ALB, ALH, ALW, ALBU, ALHU, ALWU:
				// LUI $high, TMP
				// ADD TMP, REG, TMP
				// <load> $low, TMP, TO
				p.As = ALUI
				p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: high}
				p.Reg = 0
				p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_TMP}
				p.Spadj = 0 // needed if TO is SP
				p = obj.Appendp(p, newprog)

				p.As = AADD
				p.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_TMP}
				p.Reg = q.Reg
				p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_TMP}
				p = obj.Appendp(p, newprog)

				p.As = q.As
				p.To = q.To
				p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: low}
				p.Reg = REG_TMP

			case ASD, ASB, ASH, ASW:
				// LUI $high, TMP
				// ADD TMP, TO, TMP
				// <store> $low, REG, TMP
				p.As = ALUI
				p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: high}
				p.Reg = 0
				p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_TMP}
				p.Spadj = 0 // needed if TO is SP
				p = obj.Appendp(p, newprog)

				p.As = AADD
				p.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_TMP}
				p.Reg = q.To.Reg
				p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_TMP}
				p = obj.Appendp(p, newprog)

				p.As = q.As
				p.Reg = q.Reg
				p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_TMP}
				p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: low}
			}
		}
	}

	// Compute instruction addresses.  Once we do that, we need to check for
	// overextended jumps and branches.  Within each iteration, Pc differences
	// are always lower bounds (since the program gets monotonically longer,
	// a fixed point will be reached).  No attempt to handle functions > 2GiB.
	for {
		rescan := false
		setPCs(cursym.Func.Text, 0)

		for p := cursym.Func.Text; p != nil; p = p.Link {
			switch p.As {
			case ABEQ, ABNE, ABLT, ABGE, ABLTU, ABGEU:
				if p.To.Type != obj.TYPE_BRANCH {
					panic("assemble: instruction with branch-like opcode lacks destination")
				}
				offset := p.Pcond.Pc - p.Pc
				if offset < -4096 || 4096 <= offset {
					// Branch is long.  Replace it with a jump.
					jmp := obj.Appendp(p, newprog)
					jmp.As = AJAL
					jmp.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_ZERO}
					jmp.To = obj.Addr{Type: obj.TYPE_BRANCH}
					jmp.Pcond = p.Pcond

					p.As = invertBranch(p.As)
					p.Pcond = jmp.Link

					// We may have made previous branches too long,
					// so recheck them.
					rescan = true
				}
			case AJAL:
				if p.Pcond == nil {
					panic("intersymbol jumps should be expressed as AUIPC+JALR")
				}
				offset := p.Pcond.Pc - p.Pc
				if offset < -(1<<20) || (1<<20) <= offset {
					// Replace with 2-instruction sequence. This assumes
					// that TMP is not live across J instructions, since
					// it is reserved by SSA.
					jmp := obj.Appendp(p, newprog)
					jmp.As = AJALR
					jmp.From = obj.Addr{Type: obj.TYPE_CONST, Offset: 0}
					jmp.To = p.From
					jmp.Reg = REG_TMP

					// p.From is not generally valid, however will be
					// fixed up in the next loop.
					p.As = AAUIPC
					p.From = obj.Addr{Type: obj.TYPE_BRANCH, Sym: p.From.Sym}
					p.Reg = 0
					p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_TMP}

					rescan = true
				}
			}
		}

		if !rescan {
			break
		}
	}

	// Now that there are no long branches, resolve branch and jump targets.
	// At this point, instruction rewriting which changes the number of
	// instructions will break everything--don't do it!
	for p := cursym.Func.Text; p != nil; p = p.Link {
		switch p.As {
		case AJAL, ABEQ, ABNE, ABLT, ABLTU, ABGE, ABGEU:
			switch p.To.Type {
			case obj.TYPE_BRANCH:
				p.To.Type, p.To.Offset = obj.TYPE_CONST, p.Pcond.Pc-p.Pc
			case obj.TYPE_MEM:
				panic("unhandled type")
			}

		case AAUIPC:
			if p.From.Type == obj.TYPE_BRANCH {
				low, high, err := Split32BitImmediate(p.Pcond.Pc - p.Pc)
				if err != nil {
					ctxt.Diag("%v: jump displacement %d too large", p, p.Pcond.Pc-p.Pc)
				}
				p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: high, Sym: cursym}
				p.Link.From.Offset = low
			}
		}
	}

	// Validate all instructions - this provides nice error messages.
	for p := cursym.Func.Text; p != nil; p = p.Link {
		encodingForProg(p).validate(p)
	}
}

// signExtend sign extends val starting at bit bit.
func signExtend(val int64, bit uint) int64 {
	return val << (64 - bit) >> (64 - bit)
}

// Split32BitImmediate splits a signed 32-bit immediate into a signed 20-bit
// upper immediate and a signed 12-bit lower immediate to be added to the upper
// result. For example, high may be used in LUI and low in a following ADDI to
// generate a full 32-bit constant.
func Split32BitImmediate(imm int64) (low, high int64, err error) {
	if !immIFits(imm, 32) {
		return 0, 0, fmt.Errorf("immediate does not fit in 32-bits: %d", imm)
	}

	// Nothing special needs to be done if the immediate fits in 12-bits.
	if immIFits(imm, 12) {
		return imm, 0, nil
	}

	high = imm >> 12

	// The bottom 12 bits will be treated as signed.
	//
	// If that will result in a negative 12 bit number, add 1 to
	// our upper bits to adjust for the borrow.
	//
	// It is not possible for this increment to overflow. To
	// overflow, the 20 top bits would be 1, and the sign bit for
	// the low 12 bits would be set, in which case the entire 32
	// bit pattern fits in a 12 bit signed value.
	if imm&(1<<11) != 0 {
		high++
	}

	low = signExtend(imm, 12)
	high = signExtend(high, 20)

	return low, high, nil
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

// regF returns a float register.
func regF(r int16) uint32 {
	return regVal(r, REG_F0, REG_F31)
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

// regFAddr extracts the float register from an Addr.
func regFAddr(a obj.Addr) uint32 {
	return regAddr(a, REG_F0, REG_F31)
}

// immIFits reports whether immediate value x fits in nbits bits
// as a signed integer.
func immIFits(x int64, nbits uint) bool {
	nbits--
	var min int64 = -1 << nbits
	var max int64 = 1<<nbits - 1
	return min <= x && x <= max
}

// immUFits reports whether immediate value x fits in nbits bits
// as an unsigned integer.
func immUFits(x int64, nbits uint) bool {
	var max int64 = 1<<nbits - 1
	return 0 <= x && x <= max
}

// immI extracts the signed integer literal of the specified size from an Addr.
func immI(a obj.Addr, nbits uint) uint32 {
	if a.Type != obj.TYPE_CONST {
		panic(fmt.Sprintf("ill typed: %+v", a))
	}
	if !immIFits(a.Offset, nbits) {
		panic(fmt.Sprintf("signed immediate %d in %v cannot fit in %d bits", a.Offset, a, nbits))
	}
	return uint32(a.Offset)
}

// immU extracts the unsigned integer literal of the specified size from an Addr.
func immU(a obj.Addr, nbits uint) uint32 {
	if a.Type != obj.TYPE_CONST {
		panic(fmt.Sprintf("ill typed: %+v", a))
	}
	if !immUFits(a.Offset, nbits) {
		panic(fmt.Sprintf("unsigned immediate %d in %v cannot fit in %d bits", a.Offset, a, nbits))
	}
	return uint32(a.Offset)
}

func wantImmI(p *obj.Prog, pos string, a obj.Addr, nbits uint) {
	if a.Type != obj.TYPE_CONST {
		p.Ctxt.Diag("%v\texpected immediate in %s position but got %s", p, pos, obj.Dconv(p, &a))
		return
	}
	if !immIFits(a.Offset, nbits) {
		p.Ctxt.Diag("%v\tsigned immediate in %s position cannot be larger than %d bits but got %d", p, pos, nbits, a.Offset)
	}
}

func wantImmU(p *obj.Prog, pos string, a obj.Addr, nbits uint) {
	if a.Type != obj.TYPE_CONST {
		p.Ctxt.Diag("%v\texpected immediate in %s position but got %s", p, pos, obj.Dconv(p, &a))
		return
	}
	if !immUFits(a.Offset, nbits) {
		p.Ctxt.Diag("%v\tunsigned immediate in %s position cannot be larger than %d bits but got %d", p, pos, nbits, a.Offset)
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

// wantFloatReg checks that r is a floating-point register.
func wantFloatReg(p *obj.Prog, pos string, r int16) {
	wantReg(p, pos, "float", r, REG_F0, REG_F31)
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

// wantFloatRegAddr checks that a contains a floating-point register.
func wantFloatRegAddr(p *obj.Prog, pos string, a *obj.Addr) {
	wantRegAddr(p, pos, a, "float", REG_F0, REG_F31)
}

// wantEvenJumpOffset checks that the jump offset is a multiple of two.
func wantEvenJumpOffset(p *obj.Prog) {
	if p.To.Offset%1 != 0 {
		p.Ctxt.Diag("%v\tjump offset %v must be even", p, obj.Dconv(p, &p.To))
	}
}

func validateRIII(p *obj.Prog) {
	wantIntRegAddr(p, "from", &p.From)
	wantIntReg(p, "reg", p.Reg)
	wantIntRegAddr(p, "to", &p.To)
}

func validateRFFF(p *obj.Prog) {
	wantFloatRegAddr(p, "from", &p.From)
	wantFloatReg(p, "reg", p.Reg)
	wantFloatRegAddr(p, "to", &p.To)
}

func validateRFFI(p *obj.Prog) {
	wantFloatRegAddr(p, "from", &p.From)
	wantFloatReg(p, "reg", p.Reg)
	wantIntRegAddr(p, "to", &p.To)
}

func validateRFI(p *obj.Prog) {
	wantFloatRegAddr(p, "from", &p.From)
	wantIntRegAddr(p, "to", &p.To)
}

func validateRIF(p *obj.Prog) {
	wantIntRegAddr(p, "from", &p.From)
	wantFloatRegAddr(p, "to", &p.To)
}

func validateRFF(p *obj.Prog) {
	wantFloatRegAddr(p, "from", &p.From)
	wantFloatRegAddr(p, "to", &p.To)
}

func validateII(p *obj.Prog) {
	wantImmI(p, "from", p.From, 12)
	wantIntReg(p, "reg", p.Reg)
	wantIntRegAddr(p, "to", &p.To)
}

func validateIF(p *obj.Prog) {
	wantImmI(p, "from", p.From, 12)
	wantIntReg(p, "reg", p.Reg)
	wantFloatRegAddr(p, "to", &p.To)
}

func validateSI(p *obj.Prog) {
	wantImmI(p, "from", p.From, 12)
	wantIntReg(p, "reg", p.Reg)
	wantIntRegAddr(p, "to", &p.To)
}

func validateSF(p *obj.Prog) {
	wantImmI(p, "from", p.From, 12)
	wantFloatReg(p, "reg", p.Reg)
	wantIntRegAddr(p, "to", &p.To)
}

func validateB(p *obj.Prog) {
	// Offsets are multiples of two, so accept 13 bit immediates for the
	// 12 bit slot. We implicitly drop the least significant bit in encodeB.
	wantEvenJumpOffset(p)
	wantImmI(p, "to", p.To, 13)
	wantIntReg(p, "reg", p.Reg)
	wantIntRegAddr(p, "from", &p.From)
}

func validateU(p *obj.Prog) {
	if p.As == AAUIPC && p.Mark&(NEED_PCREL_ITYPE_RELOC|NEED_PCREL_STYPE_RELOC) != 0 {
		// TODO(sorear): Hack.  The Offset is being used here to temporarily
		// store the relocation addend, not as an actual offset to assemble,
		// so it's OK for it to be out of range.  Is there a more valid way
		// to represent this state?
		return
	}
	wantImmU(p, "from", p.From, 20)
	wantIntRegAddr(p, "to", &p.To)
}

func validateJ(p *obj.Prog) {
	// Offsets are multiples of two, so accept 21 bit immediates for the
	// 20 bit slot. We implicitly drop the least significant bit in encodeJ.
	wantEvenJumpOffset(p)
	wantImmI(p, "to", p.To, 21)
	wantIntRegAddr(p, "from", &p.From)
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

func encodeRFFF(p *obj.Prog) uint32 {
	return encodeR(p, regF(p.Reg), regFAddr(p.From), regFAddr(p.To))
}

func encodeRFFI(p *obj.Prog) uint32 {
	return encodeR(p, regF(p.Reg), regFAddr(p.From), regIAddr(p.To))
}

func encodeRFI(p *obj.Prog) uint32 {
	return encodeR(p, regFAddr(p.From), 0, regIAddr(p.To))
}

func encodeRIF(p *obj.Prog) uint32 {
	return encodeR(p, regIAddr(p.From), 0, regFAddr(p.To))
}

func encodeRFF(p *obj.Prog) uint32 {
	return encodeR(p, regFAddr(p.From), 0, regFAddr(p.To))
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

func encodeIF(p *obj.Prog) uint32 {
	return encodeI(p, regFAddr(p.To))
}

// encodeS encodes an S-type RISC-V instruction.
func encodeS(p *obj.Prog, rs2 uint32) uint32 {
	imm := immI(p.From, 12)
	rs1 := regIAddr(p.To)
	ins := encode(p.As)
	if ins == nil {
		panic("encodeS: could not encode instruction")
	}
	return (imm>>5)<<25 | rs2<<20 | rs1<<15 | ins.funct3<<12 | (imm&0x1f)<<7 | ins.opcode
}

func encodeSI(p *obj.Prog) uint32 {
	return encodeS(p, regI(p.Reg))
}

func encodeSF(p *obj.Prog) uint32 {
	return encodeS(p, regF(p.Reg))
}

// encodeB encodes a B-type RISC-V instruction.
func encodeB(p *obj.Prog) uint32 {
	imm := immI(p.To, 13)
	rs2 := regI(p.Reg)
	rs1 := regIAddr(p.From)
	ins := encode(p.As)
	if ins == nil {
		panic("encodeB: could not encode instruction")
	}
	return (imm>>12)<<31 | ((imm>>5)&0x3f)<<25 | rs2<<20 | rs1<<15 | ins.funct3<<12 | ((imm>>1)&0xf)<<8 | ((imm>>11)&0x1)<<7 | ins.opcode
}

// encodeU encodes a U-type RISC-V instruction.
func encodeU(p *obj.Prog) uint32 {
	// The immediates for encodeU are the upper 20 bits of a 32 bit value.
	// Rather than have the user/compiler generate a 32 bit constant, the
	// bottommost bits of which must all be zero, instead accept just the
	// top bits.
	imm := immU(p.From, 20)
	rd := regIAddr(p.To)
	ins := encode(p.As)
	if ins == nil {
		panic("encodeU: could not encode instruction")
	}
	return imm<<12 | rd<<7 | ins.opcode
}

// encodeJ encodes a J-type RISC-V instruction.
func encodeJ(p *obj.Prog) uint32 {
	imm := immI(p.To, 21)
	rd := regIAddr(p.From)
	ins := encode(p.As)
	if ins == nil {
		panic("encodeJ: could not encode instruction")
	}
	return (imm>>20)<<31 | ((imm>>1)&0x3ff)<<21 | ((imm>>11)&0x1)<<20 | ((imm>>12)&0xff)<<12 | rd<<7 | ins.opcode
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

func EncodeIImmediate(imm int64) (int64, error) {
	if !immIFits(imm, 12) {
		return 0, fmt.Errorf("immediate %#x does not fit in 12 bits", imm)
	}
	return imm << 20, nil
}

func EncodeSImmediate(imm int64) (int64, error) {
	if !immIFits(imm, 12) {
		return 0, fmt.Errorf("immediate %#x does not fit in 12 bits", imm)
	}
	return ((imm >> 5) << 25) | ((imm & 0x1f) << 7), nil
}

func EncodeUImmediate(imm int64) (int64, error) {
	if !immUFits(imm, 20) {
		return 0, fmt.Errorf("immediate %#x does not fit in 20 bits", imm)
	}
	return imm << 12, nil
}

type encoding struct {
	encode   func(*obj.Prog) uint32 // encode returns the machine code for an *obj.Prog
	validate func(*obj.Prog)        // validate validates an *obj.Prog, calling ctxt.Diag for any issues
	length   int                    // length of encoded instruction; 0 for pseudo-ops, 4 otherwise
}

var (
	// Encodings have the following naming convention:
	//
	//  1. the instruction encoding (R/I/S/B/U/J), in lowercase
	//  2. zero or more register operand identifiers (I = integer
	//     register, F = float register), in uppercase
	//  3. the word "Encoding"
	//
	// For example, rIIIEncoding indicates an R-type instruction with two
	// integer register inputs and an integer register output; sFEncoding
	// indicates an S-type instruction with rs2 being a float register.

	rIIIEncoding = encoding{encode: encodeRIII, validate: validateRIII, length: 4}
	rFFFEncoding = encoding{encode: encodeRFFF, validate: validateRFFF, length: 4}
	rFFIEncoding = encoding{encode: encodeRFFI, validate: validateRFFI, length: 4}
	rFIEncoding  = encoding{encode: encodeRFI, validate: validateRFI, length: 4}
	rIFEncoding  = encoding{encode: encodeRIF, validate: validateRIF, length: 4}
	rFFEncoding  = encoding{encode: encodeRFF, validate: validateRFF, length: 4}

	iIEncoding = encoding{encode: encodeII, validate: validateII, length: 4}
	iFEncoding = encoding{encode: encodeIF, validate: validateIF, length: 4}

	sIEncoding = encoding{encode: encodeSI, validate: validateSI, length: 4}
	sFEncoding = encoding{encode: encodeSF, validate: validateSF, length: 4}

	bEncoding = encoding{encode: encodeB, validate: validateB, length: 4}
	uEncoding = encoding{encode: encodeU, validate: validateU, length: 4}
	jEncoding = encoding{encode: encodeJ, validate: validateJ, length: 4}

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
	ALUI & obj.AMask:   uEncoding,
	AAUIPC & obj.AMask: uEncoding,
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

	// 2.5: Control Transfer Instructions
	AJAL & obj.AMask:  jEncoding,
	AJALR & obj.AMask: iIEncoding,
	ABEQ & obj.AMask:  bEncoding,
	ABNE & obj.AMask:  bEncoding,
	ABLT & obj.AMask:  bEncoding,
	ABLTU & obj.AMask: bEncoding,
	ABGE & obj.AMask:  bEncoding,
	ABGEU & obj.AMask: bEncoding,

	// 2.6: Load and Store Instructions
	ALW & obj.AMask:  iIEncoding,
	ALWU & obj.AMask: iIEncoding,
	ALH & obj.AMask:  iIEncoding,
	ALHU & obj.AMask: iIEncoding,
	ALB & obj.AMask:  iIEncoding,
	ALBU & obj.AMask: iIEncoding,
	ASW & obj.AMask:  sIEncoding,
	ASH & obj.AMask:  sIEncoding,
	ASB & obj.AMask:  sIEncoding,

	// 5.2: Integer Computational Instructions (RV64I)
	AADDIW & obj.AMask: iIEncoding,
	ASLLIW & obj.AMask: iIEncoding,
	ASRLIW & obj.AMask: iIEncoding,
	ASRAIW & obj.AMask: iIEncoding,
	AADDW & obj.AMask:  rIIIEncoding,
	ASLLW & obj.AMask:  rIIIEncoding,
	ASRLW & obj.AMask:  rIIIEncoding,
	ASUBW & obj.AMask:  rIIIEncoding,
	ASRAW & obj.AMask:  rIIIEncoding,

	// 5.3: Load and Store Instructions (RV64I)
	ALD & obj.AMask: iIEncoding,
	ASD & obj.AMask: sIEncoding,

	// 7.1: Multiplication Operations
	AMUL & obj.AMask:    rIIIEncoding,
	AMULH & obj.AMask:   rIIIEncoding,
	AMULHU & obj.AMask:  rIIIEncoding,
	AMULHSU & obj.AMask: rIIIEncoding,
	AMULW & obj.AMask:   rIIIEncoding,
	ADIV & obj.AMask:    rIIIEncoding,
	ADIVU & obj.AMask:   rIIIEncoding,
	AREM & obj.AMask:    rIIIEncoding,
	AREMU & obj.AMask:   rIIIEncoding,
	ADIVW & obj.AMask:   rIIIEncoding,
	ADIVUW & obj.AMask:  rIIIEncoding,
	AREMW & obj.AMask:   rIIIEncoding,
	AREMUW & obj.AMask:  rIIIEncoding,

	// 10.1: Base Counters and Timers
	ARDCYCLE & obj.AMask:   iIEncoding,
	ARDTIME & obj.AMask:    iIEncoding,
	ARDINSTRET & obj.AMask: iIEncoding,

	// 11.5: Single-Precision Load and Store Instructions
	AFLW & obj.AMask: iFEncoding,
	AFSW & obj.AMask: sFEncoding,

	// 11.6: Single-Precision Floating-Point Computational Instructions
	AFADDS & obj.AMask:  rFFFEncoding,
	AFSUBS & obj.AMask:  rFFFEncoding,
	AFMULS & obj.AMask:  rFFFEncoding,
	AFDIVS & obj.AMask:  rFFFEncoding,
	AFMINS & obj.AMask:  rFFFEncoding,
	AFMAXS & obj.AMask:  rFFFEncoding,
	AFSQRTS & obj.AMask: rFFFEncoding,

	// 11.7: Single-Precision Floating-Point Conversion and Move Instructions
	AFCVTWS & obj.AMask:  rFIEncoding,
	AFCVTLS & obj.AMask:  rFIEncoding,
	AFCVTSW & obj.AMask:  rIFEncoding,
	AFCVTSL & obj.AMask:  rIFEncoding,
	AFCVTWUS & obj.AMask: rFIEncoding,
	AFCVTLUS & obj.AMask: rFIEncoding,
	AFCVTSWU & obj.AMask: rIFEncoding,
	AFCVTSLU & obj.AMask: rIFEncoding,
	AFSGNJS & obj.AMask:  rFFFEncoding,
	AFSGNJNS & obj.AMask: rFFFEncoding,
	AFSGNJXS & obj.AMask: rFFFEncoding,
	AFMVXS & obj.AMask:   rFIEncoding,
	AFMVSX & obj.AMask:   rIFEncoding,
	AFMVXW & obj.AMask:   rFIEncoding,
	AFMVWX & obj.AMask:   rIFEncoding,

	// 11.8: Single-Precision Floating-Point Compare Instructions
	AFEQS & obj.AMask: rFFIEncoding,
	AFLTS & obj.AMask: rFFIEncoding,
	AFLES & obj.AMask: rFFIEncoding,

	// 12.3: Double-Precision Load and Store Instructions
	AFLD & obj.AMask: iFEncoding,
	AFSD & obj.AMask: sFEncoding,

	// 12.4: Double-Precision Floating-Point Computational Instructions
	AFADDD & obj.AMask:  rFFFEncoding,
	AFSUBD & obj.AMask:  rFFFEncoding,
	AFMULD & obj.AMask:  rFFFEncoding,
	AFDIVD & obj.AMask:  rFFFEncoding,
	AFMIND & obj.AMask:  rFFFEncoding,
	AFMAXD & obj.AMask:  rFFFEncoding,
	AFSQRTD & obj.AMask: rFFFEncoding,

	// 12.5: Double-Precision Floating-Point Conversion and Move Instructions
	AFCVTWD & obj.AMask:  rFIEncoding,
	AFCVTLD & obj.AMask:  rFIEncoding,
	AFCVTDW & obj.AMask:  rIFEncoding,
	AFCVTDL & obj.AMask:  rIFEncoding,
	AFCVTWUD & obj.AMask: rFIEncoding,
	AFCVTLUD & obj.AMask: rFIEncoding,
	AFCVTDWU & obj.AMask: rIFEncoding,
	AFCVTDLU & obj.AMask: rIFEncoding,
	AFCVTSD & obj.AMask:  rFFEncoding,
	AFCVTDS & obj.AMask:  rFFEncoding,
	AFSGNJD & obj.AMask:  rFFFEncoding,
	AFSGNJND & obj.AMask: rFFFEncoding,
	AFSGNJXD & obj.AMask: rFFFEncoding,
	AFMVXD & obj.AMask:   rFIEncoding,
	AFMVDX & obj.AMask:   rIFEncoding,

	// 12.6: Double-Precision Floating-Point Compare Instructions
	AFEQD & obj.AMask: rFFIEncoding,
	AFLTD & obj.AMask: rFFIEncoding,
	AFLED & obj.AMask: rFFIEncoding,

	// Privileged ISA

	// 3.2.1: Environment Call and Breakpoint
	AECALL & obj.AMask:  iIEncoding,
	AEBREAK & obj.AMask: iIEncoding,

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
		switch p.As {
		case AJALR:
			if p.To.Sym != nil {
				// This is a CALL/JMP. We add a relocation only
				// for linker stack checking. No actual
				// relocation is needed.
				rel := obj.Addrel(cursym)
				rel.Off = int32(p.Pc)
				rel.Siz = 4
				rel.Sym = p.To.Sym
				rel.Add = p.To.Offset
				rel.Type = objabi.R_CALLRISCV
			}
		case AAUIPC:
			var rt objabi.RelocType
			if p.Mark&NEED_PCREL_ITYPE_RELOC == NEED_PCREL_ITYPE_RELOC {
				rt = objabi.R_RISCV_PCREL_ITYPE
			} else if p.Mark&NEED_PCREL_STYPE_RELOC == NEED_PCREL_STYPE_RELOC {
				rt = objabi.R_RISCV_PCREL_STYPE
			} else {
				break
			}
			if p.Link == nil {
				ctxt.Diag("AUIPC needing PC-relative reloc missing following instruction")
				break
			}
			if p.From.Sym == nil {
				ctxt.Diag("AUIPC needing PC-relative reloc missing symbol")
				break
			}

			// The relocation offset can be larger than the maximum
			// size of an AUIPC, so zero p.From.Offset to avoid any
			// attempt to assemble it.
			rel := obj.Addrel(cursym)
			rel.Off = int32(p.Pc)
			rel.Siz = 8
			rel.Sym = p.From.Sym
			rel.Add = p.From.Offset
			p.From.Offset = 0
			rel.Type = rt
		}

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
