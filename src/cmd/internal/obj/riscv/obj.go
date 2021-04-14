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

func buildop(ctxt *obj.Link) {}

// jalrToSym replaces p with a set of Progs needed to jump to the Sym in p.
// lr is the link register to use for the JALR.
// p must be a CALL, JMP or RET.
func jalrToSym(ctxt *obj.Link, p *obj.Prog, newprog obj.ProgAlloc, lr int16) *obj.Prog {
	if p.As != obj.ACALL && p.As != obj.AJMP && p.As != obj.ARET && p.As != obj.ADUFFZERO && p.As != obj.ADUFFCOPY {
		ctxt.Diag("unexpected Prog in jalrToSym: %v", p)
		return p
	}

	// TODO(jsing): Consider using a single JAL instruction and teaching
	// the linker to provide trampolines for the case where the destination
	// offset is too large. This would potentially reduce instructions for
	// the common case, but would require three instructions to go via the
	// trampoline.

	to := p.To

	p.As = AAUIPC
	p.Mark |= NEED_PCREL_ITYPE_RELOC
	p.SetFrom3(obj.Addr{Type: obj.TYPE_CONST, Offset: to.Offset, Sym: to.Sym})
	p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: 0}
	p.Reg = 0
	p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_TMP}
	p = obj.Appendp(p, newprog)

	// Leave Sym only for the CALL reloc in assemble.
	p.As = AJALR
	p.From.Type = obj.TYPE_REG
	p.From.Reg = lr
	p.Reg = 0
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_TMP
	p.To.Sym = to.Sym

	return p
}

// progedit is called individually for each *obj.Prog. It normalizes instruction
// formats and eliminates as many pseudo-instructions as possible.
func progedit(ctxt *obj.Link, p *obj.Prog, newprog obj.ProgAlloc) {

	// Expand binary instructions to ternary ones.
	if p.Reg == 0 {
		switch p.As {
		case AADDI, ASLTI, ASLTIU, AANDI, AORI, AXORI, ASLLI, ASRLI, ASRAI,
			AADD, AAND, AOR, AXOR, ASLL, ASRL, ASUB, ASRA,
			AMUL, AMULH, AMULHU, AMULHSU, AMULW, ADIV, ADIVU, ADIVW, ADIVUW,
			AREM, AREMU, AREMW, AREMUW:
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
	case obj.AJMP:
		// Turn JMP into JAL ZERO or JALR ZERO.
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_ZERO

		switch p.To.Type {
		case obj.TYPE_BRANCH:
			p.As = AJAL
		case obj.TYPE_MEM:
			switch p.To.Name {
			case obj.NAME_NONE:
				p.As = AJALR
			case obj.NAME_EXTERN:
				// Handled in preprocess.
			default:
				ctxt.Diag("unsupported name %d for %v", p.To.Name, p)
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
		default:
			ctxt.Diag("unknown destination type %+v in CALL: %v", p.To.Type, p)
		}

	case obj.AUNDEF:
		p.As = AEBREAK

	case ASCALL:
		// SCALL is the old name for ECALL.
		p.As = AECALL

	case ASBREAK:
		// SBREAK is the old name for EBREAK.
		p.As = AEBREAK
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
			p.From.Reg = addrToReg(p.From)

		case obj.NAME_EXTERN, obj.NAME_STATIC:
			// AUIPC $off_hi, R
			// L $off_lo, R
			as := p.As
			to := p.To

			p.As = AAUIPC
			p.Mark |= NEED_PCREL_ITYPE_RELOC
			p.SetFrom3(obj.Addr{Type: obj.TYPE_CONST, Offset: p.From.Offset, Sym: p.From.Sym})
			p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: 0}
			p.Reg = 0
			p.To = obj.Addr{Type: obj.TYPE_REG, Reg: to.Reg}
			p = obj.Appendp(p, newprog)

			p.As = movToLoad(as)
			p.From = obj.Addr{Type: obj.TYPE_MEM, Reg: to.Reg, Offset: 0}
			p.To = to

		default:
			ctxt.Diag("unsupported name %d for %v", p.From.Name, p)
		}

	case obj.TYPE_REG:
		switch p.To.Type {
		case obj.TYPE_REG:
			switch p.As {
			case AMOV, AMOVB, AMOVH, AMOVW, AMOVBU, AMOVHU, AMOVWU, AMOVF, AMOVD:
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
				p.As = movToStore(p.As)
				p.To.Reg = addrToReg(p.To)

			case obj.NAME_EXTERN:
				// AUIPC $off_hi, TMP
				// S $off_lo, TMP, R
				as := p.As
				from := p.From

				p.As = AAUIPC
				p.Mark |= NEED_PCREL_STYPE_RELOC
				p.SetFrom3(obj.Addr{Type: obj.TYPE_CONST, Offset: p.To.Offset, Sym: p.To.Sym})
				p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: 0}
				p.Reg = 0
				p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_TMP}
				p = obj.Appendp(p, newprog)

				p.As = movToStore(as)
				p.From = from
				p.To = obj.Addr{Type: obj.TYPE_MEM, Reg: REG_TMP, Offset: 0}

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

			p.As = AAUIPC
			p.Mark |= NEED_PCREL_ITYPE_RELOC
			p.SetFrom3(obj.Addr{Type: obj.TYPE_CONST, Offset: p.From.Offset, Sym: p.From.Sym})
			p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: 0}
			p.Reg = 0
			p.To = to
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

// InvertBranch inverts the condition of a conditional branch.
func InvertBranch(as obj.As) obj.As {
	switch as {
	case ABEQ:
		return ABNE
	case ABEQZ:
		return ABNEZ
	case ABGE:
		return ABLT
	case ABGEU:
		return ABLTU
	case ABGEZ:
		return ABLTZ
	case ABGT:
		return ABLE
	case ABGTU:
		return ABLEU
	case ABGTZ:
		return ABLEZ
	case ABLE:
		return ABGT
	case ABLEU:
		return ABGTU
	case ABLEZ:
		return ABGTZ
	case ABLT:
		return ABGE
	case ABLTU:
		return ABGEU
	case ABLTZ:
		return ABGEZ
	case ABNE:
		return ABEQ
	case ABNEZ:
		return ABEQZ
	default:
		panic("InvertBranch: not a branch")
	}
}

// containsCall reports whether the symbol contains a CALL (or equivalent)
// instruction. Must be called after progedit.
func containsCall(sym *obj.LSym) bool {
	// CALLs are CALL or JAL(R) with link register LR.
	for p := sym.Func().Text; p != nil; p = p.Link {
		switch p.As {
		case obj.ACALL, obj.ADUFFZERO, obj.ADUFFCOPY:
			return true
		case AJAL, AJALR:
			if p.From.Type == obj.TYPE_REG && p.From.Reg == REG_LR {
				return true
			}
		}
	}

	return false
}

// setPCs sets the Pc field in all instructions reachable from p.
// It uses pc as the initial value.
func setPCs(p *obj.Prog, pc int64) {
	for ; p != nil; p = p.Link {
		p.Pc = pc
		for _, ins := range instructionsForProg(p) {
			pc += int64(ins.length())
		}
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

// preprocess generates prologue and epilogue code, computes PC-relative branch
// and jump offsets, and resolves pseudo-registers.
//
// preprocess is called once per linker symbol.
//
// When preprocess finishes, all instructions in the symbol are either
// concrete, real RISC-V instructions or directive pseudo-ops like TEXT,
// PCDATA, and FUNCDATA.
func preprocess(ctxt *obj.Link, cursym *obj.LSym, newprog obj.ProgAlloc) {
	if cursym.Func().Text == nil || cursym.Func().Text.Link == nil {
		return
	}

	// Generate the prologue.
	text := cursym.Func().Text
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

	if !containsCall(cursym) {
		text.From.Sym.Set(obj.AttrLeaf, true)
		if stacksize == 0 {
			// A leaf function with no locals has no frame.
			text.From.Sym.Set(obj.AttrNoFrame, true)
		}
	}

	// Save LR unless there is no frame.
	if !text.From.Sym.NoFrame() {
		stacksize += ctxt.FixedFrameSize()
	}

	cursym.Func().Args = text.To.Val.(int32)
	cursym.Func().Locals = int32(stacksize)

	prologue := text

	if !cursym.Func().Text.From.Sym.NoSplit() {
		prologue = stacksplit(ctxt, prologue, cursym, newprog, stacksize) // emit split check
	}

	if stacksize != 0 {
		prologue = ctxt.StartUnsafePoint(prologue, newprog)

		// Actually save LR.
		prologue = obj.Appendp(prologue, newprog)
		prologue.As = AMOV
		prologue.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_LR}
		prologue.To = obj.Addr{Type: obj.TYPE_MEM, Reg: REG_SP, Offset: -stacksize}

		// Insert stack adjustment.
		prologue = obj.Appendp(prologue, newprog)
		prologue.As = AADDI
		prologue.From = obj.Addr{Type: obj.TYPE_CONST, Offset: -stacksize}
		prologue.Reg = REG_SP
		prologue.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_SP}
		prologue.Spadj = int32(stacksize)

		prologue = ctxt.EndUnsafePoint(prologue, newprog, -1)
	}

	if cursym.Func().Text.From.Sym.Wrapper() {
		// if(g->panic != nil && g->panic->argp == FP) g->panic->argp = bottom-of-frame
		//
		//   MOV g_panic(g), X11
		//   BNE X11, ZERO, adjust
		// end:
		//   NOP
		// ...rest of function..
		// adjust:
		//   MOV panic_argp(X11), X12
		//   ADD $(autosize+FIXED_FRAME), SP, X13
		//   BNE X12, X13, end
		//   ADD $FIXED_FRAME, SP, X12
		//   MOV X12, panic_argp(X11)
		//   JMP end
		//
		// The NOP is needed to give the jumps somewhere to land.

		ldpanic := obj.Appendp(prologue, newprog)

		ldpanic.As = AMOV
		ldpanic.From = obj.Addr{Type: obj.TYPE_MEM, Reg: REGG, Offset: 4 * int64(ctxt.Arch.PtrSize)} // G.panic
		ldpanic.Reg = 0
		ldpanic.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_X11}

		bneadj := obj.Appendp(ldpanic, newprog)
		bneadj.As = ABNE
		bneadj.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_X11}
		bneadj.Reg = REG_ZERO
		bneadj.To.Type = obj.TYPE_BRANCH

		endadj := obj.Appendp(bneadj, newprog)
		endadj.As = obj.ANOP

		last := endadj
		for last.Link != nil {
			last = last.Link
		}

		getargp := obj.Appendp(last, newprog)
		getargp.As = AMOV
		getargp.From = obj.Addr{Type: obj.TYPE_MEM, Reg: REG_X11, Offset: 0} // Panic.argp
		getargp.Reg = 0
		getargp.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_X12}

		bneadj.To.SetTarget(getargp)

		calcargp := obj.Appendp(getargp, newprog)
		calcargp.As = AADDI
		calcargp.From = obj.Addr{Type: obj.TYPE_CONST, Offset: stacksize + ctxt.FixedFrameSize()}
		calcargp.Reg = REG_SP
		calcargp.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_X13}

		testargp := obj.Appendp(calcargp, newprog)
		testargp.As = ABNE
		testargp.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_X12}
		testargp.Reg = REG_X13
		testargp.To.Type = obj.TYPE_BRANCH
		testargp.To.SetTarget(endadj)

		adjargp := obj.Appendp(testargp, newprog)
		adjargp.As = AADDI
		adjargp.From = obj.Addr{Type: obj.TYPE_CONST, Offset: int64(ctxt.Arch.PtrSize)}
		adjargp.Reg = REG_SP
		adjargp.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_X12}

		setargp := obj.Appendp(adjargp, newprog)
		setargp.As = AMOV
		setargp.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_X12}
		setargp.Reg = 0
		setargp.To = obj.Addr{Type: obj.TYPE_MEM, Reg: REG_X11, Offset: 0} // Panic.argp

		godone := obj.Appendp(setargp, newprog)
		godone.As = AJAL
		godone.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_ZERO}
		godone.To.Type = obj.TYPE_BRANCH
		godone.To.SetTarget(endadj)
	}

	// Update stack-based offsets.
	for p := cursym.Func().Text; p != nil; p = p.Link {
		stackOffset(&p.From, stacksize)
		stackOffset(&p.To, stacksize)
	}

	// Additional instruction rewriting.
	for p := cursym.Func().Text; p != nil; p = p.Link {
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

		case obj.ACALL, obj.ADUFFZERO, obj.ADUFFCOPY:
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

		case obj.ARET:
			// Replace RET with epilogue.
			retJMP := p.To.Sym

			if stacksize != 0 {
				// Restore LR.
				p.As = AMOV
				p.From = obj.Addr{Type: obj.TYPE_MEM, Reg: REG_SP, Offset: 0}
				p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_LR}
				p = obj.Appendp(p, newprog)

				p.As = AADDI
				p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: stacksize}
				p.Reg = REG_SP
				p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_SP}
				p.Spadj = int32(-stacksize)
				p = obj.Appendp(p, newprog)
			}

			if retJMP != nil {
				p.As = obj.ARET
				p.To.Sym = retJMP
				p = jalrToSym(ctxt, p, newprog, REG_ZERO)
			} else {
				p.As = AJALR
				p.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_ZERO}
				p.Reg = 0
				p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_LR}
			}

			// "Add back" the stack removed in the previous instruction.
			//
			// This is to avoid confusing pctospadj, which sums
			// Spadj from function entry to each PC, and shouldn't
			// count adjustments from earlier epilogues, since they
			// won't affect later PCs.
			p.Spadj = int32(stacksize)

		case AADDI:
			// Refine Spadjs account for adjustment via ADDI instruction.
			if p.To.Type == obj.TYPE_REG && p.To.Reg == REG_SP && p.From.Type == obj.TYPE_CONST {
				p.Spadj = int32(-p.From.Offset)
			}
		}
	}

	// Rewrite MOV pseudo-instructions. This cannot be done in
	// progedit, as SP offsets need to be applied before we split
	// up some of the Addrs.
	for p := cursym.Func().Text; p != nil; p = p.Link {
		switch p.As {
		case AMOV, AMOVB, AMOVH, AMOVW, AMOVBU, AMOVHU, AMOVWU, AMOVF, AMOVD:
			rewriteMOV(ctxt, newprog, p)
		}
	}

	// Split immediates larger than 12-bits.
	for p := cursym.Func().Text; p != nil; p = p.Link {
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
				ctxt.Diag("unsupported instruction %v for splitting", q)
			}
			p.Spadj = q.Spadj
			p.To = q.To
			p.Reg = q.Reg
			p.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_TMP}

		// <load> $imm, REG, TO (load $imm+(REG), TO)
		case ALD, ALB, ALH, ALW, ALBU, ALHU, ALWU, AFLW, AFLD:
			low, high, err := Split32BitImmediate(p.From.Offset)
			if err != nil {
				ctxt.Diag("%v: constant %d too large", p, p.From.Offset)
			}
			if high == 0 {
				break // no need to split
			}
			q := *p

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
			p.Reg = q.From.Reg
			p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_TMP}
			p = obj.Appendp(p, newprog)

			p.As = q.As
			p.To = q.To
			p.From = obj.Addr{Type: obj.TYPE_MEM, Reg: REG_TMP, Offset: low}
			p.Reg = obj.REG_NONE

		// <store> $imm, REG, TO (store $imm+(TO), REG)
		case ASD, ASB, ASH, ASW, AFSW, AFSD:
			low, high, err := Split32BitImmediate(p.To.Offset)
			if err != nil {
				ctxt.Diag("%v: constant %d too large", p, p.To.Offset)
			}
			if high == 0 {
				break // no need to split
			}
			q := *p

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
			p.From = obj.Addr{Type: obj.TYPE_REG, Reg: q.From.Reg, Offset: 0}
			p.To = obj.Addr{Type: obj.TYPE_MEM, Reg: REG_TMP, Offset: low}
		}
	}

	// Compute instruction addresses.  Once we do that, we need to check for
	// overextended jumps and branches.  Within each iteration, Pc differences
	// are always lower bounds (since the program gets monotonically longer,
	// a fixed point will be reached).  No attempt to handle functions > 2GiB.
	for {
		rescan := false
		setPCs(cursym.Func().Text, 0)

		for p := cursym.Func().Text; p != nil; p = p.Link {
			switch p.As {
			case ABEQ, ABEQZ, ABGE, ABGEU, ABGEZ, ABGT, ABGTU, ABGTZ, ABLE, ABLEU, ABLEZ, ABLT, ABLTU, ABLTZ, ABNE, ABNEZ:
				if p.To.Type != obj.TYPE_BRANCH {
					panic("assemble: instruction with branch-like opcode lacks destination")
				}
				offset := p.To.Target().Pc - p.Pc
				if offset < -4096 || 4096 <= offset {
					// Branch is long.  Replace it with a jump.
					jmp := obj.Appendp(p, newprog)
					jmp.As = AJAL
					jmp.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_ZERO}
					jmp.To = obj.Addr{Type: obj.TYPE_BRANCH}
					jmp.To.SetTarget(p.To.Target())

					p.As = InvertBranch(p.As)
					p.To.SetTarget(jmp.Link)

					// We may have made previous branches too long,
					// so recheck them.
					rescan = true
				}
			case AJAL:
				if p.To.Target() == nil {
					panic("intersymbol jumps should be expressed as AUIPC+JALR")
				}
				offset := p.To.Target().Pc - p.Pc
				if offset < -(1<<20) || (1<<20) <= offset {
					// Replace with 2-instruction sequence. This assumes
					// that TMP is not live across J instructions, since
					// it is reserved by SSA.
					jmp := obj.Appendp(p, newprog)
					jmp.As = AJALR
					jmp.From = p.From
					jmp.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_TMP}

					// p.From is not generally valid, however will be
					// fixed up in the next loop.
					p.As = AAUIPC
					p.From = obj.Addr{Type: obj.TYPE_BRANCH, Sym: p.From.Sym}
					p.From.SetTarget(p.To.Target())
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
	for p := cursym.Func().Text; p != nil; p = p.Link {
		switch p.As {
		case ABEQ, ABEQZ, ABGE, ABGEU, ABGEZ, ABGT, ABGTU, ABGTZ, ABLE, ABLEU, ABLEZ, ABLT, ABLTU, ABLTZ, ABNE, ABNEZ, AJAL:
			switch p.To.Type {
			case obj.TYPE_BRANCH:
				p.To.Type, p.To.Offset = obj.TYPE_CONST, p.To.Target().Pc-p.Pc
			case obj.TYPE_MEM:
				panic("unhandled type")
			}

		case AAUIPC:
			if p.From.Type == obj.TYPE_BRANCH {
				low, high, err := Split32BitImmediate(p.From.Target().Pc - p.Pc)
				if err != nil {
					ctxt.Diag("%v: jump displacement %d too large", p, p.To.Target().Pc-p.Pc)
				}
				p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: high, Sym: cursym}
				p.Link.From.Offset = low
			}
		}
	}

	// Validate all instructions - this provides nice error messages.
	for p := cursym.Func().Text; p != nil; p = p.Link {
		for _, ins := range instructionsForProg(p) {
			ins.validate(ctxt)
		}
	}
}

func stacksplit(ctxt *obj.Link, p *obj.Prog, cursym *obj.LSym, newprog obj.ProgAlloc, framesize int64) *obj.Prog {
	// Leaf function with no frame is effectively NOSPLIT.
	if framesize == 0 {
		return p
	}

	// MOV	g_stackguard(g), X10
	p = obj.Appendp(p, newprog)
	p.As = AMOV
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = REGG
	p.From.Offset = 2 * int64(ctxt.Arch.PtrSize) // G.stackguard0
	if cursym.CFunc() {
		p.From.Offset = 3 * int64(ctxt.Arch.PtrSize) // G.stackguard1
	}
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_X10

	var to_done, to_more *obj.Prog

	if framesize <= objabi.StackSmall {
		// small stack: SP < stackguard
		//	BLTU	SP, stackguard, done
		p = obj.Appendp(p, newprog)
		p.As = ABLTU
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_X10
		p.Reg = REG_SP
		p.To.Type = obj.TYPE_BRANCH
		to_done = p
	} else if framesize <= objabi.StackBig {
		// large stack: SP-framesize < stackguard-StackSmall
		//	ADD	$-(framesize-StackSmall), SP, X11
		//	BLTU	X11, stackguard, done
		p = obj.Appendp(p, newprog)
		// TODO(sorear): logic inconsistent with comment, but both match all non-x86 arches
		p.As = AADDI
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = -(int64(framesize) - objabi.StackSmall)
		p.Reg = REG_SP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_X11

		p = obj.Appendp(p, newprog)
		p.As = ABLTU
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_X10
		p.Reg = REG_X11
		p.To.Type = obj.TYPE_BRANCH
		to_done = p
	} else {
		// Such a large stack we need to protect against wraparound.
		// If SP is close to zero:
		//	SP-stackguard+StackGuard <= framesize + (StackGuard-StackSmall)
		// The +StackGuard on both sides is required to keep the left side positive:
		// SP is allowed to be slightly below stackguard. See stack.h.
		//
		// Preemption sets stackguard to StackPreempt, a very large value.
		// That breaks the math above, so we have to check for that explicitly.
		//	// stackguard is X10
		//	MOV	$StackPreempt, X11
		//	BEQ	X10, X11, more
		//	ADD	$StackGuard, SP, X11
		//	SUB	X10, X11
		//	MOV	$(framesize+(StackGuard-StackSmall)), X10
		//	BGTU	X11, X10, done
		p = obj.Appendp(p, newprog)
		p.As = AMOV
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = objabi.StackPreempt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_X11

		p = obj.Appendp(p, newprog)
		to_more = p
		p.As = ABEQ
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_X10
		p.Reg = REG_X11
		p.To.Type = obj.TYPE_BRANCH

		p = obj.Appendp(p, newprog)
		p.As = AADDI
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(objabi.StackGuard)
		p.Reg = REG_SP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_X11

		p = obj.Appendp(p, newprog)
		p.As = ASUB
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_X10
		p.Reg = REG_X11
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_X11

		p = obj.Appendp(p, newprog)
		p.As = AMOV
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(framesize) + int64(objabi.StackGuard) - objabi.StackSmall
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_X10

		p = obj.Appendp(p, newprog)
		p.As = ABLTU
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_X10
		p.Reg = REG_X11
		p.To.Type = obj.TYPE_BRANCH
		to_done = p
	}

	p = ctxt.EmitEntryLiveness(cursym, p, newprog)

	// CALL runtime.morestack(SB)
	p = obj.Appendp(p, newprog)
	p.As = obj.ACALL
	p.To.Type = obj.TYPE_BRANCH
	if cursym.CFunc() {
		p.To.Sym = ctxt.Lookup("runtime.morestackc")
	} else if !cursym.Func().Text.From.Sym.NeedCtxt() {
		p.To.Sym = ctxt.Lookup("runtime.morestack_noctxt")
	} else {
		p.To.Sym = ctxt.Lookup("runtime.morestack")
	}
	if to_more != nil {
		to_more.To.SetTarget(p)
	}
	p = jalrToSym(ctxt, p, newprog, REG_X5)

	// JMP start
	p = obj.Appendp(p, newprog)
	p.As = AJAL
	p.To = obj.Addr{Type: obj.TYPE_BRANCH}
	p.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_ZERO}
	p.To.SetTarget(cursym.Func().Text.Link)

	// placeholder for to_done's jump target
	p = obj.Appendp(p, newprog)
	p.As = obj.ANOP // zero-width place holder
	to_done.To.SetTarget(p)

	return p
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

func regVal(r, min, max uint32) uint32 {
	if r < min || r > max {
		panic(fmt.Sprintf("register out of range, want %d < %d < %d", min, r, max))
	}
	return r - min
}

// regI returns an integer register.
func regI(r uint32) uint32 {
	return regVal(r, REG_X0, REG_X31)
}

// regF returns a float register.
func regF(r uint32) uint32 {
	return regVal(r, REG_F0, REG_F31)
}

// regAddr extracts a register from an Addr.
func regAddr(a obj.Addr, min, max uint32) uint32 {
	if a.Type != obj.TYPE_REG {
		panic(fmt.Sprintf("ill typed: %+v", a))
	}
	return regVal(uint32(a.Reg), min, max)
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

// immI extracts the signed integer of the specified size from an immediate.
func immI(as obj.As, imm int64, nbits uint) uint32 {
	if !immIFits(imm, nbits) {
		panic(fmt.Sprintf("%v\tsigned immediate %d cannot fit in %d bits", as, imm, nbits))
	}
	return uint32(imm)
}

func wantImmI(ctxt *obj.Link, as obj.As, imm int64, nbits uint) {
	if !immIFits(imm, nbits) {
		ctxt.Diag("%v\tsigned immediate cannot be larger than %d bits but got %d", as, nbits, imm)
	}
}

func wantReg(ctxt *obj.Link, as obj.As, pos string, descr string, r, min, max uint32) {
	if r < min || r > max {
		var suffix string
		if r != obj.REG_NONE {
			suffix = fmt.Sprintf(" but got non-%s register %s", descr, RegName(int(r)))
		}
		ctxt.Diag("%v\texpected %s register in %s position%s", as, descr, pos, suffix)
	}
}

func wantNoneReg(ctxt *obj.Link, as obj.As, pos string, r uint32) {
	if r != obj.REG_NONE {
		ctxt.Diag("%v\texpected no register in %s but got register %s", as, pos, RegName(int(r)))
	}
}

// wantIntReg checks that r is an integer register.
func wantIntReg(ctxt *obj.Link, as obj.As, pos string, r uint32) {
	wantReg(ctxt, as, pos, "integer", r, REG_X0, REG_X31)
}

// wantFloatReg checks that r is a floating-point register.
func wantFloatReg(ctxt *obj.Link, as obj.As, pos string, r uint32) {
	wantReg(ctxt, as, pos, "float", r, REG_F0, REG_F31)
}

// wantEvenOffset checks that the offset is a multiple of two.
func wantEvenOffset(ctxt *obj.Link, as obj.As, offset int64) {
	if offset%1 != 0 {
		ctxt.Diag("%v\tjump offset %v must be even", as, offset)
	}
}

func validateRIII(ctxt *obj.Link, ins *instruction) {
	wantIntReg(ctxt, ins.as, "rd", ins.rd)
	wantIntReg(ctxt, ins.as, "rs1", ins.rs1)
	wantIntReg(ctxt, ins.as, "rs2", ins.rs2)
}

func validateRFFF(ctxt *obj.Link, ins *instruction) {
	wantFloatReg(ctxt, ins.as, "rd", ins.rd)
	wantFloatReg(ctxt, ins.as, "rs1", ins.rs1)
	wantFloatReg(ctxt, ins.as, "rs2", ins.rs2)
}

func validateRFFI(ctxt *obj.Link, ins *instruction) {
	wantIntReg(ctxt, ins.as, "rd", ins.rd)
	wantFloatReg(ctxt, ins.as, "rs1", ins.rs1)
	wantFloatReg(ctxt, ins.as, "rs2", ins.rs2)
}

func validateRFI(ctxt *obj.Link, ins *instruction) {
	wantIntReg(ctxt, ins.as, "rd", ins.rd)
	wantNoneReg(ctxt, ins.as, "rs1", ins.rs1)
	wantFloatReg(ctxt, ins.as, "rs2", ins.rs2)
}

func validateRIF(ctxt *obj.Link, ins *instruction) {
	wantFloatReg(ctxt, ins.as, "rd", ins.rd)
	wantNoneReg(ctxt, ins.as, "rs1", ins.rs1)
	wantIntReg(ctxt, ins.as, "rs2", ins.rs2)
}

func validateRFF(ctxt *obj.Link, ins *instruction) {
	wantFloatReg(ctxt, ins.as, "rd", ins.rd)
	wantNoneReg(ctxt, ins.as, "rs1", ins.rs1)
	wantFloatReg(ctxt, ins.as, "rs2", ins.rs2)
}

func validateII(ctxt *obj.Link, ins *instruction) {
	wantImmI(ctxt, ins.as, ins.imm, 12)
	wantIntReg(ctxt, ins.as, "rd", ins.rd)
	wantIntReg(ctxt, ins.as, "rs1", ins.rs1)
}

func validateIF(ctxt *obj.Link, ins *instruction) {
	wantImmI(ctxt, ins.as, ins.imm, 12)
	wantFloatReg(ctxt, ins.as, "rd", ins.rd)
	wantIntReg(ctxt, ins.as, "rs1", ins.rs1)
}

func validateSI(ctxt *obj.Link, ins *instruction) {
	wantImmI(ctxt, ins.as, ins.imm, 12)
	wantIntReg(ctxt, ins.as, "rd", ins.rd)
	wantIntReg(ctxt, ins.as, "rs1", ins.rs1)
}

func validateSF(ctxt *obj.Link, ins *instruction) {
	wantImmI(ctxt, ins.as, ins.imm, 12)
	wantIntReg(ctxt, ins.as, "rd", ins.rd)
	wantFloatReg(ctxt, ins.as, "rs1", ins.rs1)
}

func validateB(ctxt *obj.Link, ins *instruction) {
	// Offsets are multiples of two, so accept 13 bit immediates for the
	// 12 bit slot. We implicitly drop the least significant bit in encodeB.
	wantEvenOffset(ctxt, ins.as, ins.imm)
	wantImmI(ctxt, ins.as, ins.imm, 13)
	wantNoneReg(ctxt, ins.as, "rd", ins.rd)
	wantIntReg(ctxt, ins.as, "rs1", ins.rs1)
	wantIntReg(ctxt, ins.as, "rs2", ins.rs2)
}

func validateU(ctxt *obj.Link, ins *instruction) {
	wantImmI(ctxt, ins.as, ins.imm, 20)
	wantIntReg(ctxt, ins.as, "rd", ins.rd)
	wantNoneReg(ctxt, ins.as, "rs1", ins.rs1)
	wantNoneReg(ctxt, ins.as, "rs2", ins.rs2)
}

func validateJ(ctxt *obj.Link, ins *instruction) {
	// Offsets are multiples of two, so accept 21 bit immediates for the
	// 20 bit slot. We implicitly drop the least significant bit in encodeJ.
	wantEvenOffset(ctxt, ins.as, ins.imm)
	wantImmI(ctxt, ins.as, ins.imm, 21)
	wantIntReg(ctxt, ins.as, "rd", ins.rd)
	wantNoneReg(ctxt, ins.as, "rs1", ins.rs1)
	wantNoneReg(ctxt, ins.as, "rs2", ins.rs2)
}

func validateRaw(ctxt *obj.Link, ins *instruction) {
	// Treat the raw value specially as a 32-bit unsigned integer.
	// Nobody wants to enter negative machine code.
	if ins.imm < 0 || 1<<32 <= ins.imm {
		ctxt.Diag("%v\timmediate in raw position cannot be larger than 32 bits but got %d", ins.as, ins.imm)
	}
}

// encodeR encodes an R-type RISC-V instruction.
func encodeR(as obj.As, rs1, rs2, rd, funct3, funct7 uint32) uint32 {
	enc := encode(as)
	if enc == nil {
		panic("encodeR: could not encode instruction")
	}
	if enc.rs2 != 0 && rs2 != 0 {
		panic("encodeR: instruction uses rs2, but rs2 was nonzero")
	}
	return funct7<<25 | enc.funct7<<25 | enc.rs2<<20 | rs2<<20 | rs1<<15 | enc.funct3<<12 | funct3<<12 | rd<<7 | enc.opcode
}

func encodeRIII(ins *instruction) uint32 {
	return encodeR(ins.as, regI(ins.rs1), regI(ins.rs2), regI(ins.rd), ins.funct3, ins.funct7)
}

func encodeRFFF(ins *instruction) uint32 {
	return encodeR(ins.as, regF(ins.rs1), regF(ins.rs2), regF(ins.rd), ins.funct3, ins.funct7)
}

func encodeRFFI(ins *instruction) uint32 {
	return encodeR(ins.as, regF(ins.rs1), regF(ins.rs2), regI(ins.rd), ins.funct3, ins.funct7)
}

func encodeRFI(ins *instruction) uint32 {
	return encodeR(ins.as, regF(ins.rs2), 0, regI(ins.rd), ins.funct3, ins.funct7)
}

func encodeRIF(ins *instruction) uint32 {
	return encodeR(ins.as, regI(ins.rs2), 0, regF(ins.rd), ins.funct3, ins.funct7)
}

func encodeRFF(ins *instruction) uint32 {
	return encodeR(ins.as, regF(ins.rs2), 0, regF(ins.rd), ins.funct3, ins.funct7)
}

// encodeI encodes an I-type RISC-V instruction.
func encodeI(as obj.As, rs1, rd, imm uint32) uint32 {
	enc := encode(as)
	if enc == nil {
		panic("encodeI: could not encode instruction")
	}
	imm |= uint32(enc.csr)
	return imm<<20 | rs1<<15 | enc.funct3<<12 | rd<<7 | enc.opcode
}

func encodeII(ins *instruction) uint32 {
	return encodeI(ins.as, regI(ins.rs1), regI(ins.rd), uint32(ins.imm))
}

func encodeIF(ins *instruction) uint32 {
	return encodeI(ins.as, regI(ins.rs1), regF(ins.rd), uint32(ins.imm))
}

// encodeS encodes an S-type RISC-V instruction.
func encodeS(as obj.As, rs1, rs2, imm uint32) uint32 {
	enc := encode(as)
	if enc == nil {
		panic("encodeS: could not encode instruction")
	}
	return (imm>>5)<<25 | rs2<<20 | rs1<<15 | enc.funct3<<12 | (imm&0x1f)<<7 | enc.opcode
}

func encodeSI(ins *instruction) uint32 {
	return encodeS(ins.as, regI(ins.rd), regI(ins.rs1), uint32(ins.imm))
}

func encodeSF(ins *instruction) uint32 {
	return encodeS(ins.as, regI(ins.rd), regF(ins.rs1), uint32(ins.imm))
}

// encodeB encodes a B-type RISC-V instruction.
func encodeB(ins *instruction) uint32 {
	imm := immI(ins.as, ins.imm, 13)
	rs2 := regI(ins.rs1)
	rs1 := regI(ins.rs2)
	enc := encode(ins.as)
	if enc == nil {
		panic("encodeB: could not encode instruction")
	}
	return (imm>>12)<<31 | ((imm>>5)&0x3f)<<25 | rs2<<20 | rs1<<15 | enc.funct3<<12 | ((imm>>1)&0xf)<<8 | ((imm>>11)&0x1)<<7 | enc.opcode
}

// encodeU encodes a U-type RISC-V instruction.
func encodeU(ins *instruction) uint32 {
	// The immediates for encodeU are the upper 20 bits of a 32 bit value.
	// Rather than have the user/compiler generate a 32 bit constant, the
	// bottommost bits of which must all be zero, instead accept just the
	// top bits.
	imm := immI(ins.as, ins.imm, 20)
	rd := regI(ins.rd)
	enc := encode(ins.as)
	if enc == nil {
		panic("encodeU: could not encode instruction")
	}
	return imm<<12 | rd<<7 | enc.opcode
}

// encodeJ encodes a J-type RISC-V instruction.
func encodeJ(ins *instruction) uint32 {
	imm := immI(ins.as, ins.imm, 21)
	rd := regI(ins.rd)
	enc := encode(ins.as)
	if enc == nil {
		panic("encodeJ: could not encode instruction")
	}
	return (imm>>20)<<31 | ((imm>>1)&0x3ff)<<21 | ((imm>>11)&0x1)<<20 | ((imm>>12)&0xff)<<12 | rd<<7 | enc.opcode
}

func encodeRawIns(ins *instruction) uint32 {
	// Treat the raw value specially as a 32-bit unsigned integer.
	// Nobody wants to enter negative machine code.
	if ins.imm < 0 || 1<<32 <= ins.imm {
		panic(fmt.Sprintf("immediate %d cannot fit in 32 bits", ins.imm))
	}
	return uint32(ins.imm)
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
	if !immIFits(imm, 20) {
		return 0, fmt.Errorf("immediate %#x does not fit in 20 bits", imm)
	}
	return imm << 12, nil
}

type encoding struct {
	encode   func(*instruction) uint32     // encode returns the machine code for an instruction
	validate func(*obj.Link, *instruction) // validate validates an instruction
	length   int                           // length of encoded instruction; 0 for pseudo-ops, 4 otherwise
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
	rawEncoding = encoding{encode: encodeRawIns, validate: validateRaw, length: 4}

	// pseudoOpEncoding panics if encoding is attempted, but does no validation.
	pseudoOpEncoding = encoding{encode: nil, validate: func(*obj.Link, *instruction) {}, length: 0}

	// badEncoding is used when an invalid op is encountered.
	// An error has already been generated, so let anything else through.
	badEncoding = encoding{encode: func(*instruction) uint32 { return 0 }, validate: func(*obj.Link, *instruction) {}, length: 0}
)

// encodings contains the encodings for RISC-V instructions.
// Instructions are masked with obj.AMask to keep indices small.
var encodings = [ALAST & obj.AMask]encoding{

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

	// 2.7: Memory Ordering
	AFENCE & obj.AMask: iIEncoding,

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

	// 8.2: Load-Reserved/Store-Conditional
	ALRW & obj.AMask: rIIIEncoding,
	ALRD & obj.AMask: rIIIEncoding,
	ASCW & obj.AMask: rIIIEncoding,
	ASCD & obj.AMask: rIIIEncoding,

	// 8.3: Atomic Memory Operations
	AAMOSWAPW & obj.AMask: rIIIEncoding,
	AAMOSWAPD & obj.AMask: rIIIEncoding,
	AAMOADDW & obj.AMask:  rIIIEncoding,
	AAMOADDD & obj.AMask:  rIIIEncoding,
	AAMOANDW & obj.AMask:  rIIIEncoding,
	AAMOANDD & obj.AMask:  rIIIEncoding,
	AAMOORW & obj.AMask:   rIIIEncoding,
	AAMOORD & obj.AMask:   rIIIEncoding,
	AAMOXORW & obj.AMask:  rIIIEncoding,
	AAMOXORD & obj.AMask:  rIIIEncoding,
	AAMOMAXW & obj.AMask:  rIIIEncoding,
	AAMOMAXD & obj.AMask:  rIIIEncoding,
	AAMOMAXUW & obj.AMask: rIIIEncoding,
	AAMOMAXUD & obj.AMask: rIIIEncoding,
	AAMOMINW & obj.AMask:  rIIIEncoding,
	AAMOMIND & obj.AMask:  rIIIEncoding,
	AAMOMINUW & obj.AMask: rIIIEncoding,
	AAMOMINUD & obj.AMask: rIIIEncoding,

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

	// 11.9: Single-Precision Floating-Point Classify Instruction
	AFCLASSS & obj.AMask: rFIEncoding,

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

	// 12.7: Double-Precision Floating-Point Classify Instruction
	AFCLASSD & obj.AMask: rFIEncoding,

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
	obj.ADUFFZERO: pseudoOpEncoding,
	obj.ADUFFCOPY: pseudoOpEncoding,
}

// encodingForAs returns the encoding for an obj.As.
func encodingForAs(as obj.As) (encoding, error) {
	if base := as &^ obj.AMask; base != obj.ABaseRISCV && base != 0 {
		return badEncoding, fmt.Errorf("encodingForAs: not a RISC-V instruction %s", as)
	}
	asi := as & obj.AMask
	if int(asi) >= len(encodings) {
		return badEncoding, fmt.Errorf("encodingForAs: bad RISC-V instruction %s", as)
	}
	enc := encodings[asi]
	if enc.validate == nil {
		return badEncoding, fmt.Errorf("encodingForAs: no encoding for instruction %s", as)
	}
	return enc, nil
}

type instruction struct {
	as     obj.As // Assembler opcode
	rd     uint32 // Destination register
	rs1    uint32 // Source register 1
	rs2    uint32 // Source register 2
	imm    int64  // Immediate
	funct3 uint32 // Function 3
	funct7 uint32 // Function 7
}

func (ins *instruction) encode() (uint32, error) {
	enc, err := encodingForAs(ins.as)
	if err != nil {
		return 0, err
	}
	if enc.length > 0 {
		return enc.encode(ins), nil
	}
	return 0, fmt.Errorf("fixme")
}

func (ins *instruction) length() int {
	enc, err := encodingForAs(ins.as)
	if err != nil {
		return 0
	}
	return enc.length
}

func (ins *instruction) validate(ctxt *obj.Link) {
	enc, err := encodingForAs(ins.as)
	if err != nil {
		ctxt.Diag(err.Error())
		return
	}
	enc.validate(ctxt, ins)
}

// instructionsForProg returns the machine instructions for an *obj.Prog.
func instructionsForProg(p *obj.Prog) []*instruction {
	ins := &instruction{
		as:  p.As,
		rd:  uint32(p.To.Reg),
		rs1: uint32(p.Reg),
		rs2: uint32(p.From.Reg),
		imm: p.From.Offset,
	}

	inss := []*instruction{ins}
	switch ins.as {
	case AJAL, AJALR:
		ins.rd, ins.rs1, ins.rs2 = uint32(p.From.Reg), uint32(p.To.Reg), obj.REG_NONE
		ins.imm = p.To.Offset

	case ABEQ, ABEQZ, ABGE, ABGEU, ABGEZ, ABGT, ABGTU, ABGTZ, ABLE, ABLEU, ABLEZ, ABLT, ABLTU, ABLTZ, ABNE, ABNEZ:
		switch ins.as {
		case ABEQZ:
			ins.as, ins.rs1, ins.rs2 = ABEQ, REG_ZERO, uint32(p.From.Reg)
		case ABGEZ:
			ins.as, ins.rs1, ins.rs2 = ABGE, REG_ZERO, uint32(p.From.Reg)
		case ABGT:
			ins.as, ins.rs1, ins.rs2 = ABLT, uint32(p.From.Reg), uint32(p.Reg)
		case ABGTU:
			ins.as, ins.rs1, ins.rs2 = ABLTU, uint32(p.From.Reg), uint32(p.Reg)
		case ABGTZ:
			ins.as, ins.rs1, ins.rs2 = ABLT, uint32(p.From.Reg), REG_ZERO
		case ABLE:
			ins.as, ins.rs1, ins.rs2 = ABGE, uint32(p.From.Reg), uint32(p.Reg)
		case ABLEU:
			ins.as, ins.rs1, ins.rs2 = ABGEU, uint32(p.From.Reg), uint32(p.Reg)
		case ABLEZ:
			ins.as, ins.rs1, ins.rs2 = ABGE, uint32(p.From.Reg), REG_ZERO
		case ABLTZ:
			ins.as, ins.rs1, ins.rs2 = ABLT, REG_ZERO, uint32(p.From.Reg)
		case ABNEZ:
			ins.as, ins.rs1, ins.rs2 = ABNE, REG_ZERO, uint32(p.From.Reg)
		}
		ins.imm = p.To.Offset

	case AMOV, AMOVB, AMOVH, AMOVW, AMOVBU, AMOVHU, AMOVWU, AMOVF, AMOVD:
		// Handle register to register moves.
		if p.From.Type != obj.TYPE_REG || p.To.Type != obj.TYPE_REG {
			break
		}
		switch p.As {
		case AMOV: // MOV Ra, Rb -> ADDI $0, Ra, Rb
			ins.as, ins.rs1, ins.rs2, ins.imm = AADDI, uint32(p.From.Reg), obj.REG_NONE, 0
		case AMOVW: // MOVW Ra, Rb -> ADDIW $0, Ra, Rb
			ins.as, ins.rs1, ins.rs2, ins.imm = AADDIW, uint32(p.From.Reg), obj.REG_NONE, 0
		case AMOVBU: // MOVBU Ra, Rb -> ANDI $255, Ra, Rb
			ins.as, ins.rs1, ins.rs2, ins.imm = AANDI, uint32(p.From.Reg), obj.REG_NONE, 255
		case AMOVF: // MOVF Ra, Rb -> FSGNJS Ra, Ra, Rb
			ins.as, ins.rs1 = AFSGNJS, uint32(p.From.Reg)
		case AMOVD: // MOVD Ra, Rb -> FSGNJD Ra, Ra, Rb
			ins.as, ins.rs1 = AFSGNJD, uint32(p.From.Reg)
		case AMOVB, AMOVH:
			// Use SLLI/SRAI to extend.
			ins.as, ins.rs1, ins.rs2 = ASLLI, uint32(p.From.Reg), obj.REG_NONE
			if p.As == AMOVB {
				ins.imm = 56
			} else if p.As == AMOVH {
				ins.imm = 48
			}
			ins2 := &instruction{as: ASRAI, rd: ins.rd, rs1: ins.rd, imm: ins.imm}
			inss = append(inss, ins2)
		case AMOVHU, AMOVWU:
			// Use SLLI/SRLI to extend.
			ins.as, ins.rs1, ins.rs2 = ASLLI, uint32(p.From.Reg), obj.REG_NONE
			if p.As == AMOVHU {
				ins.imm = 48
			} else if p.As == AMOVWU {
				ins.imm = 32
			}
			ins2 := &instruction{as: ASRLI, rd: ins.rd, rs1: ins.rd, imm: ins.imm}
			inss = append(inss, ins2)
		}

	case ALW, ALWU, ALH, ALHU, ALB, ALBU, ALD, AFLW, AFLD:
		if p.From.Type != obj.TYPE_MEM {
			p.Ctxt.Diag("%v requires memory for source", p)
			return nil
		}
		ins.rs1, ins.rs2 = uint32(p.From.Reg), obj.REG_NONE
		ins.imm = p.From.Offset

	case ASW, ASH, ASB, ASD, AFSW, AFSD:
		if p.To.Type != obj.TYPE_MEM {
			p.Ctxt.Diag("%v requires memory for destination", p)
			return nil
		}
		ins.rs1, ins.rs2 = uint32(p.From.Reg), obj.REG_NONE
		ins.imm = p.To.Offset

	case ALRW, ALRD:
		// Set aq to use acquire access ordering, which matches Go's memory requirements.
		ins.funct7 = 2
		ins.rs1, ins.rs2 = uint32(p.From.Reg), REG_ZERO

	case ASCW, ASCD, AAMOSWAPW, AAMOSWAPD, AAMOADDW, AAMOADDD, AAMOANDW, AAMOANDD, AAMOORW, AAMOORD,
		AAMOXORW, AAMOXORD, AAMOMINW, AAMOMIND, AAMOMINUW, AAMOMINUD, AAMOMAXW, AAMOMAXD, AAMOMAXUW, AAMOMAXUD:
		// Set aq to use acquire access ordering, which matches Go's memory requirements.
		ins.funct7 = 2
		ins.rd, ins.rs1, ins.rs2 = uint32(p.RegTo2), uint32(p.To.Reg), uint32(p.From.Reg)

	case AECALL, AEBREAK, ARDCYCLE, ARDTIME, ARDINSTRET:
		insEnc := encode(p.As)
		if p.To.Type == obj.TYPE_NONE {
			ins.rd = REG_ZERO
		}
		ins.rs1 = REG_ZERO
		ins.imm = insEnc.csr

	case AFENCE:
		ins.rd, ins.rs1, ins.rs2 = REG_ZERO, REG_ZERO, obj.REG_NONE
		ins.imm = 0x0ff

	case AFCVTWS, AFCVTLS, AFCVTWUS, AFCVTLUS, AFCVTWD, AFCVTLD, AFCVTWUD, AFCVTLUD:
		// Set the rounding mode in funct3 to round to zero.
		ins.funct3 = 1

	case AFNES, AFNED:
		// Replace FNE[SD] with FEQ[SD] and NOT.
		if p.To.Type != obj.TYPE_REG {
			p.Ctxt.Diag("%v needs an integer register output", ins.as)
			return nil
		}
		if ins.as == AFNES {
			ins.as = AFEQS
		} else {
			ins.as = AFEQD
		}
		ins2 := &instruction{
			as:  AXORI, // [bit] xor 1 = not [bit]
			rd:  ins.rd,
			rs1: ins.rd,
			imm: 1,
		}
		inss = append(inss, ins2)

	case AFSQRTS, AFSQRTD:
		// These instructions expect a zero (i.e. float register 0)
		// to be the second input operand.
		ins.rs1 = uint32(p.From.Reg)
		ins.rs2 = REG_F0

	case ANEG, ANEGW:
		// NEG rs, rd -> SUB rs, X0, rd
		ins.as = ASUB
		if p.As == ANEGW {
			ins.as = ASUBW
		}
		ins.rs1 = REG_ZERO
		if ins.rd == obj.REG_NONE {
			ins.rd = ins.rs2
		}

	case ANOT:
		// NOT rs, rd -> XORI $-1, rs, rd
		ins.as = AXORI
		ins.rs1, ins.rs2 = uint32(p.From.Reg), obj.REG_NONE
		if ins.rd == obj.REG_NONE {
			ins.rd = ins.rs1
		}
		ins.imm = -1

	case ASEQZ:
		// SEQZ rs, rd -> SLTIU $1, rs, rd
		ins.as = ASLTIU
		ins.rs1 = uint32(p.From.Reg)
		ins.imm = 1

	case ASNEZ:
		// SNEZ rs, rd -> SLTU rs, x0, rd
		ins.as = ASLTU
		ins.rs1 = REG_ZERO

	case AFNEGS:
		// FNEGS rs, rd -> FSGNJNS rs, rs, rd
		ins.as = AFSGNJNS
		ins.rs1 = uint32(p.From.Reg)

	case AFNEGD:
		// FNEGD rs, rd -> FSGNJND rs, rs, rd
		ins.as = AFSGNJND
		ins.rs1 = uint32(p.From.Reg)
	}
	return inss
}

// assemble emits machine code.
// It is called at the very end of the assembly process.
func assemble(ctxt *obj.Link, cursym *obj.LSym, newprog obj.ProgAlloc) {
	if ctxt.Retpoline {
		ctxt.Diag("-spectre=ret not supported on riscv")
		ctxt.Retpoline = false // don't keep printing
	}

	var symcode []uint32
	for p := cursym.Func().Text; p != nil; p = p.Link {
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
			addr := p.RestArgs[0]
			if addr.Sym == nil {
				ctxt.Diag("AUIPC needing PC-relative reloc missing symbol")
				break
			}
			if addr.Sym.Type == objabi.STLSBSS {
				if rt == objabi.R_RISCV_PCREL_ITYPE {
					rt = objabi.R_RISCV_TLS_IE_ITYPE
				} else if rt == objabi.R_RISCV_PCREL_STYPE {
					rt = objabi.R_RISCV_TLS_IE_STYPE
				}
			}

			rel := obj.Addrel(cursym)
			rel.Off = int32(p.Pc)
			rel.Siz = 8
			rel.Sym = addr.Sym
			rel.Add = addr.Offset
			rel.Type = rt
		}

		for _, ins := range instructionsForProg(p) {
			ic, err := ins.encode()
			if err == nil {
				symcode = append(symcode, ic)
			}
		}
	}
	cursym.Size = int64(4 * len(symcode))

	cursym.Grow(cursym.Size)
	for p, i := cursym.P, 0; i < len(symcode); p, i = p[4:], i+1 {
		ctxt.Arch.ByteOrder.PutUint32(p, symcode[i])
	}

	obj.MarkUnsafePoints(ctxt, cursym.Func().Text, newprog, isUnsafePoint, nil)
}

func isUnsafePoint(p *obj.Prog) bool {
	return p.From.Reg == REG_TMP || p.To.Reg == REG_TMP || p.Reg == REG_TMP
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
