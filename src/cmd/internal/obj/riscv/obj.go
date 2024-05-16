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
	"internal/abi"
	"internal/buildcfg"
	"log"
	"math/bits"
	"strings"
)

func buildop(ctxt *obj.Link) {}

func jalToSym(ctxt *obj.Link, p *obj.Prog, lr int16) {
	switch p.As {
	case obj.ACALL, obj.AJMP, obj.ARET, obj.ADUFFZERO, obj.ADUFFCOPY:
	default:
		ctxt.Diag("unexpected Prog in jalToSym: %v", p)
		return
	}

	p.As = AJAL
	p.Mark |= NEED_JAL_RELOC
	p.From.Type = obj.TYPE_REG
	p.From.Reg = lr
	p.Reg = obj.REG_NONE
}

// progedit is called individually for each *obj.Prog. It normalizes instruction
// formats and eliminates as many pseudo-instructions as possible.
func progedit(ctxt *obj.Link, p *obj.Prog, newprog obj.ProgAlloc) {

	// Expand binary instructions to ternary ones.
	if p.Reg == obj.REG_NONE {
		switch p.As {
		case AADDI, ASLTI, ASLTIU, AANDI, AORI, AXORI, ASLLI, ASRLI, ASRAI,
			AADDIW, ASLLIW, ASRLIW, ASRAIW, AADDW, ASUBW, ASLLW, ASRLW, ASRAW,
			AADD, AAND, AOR, AXOR, ASLL, ASRL, ASUB, ASRA,
			AMUL, AMULH, AMULHU, AMULHSU, AMULW, ADIV, ADIVU, ADIVW, ADIVUW,
			AREM, AREMU, AREMW, AREMUW,
			AADDUW, ASH1ADD, ASH1ADDUW, ASH2ADD, ASH2ADDUW, ASH3ADD, ASH3ADDUW, ASLLIUW,
			AANDN, AORN, AXNOR, AMAX, AMAXU, AMIN, AMINU, AROL, AROLW, AROR, ARORW, ARORI, ARORIW,
			ABCLR, ABCLRI, ABEXT, ABEXTI, ABINV, ABINVI, ABSET, ABSETI:
			p.Reg = p.To.Reg
		}
	}

	// Rewrite instructions with constant operands to refer to the immediate
	// form of the instruction.
	if p.From.Type == obj.TYPE_CONST {
		switch p.As {
		case AADD:
			p.As = AADDI
		case ASUB:
			p.As, p.From.Offset = AADDI, -p.From.Offset
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
		case AADDW:
			p.As = AADDIW
		case ASUBW:
			p.As, p.From.Offset = AADDIW, -p.From.Offset
		case ASLLW:
			p.As = ASLLIW
		case ASRLW:
			p.As = ASRLIW
		case ASRAW:
			p.As = ASRAIW
		case AROR:
			p.As = ARORI
		case ARORW:
			p.As = ARORIW
		case ABCLR:
			p.As = ABCLRI
		case ABEXT:
			p.As = ABEXTI
		case ABINV:
			p.As = ABINVI
		case ABSET:
			p.As = ABSETI
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
			case obj.NAME_EXTERN, obj.NAME_STATIC:
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

	case AMOV:
		if p.From.Type == obj.TYPE_CONST && p.From.Name == obj.NAME_NONE && p.From.Reg == obj.REG_NONE && int64(int32(p.From.Offset)) != p.From.Offset {
			ctz := bits.TrailingZeros64(uint64(p.From.Offset))
			val := p.From.Offset >> ctz
			if int64(int32(val)) == val {
				// It's ok. We can handle constants with many trailing zeros.
				break
			}
			// Put >32-bit constants in memory and load them.
			p.From.Type = obj.TYPE_MEM
			p.From.Sym = ctxt.Int64Sym(p.From.Offset)
			p.From.Name = obj.NAME_EXTERN
			p.From.Offset = 0
		}
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

// markRelocs marks an obj.Prog that specifies a MOV pseudo-instruction and
// requires relocation.
func markRelocs(p *obj.Prog) {
	switch p.As {
	case AMOV, AMOVB, AMOVH, AMOVW, AMOVBU, AMOVHU, AMOVWU, AMOVF, AMOVD:
		switch {
		case p.From.Type == obj.TYPE_ADDR && p.To.Type == obj.TYPE_REG:
			switch p.From.Name {
			case obj.NAME_EXTERN, obj.NAME_STATIC:
				p.Mark |= NEED_PCREL_ITYPE_RELOC
			}
		case p.From.Type == obj.TYPE_MEM && p.To.Type == obj.TYPE_REG:
			switch p.From.Name {
			case obj.NAME_EXTERN, obj.NAME_STATIC:
				p.Mark |= NEED_PCREL_ITYPE_RELOC
			}
		case p.From.Type == obj.TYPE_REG && p.To.Type == obj.TYPE_MEM:
			switch p.To.Name {
			case obj.NAME_EXTERN, obj.NAME_STATIC:
				p.Mark |= NEED_PCREL_STYPE_RELOC
			}
		}
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
// It uses pc as the initial value and returns the next available pc.
func setPCs(p *obj.Prog, pc int64) int64 {
	for ; p != nil; p = p.Link {
		p.Pc = pc
		for _, ins := range instructionsForProg(p) {
			pc += int64(ins.length())
		}

		if p.As == obj.APCALIGN {
			alignedValue := p.From.Offset
			v := pcAlignPadLength(pc, alignedValue)
			pc += int64(v)
		}
	}
	return pc
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
// attached to https://golang.org/issue/16922#issuecomment-243748180.
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
		stacksize += ctxt.Arch.FixedFrameSize
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

		// On Linux, in a cgo binary we may get a SIGSETXID signal early on
		// before the signal stack is set, as glibc doesn't allow us to block
		// SIGSETXID. So a signal may land on the current stack and clobber
		// the content below the SP. We store the LR again after the SP is
		// decremented.
		prologue = obj.Appendp(prologue, newprog)
		prologue.As = AMOV
		prologue.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_LR}
		prologue.To = obj.Addr{Type: obj.TYPE_MEM, Reg: REG_SP, Offset: 0}
	}

	if cursym.Func().Text.From.Sym.Wrapper() {
		// if(g->panic != nil && g->panic->argp == FP) g->panic->argp = bottom-of-frame
		//
		//   MOV g_panic(g), X5
		//   BNE X5, ZERO, adjust
		// end:
		//   NOP
		// ...rest of function..
		// adjust:
		//   MOV panic_argp(X5), X6
		//   ADD $(autosize+FIXED_FRAME), SP, X7
		//   BNE X6, X7, end
		//   ADD $FIXED_FRAME, SP, X6
		//   MOV X6, panic_argp(X5)
		//   JMP end
		//
		// The NOP is needed to give the jumps somewhere to land.

		ldpanic := obj.Appendp(prologue, newprog)

		ldpanic.As = AMOV
		ldpanic.From = obj.Addr{Type: obj.TYPE_MEM, Reg: REGG, Offset: 4 * int64(ctxt.Arch.PtrSize)} // G.panic
		ldpanic.Reg = obj.REG_NONE
		ldpanic.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_X5}

		bneadj := obj.Appendp(ldpanic, newprog)
		bneadj.As = ABNE
		bneadj.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_X5}
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
		getargp.From = obj.Addr{Type: obj.TYPE_MEM, Reg: REG_X5, Offset: 0} // Panic.argp
		getargp.Reg = obj.REG_NONE
		getargp.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_X6}

		bneadj.To.SetTarget(getargp)

		calcargp := obj.Appendp(getargp, newprog)
		calcargp.As = AADDI
		calcargp.From = obj.Addr{Type: obj.TYPE_CONST, Offset: stacksize + ctxt.Arch.FixedFrameSize}
		calcargp.Reg = REG_SP
		calcargp.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_X7}

		testargp := obj.Appendp(calcargp, newprog)
		testargp.As = ABNE
		testargp.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_X6}
		testargp.Reg = REG_X7
		testargp.To.Type = obj.TYPE_BRANCH
		testargp.To.SetTarget(endadj)

		adjargp := obj.Appendp(testargp, newprog)
		adjargp.As = AADDI
		adjargp.From = obj.Addr{Type: obj.TYPE_CONST, Offset: int64(ctxt.Arch.PtrSize)}
		adjargp.Reg = REG_SP
		adjargp.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_X6}

		setargp := obj.Appendp(adjargp, newprog)
		setargp.As = AMOV
		setargp.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_X6}
		setargp.Reg = obj.REG_NONE
		setargp.To = obj.Addr{Type: obj.TYPE_MEM, Reg: REG_X5, Offset: 0} // Panic.argp

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
				jalToSym(ctxt, p, REG_LR)
			}

		case obj.AJMP:
			switch p.To.Type {
			case obj.TYPE_MEM:
				switch p.To.Name {
				case obj.NAME_EXTERN, obj.NAME_STATIC:
					jalToSym(ctxt, p, REG_ZERO)
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
				jalToSym(ctxt, p, REG_ZERO)
			} else {
				p.As = AJALR
				p.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_ZERO}
				p.Reg = obj.REG_NONE
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

		if p.To.Type == obj.TYPE_REG && p.To.Reg == REGSP && p.Spadj == 0 {
			f := cursym.Func()
			if f.FuncFlag&abi.FuncFlagSPWrite == 0 {
				f.FuncFlag |= abi.FuncFlagSPWrite
				if ctxt.Debugvlog || !ctxt.IsAsm {
					ctxt.Logf("auto-SPWRITE: %s %v\n", cursym.Name, p)
					if !ctxt.IsAsm {
						ctxt.Diag("invalid auto-SPWRITE in non-assembly")
						ctxt.DiagFlush()
						log.Fatalf("bad SPWRITE")
					}
				}
			}
		}
	}

	var callCount int
	for p := cursym.Func().Text; p != nil; p = p.Link {
		markRelocs(p)
		if p.Mark&NEED_JAL_RELOC == NEED_JAL_RELOC {
			callCount++
		}
	}
	const callTrampSize = 8 // 2 machine instructions.
	maxTrampSize := int64(callCount * callTrampSize)

	// Compute instruction addresses.  Once we do that, we need to check for
	// overextended jumps and branches.  Within each iteration, Pc differences
	// are always lower bounds (since the program gets monotonically longer,
	// a fixed point will be reached).  No attempt to handle functions > 2GiB.
	for {
		big, rescan := false, false
		maxPC := setPCs(cursym.Func().Text, 0)
		if maxPC+maxTrampSize > (1 << 20) {
			big = true
		}

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
				// Linker will handle the intersymbol case and trampolines.
				if p.To.Target() == nil {
					if !big {
						break
					}
					// This function is going to be too large for JALs
					// to reach trampolines. Replace with AUIPC+JALR.
					jmp := obj.Appendp(p, newprog)
					jmp.As = AJALR
					jmp.From = p.From
					jmp.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_TMP}

					p.As = AAUIPC
					p.Mark = (p.Mark &^ NEED_JAL_RELOC) | NEED_CALL_RELOC
					p.AddRestSource(obj.Addr{Type: obj.TYPE_CONST, Offset: p.To.Offset, Sym: p.To.Sym})
					p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: 0}
					p.Reg = obj.REG_NONE
					p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_TMP}

					rescan = true
					break
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
					p.Reg = obj.REG_NONE
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
		case ABEQ, ABEQZ, ABGE, ABGEU, ABGEZ, ABGT, ABGTU, ABGTZ, ABLE, ABLEU, ABLEZ, ABLT, ABLTU, ABLTZ, ABNE, ABNEZ:
			switch p.To.Type {
			case obj.TYPE_BRANCH:
				p.To.Type, p.To.Offset = obj.TYPE_CONST, p.To.Target().Pc-p.Pc
			case obj.TYPE_MEM:
				panic("unhandled type")
			}

		case AJAL:
			// Linker will handle the intersymbol case and trampolines.
			if p.To.Target() != nil {
				p.To.Type, p.To.Offset = obj.TYPE_CONST, p.To.Target().Pc-p.Pc
			}

		case AAUIPC:
			if p.From.Type == obj.TYPE_BRANCH {
				low, high, err := Split32BitImmediate(p.From.Target().Pc - p.Pc)
				if err != nil {
					ctxt.Diag("%v: jump displacement %d too large", p, p.To.Target().Pc-p.Pc)
				}
				p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: high, Sym: cursym}
				p.Link.To.Offset = low
			}

		case obj.APCALIGN:
			alignedValue := p.From.Offset
			if (alignedValue&(alignedValue-1) != 0) || 4 > alignedValue || alignedValue > 2048 {
				ctxt.Diag("alignment value of an instruction must be a power of two and in the range [4, 2048], got %d\n", alignedValue)
			}
			// Update the current text symbol alignment value.
			if int32(alignedValue) > cursym.Func().Align {
				cursym.Func().Align = int32(alignedValue)
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

func pcAlignPadLength(pc int64, alignedValue int64) int {
	return int(-pc & (alignedValue - 1))
}

func stacksplit(ctxt *obj.Link, p *obj.Prog, cursym *obj.LSym, newprog obj.ProgAlloc, framesize int64) *obj.Prog {
	// Leaf function with no frame is effectively NOSPLIT.
	if framesize == 0 {
		return p
	}

	if ctxt.Flag_maymorestack != "" {
		// Save LR and REGCTXT
		const frameSize = 16
		p = ctxt.StartUnsafePoint(p, newprog)

		// Spill Arguments. This has to happen before we open
		// any more frame space.
		p = cursym.Func().SpillRegisterArgs(p, newprog)

		// MOV LR, -16(SP)
		p = obj.Appendp(p, newprog)
		p.As = AMOV
		p.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_LR}
		p.To = obj.Addr{Type: obj.TYPE_MEM, Reg: REG_SP, Offset: -frameSize}
		// ADDI $-16, SP
		p = obj.Appendp(p, newprog)
		p.As = AADDI
		p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: -frameSize}
		p.Reg = REG_SP
		p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_SP}
		p.Spadj = frameSize
		// MOV REGCTXT, 8(SP)
		p = obj.Appendp(p, newprog)
		p.As = AMOV
		p.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_CTXT}
		p.To = obj.Addr{Type: obj.TYPE_MEM, Reg: REG_SP, Offset: 8}

		// CALL maymorestack
		p = obj.Appendp(p, newprog)
		p.As = obj.ACALL
		p.To.Type = obj.TYPE_BRANCH
		// See ../x86/obj6.go
		p.To.Sym = ctxt.LookupABI(ctxt.Flag_maymorestack, cursym.ABI())
		jalToSym(ctxt, p, REG_X5)

		// Restore LR and REGCTXT

		// MOV 8(SP), REGCTXT
		p = obj.Appendp(p, newprog)
		p.As = AMOV
		p.From = obj.Addr{Type: obj.TYPE_MEM, Reg: REG_SP, Offset: 8}
		p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_CTXT}
		// MOV (SP), LR
		p = obj.Appendp(p, newprog)
		p.As = AMOV
		p.From = obj.Addr{Type: obj.TYPE_MEM, Reg: REG_SP, Offset: 0}
		p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_LR}
		// ADDI $16, SP
		p = obj.Appendp(p, newprog)
		p.As = AADDI
		p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: frameSize}
		p.Reg = REG_SP
		p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_SP}
		p.Spadj = -frameSize

		// Unspill arguments
		p = cursym.Func().UnspillRegisterArgs(p, newprog)
		p = ctxt.EndUnsafePoint(p, newprog, -1)
	}

	// Jump back to here after morestack returns.
	startPred := p

	// MOV	g_stackguard(g), X6
	p = obj.Appendp(p, newprog)
	p.As = AMOV
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = REGG
	p.From.Offset = 2 * int64(ctxt.Arch.PtrSize) // G.stackguard0
	if cursym.CFunc() {
		p.From.Offset = 3 * int64(ctxt.Arch.PtrSize) // G.stackguard1
	}
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_X6

	// Mark the stack bound check and morestack call async nonpreemptible.
	// If we get preempted here, when resumed the preemption request is
	// cleared, but we'll still call morestack, which will double the stack
	// unnecessarily. See issue #35470.
	p = ctxt.StartUnsafePoint(p, newprog)

	var to_done, to_more *obj.Prog

	if framesize <= abi.StackSmall {
		// small stack
		//	// if SP > stackguard { goto done }
		//	BLTU	stackguard, SP, done
		p = obj.Appendp(p, newprog)
		p.As = ABLTU
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_X6
		p.Reg = REG_SP
		p.To.Type = obj.TYPE_BRANCH
		to_done = p
	} else {
		// large stack: SP-framesize < stackguard-StackSmall
		offset := int64(framesize) - abi.StackSmall
		if framesize > abi.StackBig {
			// Such a large stack we need to protect against underflow.
			// The runtime guarantees SP > objabi.StackBig, but
			// framesize is large enough that SP-framesize may
			// underflow, causing a direct comparison with the
			// stack guard to incorrectly succeed. We explicitly
			// guard against underflow.
			//
			//	MOV	$(framesize-StackSmall), X7
			//	BLTU	SP, X7, label-of-call-to-morestack

			p = obj.Appendp(p, newprog)
			p.As = AMOV
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = offset
			p.To.Type = obj.TYPE_REG
			p.To.Reg = REG_X7

			p = obj.Appendp(p, newprog)
			p.As = ABLTU
			p.From.Type = obj.TYPE_REG
			p.From.Reg = REG_SP
			p.Reg = REG_X7
			p.To.Type = obj.TYPE_BRANCH
			to_more = p
		}

		// Check against the stack guard. We've ensured this won't underflow.
		//	ADD	$-(framesize-StackSmall), SP, X7
		//	// if X7 > stackguard { goto done }
		//	BLTU	stackguard, X7, done
		p = obj.Appendp(p, newprog)
		p.As = AADDI
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = -offset
		p.Reg = REG_SP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_X7

		p = obj.Appendp(p, newprog)
		p.As = ABLTU
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_X6
		p.Reg = REG_X7
		p.To.Type = obj.TYPE_BRANCH
		to_done = p
	}

	// Spill the register args that could be clobbered by the
	// morestack code
	p = ctxt.EmitEntryStackMap(cursym, p, newprog)
	p = cursym.Func().SpillRegisterArgs(p, newprog)

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
	jalToSym(ctxt, p, REG_X5)

	// The instructions which unspill regs should be preemptible.
	p = ctxt.EndUnsafePoint(p, newprog, -1)
	p = cursym.Func().UnspillRegisterArgs(p, newprog)

	// JMP start
	p = obj.Appendp(p, newprog)
	p.As = AJAL
	p.To = obj.Addr{Type: obj.TYPE_BRANCH}
	p.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_ZERO}
	p.To.SetTarget(startPred.Link)

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
	if err := immIFits(imm, 32); err != nil {
		return 0, 0, err
	}

	// Nothing special needs to be done if the immediate fits in 12 bits.
	if err := immIFits(imm, 12); err == nil {
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
		panic(fmt.Sprintf("register out of range, want %d <= %d <= %d", min, r, max))
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

// immEven checks that the immediate is a multiple of two. If it
// is not, an error is returned.
func immEven(x int64) error {
	if x&1 != 0 {
		return fmt.Errorf("immediate %#x is not a multiple of two", x)
	}
	return nil
}

// immIFits checks whether the immediate value x fits in nbits bits
// as a signed integer. If it does not, an error is returned.
func immIFits(x int64, nbits uint) error {
	nbits--
	min := int64(-1) << nbits
	max := int64(1)<<nbits - 1
	if x < min || x > max {
		if nbits <= 16 {
			return fmt.Errorf("signed immediate %d must be in range [%d, %d] (%d bits)", x, min, max, nbits)
		}
		return fmt.Errorf("signed immediate %#x must be in range [%#x, %#x] (%d bits)", x, min, max, nbits)
	}
	return nil
}

// immI extracts the signed integer of the specified size from an immediate.
func immI(as obj.As, imm int64, nbits uint) uint32 {
	if err := immIFits(imm, nbits); err != nil {
		panic(fmt.Sprintf("%v: %v", as, err))
	}
	return uint32(imm)
}

func wantImmI(ctxt *obj.Link, ins *instruction, imm int64, nbits uint) {
	if err := immIFits(imm, nbits); err != nil {
		ctxt.Diag("%v: %v", ins, err)
	}
}

func wantReg(ctxt *obj.Link, ins *instruction, pos string, descr string, r, min, max uint32) {
	if r < min || r > max {
		var suffix string
		if r != obj.REG_NONE {
			suffix = fmt.Sprintf(" but got non-%s register %s", descr, RegName(int(r)))
		}
		ctxt.Diag("%v: expected %s register in %s position%s", ins, descr, pos, suffix)
	}
}

func wantNoneReg(ctxt *obj.Link, ins *instruction, pos string, r uint32) {
	if r != obj.REG_NONE {
		ctxt.Diag("%v: expected no register in %s but got register %s", ins, pos, RegName(int(r)))
	}
}

// wantIntReg checks that r is an integer register.
func wantIntReg(ctxt *obj.Link, ins *instruction, pos string, r uint32) {
	wantReg(ctxt, ins, pos, "integer", r, REG_X0, REG_X31)
}

// wantFloatReg checks that r is a floating-point register.
func wantFloatReg(ctxt *obj.Link, ins *instruction, pos string, r uint32) {
	wantReg(ctxt, ins, pos, "float", r, REG_F0, REG_F31)
}

// wantEvenOffset checks that the offset is a multiple of two.
func wantEvenOffset(ctxt *obj.Link, ins *instruction, offset int64) {
	if err := immEven(offset); err != nil {
		ctxt.Diag("%v: %v", ins, err)
	}
}

func validateRII(ctxt *obj.Link, ins *instruction) {
	wantIntReg(ctxt, ins, "rd", ins.rd)
	wantIntReg(ctxt, ins, "rs1", ins.rs1)
	wantNoneReg(ctxt, ins, "rs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateRIII(ctxt *obj.Link, ins *instruction) {
	wantIntReg(ctxt, ins, "rd", ins.rd)
	wantIntReg(ctxt, ins, "rs1", ins.rs1)
	wantIntReg(ctxt, ins, "rs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateRFFF(ctxt *obj.Link, ins *instruction) {
	wantFloatReg(ctxt, ins, "rd", ins.rd)
	wantFloatReg(ctxt, ins, "rs1", ins.rs1)
	wantFloatReg(ctxt, ins, "rs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateRFFFF(ctxt *obj.Link, ins *instruction) {
	wantFloatReg(ctxt, ins, "rd", ins.rd)
	wantFloatReg(ctxt, ins, "rs1", ins.rs1)
	wantFloatReg(ctxt, ins, "rs2", ins.rs2)
	wantFloatReg(ctxt, ins, "rs3", ins.rs3)
}

func validateRFFI(ctxt *obj.Link, ins *instruction) {
	wantIntReg(ctxt, ins, "rd", ins.rd)
	wantFloatReg(ctxt, ins, "rs1", ins.rs1)
	wantFloatReg(ctxt, ins, "rs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateRFI(ctxt *obj.Link, ins *instruction) {
	wantIntReg(ctxt, ins, "rd", ins.rd)
	wantNoneReg(ctxt, ins, "rs1", ins.rs1)
	wantFloatReg(ctxt, ins, "rs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateRIF(ctxt *obj.Link, ins *instruction) {
	wantFloatReg(ctxt, ins, "rd", ins.rd)
	wantNoneReg(ctxt, ins, "rs1", ins.rs1)
	wantIntReg(ctxt, ins, "rs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateRFF(ctxt *obj.Link, ins *instruction) {
	wantFloatReg(ctxt, ins, "rd", ins.rd)
	wantNoneReg(ctxt, ins, "rs1", ins.rs1)
	wantFloatReg(ctxt, ins, "rs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateII(ctxt *obj.Link, ins *instruction) {
	wantImmI(ctxt, ins, ins.imm, 12)
	wantIntReg(ctxt, ins, "rd", ins.rd)
	wantIntReg(ctxt, ins, "rs1", ins.rs1)
	wantNoneReg(ctxt, ins, "rs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateIF(ctxt *obj.Link, ins *instruction) {
	wantImmI(ctxt, ins, ins.imm, 12)
	wantFloatReg(ctxt, ins, "rd", ins.rd)
	wantIntReg(ctxt, ins, "rs1", ins.rs1)
	wantNoneReg(ctxt, ins, "rs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateSI(ctxt *obj.Link, ins *instruction) {
	wantImmI(ctxt, ins, ins.imm, 12)
	wantIntReg(ctxt, ins, "rd", ins.rd)
	wantIntReg(ctxt, ins, "rs1", ins.rs1)
	wantNoneReg(ctxt, ins, "rs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateSF(ctxt *obj.Link, ins *instruction) {
	wantImmI(ctxt, ins, ins.imm, 12)
	wantIntReg(ctxt, ins, "rd", ins.rd)
	wantFloatReg(ctxt, ins, "rs1", ins.rs1)
	wantNoneReg(ctxt, ins, "rs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateB(ctxt *obj.Link, ins *instruction) {
	// Offsets are multiples of two, so accept 13 bit immediates for the
	// 12 bit slot. We implicitly drop the least significant bit in encodeB.
	wantEvenOffset(ctxt, ins, ins.imm)
	wantImmI(ctxt, ins, ins.imm, 13)
	wantNoneReg(ctxt, ins, "rd", ins.rd)
	wantIntReg(ctxt, ins, "rs1", ins.rs1)
	wantIntReg(ctxt, ins, "rs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateU(ctxt *obj.Link, ins *instruction) {
	wantImmI(ctxt, ins, ins.imm, 20)
	wantIntReg(ctxt, ins, "rd", ins.rd)
	wantNoneReg(ctxt, ins, "rs1", ins.rs1)
	wantNoneReg(ctxt, ins, "rs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateJ(ctxt *obj.Link, ins *instruction) {
	// Offsets are multiples of two, so accept 21 bit immediates for the
	// 20 bit slot. We implicitly drop the least significant bit in encodeJ.
	wantEvenOffset(ctxt, ins, ins.imm)
	wantImmI(ctxt, ins, ins.imm, 21)
	wantIntReg(ctxt, ins, "rd", ins.rd)
	wantNoneReg(ctxt, ins, "rs1", ins.rs1)
	wantNoneReg(ctxt, ins, "rs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateRaw(ctxt *obj.Link, ins *instruction) {
	// Treat the raw value specially as a 32-bit unsigned integer.
	// Nobody wants to enter negative machine code.
	if ins.imm < 0 || 1<<32 <= ins.imm {
		ctxt.Diag("%v: immediate %d in raw position cannot be larger than 32 bits", ins.as, ins.imm)
	}
}

// extractBitAndShift extracts the specified bit from the given immediate,
// before shifting it to the requested position and returning it.
func extractBitAndShift(imm uint32, bit, pos int) uint32 {
	return ((imm >> bit) & 1) << pos
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

// encodeR4 encodes an R4-type RISC-V instruction.
func encodeR4(as obj.As, rs1, rs2, rs3, rd, funct3, funct2 uint32) uint32 {
	enc := encode(as)
	if enc == nil {
		panic("encodeR4: could not encode instruction")
	}
	if enc.rs2 != 0 {
		panic("encodeR4: instruction uses rs2")
	}
	funct2 |= enc.funct7
	if funct2&^3 != 0 {
		panic("encodeR4: funct2 requires more than 2 bits")
	}
	return rs3<<27 | funct2<<25 | rs2<<20 | rs1<<15 | enc.funct3<<12 | funct3<<12 | rd<<7 | enc.opcode
}

func encodeRII(ins *instruction) uint32 {
	return encodeR(ins.as, regI(ins.rs1), 0, regI(ins.rd), ins.funct3, ins.funct7)
}

func encodeRIII(ins *instruction) uint32 {
	return encodeR(ins.as, regI(ins.rs1), regI(ins.rs2), regI(ins.rd), ins.funct3, ins.funct7)
}

func encodeRFFF(ins *instruction) uint32 {
	return encodeR(ins.as, regF(ins.rs1), regF(ins.rs2), regF(ins.rd), ins.funct3, ins.funct7)
}

func encodeRFFFF(ins *instruction) uint32 {
	return encodeR4(ins.as, regF(ins.rs1), regF(ins.rs2), regF(ins.rs3), regF(ins.rd), ins.funct3, ins.funct7)
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

// encodeBImmediate encodes an immediate for a B-type RISC-V instruction.
func encodeBImmediate(imm uint32) uint32 {
	return (imm>>12)<<31 | ((imm>>5)&0x3f)<<25 | ((imm>>1)&0xf)<<8 | ((imm>>11)&0x1)<<7
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
	return encodeBImmediate(imm) | rs2<<20 | rs1<<15 | enc.funct3<<12 | enc.opcode
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

// encodeJImmediate encodes an immediate for a J-type RISC-V instruction.
func encodeJImmediate(imm uint32) uint32 {
	return (imm>>20)<<31 | ((imm>>1)&0x3ff)<<21 | ((imm>>11)&0x1)<<20 | ((imm>>12)&0xff)<<12
}

// encodeJ encodes a J-type RISC-V instruction.
func encodeJ(ins *instruction) uint32 {
	imm := immI(ins.as, ins.imm, 21)
	rd := regI(ins.rd)
	enc := encode(ins.as)
	if enc == nil {
		panic("encodeJ: could not encode instruction")
	}
	return encodeJImmediate(imm) | rd<<7 | enc.opcode
}

// encodeCBImmediate encodes an immediate for a CB-type RISC-V instruction.
func encodeCBImmediate(imm uint32) uint32 {
	// Bit order - [8|4:3|7:6|2:1|5]
	bits := extractBitAndShift(imm, 8, 7)
	bits |= extractBitAndShift(imm, 4, 6)
	bits |= extractBitAndShift(imm, 3, 5)
	bits |= extractBitAndShift(imm, 7, 4)
	bits |= extractBitAndShift(imm, 6, 3)
	bits |= extractBitAndShift(imm, 2, 2)
	bits |= extractBitAndShift(imm, 1, 1)
	bits |= extractBitAndShift(imm, 5, 0)
	return (bits>>5)<<10 | (bits&0x1f)<<2
}

// encodeCJImmediate encodes an immediate for a CJ-type RISC-V instruction.
func encodeCJImmediate(imm uint32) uint32 {
	// Bit order - [11|4|9:8|10|6|7|3:1|5]
	bits := extractBitAndShift(imm, 11, 10)
	bits |= extractBitAndShift(imm, 4, 9)
	bits |= extractBitAndShift(imm, 9, 8)
	bits |= extractBitAndShift(imm, 8, 7)
	bits |= extractBitAndShift(imm, 10, 6)
	bits |= extractBitAndShift(imm, 6, 5)
	bits |= extractBitAndShift(imm, 7, 4)
	bits |= extractBitAndShift(imm, 3, 3)
	bits |= extractBitAndShift(imm, 2, 2)
	bits |= extractBitAndShift(imm, 1, 1)
	bits |= extractBitAndShift(imm, 5, 0)
	return bits << 2
}

func encodeRawIns(ins *instruction) uint32 {
	// Treat the raw value specially as a 32-bit unsigned integer.
	// Nobody wants to enter negative machine code.
	if ins.imm < 0 || 1<<32 <= ins.imm {
		panic(fmt.Sprintf("immediate %d cannot fit in 32 bits", ins.imm))
	}
	return uint32(ins.imm)
}

func EncodeBImmediate(imm int64) (int64, error) {
	if err := immIFits(imm, 13); err != nil {
		return 0, err
	}
	if err := immEven(imm); err != nil {
		return 0, err
	}
	return int64(encodeBImmediate(uint32(imm))), nil
}

func EncodeCBImmediate(imm int64) (int64, error) {
	if err := immIFits(imm, 9); err != nil {
		return 0, err
	}
	if err := immEven(imm); err != nil {
		return 0, err
	}
	return int64(encodeCBImmediate(uint32(imm))), nil
}

func EncodeCJImmediate(imm int64) (int64, error) {
	if err := immIFits(imm, 12); err != nil {
		return 0, err
	}
	if err := immEven(imm); err != nil {
		return 0, err
	}
	return int64(encodeCJImmediate(uint32(imm))), nil
}

func EncodeIImmediate(imm int64) (int64, error) {
	if err := immIFits(imm, 12); err != nil {
		return 0, err
	}
	return imm << 20, nil
}

func EncodeJImmediate(imm int64) (int64, error) {
	if err := immIFits(imm, 21); err != nil {
		return 0, err
	}
	if err := immEven(imm); err != nil {
		return 0, err
	}
	return int64(encodeJImmediate(uint32(imm))), nil
}

func EncodeSImmediate(imm int64) (int64, error) {
	if err := immIFits(imm, 12); err != nil {
		return 0, err
	}
	return ((imm >> 5) << 25) | ((imm & 0x1f) << 7), nil
}

func EncodeUImmediate(imm int64) (int64, error) {
	if err := immIFits(imm, 20); err != nil {
		return 0, err
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

	rIIIEncoding  = encoding{encode: encodeRIII, validate: validateRIII, length: 4}
	rIIEncoding   = encoding{encode: encodeRII, validate: validateRII, length: 4}
	rFFFEncoding  = encoding{encode: encodeRFFF, validate: validateRFFF, length: 4}
	rFFFFEncoding = encoding{encode: encodeRFFFF, validate: validateRFFFF, length: 4}
	rFFIEncoding  = encoding{encode: encodeRFFI, validate: validateRFFI, length: 4}
	rFIEncoding   = encoding{encode: encodeRFI, validate: validateRFI, length: 4}
	rIFEncoding   = encoding{encode: encodeRIF, validate: validateRIF, length: 4}
	rFFEncoding   = encoding{encode: encodeRFF, validate: validateRFF, length: 4}

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
	AFADDS & obj.AMask:   rFFFEncoding,
	AFSUBS & obj.AMask:   rFFFEncoding,
	AFMULS & obj.AMask:   rFFFEncoding,
	AFDIVS & obj.AMask:   rFFFEncoding,
	AFMINS & obj.AMask:   rFFFEncoding,
	AFMAXS & obj.AMask:   rFFFEncoding,
	AFSQRTS & obj.AMask:  rFFFEncoding,
	AFMADDS & obj.AMask:  rFFFFEncoding,
	AFMSUBS & obj.AMask:  rFFFFEncoding,
	AFNMSUBS & obj.AMask: rFFFFEncoding,
	AFNMADDS & obj.AMask: rFFFFEncoding,

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
	AFADDD & obj.AMask:   rFFFEncoding,
	AFSUBD & obj.AMask:   rFFFEncoding,
	AFMULD & obj.AMask:   rFFFEncoding,
	AFDIVD & obj.AMask:   rFFFEncoding,
	AFMIND & obj.AMask:   rFFFEncoding,
	AFMAXD & obj.AMask:   rFFFEncoding,
	AFSQRTD & obj.AMask:  rFFFEncoding,
	AFMADDD & obj.AMask:  rFFFFEncoding,
	AFMSUBD & obj.AMask:  rFFFFEncoding,
	AFNMSUBD & obj.AMask: rFFFFEncoding,
	AFNMADDD & obj.AMask: rFFFFEncoding,

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

	//
	// RISC-V Bit-Manipulation ISA-extensions (1.0)
	//

	// 1.1: Address Generation Instructions (Zba)
	AADDUW & obj.AMask:    rIIIEncoding,
	ASH1ADD & obj.AMask:   rIIIEncoding,
	ASH1ADDUW & obj.AMask: rIIIEncoding,
	ASH2ADD & obj.AMask:   rIIIEncoding,
	ASH2ADDUW & obj.AMask: rIIIEncoding,
	ASH3ADD & obj.AMask:   rIIIEncoding,
	ASH3ADDUW & obj.AMask: rIIIEncoding,
	ASLLIUW & obj.AMask:   iIEncoding,

	// 1.2: Basic Bit Manipulation (Zbb)
	AANDN & obj.AMask:  rIIIEncoding,
	ACLZ & obj.AMask:   rIIEncoding,
	ACLZW & obj.AMask:  rIIEncoding,
	ACPOP & obj.AMask:  rIIEncoding,
	ACPOPW & obj.AMask: rIIEncoding,
	ACTZ & obj.AMask:   rIIEncoding,
	ACTZW & obj.AMask:  rIIEncoding,
	AMAX & obj.AMask:   rIIIEncoding,
	AMAXU & obj.AMask:  rIIIEncoding,
	AMIN & obj.AMask:   rIIIEncoding,
	AMINU & obj.AMask:  rIIIEncoding,
	AORN & obj.AMask:   rIIIEncoding,
	ASEXTB & obj.AMask: rIIEncoding,
	ASEXTH & obj.AMask: rIIEncoding,
	AXNOR & obj.AMask:  rIIIEncoding,
	AZEXTH & obj.AMask: rIIEncoding,

	// 1.3: Bitwise Rotation (Zbb)
	AROL & obj.AMask:   rIIIEncoding,
	AROLW & obj.AMask:  rIIIEncoding,
	AROR & obj.AMask:   rIIIEncoding,
	ARORI & obj.AMask:  iIEncoding,
	ARORIW & obj.AMask: iIEncoding,
	ARORW & obj.AMask:  rIIIEncoding,
	AORCB & obj.AMask:  iIEncoding,
	AREV8 & obj.AMask:  iIEncoding,

	// 1.5: Single-bit Instructions (Zbs)
	ABCLR & obj.AMask:  rIIIEncoding,
	ABCLRI & obj.AMask: iIEncoding,
	ABEXT & obj.AMask:  rIIIEncoding,
	ABEXTI & obj.AMask: iIEncoding,
	ABINV & obj.AMask:  rIIIEncoding,
	ABINVI & obj.AMask: iIEncoding,
	ABSET & obj.AMask:  rIIIEncoding,
	ABSETI & obj.AMask: iIEncoding,

	// Escape hatch
	AWORD & obj.AMask: rawEncoding,

	// Pseudo-operations
	obj.AFUNCDATA: pseudoOpEncoding,
	obj.APCDATA:   pseudoOpEncoding,
	obj.ATEXT:     pseudoOpEncoding,
	obj.ANOP:      pseudoOpEncoding,
	obj.ADUFFZERO: pseudoOpEncoding,
	obj.ADUFFCOPY: pseudoOpEncoding,
	obj.APCALIGN:  pseudoOpEncoding,
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
	p      *obj.Prog // Prog that instruction is for
	as     obj.As    // Assembler opcode
	rd     uint32    // Destination register
	rs1    uint32    // Source register 1
	rs2    uint32    // Source register 2
	rs3    uint32    // Source register 3
	imm    int64     // Immediate
	funct3 uint32    // Function 3
	funct7 uint32    // Function 7 (or Function 2)
}

func (ins *instruction) String() string {
	if ins.p == nil {
		return ins.as.String()
	}
	var suffix string
	if ins.p.As != ins.as {
		suffix = fmt.Sprintf(" (%v)", ins.as)
	}
	return fmt.Sprintf("%v%v", ins.p, suffix)
}

func (ins *instruction) encode() (uint32, error) {
	enc, err := encodingForAs(ins.as)
	if err != nil {
		return 0, err
	}
	if enc.length <= 0 {
		return 0, fmt.Errorf("%v: encoding called for a pseudo instruction", ins.as)
	}
	return enc.encode(ins), nil
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

func (ins *instruction) usesRegTmp() bool {
	return ins.rd == REG_TMP || ins.rs1 == REG_TMP || ins.rs2 == REG_TMP
}

// instructionForProg returns the default *obj.Prog to instruction mapping.
func instructionForProg(p *obj.Prog) *instruction {
	ins := &instruction{
		as:  p.As,
		rd:  uint32(p.To.Reg),
		rs1: uint32(p.Reg),
		rs2: uint32(p.From.Reg),
		imm: p.From.Offset,
	}
	if len(p.RestArgs) == 1 {
		ins.rs3 = uint32(p.RestArgs[0].Reg)
	}
	return ins
}

// instructionsForOpImmediate returns the machine instructions for an immediate
// operand. The instruction is specified by as and the source register is
// specified by rs, instead of the obj.Prog.
func instructionsForOpImmediate(p *obj.Prog, as obj.As, rs int16) []*instruction {
	// <opi> $imm, REG, TO
	ins := instructionForProg(p)
	ins.as, ins.rs1, ins.rs2 = as, uint32(rs), obj.REG_NONE

	low, high, err := Split32BitImmediate(ins.imm)
	if err != nil {
		p.Ctxt.Diag("%v: constant %d too large", p, ins.imm, err)
		return nil
	}
	if high == 0 {
		return []*instruction{ins}
	}

	// Split into two additions, if possible.
	// Do not split SP-writing instructions, as otherwise the recorded SP delta may be wrong.
	if p.Spadj == 0 && ins.as == AADDI && ins.imm >= -(1<<12) && ins.imm < 1<<12-1 {
		imm0 := ins.imm / 2
		imm1 := ins.imm - imm0

		// ADDI $(imm/2), REG, TO
		// ADDI $(imm-imm/2), TO, TO
		ins.imm = imm0
		insADDI := &instruction{as: AADDI, rd: ins.rd, rs1: ins.rd, imm: imm1}
		return []*instruction{ins, insADDI}
	}

	// LUI $high, TMP
	// ADDIW $low, TMP, TMP
	// <op> TMP, REG, TO
	insLUI := &instruction{as: ALUI, rd: REG_TMP, imm: high}
	insADDIW := &instruction{as: AADDIW, rd: REG_TMP, rs1: REG_TMP, imm: low}
	switch ins.as {
	case AADDI:
		ins.as = AADD
	case AANDI:
		ins.as = AAND
	case AORI:
		ins.as = AOR
	case AXORI:
		ins.as = AXOR
	default:
		p.Ctxt.Diag("unsupported immediate instruction %v for splitting", p)
		return nil
	}
	ins.rs2 = REG_TMP
	if low == 0 {
		return []*instruction{insLUI, ins}
	}
	return []*instruction{insLUI, insADDIW, ins}
}

// instructionsForLoad returns the machine instructions for a load. The load
// instruction is specified by as and the base/source register is specified
// by rs, instead of the obj.Prog.
func instructionsForLoad(p *obj.Prog, as obj.As, rs int16) []*instruction {
	if p.From.Type != obj.TYPE_MEM {
		p.Ctxt.Diag("%v requires memory for source", p)
		return nil
	}

	switch as {
	case ALD, ALB, ALH, ALW, ALBU, ALHU, ALWU, AFLW, AFLD:
	default:
		p.Ctxt.Diag("%v: unknown load instruction %v", p, as)
		return nil
	}

	// <load> $imm, REG, TO (load $imm+(REG), TO)
	ins := instructionForProg(p)
	ins.as, ins.rs1, ins.rs2 = as, uint32(rs), obj.REG_NONE
	ins.imm = p.From.Offset

	low, high, err := Split32BitImmediate(ins.imm)
	if err != nil {
		p.Ctxt.Diag("%v: constant %d too large", p, ins.imm)
		return nil
	}
	if high == 0 {
		return []*instruction{ins}
	}

	// LUI $high, TMP
	// ADD TMP, REG, TMP
	// <load> $low, TMP, TO
	insLUI := &instruction{as: ALUI, rd: REG_TMP, imm: high}
	insADD := &instruction{as: AADD, rd: REG_TMP, rs1: REG_TMP, rs2: ins.rs1}
	ins.rs1, ins.imm = REG_TMP, low

	return []*instruction{insLUI, insADD, ins}
}

// instructionsForStore returns the machine instructions for a store. The store
// instruction is specified by as and the target/source register is specified
// by rd, instead of the obj.Prog.
func instructionsForStore(p *obj.Prog, as obj.As, rd int16) []*instruction {
	if p.To.Type != obj.TYPE_MEM {
		p.Ctxt.Diag("%v requires memory for destination", p)
		return nil
	}

	switch as {
	case ASW, ASH, ASB, ASD, AFSW, AFSD:
	default:
		p.Ctxt.Diag("%v: unknown store instruction %v", p, as)
		return nil
	}

	// <store> $imm, REG, TO (store $imm+(TO), REG)
	ins := instructionForProg(p)
	ins.as, ins.rd, ins.rs1, ins.rs2 = as, uint32(rd), uint32(p.From.Reg), obj.REG_NONE
	ins.imm = p.To.Offset

	low, high, err := Split32BitImmediate(ins.imm)
	if err != nil {
		p.Ctxt.Diag("%v: constant %d too large", p, ins.imm)
		return nil
	}
	if high == 0 {
		return []*instruction{ins}
	}

	// LUI $high, TMP
	// ADD TMP, TO, TMP
	// <store> $low, REG, TMP
	insLUI := &instruction{as: ALUI, rd: REG_TMP, imm: high}
	insADD := &instruction{as: AADD, rd: REG_TMP, rs1: REG_TMP, rs2: ins.rd}
	ins.rd, ins.imm = REG_TMP, low

	return []*instruction{insLUI, insADD, ins}
}

func instructionsForTLS(p *obj.Prog, ins *instruction) []*instruction {
	insAddTP := &instruction{as: AADD, rd: REG_TMP, rs1: REG_TMP, rs2: REG_TP}

	var inss []*instruction
	if p.Ctxt.Flag_shared {
		// TLS initial-exec mode - load TLS offset from GOT, add the thread pointer
		// register, then load from or store to the resulting memory location.
		insAUIPC := &instruction{as: AAUIPC, rd: REG_TMP}
		insLoadTLSOffset := &instruction{as: ALD, rd: REG_TMP, rs1: REG_TMP}
		inss = []*instruction{insAUIPC, insLoadTLSOffset, insAddTP, ins}
	} else {
		// TLS local-exec mode - load upper TLS offset, add the lower TLS offset,
		// add the thread pointer register, then load from or store to the resulting
		// memory location. Note that this differs from the suggested three
		// instruction sequence, as the Go linker does not currently have an
		// easy way to handle relocation across 12 bytes of machine code.
		insLUI := &instruction{as: ALUI, rd: REG_TMP}
		insADDIW := &instruction{as: AADDIW, rd: REG_TMP, rs1: REG_TMP}
		inss = []*instruction{insLUI, insADDIW, insAddTP, ins}
	}
	return inss
}

func instructionsForTLSLoad(p *obj.Prog) []*instruction {
	if p.From.Sym.Type != objabi.STLSBSS {
		p.Ctxt.Diag("%v: %v is not a TLS symbol", p, p.From.Sym)
		return nil
	}

	ins := instructionForProg(p)
	ins.as, ins.rs1, ins.rs2, ins.imm = movToLoad(p.As), REG_TMP, obj.REG_NONE, 0

	return instructionsForTLS(p, ins)
}

func instructionsForTLSStore(p *obj.Prog) []*instruction {
	if p.To.Sym.Type != objabi.STLSBSS {
		p.Ctxt.Diag("%v: %v is not a TLS symbol", p, p.To.Sym)
		return nil
	}

	ins := instructionForProg(p)
	ins.as, ins.rd, ins.rs1, ins.rs2, ins.imm = movToStore(p.As), REG_TMP, uint32(p.From.Reg), obj.REG_NONE, 0

	return instructionsForTLS(p, ins)
}

// instructionsForMOV returns the machine instructions for an *obj.Prog that
// uses a MOV pseudo-instruction.
func instructionsForMOV(p *obj.Prog) []*instruction {
	ins := instructionForProg(p)
	inss := []*instruction{ins}

	if p.Reg != 0 {
		p.Ctxt.Diag("%v: illegal MOV instruction", p)
		return nil
	}

	switch {
	case p.From.Type == obj.TYPE_CONST && p.To.Type == obj.TYPE_REG:
		// Handle constant to register moves.
		if p.As != AMOV {
			p.Ctxt.Diag("%v: unsupported constant load", p)
			return nil
		}

		// For constants larger than 32 bits in size that have trailing zeros,
		// use the value with the trailing zeros removed and then use a SLLI
		// instruction to restore the original constant.
		// For example:
		// 	MOV $0x8000000000000000, X10
		// becomes
		// 	MOV $1, X10
		// 	SLLI $63, X10, X10
		var insSLLI *instruction
		if err := immIFits(ins.imm, 32); err != nil {
			ctz := bits.TrailingZeros64(uint64(ins.imm))
			if err := immIFits(ins.imm>>ctz, 32); err == nil {
				ins.imm = ins.imm >> ctz
				insSLLI = &instruction{as: ASLLI, rd: ins.rd, rs1: ins.rd, imm: int64(ctz)}
			}
		}

		low, high, err := Split32BitImmediate(ins.imm)
		if err != nil {
			p.Ctxt.Diag("%v: constant %d too large: %v", p, ins.imm, err)
			return nil
		}

		// MOV $c, R -> ADD $c, ZERO, R
		ins.as, ins.rs1, ins.rs2, ins.imm = AADDI, REG_ZERO, obj.REG_NONE, low

		// LUI is only necessary if the constant does not fit in 12 bits.
		if high != 0 {
			// LUI top20bits(c), R
			// ADD bottom12bits(c), R, R
			insLUI := &instruction{as: ALUI, rd: ins.rd, imm: high}
			inss = []*instruction{insLUI}
			if low != 0 {
				ins.as, ins.rs1 = AADDIW, ins.rd
				inss = append(inss, ins)
			}
		}
		if insSLLI != nil {
			inss = append(inss, insSLLI)
		}

	case p.From.Type == obj.TYPE_CONST && p.To.Type != obj.TYPE_REG:
		p.Ctxt.Diag("%v: constant load must target register", p)
		return nil

	case p.From.Type == obj.TYPE_REG && p.To.Type == obj.TYPE_REG:
		// Handle register to register moves.
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
			if buildcfg.GORISCV64 >= 22 {
				// Use SEXTB or SEXTH to extend.
				ins.as, ins.rs1, ins.rs2 = ASEXTB, uint32(p.From.Reg), obj.REG_NONE
				if p.As == AMOVH {
					ins.as = ASEXTH
				}
			} else {
				// Use SLLI/SRAI sequence to extend.
				ins.as, ins.rs1, ins.rs2 = ASLLI, uint32(p.From.Reg), obj.REG_NONE
				if p.As == AMOVB {
					ins.imm = 56
				} else if p.As == AMOVH {
					ins.imm = 48
				}
				ins2 := &instruction{as: ASRAI, rd: ins.rd, rs1: ins.rd, imm: ins.imm}
				inss = append(inss, ins2)
			}
		case AMOVHU, AMOVWU:
			if buildcfg.GORISCV64 >= 22 {
				// Use ZEXTH or ADDUW to extend.
				ins.as, ins.rs1, ins.rs2, ins.imm = AZEXTH, uint32(p.From.Reg), obj.REG_NONE, 0
				if p.As == AMOVWU {
					ins.as, ins.rs2 = AADDUW, REG_ZERO
				}
			} else {
				// Use SLLI/SRLI sequence to extend.
				ins.as, ins.rs1, ins.rs2 = ASLLI, uint32(p.From.Reg), obj.REG_NONE
				if p.As == AMOVHU {
					ins.imm = 48
				} else if p.As == AMOVWU {
					ins.imm = 32
				}
				ins2 := &instruction{as: ASRLI, rd: ins.rd, rs1: ins.rd, imm: ins.imm}
				inss = append(inss, ins2)
			}
		}

	case p.From.Type == obj.TYPE_MEM && p.To.Type == obj.TYPE_REG:
		// Memory to register loads.
		switch p.From.Name {
		case obj.NAME_AUTO, obj.NAME_PARAM, obj.NAME_NONE:
			// MOV c(Rs), Rd -> L $c, Rs, Rd
			inss = instructionsForLoad(p, movToLoad(p.As), addrToReg(p.From))

		case obj.NAME_EXTERN, obj.NAME_STATIC:
			if p.From.Sym.Type == objabi.STLSBSS {
				return instructionsForTLSLoad(p)
			}

			// Note that the values for $off_hi and $off_lo are currently
			// zero and will be assigned during relocation.
			//
			// AUIPC $off_hi, Rd
			// L $off_lo, Rd, Rd
			insAUIPC := &instruction{as: AAUIPC, rd: ins.rd}
			ins.as, ins.rs1, ins.rs2, ins.imm = movToLoad(p.As), ins.rd, obj.REG_NONE, 0
			inss = []*instruction{insAUIPC, ins}

		default:
			p.Ctxt.Diag("unsupported name %d for %v", p.From.Name, p)
			return nil
		}

	case p.From.Type == obj.TYPE_REG && p.To.Type == obj.TYPE_MEM:
		// Register to memory stores.
		switch p.As {
		case AMOVBU, AMOVHU, AMOVWU:
			p.Ctxt.Diag("%v: unsupported unsigned store", p)
			return nil
		}
		switch p.To.Name {
		case obj.NAME_AUTO, obj.NAME_PARAM, obj.NAME_NONE:
			// MOV Rs, c(Rd) -> S $c, Rs, Rd
			inss = instructionsForStore(p, movToStore(p.As), addrToReg(p.To))

		case obj.NAME_EXTERN, obj.NAME_STATIC:
			if p.To.Sym.Type == objabi.STLSBSS {
				return instructionsForTLSStore(p)
			}

			// Note that the values for $off_hi and $off_lo are currently
			// zero and will be assigned during relocation.
			//
			// AUIPC $off_hi, Rtmp
			// S $off_lo, Rtmp, Rd
			insAUIPC := &instruction{as: AAUIPC, rd: REG_TMP}
			ins.as, ins.rd, ins.rs1, ins.rs2, ins.imm = movToStore(p.As), REG_TMP, uint32(p.From.Reg), obj.REG_NONE, 0
			inss = []*instruction{insAUIPC, ins}

		default:
			p.Ctxt.Diag("unsupported name %d for %v", p.From.Name, p)
			return nil
		}

	case p.From.Type == obj.TYPE_ADDR && p.To.Type == obj.TYPE_REG:
		// MOV $sym+off(SP/SB), R
		if p.As != AMOV {
			p.Ctxt.Diag("%v: unsupported address load", p)
			return nil
		}
		switch p.From.Name {
		case obj.NAME_AUTO, obj.NAME_PARAM, obj.NAME_NONE:
			inss = instructionsForOpImmediate(p, AADDI, addrToReg(p.From))

		case obj.NAME_EXTERN, obj.NAME_STATIC:
			// Note that the values for $off_hi and $off_lo are currently
			// zero and will be assigned during relocation.
			//
			// AUIPC $off_hi, R
			// ADDI $off_lo, R
			insAUIPC := &instruction{as: AAUIPC, rd: ins.rd}
			ins.as, ins.rs1, ins.rs2, ins.imm = AADDI, ins.rd, obj.REG_NONE, 0
			inss = []*instruction{insAUIPC, ins}

		default:
			p.Ctxt.Diag("unsupported name %d for %v", p.From.Name, p)
			return nil
		}

	case p.From.Type == obj.TYPE_ADDR && p.To.Type != obj.TYPE_REG:
		p.Ctxt.Diag("%v: address load must target register", p)
		return nil

	default:
		p.Ctxt.Diag("%v: unsupported MOV", p)
		return nil
	}

	return inss
}

// instructionsForRotate returns the machine instructions for a bitwise rotation.
func instructionsForRotate(p *obj.Prog, ins *instruction) []*instruction {
	if buildcfg.GORISCV64 >= 22 {
		// Rotation instructions are supported natively.
		return []*instruction{ins}
	}

	switch ins.as {
	case AROL, AROLW, AROR, ARORW:
		// ROL -> OR (SLL x y) (SRL x (NEG y))
		// ROR -> OR (SRL x y) (SLL x (NEG y))
		sllOp, srlOp := ASLL, ASRL
		if ins.as == AROLW || ins.as == ARORW {
			sllOp, srlOp = ASLLW, ASRLW
		}
		shift1, shift2 := sllOp, srlOp
		if ins.as == AROR || ins.as == ARORW {
			shift1, shift2 = shift2, shift1
		}
		return []*instruction{
			&instruction{as: ASUB, rs1: REG_ZERO, rs2: ins.rs2, rd: REG_TMP},
			&instruction{as: shift2, rs1: ins.rs1, rs2: REG_TMP, rd: REG_TMP},
			&instruction{as: shift1, rs1: ins.rs1, rs2: ins.rs2, rd: ins.rd},
			&instruction{as: AOR, rs1: REG_TMP, rs2: ins.rd, rd: ins.rd},
		}

	case ARORI, ARORIW:
		// ROR -> OR (SLLI -x y) (SRLI x y)
		sllOp, srlOp := ASLLI, ASRLI
		sllImm := int64(int8(-ins.imm) & 63)
		if ins.as == ARORIW {
			sllOp, srlOp = ASLLIW, ASRLIW
			sllImm = int64(int8(-ins.imm) & 31)
		}
		return []*instruction{
			&instruction{as: srlOp, rs1: ins.rs1, rd: REG_TMP, imm: ins.imm},
			&instruction{as: sllOp, rs1: ins.rs1, rd: ins.rd, imm: sllImm},
			&instruction{as: AOR, rs1: REG_TMP, rs2: ins.rd, rd: ins.rd},
		}

	default:
		p.Ctxt.Diag("%v: unknown rotation", p)
		return nil
	}
}

// instructionsForProg returns the machine instructions for an *obj.Prog.
func instructionsForProg(p *obj.Prog) []*instruction {
	ins := instructionForProg(p)
	inss := []*instruction{ins}

	if len(p.RestArgs) > 1 {
		p.Ctxt.Diag("too many source registers")
		return nil
	}

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
		inss = instructionsForMOV(p)

	case ALW, ALWU, ALH, ALHU, ALB, ALBU, ALD, AFLW, AFLD:
		inss = instructionsForLoad(p, ins.as, p.From.Reg)

	case ASW, ASH, ASB, ASD, AFSW, AFSD:
		inss = instructionsForStore(p, ins.as, p.To.Reg)

	case ALRW, ALRD:
		// Set aq to use acquire access ordering
		ins.funct7 = 2
		ins.rs1, ins.rs2 = uint32(p.From.Reg), REG_ZERO

	case AADDI, AANDI, AORI, AXORI:
		inss = instructionsForOpImmediate(p, ins.as, p.Reg)

	case ASCW, ASCD:
		// Set release access ordering
		ins.funct7 = 1
		ins.rd, ins.rs1, ins.rs2 = uint32(p.RegTo2), uint32(p.To.Reg), uint32(p.From.Reg)

	case AAMOSWAPW, AAMOSWAPD, AAMOADDW, AAMOADDD, AAMOANDW, AAMOANDD, AAMOORW, AAMOORD,
		AAMOXORW, AAMOXORD, AAMOMINW, AAMOMIND, AAMOMINUW, AAMOMINUD, AAMOMAXW, AAMOMAXD, AAMOMAXUW, AAMOMAXUD:
		// Set aqrl to use acquire & release access ordering
		ins.funct7 = 3
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
		// Set the default rounding mode in funct3 to round to zero.
		if p.Scond&rmSuffixBit == 0 {
			ins.funct3 = uint32(RM_RTZ)
		} else {
			ins.funct3 = uint32(p.Scond &^ rmSuffixBit)
		}

	case AFNES, AFNED:
		// Replace FNE[SD] with FEQ[SD] and NOT.
		if p.To.Type != obj.TYPE_REG {
			p.Ctxt.Diag("%v needs an integer register output", p)
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

	case AFMADDS, AFMSUBS, AFNMADDS, AFNMSUBS,
		AFMADDD, AFMSUBD, AFNMADDD, AFNMSUBD:
		// Swap the first two operands so that the operands are in the same
		// order as they are in the specification: RS1, RS2, RS3, RD.
		ins.rs1, ins.rs2 = ins.rs2, ins.rs1

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
		ins.rs1, ins.rs2 = uint32(p.From.Reg), obj.REG_NONE
		ins.imm = 1

	case ASNEZ:
		// SNEZ rs, rd -> SLTU rs, x0, rd
		ins.as = ASLTU
		ins.rs1 = REG_ZERO

	case AFABSS:
		// FABSS rs, rd -> FSGNJXS rs, rs, rd
		ins.as = AFSGNJXS
		ins.rs1 = uint32(p.From.Reg)

	case AFABSD:
		// FABSD rs, rd -> FSGNJXD rs, rs, rd
		ins.as = AFSGNJXD
		ins.rs1 = uint32(p.From.Reg)

	case AFNEGS:
		// FNEGS rs, rd -> FSGNJNS rs, rs, rd
		ins.as = AFSGNJNS
		ins.rs1 = uint32(p.From.Reg)

	case AFNEGD:
		// FNEGD rs, rd -> FSGNJND rs, rs, rd
		ins.as = AFSGNJND
		ins.rs1 = uint32(p.From.Reg)

	case AROL, AROLW, AROR, ARORW:
		inss = instructionsForRotate(p, ins)

	case ARORI:
		if ins.imm < 0 || ins.imm > 63 {
			p.Ctxt.Diag("%v: immediate out of range 0 to 63", p)
		}
		inss = instructionsForRotate(p, ins)

	case ARORIW:
		if ins.imm < 0 || ins.imm > 31 {
			p.Ctxt.Diag("%v: immediate out of range 0 to 31", p)
		}
		inss = instructionsForRotate(p, ins)

	case ASLLI, ASRLI, ASRAI:
		if ins.imm < 0 || ins.imm > 63 {
			p.Ctxt.Diag("%v: immediate out of range 0 to 63", p)
		}

	case ASLLIW, ASRLIW, ASRAIW:
		if ins.imm < 0 || ins.imm > 31 {
			p.Ctxt.Diag("%v: immediate out of range 0 to 31", p)
		}

	case ACLZ, ACLZW, ACTZ, ACTZW, ACPOP, ACPOPW, ASEXTB, ASEXTH, AZEXTH:
		ins.rs1, ins.rs2 = uint32(p.From.Reg), obj.REG_NONE

	case AORCB, AREV8:
		ins.rd, ins.rs1, ins.rs2 = uint32(p.To.Reg), uint32(p.From.Reg), obj.REG_NONE
	}

	for _, ins := range inss {
		ins.p = p
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

	// If errors were encountered during preprocess/validation, proceeding
	// and attempting to encode said instructions will only lead to panics.
	if ctxt.Errors > 0 {
		return
	}

	for p := cursym.Func().Text; p != nil; p = p.Link {
		switch p.As {
		case AJAL:
			if p.Mark&NEED_JAL_RELOC == NEED_JAL_RELOC {
				rel := obj.Addrel(cursym)
				rel.Off = int32(p.Pc)
				rel.Siz = 4
				rel.Sym = p.To.Sym
				rel.Add = p.To.Offset
				rel.Type = objabi.R_RISCV_JAL
			}
		case AJALR:
			if p.To.Sym != nil {
				ctxt.Diag("%v: unexpected AJALR with to symbol", p)
			}

		case AAUIPC, AMOV, AMOVB, AMOVH, AMOVW, AMOVBU, AMOVHU, AMOVWU, AMOVF, AMOVD:
			var addr *obj.Addr
			var rt objabi.RelocType
			if p.Mark&NEED_CALL_RELOC == NEED_CALL_RELOC {
				rt = objabi.R_RISCV_CALL
				addr = &p.From
			} else if p.Mark&NEED_PCREL_ITYPE_RELOC == NEED_PCREL_ITYPE_RELOC {
				rt = objabi.R_RISCV_PCREL_ITYPE
				addr = &p.From
			} else if p.Mark&NEED_PCREL_STYPE_RELOC == NEED_PCREL_STYPE_RELOC {
				rt = objabi.R_RISCV_PCREL_STYPE
				addr = &p.To
			} else {
				break
			}
			if p.As == AAUIPC {
				if p.Link == nil {
					ctxt.Diag("AUIPC needing PC-relative reloc missing following instruction")
					break
				}
				addr = &p.RestArgs[0].Addr
			}
			if addr.Sym == nil {
				ctxt.Diag("PC-relative relocation missing symbol")
				break
			}
			if addr.Sym.Type == objabi.STLSBSS {
				if ctxt.Flag_shared {
					rt = objabi.R_RISCV_TLS_IE
				} else {
					rt = objabi.R_RISCV_TLS_LE
				}
			}

			rel := obj.Addrel(cursym)
			rel.Off = int32(p.Pc)
			rel.Siz = 8
			rel.Sym = addr.Sym
			rel.Add = addr.Offset
			rel.Type = rt

		case obj.APCALIGN:
			alignedValue := p.From.Offset
			v := pcAlignPadLength(p.Pc, alignedValue)
			offset := p.Pc
			for ; v >= 4; v -= 4 {
				// NOP
				cursym.WriteBytes(ctxt, offset, []byte{0x13, 0, 0, 0})
				offset += 4
			}
			continue
		}

		offset := p.Pc
		for _, ins := range instructionsForProg(p) {
			if ic, err := ins.encode(); err == nil {
				cursym.WriteInt(ctxt, offset, ins.length(), int64(ic))
				offset += int64(ins.length())
			}
			if ins.usesRegTmp() {
				p.Mark |= USES_REG_TMP
			}
		}
	}

	obj.MarkUnsafePoints(ctxt, cursym.Func().Text, newprog, isUnsafePoint, nil)
}

func isUnsafePoint(p *obj.Prog) bool {
	return p.Mark&USES_REG_TMP == USES_REG_TMP || p.From.Reg == REG_TMP || p.To.Reg == REG_TMP || p.Reg == REG_TMP
}

func ParseSuffix(prog *obj.Prog, cond string) (err error) {
	switch prog.As {
	case AFCVTWS, AFCVTLS, AFCVTWUS, AFCVTLUS, AFCVTWD, AFCVTLD, AFCVTWUD, AFCVTLUD:
		prog.Scond, err = rmSuffixEncode(strings.TrimPrefix(cond, "."))
	}
	return
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
