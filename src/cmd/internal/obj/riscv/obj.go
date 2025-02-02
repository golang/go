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
	"cmd/internal/src"
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
	insData, err := instructionDataForAs(p.As)
	if err != nil {
		panic(fmt.Sprintf("failed to lookup instruction data for %v: %v", p.As, err))
	}

	// Expand binary instructions to ternary ones.
	if p.Reg == obj.REG_NONE {
		if insData.ternary {
			p.Reg = p.To.Reg
		}
	}

	// Rewrite instructions with constant operands to refer to the immediate
	// form of the instruction.
	if p.From.Type == obj.TYPE_CONST {
		switch p.As {
		case ASUB:
			p.As, p.From.Offset = AADDI, -p.From.Offset
		case ASUBW:
			p.As, p.From.Offset = AADDIW, -p.From.Offset
		default:
			if insData.immForm != obj.AXXX {
				p.As = insData.immForm
			}
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

	case AFMVXS:
		// FMVXS is the old name for FMVXW.
		p.As = AFMVXW

	case AFMVSX:
		// FMVSX is the old name for FMVWX.
		p.As = AFMVWX

	case ASCALL:
		// SCALL is the old name for ECALL.
		p.As = AECALL

	case ASBREAK:
		// SBREAK is the old name for EBREAK.
		p.As = AEBREAK

	case AMOV:
		if p.From.Type == obj.TYPE_CONST && p.From.Name == obj.NAME_NONE && p.From.Reg == obj.REG_NONE && int64(int32(p.From.Offset)) != p.From.Offset {
			if isShiftConst(p.From.Offset) {
				break
			}
			// Put >32-bit constants in memory and load them.
			p.From.Type = obj.TYPE_MEM
			p.From.Sym = ctxt.Int64Sym(p.From.Offset)
			p.From.Name = obj.NAME_EXTERN
			p.From.Offset = 0
		}

	case AMOVD:
		if p.From.Type == obj.TYPE_FCONST && p.From.Name == obj.NAME_NONE && p.From.Reg == obj.REG_NONE {
			f64 := p.From.Val.(float64)
			p.From.Type = obj.TYPE_MEM
			p.From.Sym = ctxt.Float64Sym(f64)
			p.From.Name = obj.NAME_EXTERN
			p.From.Offset = 0
		}
	}

	if ctxt.Flag_dynlink {
		rewriteToUseGot(ctxt, p, newprog)
	}
}

// Rewrite p, if necessary, to access global data via the global offset table.
func rewriteToUseGot(ctxt *obj.Link, p *obj.Prog, newprog obj.ProgAlloc) {
	if p.As == obj.ADUFFCOPY || p.As == obj.ADUFFZERO {
		//     ADUFFxxx $offset
		// becomes
		//     MOV runtime.duffxxx@GOT, REG_TMP
		//     ADD $offset, REG_TMP
		//     CALL REG_TMP
		var sym *obj.LSym
		if p.As == obj.ADUFFCOPY {
			sym = ctxt.LookupABI("runtime.duffcopy", obj.ABIInternal)
		} else {
			sym = ctxt.LookupABI("runtime.duffzero", obj.ABIInternal)
		}
		offset := p.To.Offset
		p.As = AMOV
		p.From.Type = obj.TYPE_MEM
		p.From.Name = obj.NAME_GOTREF
		p.From.Sym = sym
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_TMP
		p.To.Name = obj.NAME_NONE
		p.To.Offset = 0
		p.To.Sym = nil

		p1 := obj.Appendp(p, newprog)
		p1.As = AADD
		p1.From.Type = obj.TYPE_CONST
		p1.From.Offset = offset
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = REG_TMP

		p2 := obj.Appendp(p1, newprog)
		p2.As = obj.ACALL
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = REG_TMP
	}

	// We only care about global data: NAME_EXTERN means a global
	// symbol in the Go sense and p.Sym.Local is true for a few internally
	// defined symbols.
	if p.From.Type == obj.TYPE_ADDR && p.From.Name == obj.NAME_EXTERN && !p.From.Sym.Local() {
		// MOV $sym, Rx becomes MOV sym@GOT, Rx
		// MOV $sym+<off>, Rx becomes MOV sym@GOT, Rx; ADD <off>, Rx
		if p.As != AMOV {
			ctxt.Diag("don't know how to handle TYPE_ADDR in %v with -dynlink", p)
		}
		if p.To.Type != obj.TYPE_REG {
			ctxt.Diag("don't know how to handle LD instruction to non-register in %v with -dynlink", p)
		}
		p.From.Type = obj.TYPE_MEM
		p.From.Name = obj.NAME_GOTREF
		if p.From.Offset != 0 {
			q := obj.Appendp(p, newprog)
			q.As = AADD
			q.From.Type = obj.TYPE_CONST
			q.From.Offset = p.From.Offset
			q.To = p.To
			p.From.Offset = 0
		}

	}

	if p.GetFrom3() != nil && p.GetFrom3().Name == obj.NAME_EXTERN {
		ctxt.Diag("don't know how to handle %v with -dynlink", p)
	}

	var source *obj.Addr
	// MOVx sym, Ry becomes MOV sym@GOT, X31; MOVx (X31), Ry
	// MOVx Ry, sym becomes MOV sym@GOT, X31; MOV Ry, (X31)
	// An addition may be inserted between the two MOVs if there is an offset.
	if p.From.Name == obj.NAME_EXTERN && !p.From.Sym.Local() {
		if p.To.Name == obj.NAME_EXTERN && !p.To.Sym.Local() {
			ctxt.Diag("cannot handle NAME_EXTERN on both sides in %v with -dynlink", p)
		}
		source = &p.From
	} else if p.To.Name == obj.NAME_EXTERN && !p.To.Sym.Local() {
		source = &p.To
	} else {
		return
	}
	if p.As == obj.ATEXT || p.As == obj.AFUNCDATA || p.As == obj.ACALL || p.As == obj.ARET || p.As == obj.AJMP {
		return
	}
	if source.Sym.Type == objabi.STLSBSS {
		return
	}
	if source.Type != obj.TYPE_MEM {
		ctxt.Diag("don't know how to handle %v with -dynlink", p)
	}
	p1 := obj.Appendp(p, newprog)
	p1.As = AMOV
	p1.From.Type = obj.TYPE_MEM
	p1.From.Sym = source.Sym
	p1.From.Name = obj.NAME_GOTREF
	p1.To.Type = obj.TYPE_REG
	p1.To.Reg = REG_TMP

	p2 := obj.Appendp(p1, newprog)
	p2.As = p.As
	p2.From = p.From
	p2.To = p.To
	if p.From.Name == obj.NAME_EXTERN {
		p2.From.Reg = REG_TMP
		p2.From.Name = obj.NAME_NONE
		p2.From.Sym = nil
	} else if p.To.Name == obj.NAME_EXTERN {
		p2.To.Reg = REG_TMP
		p2.To.Name = obj.NAME_NONE
		p2.To.Sym = nil
	} else {
		return
	}
	obj.Nopout(p)

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
			case obj.NAME_GOTREF:
				p.Mark |= NEED_GOT_PCREL_ITYPE_RELOC
			}
		case p.From.Type == obj.TYPE_MEM && p.To.Type == obj.TYPE_REG:
			switch p.From.Name {
			case obj.NAME_EXTERN, obj.NAME_STATIC:
				p.Mark |= NEED_PCREL_ITYPE_RELOC
			case obj.NAME_GOTREF:
				p.Mark |= NEED_GOT_PCREL_ITYPE_RELOC
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

	q := prologue

	if stacksize != 0 {
		prologue = ctxt.StartUnsafePoint(prologue, newprog)

		// Actually save LR.
		prologue = obj.Appendp(prologue, newprog)
		prologue.As = AMOV
		prologue.Pos = q.Pos
		prologue.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_LR}
		prologue.To = obj.Addr{Type: obj.TYPE_MEM, Reg: REG_SP, Offset: -stacksize}

		// Insert stack adjustment.
		prologue = obj.Appendp(prologue, newprog)
		prologue.As = AADDI
		prologue.Pos = q.Pos
		prologue.Pos = prologue.Pos.WithXlogue(src.PosPrologueEnd)
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
					ctxt.Diag("%v: instruction with branch-like opcode lacks destination", p)
					break
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

		// Return if errors have been detected up to this point. Continuing
		// may lead to duplicate errors being output.
		if ctxt.Errors > 0 {
			return
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
				if ctxt.Errors == 0 {
					// An error should have already been reported for this instruction
					panic("unhandled type")
				}
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

// regV returns a vector register.
func regV(r uint32) uint32 {
	return regVal(r, REG_V0, REG_V31)
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

func immFits(x int64, nbits uint, signed bool) error {
	label := "unsigned"
	min, max := int64(0), int64(1)<<nbits-1
	if signed {
		label = "signed"
		sbits := nbits - 1
		min, max = int64(-1)<<sbits, int64(1)<<sbits-1
	}
	if x < min || x > max {
		if nbits <= 16 {
			return fmt.Errorf("%s immediate %d must be in range [%d, %d] (%d bits)", label, x, min, max, nbits)
		}
		return fmt.Errorf("%s immediate %#x must be in range [%#x, %#x] (%d bits)", label, x, min, max, nbits)
	}
	return nil
}

// immIFits checks whether the immediate value x fits in nbits bits
// as a signed integer. If it does not, an error is returned.
func immIFits(x int64, nbits uint) error {
	return immFits(x, nbits, true)
}

// immI extracts the signed integer of the specified size from an immediate.
func immI(as obj.As, imm int64, nbits uint) uint32 {
	if err := immIFits(imm, nbits); err != nil {
		panic(fmt.Sprintf("%v: %v", as, err))
	}
	return uint32(imm) & ((1 << nbits) - 1)
}

func wantImmI(ctxt *obj.Link, ins *instruction, imm int64, nbits uint) {
	if err := immIFits(imm, nbits); err != nil {
		ctxt.Diag("%v: %v", ins, err)
	}
}

// immUFits checks whether the immediate value x fits in nbits bits
// as an unsigned integer. If it does not, an error is returned.
func immUFits(x int64, nbits uint) error {
	return immFits(x, nbits, false)
}

// immU extracts the unsigned integer of the specified size from an immediate.
func immU(as obj.As, imm int64, nbits uint) uint32 {
	if err := immUFits(imm, nbits); err != nil {
		panic(fmt.Sprintf("%v: %v", as, err))
	}
	return uint32(imm) & ((1 << nbits) - 1)
}

func wantImmU(ctxt *obj.Link, ins *instruction, imm int64, nbits uint) {
	if err := immUFits(imm, nbits); err != nil {
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

// wantVectorReg checks that r is a vector register.
func wantVectorReg(ctxt *obj.Link, ins *instruction, pos string, r uint32) {
	wantReg(ctxt, ins, pos, "vector", r, REG_V0, REG_V31)
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

func validateRFF(ctxt *obj.Link, ins *instruction) {
	wantFloatReg(ctxt, ins, "rd", ins.rd)
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

func validateRVFV(ctxt *obj.Link, ins *instruction) {
	wantVectorReg(ctxt, ins, "vd", ins.rd)
	wantFloatReg(ctxt, ins, "rs1", ins.rs1)
	wantVectorReg(ctxt, ins, "vs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateRVIV(ctxt *obj.Link, ins *instruction) {
	wantVectorReg(ctxt, ins, "vd", ins.rd)
	wantIntReg(ctxt, ins, "rs1", ins.rs1)
	wantVectorReg(ctxt, ins, "vs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateRVV(ctxt *obj.Link, ins *instruction) {
	wantVectorReg(ctxt, ins, "vd", ins.rd)
	wantNoneReg(ctxt, ins, "rs1", ins.rs1)
	wantVectorReg(ctxt, ins, "vs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateRVVi(ctxt *obj.Link, ins *instruction) {
	wantImmI(ctxt, ins, ins.imm, 5)
	wantVectorReg(ctxt, ins, "vd", ins.rd)
	wantNoneReg(ctxt, ins, "rs1", ins.rs1)
	wantVectorReg(ctxt, ins, "vs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateRVVu(ctxt *obj.Link, ins *instruction) {
	wantImmU(ctxt, ins, ins.imm, 5)
	wantVectorReg(ctxt, ins, "vd", ins.rd)
	wantNoneReg(ctxt, ins, "rs1", ins.rs1)
	wantVectorReg(ctxt, ins, "vs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateRVVV(ctxt *obj.Link, ins *instruction) {
	wantVectorReg(ctxt, ins, "vd", ins.rd)
	wantVectorReg(ctxt, ins, "vs1", ins.rs1)
	wantVectorReg(ctxt, ins, "vs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateIII(ctxt *obj.Link, ins *instruction) {
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

func validateIV(ctxt *obj.Link, ins *instruction) {
	wantVectorReg(ctxt, ins, "vd", ins.rd)
	wantIntReg(ctxt, ins, "rs1", ins.rs1)
	wantNoneReg(ctxt, ins, "rs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateIIIV(ctxt *obj.Link, ins *instruction) {
	wantVectorReg(ctxt, ins, "vd", ins.rd)
	wantIntReg(ctxt, ins, "rs1", ins.rs1)
	wantIntReg(ctxt, ins, "rs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateIVIV(ctxt *obj.Link, ins *instruction) {
	wantVectorReg(ctxt, ins, "vd", ins.rd)
	wantIntReg(ctxt, ins, "rs1", ins.rs1)
	wantVectorReg(ctxt, ins, "vs2", ins.rs2)
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

func validateSV(ctxt *obj.Link, ins *instruction) {
	wantIntReg(ctxt, ins, "rd", ins.rd)
	wantVectorReg(ctxt, ins, "vs1", ins.rs1)
	wantNoneReg(ctxt, ins, "rs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateSVII(ctxt *obj.Link, ins *instruction) {
	wantVectorReg(ctxt, ins, "vd", ins.rd)
	wantIntReg(ctxt, ins, "rs1", ins.rs1)
	wantIntReg(ctxt, ins, "rs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateSVIV(ctxt *obj.Link, ins *instruction) {
	wantVectorReg(ctxt, ins, "vd", ins.rd)
	wantIntReg(ctxt, ins, "rs1", ins.rs1)
	wantVectorReg(ctxt, ins, "vs2", ins.rs2)
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

func validateVsetvli(ctxt *obj.Link, ins *instruction) {
	wantImmU(ctxt, ins, ins.imm, 11)
	wantIntReg(ctxt, ins, "rd", ins.rd)
	wantIntReg(ctxt, ins, "rs1", ins.rs1)
	wantNoneReg(ctxt, ins, "rs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateVsetivli(ctxt *obj.Link, ins *instruction) {
	wantImmU(ctxt, ins, ins.imm, 10)
	wantIntReg(ctxt, ins, "rd", ins.rd)
	wantImmU(ctxt, ins, int64(ins.rs1), 5)
	wantNoneReg(ctxt, ins, "rs2", ins.rs2)
	wantNoneReg(ctxt, ins, "rs3", ins.rs3)
}

func validateVsetvl(ctxt *obj.Link, ins *instruction) {
	wantIntReg(ctxt, ins, "rd", ins.rd)
	wantIntReg(ctxt, ins, "rs1", ins.rs1)
	wantIntReg(ctxt, ins, "rs2", ins.rs2)
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
	if enc.rs1 != 0 && rs1 != 0 {
		panic("encodeR: instruction uses rs1, but rs1 is nonzero")
	}
	if enc.rs2 != 0 && rs2 != 0 {
		panic("encodeR: instruction uses rs2, but rs2 is nonzero")
	}
	funct3 |= enc.funct3
	funct7 |= enc.funct7
	rs1 |= enc.rs1
	rs2 |= enc.rs2
	return funct7<<25 | rs2<<20 | rs1<<15 | funct3<<12 | rd<<7 | enc.opcode
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

func encodeRFF(ins *instruction) uint32 {
	return encodeR(ins.as, regF(ins.rs2), 0, regF(ins.rd), ins.funct3, ins.funct7)
}

func encodeRIF(ins *instruction) uint32 {
	return encodeR(ins.as, regI(ins.rs2), 0, regF(ins.rd), ins.funct3, ins.funct7)
}

func encodeRVFV(ins *instruction) uint32 {
	return encodeR(ins.as, regF(ins.rs1), regV(ins.rs2), regV(ins.rd), ins.funct3, ins.funct7)
}

func encodeRVIV(ins *instruction) uint32 {
	return encodeR(ins.as, regI(ins.rs1), regV(ins.rs2), regV(ins.rd), ins.funct3, ins.funct7)
}

func encodeRVV(ins *instruction) uint32 {
	return encodeR(ins.as, 0, regV(ins.rs2), regV(ins.rd), ins.funct3, ins.funct7)
}

func encodeRVVi(ins *instruction) uint32 {
	return encodeR(ins.as, immI(ins.as, ins.imm, 5), regV(ins.rs2), regV(ins.rd), ins.funct3, ins.funct7)
}

func encodeRVVu(ins *instruction) uint32 {
	return encodeR(ins.as, immU(ins.as, ins.imm, 5), regV(ins.rs2), regV(ins.rd), ins.funct3, ins.funct7)
}

func encodeRVVV(ins *instruction) uint32 {
	return encodeR(ins.as, regV(ins.rs1), regV(ins.rs2), regV(ins.rd), ins.funct3, ins.funct7)
}

// encodeI encodes an I-type RISC-V instruction.
func encodeI(as obj.As, rs1, rd, imm, funct7 uint32) uint32 {
	enc := encode(as)
	if enc == nil {
		panic("encodeI: could not encode instruction")
	}
	imm |= uint32(enc.csr)
	return funct7<<25 | imm<<20 | rs1<<15 | enc.funct3<<12 | rd<<7 | enc.opcode
}

func encodeIII(ins *instruction) uint32 {
	return encodeI(ins.as, regI(ins.rs1), regI(ins.rd), uint32(ins.imm), 0)
}

func encodeIF(ins *instruction) uint32 {
	return encodeI(ins.as, regI(ins.rs1), regF(ins.rd), uint32(ins.imm), 0)
}

func encodeIV(ins *instruction) uint32 {
	return encodeI(ins.as, regI(ins.rs1), regV(ins.rd), uint32(ins.imm), ins.funct7)
}

func encodeIIIV(ins *instruction) uint32 {
	return encodeI(ins.as, regI(ins.rs1), regV(ins.rd), regI(ins.rs2), ins.funct7)
}

func encodeIVIV(ins *instruction) uint32 {
	return encodeI(ins.as, regI(ins.rs1), regV(ins.rd), regV(ins.rs2), ins.funct7)
}

// encodeS encodes an S-type RISC-V instruction.
func encodeS(as obj.As, rs1, rs2, imm, funct7 uint32) uint32 {
	enc := encode(as)
	if enc == nil {
		panic("encodeS: could not encode instruction")
	}
	if enc.rs2 != 0 && rs2 != 0 {
		panic("encodeS: instruction uses rs2, but rs2 was nonzero")
	}
	rs2 |= enc.rs2
	imm |= uint32(enc.csr) &^ 0x1f
	return funct7<<25 | (imm>>5)<<25 | rs2<<20 | rs1<<15 | enc.funct3<<12 | (imm&0x1f)<<7 | enc.opcode
}

func encodeSI(ins *instruction) uint32 {
	return encodeS(ins.as, regI(ins.rd), regI(ins.rs1), uint32(ins.imm), 0)
}

func encodeSF(ins *instruction) uint32 {
	return encodeS(ins.as, regI(ins.rd), regF(ins.rs1), uint32(ins.imm), 0)
}

func encodeSV(ins *instruction) uint32 {
	return encodeS(ins.as, regI(ins.rd), 0, regV(ins.rs1), ins.funct7)
}

func encodeSVII(ins *instruction) uint32 {
	return encodeS(ins.as, regI(ins.rs1), regI(ins.rs2), regV(ins.rd), ins.funct7)
}

func encodeSVIV(ins *instruction) uint32 {
	return encodeS(ins.as, regI(ins.rs1), regV(ins.rs2), regV(ins.rd), ins.funct7)
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

func encodeVset(as obj.As, rs1, rs2, rd uint32) uint32 {
	enc := encode(as)
	if enc == nil {
		panic("encodeVset: could not encode instruction")
	}
	return enc.funct7<<25 | rs2<<20 | rs1<<15 | enc.funct3<<12 | rd<<7 | enc.opcode
}

func encodeVsetvli(ins *instruction) uint32 {
	vtype := immU(ins.as, ins.imm, 11)
	return encodeVset(ins.as, regI(ins.rs1), vtype, regI(ins.rd))
}

func encodeVsetivli(ins *instruction) uint32 {
	vtype := immU(ins.as, ins.imm, 10)
	avl := immU(ins.as, int64(ins.rs1), 5)
	return encodeVset(ins.as, avl, vtype, regI(ins.rd))
}

func encodeVsetvl(ins *instruction) uint32 {
	return encodeVset(ins.as, regI(ins.rs1), regI(ins.rs2), regI(ins.rd))
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

func EncodeVectorType(vsew, vlmul, vtail, vmask int64) (int64, error) {
	vsewSO := SpecialOperand(vsew)
	if vsewSO < SPOP_E8 || vsewSO > SPOP_E64 {
		return -1, fmt.Errorf("invalid vector selected element width %q", vsewSO)
	}
	vlmulSO := SpecialOperand(vlmul)
	if vlmulSO < SPOP_M1 || vlmulSO > SPOP_MF8 {
		return -1, fmt.Errorf("invalid vector register group multiplier %q", vlmulSO)
	}
	vtailSO := SpecialOperand(vtail)
	if vtailSO != SPOP_TA && vtailSO != SPOP_TU {
		return -1, fmt.Errorf("invalid vector tail policy %q", vtailSO)
	}
	vmaskSO := SpecialOperand(vmask)
	if vmaskSO != SPOP_MA && vmaskSO != SPOP_MU {
		return -1, fmt.Errorf("invalid vector mask policy %q", vmaskSO)
	}
	vtype := vmaskSO.encode()<<7 | vtailSO.encode()<<6 | vsewSO.encode()<<3 | vlmulSO.encode()
	return int64(vtype), nil
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
	//     register, F = float register, V = vector register), in uppercase
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
	rVFVEncoding  = encoding{encode: encodeRVFV, validate: validateRVFV, length: 4}
	rVIVEncoding  = encoding{encode: encodeRVIV, validate: validateRVIV, length: 4}
	rVVEncoding   = encoding{encode: encodeRVV, validate: validateRVV, length: 4}
	rVViEncoding  = encoding{encode: encodeRVVi, validate: validateRVVi, length: 4}
	rVVuEncoding  = encoding{encode: encodeRVVu, validate: validateRVVu, length: 4}
	rVVVEncoding  = encoding{encode: encodeRVVV, validate: validateRVVV, length: 4}

	iIIEncoding  = encoding{encode: encodeIII, validate: validateIII, length: 4}
	iFEncoding   = encoding{encode: encodeIF, validate: validateIF, length: 4}
	iVEncoding   = encoding{encode: encodeIV, validate: validateIV, length: 4}
	iIIVEncoding = encoding{encode: encodeIIIV, validate: validateIIIV, length: 4}
	iVIVEncoding = encoding{encode: encodeIVIV, validate: validateIVIV, length: 4}

	sIEncoding   = encoding{encode: encodeSI, validate: validateSI, length: 4}
	sFEncoding   = encoding{encode: encodeSF, validate: validateSF, length: 4}
	sVEncoding   = encoding{encode: encodeSV, validate: validateSV, length: 4}
	sVIIEncoding = encoding{encode: encodeSVII, validate: validateSVII, length: 4}
	sVIVEncoding = encoding{encode: encodeSVIV, validate: validateSVIV, length: 4}

	bEncoding = encoding{encode: encodeB, validate: validateB, length: 4}
	uEncoding = encoding{encode: encodeU, validate: validateU, length: 4}
	jEncoding = encoding{encode: encodeJ, validate: validateJ, length: 4}

	// Encodings for vector configuration setting instruction.
	vsetvliEncoding  = encoding{encode: encodeVsetvli, validate: validateVsetvli, length: 4}
	vsetivliEncoding = encoding{encode: encodeVsetivli, validate: validateVsetivli, length: 4}
	vsetvlEncoding   = encoding{encode: encodeVsetvl, validate: validateVsetvl, length: 4}

	// rawEncoding encodes a raw instruction byte sequence.
	rawEncoding = encoding{encode: encodeRawIns, validate: validateRaw, length: 4}

	// pseudoOpEncoding panics if encoding is attempted, but does no validation.
	pseudoOpEncoding = encoding{encode: nil, validate: func(*obj.Link, *instruction) {}, length: 0}

	// badEncoding is used when an invalid op is encountered.
	// An error has already been generated, so let anything else through.
	badEncoding = encoding{encode: func(*instruction) uint32 { return 0 }, validate: func(*obj.Link, *instruction) {}, length: 0}
)

// instructionData specifies details relating to a RISC-V instruction.
type instructionData struct {
	enc     encoding
	immForm obj.As // immediate form of this instruction
	ternary bool
}

// instructions contains details of RISC-V instructions, including
// their encoding type. Entries are masked with obj.AMask to keep
// indices small.
var instructions = [ALAST & obj.AMask]instructionData{
	//
	// Unprivileged ISA
	//

	// 2.4: Integer Computational Instructions
	AADDI & obj.AMask:  {enc: iIIEncoding, ternary: true},
	ASLTI & obj.AMask:  {enc: iIIEncoding, ternary: true},
	ASLTIU & obj.AMask: {enc: iIIEncoding, ternary: true},
	AANDI & obj.AMask:  {enc: iIIEncoding, ternary: true},
	AORI & obj.AMask:   {enc: iIIEncoding, ternary: true},
	AXORI & obj.AMask:  {enc: iIIEncoding, ternary: true},
	ASLLI & obj.AMask:  {enc: iIIEncoding, ternary: true},
	ASRLI & obj.AMask:  {enc: iIIEncoding, ternary: true},
	ASRAI & obj.AMask:  {enc: iIIEncoding, ternary: true},
	ALUI & obj.AMask:   {enc: uEncoding},
	AAUIPC & obj.AMask: {enc: uEncoding},
	AADD & obj.AMask:   {enc: rIIIEncoding, immForm: AADDI, ternary: true},
	ASLT & obj.AMask:   {enc: rIIIEncoding, immForm: ASLTI, ternary: true},
	ASLTU & obj.AMask:  {enc: rIIIEncoding, immForm: ASLTIU, ternary: true},
	AAND & obj.AMask:   {enc: rIIIEncoding, immForm: AANDI, ternary: true},
	AOR & obj.AMask:    {enc: rIIIEncoding, immForm: AORI, ternary: true},
	AXOR & obj.AMask:   {enc: rIIIEncoding, immForm: AXORI, ternary: true},
	ASLL & obj.AMask:   {enc: rIIIEncoding, immForm: ASLLI, ternary: true},
	ASRL & obj.AMask:   {enc: rIIIEncoding, immForm: ASRLI, ternary: true},
	ASUB & obj.AMask:   {enc: rIIIEncoding, ternary: true},
	ASRA & obj.AMask:   {enc: rIIIEncoding, immForm: ASRAI, ternary: true},

	// 2.5: Control Transfer Instructions
	AJAL & obj.AMask:  {enc: jEncoding},
	AJALR & obj.AMask: {enc: iIIEncoding},
	ABEQ & obj.AMask:  {enc: bEncoding},
	ABNE & obj.AMask:  {enc: bEncoding},
	ABLT & obj.AMask:  {enc: bEncoding},
	ABLTU & obj.AMask: {enc: bEncoding},
	ABGE & obj.AMask:  {enc: bEncoding},
	ABGEU & obj.AMask: {enc: bEncoding},

	// 2.6: Load and Store Instructions
	ALW & obj.AMask:  {enc: iIIEncoding},
	ALWU & obj.AMask: {enc: iIIEncoding},
	ALH & obj.AMask:  {enc: iIIEncoding},
	ALHU & obj.AMask: {enc: iIIEncoding},
	ALB & obj.AMask:  {enc: iIIEncoding},
	ALBU & obj.AMask: {enc: iIIEncoding},
	ASW & obj.AMask:  {enc: sIEncoding},
	ASH & obj.AMask:  {enc: sIEncoding},
	ASB & obj.AMask:  {enc: sIEncoding},

	// 2.7: Memory Ordering
	AFENCE & obj.AMask: {enc: iIIEncoding},

	// 4.2: Integer Computational Instructions (RV64I)
	AADDIW & obj.AMask: {enc: iIIEncoding, ternary: true},
	ASLLIW & obj.AMask: {enc: iIIEncoding, ternary: true},
	ASRLIW & obj.AMask: {enc: iIIEncoding, ternary: true},
	ASRAIW & obj.AMask: {enc: iIIEncoding, ternary: true},
	AADDW & obj.AMask:  {enc: rIIIEncoding, immForm: AADDIW, ternary: true},
	ASLLW & obj.AMask:  {enc: rIIIEncoding, immForm: ASLLIW, ternary: true},
	ASRLW & obj.AMask:  {enc: rIIIEncoding, immForm: ASRLIW, ternary: true},
	ASUBW & obj.AMask:  {enc: rIIIEncoding, ternary: true},
	ASRAW & obj.AMask:  {enc: rIIIEncoding, immForm: ASRAIW, ternary: true},

	// 4.3: Load and Store Instructions (RV64I)
	ALD & obj.AMask: {enc: iIIEncoding},
	ASD & obj.AMask: {enc: sIEncoding},

	// 7.1: CSR Instructions
	ACSRRS & obj.AMask: {enc: iIIEncoding},

	// 13.1: Multiplication Operations
	AMUL & obj.AMask:    {enc: rIIIEncoding, ternary: true},
	AMULH & obj.AMask:   {enc: rIIIEncoding, ternary: true},
	AMULHU & obj.AMask:  {enc: rIIIEncoding, ternary: true},
	AMULHSU & obj.AMask: {enc: rIIIEncoding, ternary: true},
	AMULW & obj.AMask:   {enc: rIIIEncoding, ternary: true},
	ADIV & obj.AMask:    {enc: rIIIEncoding, ternary: true},
	ADIVU & obj.AMask:   {enc: rIIIEncoding, ternary: true},
	AREM & obj.AMask:    {enc: rIIIEncoding, ternary: true},
	AREMU & obj.AMask:   {enc: rIIIEncoding, ternary: true},
	ADIVW & obj.AMask:   {enc: rIIIEncoding, ternary: true},
	ADIVUW & obj.AMask:  {enc: rIIIEncoding, ternary: true},
	AREMW & obj.AMask:   {enc: rIIIEncoding, ternary: true},
	AREMUW & obj.AMask:  {enc: rIIIEncoding, ternary: true},

	// 14.2: Load-Reserved/Store-Conditional Instructions (Zalrsc)
	ALRW & obj.AMask: {enc: rIIIEncoding},
	ALRD & obj.AMask: {enc: rIIIEncoding},
	ASCW & obj.AMask: {enc: rIIIEncoding},
	ASCD & obj.AMask: {enc: rIIIEncoding},

	// 14.4: Atomic Memory Operations (Zaamo)
	AAMOSWAPW & obj.AMask: {enc: rIIIEncoding},
	AAMOSWAPD & obj.AMask: {enc: rIIIEncoding},
	AAMOADDW & obj.AMask:  {enc: rIIIEncoding},
	AAMOADDD & obj.AMask:  {enc: rIIIEncoding},
	AAMOANDW & obj.AMask:  {enc: rIIIEncoding},
	AAMOANDD & obj.AMask:  {enc: rIIIEncoding},
	AAMOORW & obj.AMask:   {enc: rIIIEncoding},
	AAMOORD & obj.AMask:   {enc: rIIIEncoding},
	AAMOXORW & obj.AMask:  {enc: rIIIEncoding},
	AAMOXORD & obj.AMask:  {enc: rIIIEncoding},
	AAMOMAXW & obj.AMask:  {enc: rIIIEncoding},
	AAMOMAXD & obj.AMask:  {enc: rIIIEncoding},
	AAMOMAXUW & obj.AMask: {enc: rIIIEncoding},
	AAMOMAXUD & obj.AMask: {enc: rIIIEncoding},
	AAMOMINW & obj.AMask:  {enc: rIIIEncoding},
	AAMOMIND & obj.AMask:  {enc: rIIIEncoding},
	AAMOMINUW & obj.AMask: {enc: rIIIEncoding},
	AAMOMINUD & obj.AMask: {enc: rIIIEncoding},

	// 20.5: Single-Precision Load and Store Instructions
	AFLW & obj.AMask: {enc: iFEncoding},
	AFSW & obj.AMask: {enc: sFEncoding},

	// 20.6: Single-Precision Floating-Point Computational Instructions
	AFADDS & obj.AMask:   {enc: rFFFEncoding},
	AFSUBS & obj.AMask:   {enc: rFFFEncoding},
	AFMULS & obj.AMask:   {enc: rFFFEncoding},
	AFDIVS & obj.AMask:   {enc: rFFFEncoding},
	AFMINS & obj.AMask:   {enc: rFFFEncoding},
	AFMAXS & obj.AMask:   {enc: rFFFEncoding},
	AFSQRTS & obj.AMask:  {enc: rFFFEncoding},
	AFMADDS & obj.AMask:  {enc: rFFFFEncoding},
	AFMSUBS & obj.AMask:  {enc: rFFFFEncoding},
	AFNMSUBS & obj.AMask: {enc: rFFFFEncoding},
	AFNMADDS & obj.AMask: {enc: rFFFFEncoding},

	// 20.7: Single-Precision Floating-Point Conversion and Move Instructions
	AFCVTWS & obj.AMask:  {enc: rFIEncoding},
	AFCVTLS & obj.AMask:  {enc: rFIEncoding},
	AFCVTSW & obj.AMask:  {enc: rIFEncoding},
	AFCVTSL & obj.AMask:  {enc: rIFEncoding},
	AFCVTWUS & obj.AMask: {enc: rFIEncoding},
	AFCVTLUS & obj.AMask: {enc: rFIEncoding},
	AFCVTSWU & obj.AMask: {enc: rIFEncoding},
	AFCVTSLU & obj.AMask: {enc: rIFEncoding},
	AFSGNJS & obj.AMask:  {enc: rFFFEncoding},
	AFSGNJNS & obj.AMask: {enc: rFFFEncoding},
	AFSGNJXS & obj.AMask: {enc: rFFFEncoding},
	AFMVXW & obj.AMask:   {enc: rFIEncoding},
	AFMVWX & obj.AMask:   {enc: rIFEncoding},

	// 20.8: Single-Precision Floating-Point Compare Instructions
	AFEQS & obj.AMask: {enc: rFFIEncoding},
	AFLTS & obj.AMask: {enc: rFFIEncoding},
	AFLES & obj.AMask: {enc: rFFIEncoding},

	// 20.9: Single-Precision Floating-Point Classify Instruction
	AFCLASSS & obj.AMask: {enc: rFIEncoding},

	// 12.3: Double-Precision Load and Store Instructions
	AFLD & obj.AMask: {enc: iFEncoding},
	AFSD & obj.AMask: {enc: sFEncoding},

	// 21.4: Double-Precision Floating-Point Computational Instructions
	AFADDD & obj.AMask:   {enc: rFFFEncoding},
	AFSUBD & obj.AMask:   {enc: rFFFEncoding},
	AFMULD & obj.AMask:   {enc: rFFFEncoding},
	AFDIVD & obj.AMask:   {enc: rFFFEncoding},
	AFMIND & obj.AMask:   {enc: rFFFEncoding},
	AFMAXD & obj.AMask:   {enc: rFFFEncoding},
	AFSQRTD & obj.AMask:  {enc: rFFFEncoding},
	AFMADDD & obj.AMask:  {enc: rFFFFEncoding},
	AFMSUBD & obj.AMask:  {enc: rFFFFEncoding},
	AFNMSUBD & obj.AMask: {enc: rFFFFEncoding},
	AFNMADDD & obj.AMask: {enc: rFFFFEncoding},

	// 21.5: Double-Precision Floating-Point Conversion and Move Instructions
	AFCVTWD & obj.AMask:  {enc: rFIEncoding},
	AFCVTLD & obj.AMask:  {enc: rFIEncoding},
	AFCVTDW & obj.AMask:  {enc: rIFEncoding},
	AFCVTDL & obj.AMask:  {enc: rIFEncoding},
	AFCVTWUD & obj.AMask: {enc: rFIEncoding},
	AFCVTLUD & obj.AMask: {enc: rFIEncoding},
	AFCVTDWU & obj.AMask: {enc: rIFEncoding},
	AFCVTDLU & obj.AMask: {enc: rIFEncoding},
	AFCVTSD & obj.AMask:  {enc: rFFEncoding},
	AFCVTDS & obj.AMask:  {enc: rFFEncoding},
	AFSGNJD & obj.AMask:  {enc: rFFFEncoding},
	AFSGNJND & obj.AMask: {enc: rFFFEncoding},
	AFSGNJXD & obj.AMask: {enc: rFFFEncoding},
	AFMVXD & obj.AMask:   {enc: rFIEncoding},
	AFMVDX & obj.AMask:   {enc: rIFEncoding},

	// 21.6: Double-Precision Floating-Point Compare Instructions
	AFEQD & obj.AMask: {enc: rFFIEncoding},
	AFLTD & obj.AMask: {enc: rFFIEncoding},
	AFLED & obj.AMask: {enc: rFFIEncoding},

	// 21.7: Double-Precision Floating-Point Classify Instruction
	AFCLASSD & obj.AMask: {enc: rFIEncoding},

	//
	// "B" Extension for Bit Manipulation, Version 1.0.0
	//

	// 28.4.1: Address Generation Instructions (Zba)
	AADDUW & obj.AMask:    {enc: rIIIEncoding, ternary: true},
	ASH1ADD & obj.AMask:   {enc: rIIIEncoding, ternary: true},
	ASH1ADDUW & obj.AMask: {enc: rIIIEncoding, ternary: true},
	ASH2ADD & obj.AMask:   {enc: rIIIEncoding, ternary: true},
	ASH2ADDUW & obj.AMask: {enc: rIIIEncoding, ternary: true},
	ASH3ADD & obj.AMask:   {enc: rIIIEncoding, ternary: true},
	ASH3ADDUW & obj.AMask: {enc: rIIIEncoding, ternary: true},
	ASLLIUW & obj.AMask:   {enc: iIIEncoding, ternary: true},

	// 28.4.2: Basic Bit Manipulation (Zbb)
	AANDN & obj.AMask:  {enc: rIIIEncoding, ternary: true},
	ACLZ & obj.AMask:   {enc: rIIEncoding},
	ACLZW & obj.AMask:  {enc: rIIEncoding},
	ACPOP & obj.AMask:  {enc: rIIEncoding},
	ACPOPW & obj.AMask: {enc: rIIEncoding},
	ACTZ & obj.AMask:   {enc: rIIEncoding},
	ACTZW & obj.AMask:  {enc: rIIEncoding},
	AMAX & obj.AMask:   {enc: rIIIEncoding, ternary: true},
	AMAXU & obj.AMask:  {enc: rIIIEncoding, ternary: true},
	AMIN & obj.AMask:   {enc: rIIIEncoding, ternary: true},
	AMINU & obj.AMask:  {enc: rIIIEncoding, ternary: true},
	AORN & obj.AMask:   {enc: rIIIEncoding, ternary: true},
	ASEXTB & obj.AMask: {enc: rIIEncoding},
	ASEXTH & obj.AMask: {enc: rIIEncoding},
	AXNOR & obj.AMask:  {enc: rIIIEncoding, ternary: true},
	AZEXTH & obj.AMask: {enc: rIIEncoding},

	// 28.4.3: Bitwise Rotation (Zbb)
	AROL & obj.AMask:   {enc: rIIIEncoding, ternary: true},
	AROLW & obj.AMask:  {enc: rIIIEncoding, ternary: true},
	AROR & obj.AMask:   {enc: rIIIEncoding, immForm: ARORI, ternary: true},
	ARORI & obj.AMask:  {enc: iIIEncoding, ternary: true},
	ARORIW & obj.AMask: {enc: iIIEncoding, ternary: true},
	ARORW & obj.AMask:  {enc: rIIIEncoding, immForm: ARORIW, ternary: true},
	AORCB & obj.AMask:  {enc: rIIEncoding},
	AREV8 & obj.AMask:  {enc: rIIEncoding},

	// 28.4.4: Single-bit Instructions (Zbs)
	ABCLR & obj.AMask:  {enc: rIIIEncoding, immForm: ABCLRI, ternary: true},
	ABCLRI & obj.AMask: {enc: iIIEncoding, ternary: true},
	ABEXT & obj.AMask:  {enc: rIIIEncoding, immForm: ABEXTI, ternary: true},
	ABEXTI & obj.AMask: {enc: iIIEncoding, ternary: true},
	ABINV & obj.AMask:  {enc: rIIIEncoding, immForm: ABINVI, ternary: true},
	ABINVI & obj.AMask: {enc: iIIEncoding, ternary: true},
	ABSET & obj.AMask:  {enc: rIIIEncoding, immForm: ABSETI, ternary: true},
	ABSETI & obj.AMask: {enc: iIIEncoding, ternary: true},

	//
	// "V" Standard Extension for Vector Operations, Version 1.0
	//

	// 31.6: Vector Configuration-Setting Instructions
	AVSETVLI & obj.AMask:  {enc: vsetvliEncoding, immForm: AVSETIVLI},
	AVSETIVLI & obj.AMask: {enc: vsetivliEncoding},
	AVSETVL & obj.AMask:   {enc: vsetvlEncoding},

	// 31.7.4: Vector Unit-Stride Instructions
	AVLE8V & obj.AMask:  {enc: iVEncoding},
	AVLE16V & obj.AMask: {enc: iVEncoding},
	AVLE32V & obj.AMask: {enc: iVEncoding},
	AVLE64V & obj.AMask: {enc: iVEncoding},
	AVSE8V & obj.AMask:  {enc: sVEncoding},
	AVSE16V & obj.AMask: {enc: sVEncoding},
	AVSE32V & obj.AMask: {enc: sVEncoding},
	AVSE64V & obj.AMask: {enc: sVEncoding},
	AVLMV & obj.AMask:   {enc: iVEncoding},
	AVSMV & obj.AMask:   {enc: sVEncoding},

	// 31.7.5: Vector Strided Instructions
	AVLSE8V & obj.AMask:  {enc: iIIVEncoding},
	AVLSE16V & obj.AMask: {enc: iIIVEncoding},
	AVLSE32V & obj.AMask: {enc: iIIVEncoding},
	AVLSE64V & obj.AMask: {enc: iIIVEncoding},
	AVSSE8V & obj.AMask:  {enc: sVIIEncoding},
	AVSSE16V & obj.AMask: {enc: sVIIEncoding},
	AVSSE32V & obj.AMask: {enc: sVIIEncoding},
	AVSSE64V & obj.AMask: {enc: sVIIEncoding},

	// 31.7.6: Vector Indexed Instructions
	AVLUXEI8V & obj.AMask:  {enc: iVIVEncoding},
	AVLUXEI16V & obj.AMask: {enc: iVIVEncoding},
	AVLUXEI32V & obj.AMask: {enc: iVIVEncoding},
	AVLUXEI64V & obj.AMask: {enc: iVIVEncoding},
	AVLOXEI8V & obj.AMask:  {enc: iVIVEncoding},
	AVLOXEI16V & obj.AMask: {enc: iVIVEncoding},
	AVLOXEI32V & obj.AMask: {enc: iVIVEncoding},
	AVLOXEI64V & obj.AMask: {enc: iVIVEncoding},
	AVSUXEI8V & obj.AMask:  {enc: sVIVEncoding},
	AVSUXEI16V & obj.AMask: {enc: sVIVEncoding},
	AVSUXEI32V & obj.AMask: {enc: sVIVEncoding},
	AVSUXEI64V & obj.AMask: {enc: sVIVEncoding},
	AVSOXEI8V & obj.AMask:  {enc: sVIVEncoding},
	AVSOXEI16V & obj.AMask: {enc: sVIVEncoding},
	AVSOXEI32V & obj.AMask: {enc: sVIVEncoding},
	AVSOXEI64V & obj.AMask: {enc: sVIVEncoding},

	// 31.7.9: Vector Load/Store Whole Register Instructions
	AVL1RE8V & obj.AMask:  {enc: iVEncoding},
	AVL1RE16V & obj.AMask: {enc: iVEncoding},
	AVL1RE32V & obj.AMask: {enc: iVEncoding},
	AVL1RE64V & obj.AMask: {enc: iVEncoding},
	AVL2RE8V & obj.AMask:  {enc: iVEncoding},
	AVL2RE16V & obj.AMask: {enc: iVEncoding},
	AVL2RE32V & obj.AMask: {enc: iVEncoding},
	AVL2RE64V & obj.AMask: {enc: iVEncoding},
	AVL4RE8V & obj.AMask:  {enc: iVEncoding},
	AVL4RE16V & obj.AMask: {enc: iVEncoding},
	AVL4RE32V & obj.AMask: {enc: iVEncoding},
	AVL4RE64V & obj.AMask: {enc: iVEncoding},
	AVL8RE8V & obj.AMask:  {enc: iVEncoding},
	AVL8RE16V & obj.AMask: {enc: iVEncoding},
	AVL8RE32V & obj.AMask: {enc: iVEncoding},
	AVL8RE64V & obj.AMask: {enc: iVEncoding},
	AVS1RV & obj.AMask:    {enc: sVEncoding},
	AVS2RV & obj.AMask:    {enc: sVEncoding},
	AVS4RV & obj.AMask:    {enc: sVEncoding},
	AVS8RV & obj.AMask:    {enc: sVEncoding},

	// 31.11.1: Vector Single-Width Integer Add and Subtract
	AVADDVV & obj.AMask:  {enc: rVVVEncoding},
	AVADDVX & obj.AMask:  {enc: rVIVEncoding},
	AVADDVI & obj.AMask:  {enc: rVViEncoding},
	AVSUBVV & obj.AMask:  {enc: rVVVEncoding},
	AVSUBVX & obj.AMask:  {enc: rVIVEncoding},
	AVRSUBVX & obj.AMask: {enc: rVIVEncoding},
	AVRSUBVI & obj.AMask: {enc: rVViEncoding},

	// 31.11.2: Vector Widening Integer Add/Subtract
	AVWADDUVV & obj.AMask: {enc: rVVVEncoding},
	AVWADDUVX & obj.AMask: {enc: rVIVEncoding},
	AVWSUBUVV & obj.AMask: {enc: rVVVEncoding},
	AVWSUBUVX & obj.AMask: {enc: rVIVEncoding},
	AVWADDVV & obj.AMask:  {enc: rVVVEncoding},
	AVWADDVX & obj.AMask:  {enc: rVIVEncoding},
	AVWSUBVV & obj.AMask:  {enc: rVVVEncoding},
	AVWSUBVX & obj.AMask:  {enc: rVIVEncoding},
	AVWADDUWV & obj.AMask: {enc: rVVVEncoding},
	AVWADDUWX & obj.AMask: {enc: rVIVEncoding},
	AVWSUBUWV & obj.AMask: {enc: rVVVEncoding},
	AVWSUBUWX & obj.AMask: {enc: rVIVEncoding},
	AVWADDWV & obj.AMask:  {enc: rVVVEncoding},
	AVWADDWX & obj.AMask:  {enc: rVIVEncoding},
	AVWSUBWV & obj.AMask:  {enc: rVVVEncoding},
	AVWSUBWX & obj.AMask:  {enc: rVIVEncoding},

	// 31.11.3: Vector Integer Extension
	AVZEXTVF2 & obj.AMask: {enc: rVVEncoding},
	AVSEXTVF2 & obj.AMask: {enc: rVVEncoding},
	AVZEXTVF4 & obj.AMask: {enc: rVVEncoding},
	AVSEXTVF4 & obj.AMask: {enc: rVVEncoding},
	AVZEXTVF8 & obj.AMask: {enc: rVVEncoding},
	AVSEXTVF8 & obj.AMask: {enc: rVVEncoding},

	// 31.11.4: Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions
	AVADCVVM & obj.AMask:  {enc: rVVVEncoding},
	AVADCVXM & obj.AMask:  {enc: rVIVEncoding},
	AVADCVIM & obj.AMask:  {enc: rVViEncoding},
	AVMADCVVM & obj.AMask: {enc: rVVVEncoding},
	AVMADCVXM & obj.AMask: {enc: rVIVEncoding},
	AVMADCVIM & obj.AMask: {enc: rVViEncoding},
	AVMADCVV & obj.AMask:  {enc: rVVVEncoding},
	AVMADCVX & obj.AMask:  {enc: rVIVEncoding},
	AVMADCVI & obj.AMask:  {enc: rVViEncoding},
	AVSBCVVM & obj.AMask:  {enc: rVVVEncoding},
	AVSBCVXM & obj.AMask:  {enc: rVIVEncoding},
	AVMSBCVVM & obj.AMask: {enc: rVVVEncoding},
	AVMSBCVXM & obj.AMask: {enc: rVIVEncoding},
	AVMSBCVV & obj.AMask:  {enc: rVVVEncoding},
	AVMSBCVX & obj.AMask:  {enc: rVIVEncoding},

	// 31.11.5: Vector Bitwise Logical Instructions
	AVANDVV & obj.AMask: {enc: rVVVEncoding},
	AVANDVX & obj.AMask: {enc: rVIVEncoding},
	AVANDVI & obj.AMask: {enc: rVViEncoding},
	AVORVV & obj.AMask:  {enc: rVVVEncoding},
	AVORVX & obj.AMask:  {enc: rVIVEncoding},
	AVORVI & obj.AMask:  {enc: rVViEncoding},
	AVXORVV & obj.AMask: {enc: rVVVEncoding},
	AVXORVX & obj.AMask: {enc: rVIVEncoding},
	AVXORVI & obj.AMask: {enc: rVViEncoding},

	// 31.11.6: Vector Single-Width Shift Instructions
	AVSLLVV & obj.AMask: {enc: rVVVEncoding},
	AVSLLVX & obj.AMask: {enc: rVIVEncoding},
	AVSLLVI & obj.AMask: {enc: rVVuEncoding},
	AVSRLVV & obj.AMask: {enc: rVVVEncoding},
	AVSRLVX & obj.AMask: {enc: rVIVEncoding},
	AVSRLVI & obj.AMask: {enc: rVVuEncoding},
	AVSRAVV & obj.AMask: {enc: rVVVEncoding},
	AVSRAVX & obj.AMask: {enc: rVIVEncoding},
	AVSRAVI & obj.AMask: {enc: rVVuEncoding},

	// 31.11.7: Vector Narrowing Integer Right Shift Instructions
	AVNSRLWV & obj.AMask: {enc: rVVVEncoding},
	AVNSRLWX & obj.AMask: {enc: rVIVEncoding},
	AVNSRLWI & obj.AMask: {enc: rVVuEncoding},
	AVNSRAWV & obj.AMask: {enc: rVVVEncoding},
	AVNSRAWX & obj.AMask: {enc: rVIVEncoding},
	AVNSRAWI & obj.AMask: {enc: rVVuEncoding},

	// 31.11.8: Vector Integer Compare Instructions
	AVMSEQVV & obj.AMask:  {enc: rVVVEncoding},
	AVMSEQVX & obj.AMask:  {enc: rVIVEncoding},
	AVMSEQVI & obj.AMask:  {enc: rVViEncoding},
	AVMSNEVV & obj.AMask:  {enc: rVVVEncoding},
	AVMSNEVX & obj.AMask:  {enc: rVIVEncoding},
	AVMSNEVI & obj.AMask:  {enc: rVViEncoding},
	AVMSLTUVV & obj.AMask: {enc: rVVVEncoding},
	AVMSLTUVX & obj.AMask: {enc: rVIVEncoding},
	AVMSLTVV & obj.AMask:  {enc: rVVVEncoding},
	AVMSLTVX & obj.AMask:  {enc: rVIVEncoding},
	AVMSLEUVV & obj.AMask: {enc: rVVVEncoding},
	AVMSLEUVX & obj.AMask: {enc: rVIVEncoding},
	AVMSLEUVI & obj.AMask: {enc: rVViEncoding},
	AVMSLEVV & obj.AMask:  {enc: rVVVEncoding},
	AVMSLEVX & obj.AMask:  {enc: rVIVEncoding},
	AVMSLEVI & obj.AMask:  {enc: rVViEncoding},
	AVMSGTUVX & obj.AMask: {enc: rVIVEncoding},
	AVMSGTUVI & obj.AMask: {enc: rVViEncoding},
	AVMSGTVX & obj.AMask:  {enc: rVIVEncoding},
	AVMSGTVI & obj.AMask:  {enc: rVViEncoding},

	// 31.11.9: Vector Integer Min/Max Instructions
	AVMINUVV & obj.AMask: {enc: rVVVEncoding},
	AVMINUVX & obj.AMask: {enc: rVIVEncoding},
	AVMINVV & obj.AMask:  {enc: rVVVEncoding},
	AVMINVX & obj.AMask:  {enc: rVIVEncoding},
	AVMAXUVV & obj.AMask: {enc: rVVVEncoding},
	AVMAXUVX & obj.AMask: {enc: rVIVEncoding},
	AVMAXVV & obj.AMask:  {enc: rVVVEncoding},
	AVMAXVX & obj.AMask:  {enc: rVIVEncoding},

	// 31.11.10: Vector Single-Width Integer Multiply Instructions
	AVMULVV & obj.AMask:    {enc: rVVVEncoding},
	AVMULVX & obj.AMask:    {enc: rVIVEncoding},
	AVMULHVV & obj.AMask:   {enc: rVVVEncoding},
	AVMULHVX & obj.AMask:   {enc: rVIVEncoding},
	AVMULHUVV & obj.AMask:  {enc: rVVVEncoding},
	AVMULHUVX & obj.AMask:  {enc: rVIVEncoding},
	AVMULHSUVV & obj.AMask: {enc: rVVVEncoding},
	AVMULHSUVX & obj.AMask: {enc: rVIVEncoding},

	// 31.11.11: Vector Integer Divide Instructions
	AVDIVUVV & obj.AMask: {enc: rVVVEncoding},
	AVDIVUVX & obj.AMask: {enc: rVIVEncoding},
	AVDIVVV & obj.AMask:  {enc: rVVVEncoding},
	AVDIVVX & obj.AMask:  {enc: rVIVEncoding},
	AVREMUVV & obj.AMask: {enc: rVVVEncoding},
	AVREMUVX & obj.AMask: {enc: rVIVEncoding},
	AVREMVV & obj.AMask:  {enc: rVVVEncoding},
	AVREMVX & obj.AMask:  {enc: rVIVEncoding},

	// 31.11.12: Vector Widening Integer Multiply Instructions
	AVWMULVV & obj.AMask:   {enc: rVVVEncoding},
	AVWMULVX & obj.AMask:   {enc: rVIVEncoding},
	AVWMULUVV & obj.AMask:  {enc: rVVVEncoding},
	AVWMULUVX & obj.AMask:  {enc: rVIVEncoding},
	AVWMULSUVV & obj.AMask: {enc: rVVVEncoding},
	AVWMULSUVX & obj.AMask: {enc: rVIVEncoding},

	// 31.11.13: Vector Single-Width Integer Multiply-Add Instructions
	AVMACCVV & obj.AMask:  {enc: rVVVEncoding},
	AVMACCVX & obj.AMask:  {enc: rVIVEncoding},
	AVNMSACVV & obj.AMask: {enc: rVVVEncoding},
	AVNMSACVX & obj.AMask: {enc: rVIVEncoding},
	AVMADDVV & obj.AMask:  {enc: rVVVEncoding},
	AVMADDVX & obj.AMask:  {enc: rVIVEncoding},
	AVNMSUBVV & obj.AMask: {enc: rVVVEncoding},
	AVNMSUBVX & obj.AMask: {enc: rVIVEncoding},

	// 31.11.14: Vector Widening Integer Multiply-Add Instructions
	AVWMACCUVV & obj.AMask:  {enc: rVVVEncoding},
	AVWMACCUVX & obj.AMask:  {enc: rVIVEncoding},
	AVWMACCVV & obj.AMask:   {enc: rVVVEncoding},
	AVWMACCVX & obj.AMask:   {enc: rVIVEncoding},
	AVWMACCSUVV & obj.AMask: {enc: rVVVEncoding},
	AVWMACCSUVX & obj.AMask: {enc: rVIVEncoding},
	AVWMACCUSVX & obj.AMask: {enc: rVIVEncoding},

	// 31.11.15: Vector Integer Merge Instructions
	AVMERGEVVM & obj.AMask: {enc: rVVVEncoding},
	AVMERGEVXM & obj.AMask: {enc: rVIVEncoding},
	AVMERGEVIM & obj.AMask: {enc: rVViEncoding},

	// 31.11.16: Vector Integer Move Instructions
	AVMVVV & obj.AMask: {enc: rVVVEncoding},
	AVMVVX & obj.AMask: {enc: rVIVEncoding},
	AVMVVI & obj.AMask: {enc: rVViEncoding},

	// 31.12.1: Vector Single-Width Saturating Add and Subtract
	AVSADDUVV & obj.AMask: {enc: rVVVEncoding},
	AVSADDUVX & obj.AMask: {enc: rVIVEncoding},
	AVSADDUVI & obj.AMask: {enc: rVViEncoding},
	AVSADDVV & obj.AMask:  {enc: rVVVEncoding},
	AVSADDVX & obj.AMask:  {enc: rVIVEncoding},
	AVSADDVI & obj.AMask:  {enc: rVViEncoding},
	AVSSUBUVV & obj.AMask: {enc: rVVVEncoding},
	AVSSUBUVX & obj.AMask: {enc: rVIVEncoding},
	AVSSUBVV & obj.AMask:  {enc: rVVVEncoding},
	AVSSUBVX & obj.AMask:  {enc: rVIVEncoding},

	// 31.12.2: Vector Single-Width Averaging Add and Subtract
	AVAADDUVV & obj.AMask: {enc: rVVVEncoding},
	AVAADDUVX & obj.AMask: {enc: rVIVEncoding},
	AVAADDVV & obj.AMask:  {enc: rVVVEncoding},
	AVAADDVX & obj.AMask:  {enc: rVIVEncoding},
	AVASUBUVV & obj.AMask: {enc: rVVVEncoding},
	AVASUBUVX & obj.AMask: {enc: rVIVEncoding},
	AVASUBVV & obj.AMask:  {enc: rVVVEncoding},
	AVASUBVX & obj.AMask:  {enc: rVIVEncoding},

	// 31.12.3: Vector Single-Width Fractional Multiply with Rounding and Saturation
	AVSMULVV & obj.AMask: {enc: rVVVEncoding},
	AVSMULVX & obj.AMask: {enc: rVIVEncoding},

	// 31.12.4: Vector Single-Width Scaling Shift Instructions
	AVSSRLVV & obj.AMask: {enc: rVVVEncoding},
	AVSSRLVX & obj.AMask: {enc: rVIVEncoding},
	AVSSRLVI & obj.AMask: {enc: rVVuEncoding},
	AVSSRAVV & obj.AMask: {enc: rVVVEncoding},
	AVSSRAVX & obj.AMask: {enc: rVIVEncoding},
	AVSSRAVI & obj.AMask: {enc: rVVuEncoding},

	// 31.12.5: Vector Narrowing Fixed-Point Clip Instructions
	AVNCLIPUWV & obj.AMask: {enc: rVVVEncoding},
	AVNCLIPUWX & obj.AMask: {enc: rVIVEncoding},
	AVNCLIPUWI & obj.AMask: {enc: rVVuEncoding},
	AVNCLIPWV & obj.AMask:  {enc: rVVVEncoding},
	AVNCLIPWX & obj.AMask:  {enc: rVIVEncoding},
	AVNCLIPWI & obj.AMask:  {enc: rVVuEncoding},

	// 31.13.2: Vector Single-Width Floating-Point Add/Subtract Instructions
	AVFADDVV & obj.AMask:  {enc: rVVVEncoding},
	AVFADDVF & obj.AMask:  {enc: rVFVEncoding},
	AVFSUBVV & obj.AMask:  {enc: rVVVEncoding},
	AVFSUBVF & obj.AMask:  {enc: rVFVEncoding},
	AVFRSUBVF & obj.AMask: {enc: rVFVEncoding},

	// 31.13.3: Vector Widening Floating-Point Add/Subtract Instructions
	AVFWADDVV & obj.AMask: {enc: rVVVEncoding},
	AVFWADDVF & obj.AMask: {enc: rVFVEncoding},
	AVFWSUBVV & obj.AMask: {enc: rVVVEncoding},
	AVFWSUBVF & obj.AMask: {enc: rVFVEncoding},
	AVFWADDWV & obj.AMask: {enc: rVVVEncoding},
	AVFWADDWF & obj.AMask: {enc: rVFVEncoding},
	AVFWSUBWV & obj.AMask: {enc: rVVVEncoding},
	AVFWSUBWF & obj.AMask: {enc: rVFVEncoding},

	// 31.13.4: Vector Single-Width Floating-Point Multiply/Divide Instructions
	AVFMULVV & obj.AMask:  {enc: rVVVEncoding},
	AVFMULVF & obj.AMask:  {enc: rVFVEncoding},
	AVFDIVVV & obj.AMask:  {enc: rVVVEncoding},
	AVFDIVVF & obj.AMask:  {enc: rVFVEncoding},
	AVFRDIVVF & obj.AMask: {enc: rVFVEncoding},

	// 31.13.5: Vector Widening Floating-Point Multiply
	AVFWMULVV & obj.AMask: {enc: rVVVEncoding},
	AVFWMULVF & obj.AMask: {enc: rVFVEncoding},

	// 31.13.6: Vector Single-Width Floating-Point Fused Multiply-Add Instructions
	AVFMACCVV & obj.AMask:  {enc: rVVVEncoding},
	AVFMACCVF & obj.AMask:  {enc: rVFVEncoding},
	AVFNMACCVV & obj.AMask: {enc: rVVVEncoding},
	AVFNMACCVF & obj.AMask: {enc: rVFVEncoding},
	AVFMSACVV & obj.AMask:  {enc: rVVVEncoding},
	AVFMSACVF & obj.AMask:  {enc: rVFVEncoding},
	AVFNMSACVV & obj.AMask: {enc: rVVVEncoding},
	AVFNMSACVF & obj.AMask: {enc: rVFVEncoding},
	AVFMADDVV & obj.AMask:  {enc: rVVVEncoding},
	AVFMADDVF & obj.AMask:  {enc: rVFVEncoding},
	AVFNMADDVV & obj.AMask: {enc: rVVVEncoding},
	AVFNMADDVF & obj.AMask: {enc: rVFVEncoding},
	AVFMSUBVV & obj.AMask:  {enc: rVVVEncoding},
	AVFMSUBVF & obj.AMask:  {enc: rVFVEncoding},
	AVFNMSUBVV & obj.AMask: {enc: rVVVEncoding},
	AVFNMSUBVF & obj.AMask: {enc: rVFVEncoding},

	// 31.13.7: Vector Widening Floating-Point Fused Multiply-Add Instructions
	AVFWMACCVV & obj.AMask:  {enc: rVVVEncoding},
	AVFWMACCVF & obj.AMask:  {enc: rVFVEncoding},
	AVFWNMACCVV & obj.AMask: {enc: rVVVEncoding},
	AVFWNMACCVF & obj.AMask: {enc: rVFVEncoding},
	AVFWMSACVV & obj.AMask:  {enc: rVVVEncoding},
	AVFWMSACVF & obj.AMask:  {enc: rVFVEncoding},
	AVFWNMSACVV & obj.AMask: {enc: rVVVEncoding},
	AVFWNMSACVF & obj.AMask: {enc: rVFVEncoding},

	// 31.13.8: Vector Floating-Point Square-Root Instruction
	AVFSQRTV & obj.AMask: {enc: rVVEncoding},

	// 31.13.9: Vector Floating-Point Reciprocal Square-Root Estimate Instruction
	AVFRSQRT7V & obj.AMask: {enc: rVVEncoding},

	// 31.13.10: Vector Floating-Point Reciprocal Estimate Instruction
	AVFREC7V & obj.AMask: {enc: rVVEncoding},

	// 31.13.11: Vector Floating-Point MIN/MAX Instructions
	AVFMINVV & obj.AMask: {enc: rVVVEncoding},
	AVFMINVF & obj.AMask: {enc: rVFVEncoding},
	AVFMAXVV & obj.AMask: {enc: rVVVEncoding},
	AVFMAXVF & obj.AMask: {enc: rVFVEncoding},

	// 31.13.12: Vector Floating-Point Sign-Injection Instructions
	AVFSGNJVV & obj.AMask:  {enc: rVVVEncoding},
	AVFSGNJVF & obj.AMask:  {enc: rVFVEncoding},
	AVFSGNJNVV & obj.AMask: {enc: rVVVEncoding},
	AVFSGNJNVF & obj.AMask: {enc: rVFVEncoding},
	AVFSGNJXVV & obj.AMask: {enc: rVVVEncoding},
	AVFSGNJXVF & obj.AMask: {enc: rVFVEncoding},

	// 31.13.13: Vector Floating-Point Compare Instructions
	AVMFEQVV & obj.AMask: {enc: rVVVEncoding},
	AVMFEQVF & obj.AMask: {enc: rVFVEncoding},
	AVMFNEVV & obj.AMask: {enc: rVVVEncoding},
	AVMFNEVF & obj.AMask: {enc: rVFVEncoding},
	AVMFLTVV & obj.AMask: {enc: rVVVEncoding},
	AVMFLTVF & obj.AMask: {enc: rVFVEncoding},
	AVMFLEVV & obj.AMask: {enc: rVVVEncoding},
	AVMFLEVF & obj.AMask: {enc: rVFVEncoding},
	AVMFGTVF & obj.AMask: {enc: rVFVEncoding},
	AVMFGEVF & obj.AMask: {enc: rVFVEncoding},

	// 31.13.14: Vector Floating-Point Classify Instruction
	AVFCLASSV & obj.AMask: {enc: rVVEncoding},

	// 31.13.15: Vector Floating-Point Merge Instruction
	AVFMERGEVFM & obj.AMask: {enc: rVFVEncoding},

	// 31.13.16: Vector Floating-Point Move Instruction
	AVFMVVF & obj.AMask: {enc: rVFVEncoding},

	// 31.13.17: Single-Width Floating-Point/Integer Type-Convert Instructions
	AVFCVTXUFV & obj.AMask:    {enc: rVVEncoding},
	AVFCVTXFV & obj.AMask:     {enc: rVVEncoding},
	AVFCVTRTZXUFV & obj.AMask: {enc: rVVEncoding},
	AVFCVTRTZXFV & obj.AMask:  {enc: rVVEncoding},
	AVFCVTFXUV & obj.AMask:    {enc: rVVEncoding},
	AVFCVTFXV & obj.AMask:     {enc: rVVEncoding},

	// 31.13.18: Widening Floating-Point/Integer Type-Convert Instructions
	AVFWCVTXUFV & obj.AMask:    {enc: rVVEncoding},
	AVFWCVTXFV & obj.AMask:     {enc: rVVEncoding},
	AVFWCVTRTZXUFV & obj.AMask: {enc: rVVEncoding},
	AVFWCVTRTZXFV & obj.AMask:  {enc: rVVEncoding},
	AVFWCVTFXUV & obj.AMask:    {enc: rVVEncoding},
	AVFWCVTFXV & obj.AMask:     {enc: rVVEncoding},
	AVFWCVTFFV & obj.AMask:     {enc: rVVEncoding},

	// 31.13.19: Narrowing Floating-Point/Integer Type-Convert Instructions
	AVFNCVTXUFW & obj.AMask:    {enc: rVVEncoding},
	AVFNCVTXFW & obj.AMask:     {enc: rVVEncoding},
	AVFNCVTRTZXUFW & obj.AMask: {enc: rVVEncoding},
	AVFNCVTRTZXFW & obj.AMask:  {enc: rVVEncoding},
	AVFNCVTFXUW & obj.AMask:    {enc: rVVEncoding},
	AVFNCVTFXW & obj.AMask:     {enc: rVVEncoding},
	AVFNCVTFFW & obj.AMask:     {enc: rVVEncoding},
	AVFNCVTRODFFW & obj.AMask:  {enc: rVVEncoding},

	// 31.14.1: Vector Single-Width Integer Reduction Instructions
	AVREDSUMVS & obj.AMask:  {enc: rVVVEncoding},
	AVREDMAXUVS & obj.AMask: {enc: rVVVEncoding},
	AVREDMAXVS & obj.AMask:  {enc: rVVVEncoding},
	AVREDMINUVS & obj.AMask: {enc: rVVVEncoding},
	AVREDMINVS & obj.AMask:  {enc: rVVVEncoding},
	AVREDANDVS & obj.AMask:  {enc: rVVVEncoding},
	AVREDORVS & obj.AMask:   {enc: rVVVEncoding},
	AVREDXORVS & obj.AMask:  {enc: rVVVEncoding},

	// 31.14.2: Vector Widening Integer Reduction Instructions
	AVWREDSUMUVS & obj.AMask: {enc: rVVVEncoding},
	AVWREDSUMVS & obj.AMask:  {enc: rVVVEncoding},

	// 31.14.3: Vector Single-Width Floating-Point Reduction Instructions
	AVFREDOSUMVS & obj.AMask: {enc: rVVVEncoding},
	AVFREDUSUMVS & obj.AMask: {enc: rVVVEncoding},
	AVFREDMAXVS & obj.AMask:  {enc: rVVVEncoding},
	AVFREDMINVS & obj.AMask:  {enc: rVVVEncoding},

	// 31.14.4: Vector Widening Floating-Point Reduction Instructions
	AVFWREDOSUMVS & obj.AMask: {enc: rVVVEncoding},
	AVFWREDUSUMVS & obj.AMask: {enc: rVVVEncoding},

	//
	// Privileged ISA
	//

	// 3.3.1: Environment Call and Breakpoint
	AECALL & obj.AMask:  {enc: iIIEncoding},
	AEBREAK & obj.AMask: {enc: iIIEncoding},

	// Escape hatch
	AWORD & obj.AMask: {enc: rawEncoding},

	// Pseudo-operations
	obj.AFUNCDATA: {enc: pseudoOpEncoding},
	obj.APCDATA:   {enc: pseudoOpEncoding},
	obj.ATEXT:     {enc: pseudoOpEncoding},
	obj.ANOP:      {enc: pseudoOpEncoding},
	obj.ADUFFZERO: {enc: pseudoOpEncoding},
	obj.ADUFFCOPY: {enc: pseudoOpEncoding},
	obj.APCALIGN:  {enc: pseudoOpEncoding},
}

// instructionDataForAs returns the instruction data for an obj.As.
func instructionDataForAs(as obj.As) (*instructionData, error) {
	if base := as &^ obj.AMask; base != obj.ABaseRISCV && base != 0 {
		return nil, fmt.Errorf("%v is not a RISC-V instruction", as)
	}
	asi := as & obj.AMask
	if int(asi) >= len(instructions) {
		return nil, fmt.Errorf("bad RISC-V instruction %v", as)
	}
	return &instructions[asi], nil
}

// encodingForAs returns the encoding for an obj.As.
func encodingForAs(as obj.As) (*encoding, error) {
	insData, err := instructionDataForAs(as)
	if err != nil {
		return &badEncoding, err
	}
	if insData.enc.validate == nil {
		return &badEncoding, fmt.Errorf("no encoding for instruction %s", as)
	}
	return &insData.enc, nil
}

// splitShiftConst attempts to split a constant into a signed 12 bit or
// 32 bit integer, with corresponding logical right shift and/or left shift.
func splitShiftConst(v int64) (imm int64, lsh int, rsh int, ok bool) {
	// See if we can reconstruct this value from a signed 32 bit integer.
	lsh = bits.TrailingZeros64(uint64(v))
	c := v >> lsh
	if int64(int32(c)) == c {
		return c, lsh, 0, true
	}

	// See if we can reconstruct this value from a small negative constant.
	rsh = bits.LeadingZeros64(uint64(v))
	ones := bits.OnesCount64((uint64(v) >> lsh) >> 11)
	c = signExtend(1<<11|((v>>lsh)&0x7ff), 12)
	if rsh+ones+lsh+11 == 64 {
		if lsh > 0 || c != -1 {
			lsh += rsh
		}
		return c, lsh, rsh, true
	}

	return 0, 0, 0, false
}

// isShiftConst indicates whether a constant can be represented as a signed
// 32 bit integer that is left shifted.
func isShiftConst(v int64) bool {
	_, lsh, rsh, ok := splitShiftConst(v)
	return ok && (lsh > 0 || rsh > 0)
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
		//
		// For example:
		//     MOV $0x8000000000000000, X10
		// becomes
		//     MOV $1, X10
		//     SLLI $63, X10, X10
		//
		// Similarly, we can construct large constants that have a consecutive
		// sequence of ones from a small negative constant, with a right and/or
		// left shift.
		//
		// For example:
		//     MOV $0x000fffffffffffda, X10
		// becomes
		//     MOV $-19, X10
		//     SLLI $13, X10
		//     SRLI $12, X10
		//
		var insSLLI, insSRLI *instruction
		if err := immIFits(ins.imm, 32); err != nil {
			if c, lsh, rsh, ok := splitShiftConst(ins.imm); ok {
				ins.imm = c
				if lsh > 0 {
					insSLLI = &instruction{as: ASLLI, rd: ins.rd, rs1: ins.rd, imm: int64(lsh)}
				}
				if rsh > 0 {
					insSRLI = &instruction{as: ASRLI, rd: ins.rd, rs1: ins.rd, imm: int64(rsh)}
				}
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
		if insSRLI != nil {
			inss = append(inss, insSRLI)
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

		case obj.NAME_EXTERN, obj.NAME_STATIC, obj.NAME_GOTREF:
			if p.From.Sym.Type == objabi.STLSBSS {
				return instructionsForTLSLoad(p)
			}

			// Note that the values for $off_hi and $off_lo are currently
			// zero and will be assigned during relocation. If the destination
			// is an integer register then we can use the same register for the
			// address computation, otherwise we need to use the temporary register.
			//
			// AUIPC $off_hi, Rd
			// L $off_lo, Rd, Rd
			//
			addrReg := ins.rd
			if addrReg < REG_X0 || addrReg > REG_X31 {
				addrReg = REG_TMP
			}
			insAUIPC := &instruction{as: AAUIPC, rd: addrReg}
			ins.as, ins.rs1, ins.rs2, ins.imm = movToLoad(p.As), addrReg, obj.REG_NONE, 0
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

// instructionsForMinMax returns the machine instructions for an integer minimum or maximum.
func instructionsForMinMax(p *obj.Prog, ins *instruction) []*instruction {
	if buildcfg.GORISCV64 >= 22 {
		// Minimum and maximum instructions are supported natively.
		return []*instruction{ins}
	}

	// Generate a move for identical inputs.
	if ins.rs1 == ins.rs2 {
		ins.as, ins.rs2, ins.imm = AADDI, obj.REG_NONE, 0
		return []*instruction{ins}
	}

	// Ensure that if one of the source registers is the same as the destination,
	// it is processed first.
	if ins.rs1 == ins.rd {
		ins.rs1, ins.rs2 = ins.rs2, ins.rs1
	}
	sltReg1, sltReg2 := ins.rs2, ins.rs1

	// MIN -> SLT/SUB/XOR/AND/XOR
	// MAX -> SLT/SUB/XOR/AND/XOR with swapped inputs to SLT
	switch ins.as {
	case AMIN:
		ins.as = ASLT
	case AMAX:
		ins.as, sltReg1, sltReg2 = ASLT, sltReg2, sltReg1
	case AMINU:
		ins.as = ASLTU
	case AMAXU:
		ins.as, sltReg1, sltReg2 = ASLTU, sltReg2, sltReg1
	}
	return []*instruction{
		&instruction{as: ins.as, rs1: sltReg1, rs2: sltReg2, rd: REG_TMP},
		&instruction{as: ASUB, rs1: REG_ZERO, rs2: REG_TMP, rd: REG_TMP},
		&instruction{as: AXOR, rs1: ins.rs1, rs2: ins.rs2, rd: ins.rd},
		&instruction{as: AAND, rs1: REG_TMP, rs2: ins.rd, rd: ins.rd},
		&instruction{as: AXOR, rs1: ins.rs1, rs2: ins.rd, rd: ins.rd},
	}
}

// instructionsForProg returns the machine instructions for an *obj.Prog.
func instructionsForProg(p *obj.Prog) []*instruction {
	ins := instructionForProg(p)
	inss := []*instruction{ins}

	if ins.as == AVSETVLI || ins.as == AVSETIVLI {
		if len(p.RestArgs) != 4 {
			p.Ctxt.Diag("incorrect number of arguments for instruction")
			return nil
		}
	} else if len(p.RestArgs) > 1 {
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

	case AECALL, AEBREAK:
		insEnc := encode(p.As)
		if p.To.Type == obj.TYPE_NONE {
			ins.rd = REG_ZERO
		}
		ins.rs1 = REG_ZERO
		ins.imm = insEnc.csr

	case ARDCYCLE, ARDTIME, ARDINSTRET:
		ins.as = ACSRRS
		if p.To.Type == obj.TYPE_NONE {
			ins.rd = REG_ZERO
		}
		ins.rs1 = REG_ZERO
		switch p.As {
		case ARDCYCLE:
			ins.imm = -1024
		case ARDTIME:
			ins.imm = -1023
		case ARDINSTRET:
			ins.imm = -1022
		}

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

	case AANDN, AORN:
		if buildcfg.GORISCV64 >= 22 {
			// ANDN and ORN instructions are supported natively.
			break
		}
		// ANDN -> (AND (NOT x) y)
		// ORN  -> (OR  (NOT x) y)
		bitwiseOp, notReg := AAND, ins.rd
		if ins.as == AORN {
			bitwiseOp = AOR
		}
		if ins.rs1 == notReg {
			notReg = REG_TMP
		}
		inss = []*instruction{
			&instruction{as: AXORI, rs1: ins.rs2, rs2: obj.REG_NONE, rd: notReg, imm: -1},
			&instruction{as: bitwiseOp, rs1: ins.rs1, rs2: notReg, rd: ins.rd},
		}

	case AXNOR:
		if buildcfg.GORISCV64 >= 22 {
			// XNOR instruction is supported natively.
			break
		}
		// XNOR -> (NOT (XOR x y))
		ins.as = AXOR
		inss = append(inss, &instruction{as: AXORI, rs1: ins.rd, rs2: obj.REG_NONE, rd: ins.rd, imm: -1})

	case AMIN, AMAX, AMINU, AMAXU:
		inss = instructionsForMinMax(p, ins)

	case AVSETVLI, AVSETIVLI:
		ins.rs1, ins.rs2 = ins.rs2, obj.REG_NONE
		vtype, err := EncodeVectorType(p.RestArgs[0].Offset, p.RestArgs[1].Offset, p.RestArgs[2].Offset, p.RestArgs[3].Offset)
		if err != nil {
			p.Ctxt.Diag("%v: %v", p, err)
		}
		ins.imm = int64(vtype)
		if ins.as == AVSETIVLI {
			if p.From.Type != obj.TYPE_CONST {
				p.Ctxt.Diag("%v: expected immediate value", p)
			}
			ins.rs1 = uint32(p.From.Offset)
		}

	case AVLE8V, AVLE16V, AVLE32V, AVLE64V, AVSE8V, AVSE16V, AVSE32V, AVSE64V, AVLMV, AVSMV:
		// Set mask bit
		switch {
		case ins.rs1 == obj.REG_NONE:
			ins.funct7 |= 1 // unmasked
		case ins.rs1 != REG_V0:
			p.Ctxt.Diag("%v: invalid vector mask register", p)
		}
		ins.rd, ins.rs1, ins.rs2 = uint32(p.To.Reg), uint32(p.From.Reg), obj.REG_NONE

	case AVLSE8V, AVLSE16V, AVLSE32V, AVLSE64V,
		AVLUXEI8V, AVLUXEI16V, AVLUXEI32V, AVLUXEI64V, AVLOXEI8V, AVLOXEI16V, AVLOXEI32V, AVLOXEI64V:
		// Set mask bit
		switch {
		case ins.rs3 == obj.REG_NONE:
			ins.funct7 |= 1 // unmasked
		case ins.rs3 != REG_V0:
			p.Ctxt.Diag("%v: invalid vector mask register", p)
		}
		ins.rs1, ins.rs2, ins.rs3 = ins.rs2, ins.rs1, obj.REG_NONE

	case AVSSE8V, AVSSE16V, AVSSE32V, AVSSE64V,
		AVSUXEI8V, AVSUXEI16V, AVSUXEI32V, AVSUXEI64V, AVSOXEI8V, AVSOXEI16V, AVSOXEI32V, AVSOXEI64V:
		// Set mask bit
		switch {
		case ins.rs3 == obj.REG_NONE:
			ins.funct7 |= 1 // unmasked
		case ins.rs3 != REG_V0:
			p.Ctxt.Diag("%v: invalid vector mask register", p)
		}
		ins.rd, ins.rs1, ins.rs2, ins.rs3 = ins.rs2, ins.rd, ins.rs1, obj.REG_NONE

	case AVL1RV, AVL1RE8V, AVL1RE16V, AVL1RE32V, AVL1RE64V, AVL2RV, AVL2RE8V, AVL2RE16V, AVL2RE32V, AVL2RE64V,
		AVL4RV, AVL4RE8V, AVL4RE16V, AVL4RE32V, AVL4RE64V, AVL8RV, AVL8RE8V, AVL8RE16V, AVL8RE32V, AVL8RE64V:
		switch ins.as {
		case AVL1RV:
			ins.as = AVL1RE8V
		case AVL2RV:
			ins.as = AVL2RE8V
		case AVL4RV:
			ins.as = AVL4RE8V
		case AVL8RV:
			ins.as = AVL8RE8V
		}
		if ins.rs1 != obj.REG_NONE {
			p.Ctxt.Diag("%v: too many operands for instruction", p)
		}
		ins.rd, ins.rs1, ins.rs2 = uint32(p.To.Reg), uint32(p.From.Reg), obj.REG_NONE

	case AVS1RV, AVS2RV, AVS4RV, AVS8RV:
		if ins.rs1 != obj.REG_NONE {
			p.Ctxt.Diag("%v: too many operands for instruction", p)
		}
		ins.rd, ins.rs1, ins.rs2 = uint32(p.To.Reg), uint32(p.From.Reg), obj.REG_NONE

	case AVADDVV, AVADDVX, AVSUBVV, AVSUBVX, AVRSUBVX, AVWADDUVV, AVWADDUVX, AVWSUBUVV, AVWSUBUVX,
		AVWADDVV, AVWADDVX, AVWSUBVV, AVWSUBVX, AVWADDUWV, AVWADDUWX, AVWSUBUWV, AVWSUBUWX,
		AVWADDWV, AVWADDWX, AVWSUBWV, AVWSUBWX, AVANDVV, AVANDVX, AVORVV, AVORVX, AVXORVV, AVXORVX,
		AVSLLVV, AVSLLVX, AVSRLVV, AVSRLVX, AVSRAVV, AVSRAVX,
		AVMSEQVV, AVMSEQVX, AVMSNEVV, AVMSNEVX, AVMSLTUVV, AVMSLTUVX, AVMSLTVV, AVMSLTVX,
		AVMSLEUVV, AVMSLEUVX, AVMSLEVV, AVMSLEVX, AVMSGTUVX, AVMSGTVX,
		AVMINUVV, AVMINUVX, AVMINVV, AVMINVX, AVMAXUVV, AVMAXUVX, AVMAXVV, AVMAXVX,
		AVMULVV, AVMULVX, AVMULHVV, AVMULHVX, AVMULHUVV, AVMULHUVX, AVMULHSUVV, AVMULHSUVX,
		AVDIVUVV, AVDIVUVX, AVDIVVV, AVDIVVX, AVREMUVV, AVREMUVX, AVREMVV, AVREMVX,
		AVWMULVV, AVWMULVX, AVWMULUVV, AVWMULUVX, AVWMULSUVV, AVWMULSUVX, AVNSRLWV, AVNSRLWX, AVNSRAWV, AVNSRAWX,
		AVMACCVV, AVMACCVX, AVNMSACVV, AVNMSACVX, AVMADDVV, AVMADDVX, AVNMSUBVV, AVNMSUBVX,
		AVWMACCUVV, AVWMACCUVX, AVWMACCVV, AVWMACCVX, AVWMACCSUVV, AVWMACCSUVX, AVWMACCUSVX,
		AVSADDUVV, AVSADDUVX, AVSADDUVI, AVSADDVV, AVSADDVX, AVSADDVI, AVSSUBUVV, AVSSUBUVX, AVSSUBVV, AVSSUBVX,
		AVAADDUVV, AVAADDUVX, AVAADDVV, AVAADDVX, AVASUBUVV, AVASUBUVX, AVASUBVV, AVASUBVX,
		AVSMULVV, AVSMULVX, AVSSRLVV, AVSSRLVX, AVSSRLVI, AVSSRAVV, AVSSRAVX, AVSSRAVI,
		AVNCLIPUWV, AVNCLIPUWX, AVNCLIPUWI, AVNCLIPWV, AVNCLIPWX, AVNCLIPWI,
		AVFADDVV, AVFADDVF, AVFSUBVV, AVFSUBVF, AVFRSUBVF,
		AVFWADDVV, AVFWADDVF, AVFWSUBVV, AVFWSUBVF, AVFWADDWV, AVFWADDWF, AVFWSUBWV, AVFWSUBWF,
		AVFMULVV, AVFMULVF, AVFDIVVV, AVFDIVVF, AVFRDIVVF, AVFWMULVV, AVFWMULVF,
		AVFMINVV, AVFMINVF, AVFMAXVV, AVFMAXVF,
		AVFSGNJVV, AVFSGNJVF, AVFSGNJNVV, AVFSGNJNVF, AVFSGNJXVV, AVFSGNJXVF,
		AVMFEQVV, AVMFEQVF, AVMFNEVV, AVMFNEVF, AVMFLTVV, AVMFLTVF, AVMFLEVV, AVMFLEVF, AVMFGTVF, AVMFGEVF,
		AVREDSUMVS, AVREDMAXUVS, AVREDMAXVS, AVREDMINUVS, AVREDMINVS, AVREDANDVS, AVREDORVS, AVREDXORVS,
		AVWREDSUMUVS, AVWREDSUMVS, AVFREDOSUMVS, AVFREDUSUMVS, AVFREDMAXVS, AVFREDMINVS, AVFWREDOSUMVS, AVFWREDUSUMVS:
		// Set mask bit
		switch {
		case ins.rs3 == obj.REG_NONE:
			ins.funct7 |= 1 // unmasked
		case ins.rs3 != REG_V0:
			p.Ctxt.Diag("%v: invalid vector mask register", p)
		}
		ins.rd, ins.rs1, ins.rs2, ins.rs3 = uint32(p.To.Reg), uint32(p.From.Reg), uint32(p.Reg), obj.REG_NONE

	case AVFMACCVV, AVFMACCVF, AVFNMACCVV, AVFNMACCVF, AVFMSACVV, AVFMSACVF, AVFNMSACVV, AVFNMSACVF,
		AVFMADDVV, AVFMADDVF, AVFNMADDVV, AVFNMADDVF, AVFMSUBVV, AVFMSUBVF, AVFNMSUBVV, AVFNMSUBVF,
		AVFWMACCVV, AVFWMACCVF, AVFWNMACCVV, AVFWNMACCVF, AVFWMSACVV, AVFWMSACVF, AVFWNMSACVV, AVFWNMSACVF:
		switch {
		case ins.rs3 == obj.REG_NONE:
			ins.funct7 |= 1 // unmasked
		case ins.rs3 != REG_V0:
			p.Ctxt.Diag("%v: invalid vector mask register", p)
		}
		ins.rd, ins.rs1, ins.rs2, ins.rs3 = uint32(p.To.Reg), uint32(p.Reg), uint32(p.From.Reg), obj.REG_NONE

	case AVADDVI, AVRSUBVI, AVANDVI, AVORVI, AVXORVI, AVMSEQVI, AVMSNEVI, AVMSLEUVI, AVMSLEVI, AVMSGTUVI, AVMSGTVI,
		AVSLLVI, AVSRLVI, AVSRAVI, AVNSRLWI, AVNSRAWI:
		// Set mask bit
		switch {
		case ins.rs3 == obj.REG_NONE:
			ins.funct7 |= 1 // unmasked
		case ins.rs3 != REG_V0:
			p.Ctxt.Diag("%v: invalid vector mask register", p)
		}
		ins.rd, ins.rs1, ins.rs2, ins.rs3 = uint32(p.To.Reg), obj.REG_NONE, uint32(p.Reg), obj.REG_NONE

	case AVZEXTVF2, AVSEXTVF2, AVZEXTVF4, AVSEXTVF4, AVZEXTVF8, AVSEXTVF8, AVFSQRTV, AVFRSQRT7V, AVFREC7V, AVFCLASSV,
		AVFCVTXUFV, AVFCVTXFV, AVFCVTRTZXUFV, AVFCVTRTZXFV, AVFCVTFXUV, AVFCVTFXV,
		AVFWCVTXUFV, AVFWCVTXFV, AVFWCVTRTZXUFV, AVFWCVTRTZXFV, AVFWCVTFXUV, AVFWCVTFXV, AVFWCVTFFV,
		AVFNCVTXUFW, AVFNCVTXFW, AVFNCVTRTZXUFW, AVFNCVTRTZXFW, AVFNCVTFXUW, AVFNCVTFXW, AVFNCVTFFW, AVFNCVTRODFFW:
		// Set mask bit
		switch {
		case ins.rs1 == obj.REG_NONE:
			ins.funct7 |= 1 // unmasked
		case ins.rs1 != REG_V0:
			p.Ctxt.Diag("%v: invalid vector mask register", p)
		}
		ins.rs1 = obj.REG_NONE

	case AVMVVV, AVMVVX:
		if ins.rs1 != obj.REG_NONE {
			p.Ctxt.Diag("%v: too many operands for instruction", p)
		}
		ins.rd, ins.rs1, ins.rs2 = uint32(p.To.Reg), uint32(p.From.Reg), REG_V0

	case AVMVVI:
		if ins.rs1 != obj.REG_NONE {
			p.Ctxt.Diag("%v: too many operands for instruction", p)
		}
		ins.rd, ins.rs1, ins.rs2 = uint32(p.To.Reg), obj.REG_NONE, REG_V0

	case AVFMVVF:
		ins.funct7 |= 1 // unmasked
		ins.rd, ins.rs1, ins.rs2 = uint32(p.To.Reg), uint32(p.From.Reg), REG_V0

	case AVADCVVM, AVADCVXM, AVMADCVVM, AVMADCVXM, AVSBCVVM, AVSBCVXM, AVMSBCVVM, AVMSBCVXM, AVADCVIM, AVMADCVIM,
		AVMERGEVVM, AVMERGEVXM, AVMERGEVIM, AVFMERGEVFM:
		if ins.rs3 != REG_V0 {
			p.Ctxt.Diag("%v: invalid vector mask register", p)
		}
		ins.rd, ins.rs1, ins.rs2, ins.rs3 = uint32(p.To.Reg), uint32(p.From.Reg), uint32(p.Reg), obj.REG_NONE

	case AVMADCVV, AVMADCVX, AVMSBCVV, AVMSBCVX, AVMADCVI:
		ins.rd, ins.rs1, ins.rs2 = uint32(p.To.Reg), uint32(p.From.Reg), uint32(p.Reg)

	case AVNEGV, AVWCVTXXV, AVWCVTUXXV, AVNCVTXXW:
		// Set mask bit
		switch {
		case ins.rs1 == obj.REG_NONE:
			ins.funct7 |= 1 // unmasked
		case ins.rs1 != REG_V0:
			p.Ctxt.Diag("%v: invalid vector mask register", p)
		}
		switch ins.as {
		case AVNEGV:
			ins.as = AVRSUBVX
		case AVWCVTXXV:
			ins.as = AVWADDVX
		case AVWCVTUXXV:
			ins.as = AVWADDUVX
		case AVNCVTXXW:
			ins.as = AVNSRLWX
		}
		ins.rd, ins.rs1, ins.rs2 = uint32(p.To.Reg), REG_X0, uint32(p.From.Reg)

	case AVNOTV:
		// Set mask bit
		switch {
		case ins.rs1 == obj.REG_NONE:
			ins.funct7 |= 1 // unmasked
		case ins.rs1 != REG_V0:
			p.Ctxt.Diag("%v: invalid vector mask register", p)
		}
		ins.as = AVXORVI
		ins.rd, ins.rs1, ins.rs2, ins.imm = uint32(p.To.Reg), obj.REG_NONE, uint32(p.From.Reg), -1

	case AVMSGTVV, AVMSGTUVV, AVMSGEVV, AVMSGEUVV, AVMFGTVV, AVMFGEVV:
		// Set mask bit
		switch {
		case ins.rs3 == obj.REG_NONE:
			ins.funct7 |= 1 // unmasked
		case ins.rs3 != REG_V0:
			p.Ctxt.Diag("%v: invalid vector mask register", p)
		}
		switch ins.as {
		case AVMSGTVV:
			ins.as = AVMSLTVV
		case AVMSGTUVV:
			ins.as = AVMSLTUVV
		case AVMSGEVV:
			ins.as = AVMSLEVV
		case AVMSGEUVV:
			ins.as = AVMSLEUVV
		case AVMFGTVV:
			ins.as = AVMFLTVV
		case AVMFGEVV:
			ins.as = AVMFLEVV
		}
		ins.rd, ins.rs1, ins.rs2, ins.rs3 = uint32(p.To.Reg), uint32(p.Reg), uint32(p.From.Reg), obj.REG_NONE

	case AVMSLTVI, AVMSLTUVI, AVMSGEVI, AVMSGEUVI:
		// Set mask bit
		switch {
		case ins.rs3 == obj.REG_NONE:
			ins.funct7 |= 1 // unmasked
		case ins.rs3 != REG_V0:
			p.Ctxt.Diag("%v: invalid vector mask register", p)
		}
		switch ins.as {
		case AVMSLTVI:
			ins.as = AVMSLEVI
		case AVMSLTUVI:
			ins.as = AVMSLEUVI
		case AVMSGEVI:
			ins.as = AVMSGTVI
		case AVMSGEUVI:
			ins.as = AVMSGTUVI
		}
		ins.rd, ins.rs1, ins.rs2, ins.rs3, ins.imm = uint32(p.To.Reg), obj.REG_NONE, uint32(p.Reg), obj.REG_NONE, ins.imm-1

	case AVFABSV, AVFNEGV:
		// Set mask bit
		switch {
		case ins.rs1 == obj.REG_NONE:
			ins.funct7 |= 1 // unmasked
		case ins.rs1 != REG_V0:
			p.Ctxt.Diag("%v: invalid vector mask register", p)
		}
		switch ins.as {
		case AVFABSV:
			ins.as = AVFSGNJXVV
		case AVFNEGV:
			ins.as = AVFSGNJNVV
		}
		ins.rd, ins.rs1, ins.rs2 = uint32(p.To.Reg), uint32(p.From.Reg), uint32(p.From.Reg)
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
				cursym.AddRel(ctxt, obj.Reloc{
					Type: objabi.R_RISCV_JAL,
					Off:  int32(p.Pc),
					Siz:  4,
					Sym:  p.To.Sym,
					Add:  p.To.Offset,
				})
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
			} else if p.Mark&NEED_GOT_PCREL_ITYPE_RELOC == NEED_GOT_PCREL_ITYPE_RELOC {
				rt = objabi.R_RISCV_GOT_PCREL_ITYPE
				addr = &p.From
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

			cursym.AddRel(ctxt, obj.Reloc{
				Type: rt,
				Off:  int32(p.Pc),
				Siz:  8,
				Sym:  addr.Sym,
				Add:  addr.Offset,
			})

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
