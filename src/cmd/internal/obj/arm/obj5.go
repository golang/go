// Derived from Inferno utils/5c/swt.c
// https://bitbucket.org/inferno-os/inferno-os/src/master/utils/5c/swt.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors. All rights reserved.
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

package arm

import (
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"internal/abi"
	"internal/buildcfg"
	"log"
)

var progedit_tlsfallback *obj.LSym

func progedit(ctxt *obj.Link, p *obj.Prog, newprog obj.ProgAlloc) {
	p.From.Class = 0
	p.To.Class = 0

	c := ctxt5{ctxt: ctxt, newprog: newprog}

	// Rewrite B/BL to symbol as TYPE_BRANCH.
	switch p.As {
	case AB, ABL, obj.ADUFFZERO, obj.ADUFFCOPY:
		if p.To.Type == obj.TYPE_MEM && (p.To.Name == obj.NAME_EXTERN || p.To.Name == obj.NAME_STATIC) && p.To.Sym != nil {
			p.To.Type = obj.TYPE_BRANCH
		}
	}

	// Replace TLS register fetches on older ARM processors.
	switch p.As {
	// Treat MRC 15, 0, <reg>, C13, C0, 3 specially.
	case AMRC:
		if p.To.Offset&0xffff0fff == 0xee1d0f70 {
			// Because the instruction might be rewritten to a BL which returns in R0
			// the register must be zero.
			if p.To.Offset&0xf000 != 0 {
				ctxt.Diag("%v: TLS MRC instruction must write to R0 as it might get translated into a BL instruction", p.Line())
			}

			if buildcfg.GOARM.Version < 7 {
				// Replace it with BL runtime.read_tls_fallback(SB) for ARM CPUs that lack the tls extension.
				if progedit_tlsfallback == nil {
					progedit_tlsfallback = ctxt.Lookup("runtime.read_tls_fallback")
				}

				// MOVW	LR, R11
				p.As = AMOVW

				p.From.Type = obj.TYPE_REG
				p.From.Reg = REGLINK
				p.To.Type = obj.TYPE_REG
				p.To.Reg = REGTMP

				// BL	runtime.read_tls_fallback(SB)
				p = obj.Appendp(p, newprog)

				p.As = ABL
				p.To.Type = obj.TYPE_BRANCH
				p.To.Sym = progedit_tlsfallback
				p.To.Offset = 0

				// MOVW	R11, LR
				p = obj.Appendp(p, newprog)

				p.As = AMOVW
				p.From.Type = obj.TYPE_REG
				p.From.Reg = REGTMP
				p.To.Type = obj.TYPE_REG
				p.To.Reg = REGLINK
				break
			}
		}

		// Otherwise, MRC/MCR instructions need no further treatment.
		p.As = AWORD
	}

	// Rewrite float constants to values stored in memory.
	switch p.As {
	case AMOVF:
		if p.From.Type == obj.TYPE_FCONST && c.chipfloat5(p.From.Val.(float64)) < 0 && (c.chipzero5(p.From.Val.(float64)) < 0 || p.Scond&C_SCOND != C_SCOND_NONE) {
			f32 := float32(p.From.Val.(float64))
			p.From.Type = obj.TYPE_MEM
			p.From.Sym = ctxt.Float32Sym(f32)
			p.From.Name = obj.NAME_EXTERN
			p.From.Offset = 0
		}

	case AMOVD:
		if p.From.Type == obj.TYPE_FCONST && c.chipfloat5(p.From.Val.(float64)) < 0 && (c.chipzero5(p.From.Val.(float64)) < 0 || p.Scond&C_SCOND != C_SCOND_NONE) {
			p.From.Type = obj.TYPE_MEM
			p.From.Sym = ctxt.Float64Sym(p.From.Val.(float64))
			p.From.Name = obj.NAME_EXTERN
			p.From.Offset = 0
		}
	}

	if ctxt.Flag_dynlink {
		c.rewriteToUseGot(p)
	}
}

// Rewrite p, if necessary, to access global data via the global offset table.
func (c *ctxt5) rewriteToUseGot(p *obj.Prog) {
	if p.As == obj.ADUFFCOPY || p.As == obj.ADUFFZERO {
		//     ADUFFxxx $offset
		// becomes
		//     MOVW runtime.duffxxx@GOT, R9
		//     ADD $offset, R9
		//     CALL (R9)
		var sym *obj.LSym
		if p.As == obj.ADUFFZERO {
			sym = c.ctxt.Lookup("runtime.duffzero")
		} else {
			sym = c.ctxt.Lookup("runtime.duffcopy")
		}
		offset := p.To.Offset
		p.As = AMOVW
		p.From.Type = obj.TYPE_MEM
		p.From.Name = obj.NAME_GOTREF
		p.From.Sym = sym
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R9
		p.To.Name = obj.NAME_NONE
		p.To.Offset = 0
		p.To.Sym = nil
		p1 := obj.Appendp(p, c.newprog)
		p1.As = AADD
		p1.From.Type = obj.TYPE_CONST
		p1.From.Offset = offset
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = REG_R9
		p2 := obj.Appendp(p1, c.newprog)
		p2.As = obj.ACALL
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = REG_R9
		return
	}

	// We only care about global data: NAME_EXTERN means a global
	// symbol in the Go sense, and p.Sym.Local is true for a few
	// internally defined symbols.
	if p.From.Type == obj.TYPE_ADDR && p.From.Name == obj.NAME_EXTERN && !p.From.Sym.Local() {
		// MOVW $sym, Rx becomes MOVW sym@GOT, Rx
		// MOVW $sym+<off>, Rx becomes MOVW sym@GOT, Rx; ADD <off>, Rx
		if p.As != AMOVW {
			c.ctxt.Diag("do not know how to handle TYPE_ADDR in %v with -dynlink", p)
		}
		if p.To.Type != obj.TYPE_REG {
			c.ctxt.Diag("do not know how to handle LEAQ-type insn to non-register in %v with -dynlink", p)
		}
		p.From.Type = obj.TYPE_MEM
		p.From.Name = obj.NAME_GOTREF
		if p.From.Offset != 0 {
			q := obj.Appendp(p, c.newprog)
			q.As = AADD
			q.From.Type = obj.TYPE_CONST
			q.From.Offset = p.From.Offset
			q.To = p.To
			p.From.Offset = 0
		}
	}
	if p.GetFrom3() != nil && p.GetFrom3().Name == obj.NAME_EXTERN {
		c.ctxt.Diag("don't know how to handle %v with -dynlink", p)
	}
	var source *obj.Addr
	// MOVx sym, Ry becomes MOVW sym@GOT, R9; MOVx (R9), Ry
	// MOVx Ry, sym becomes MOVW sym@GOT, R9; MOVx Ry, (R9)
	// An addition may be inserted between the two MOVs if there is an offset.
	if p.From.Name == obj.NAME_EXTERN && !p.From.Sym.Local() {
		if p.To.Name == obj.NAME_EXTERN && !p.To.Sym.Local() {
			c.ctxt.Diag("cannot handle NAME_EXTERN on both sides in %v with -dynlink", p)
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
		c.ctxt.Diag("don't know how to handle %v with -dynlink", p)
	}
	p1 := obj.Appendp(p, c.newprog)
	p2 := obj.Appendp(p1, c.newprog)

	p1.As = AMOVW
	p1.From.Type = obj.TYPE_MEM
	p1.From.Sym = source.Sym
	p1.From.Name = obj.NAME_GOTREF
	p1.To.Type = obj.TYPE_REG
	p1.To.Reg = REG_R9

	p2.As = p.As
	p2.From = p.From
	p2.To = p.To
	if p.From.Name == obj.NAME_EXTERN {
		p2.From.Reg = REG_R9
		p2.From.Name = obj.NAME_NONE
		p2.From.Sym = nil
	} else if p.To.Name == obj.NAME_EXTERN {
		p2.To.Reg = REG_R9
		p2.To.Name = obj.NAME_NONE
		p2.To.Sym = nil
	} else {
		return
	}
	obj.Nopout(p)
}

// Prog.mark
const (
	FOLL  = 1 << 0
	LABEL = 1 << 1
	LEAF  = 1 << 2
)

func preprocess(ctxt *obj.Link, cursym *obj.LSym, newprog obj.ProgAlloc) {
	autosize := int32(0)

	if cursym.Func().Text == nil || cursym.Func().Text.Link == nil {
		return
	}

	c := ctxt5{ctxt: ctxt, cursym: cursym, newprog: newprog}

	p := c.cursym.Func().Text
	autoffset := int32(p.To.Offset)
	if autoffset == -4 {
		// Historical way to mark NOFRAME.
		p.From.Sym.Set(obj.AttrNoFrame, true)
		autoffset = 0
	}
	if autoffset < 0 || autoffset%4 != 0 {
		c.ctxt.Diag("frame size %d not 0 or a positive multiple of 4", autoffset)
	}
	if p.From.Sym.NoFrame() {
		if autoffset != 0 {
			c.ctxt.Diag("NOFRAME functions must have a frame size of 0, not %d", autoffset)
		}
	}

	cursym.Func().Locals = autoffset
	cursym.Func().Args = p.To.Val.(int32)

	/*
	 * find leaf subroutines
	 */
	for p := cursym.Func().Text; p != nil; p = p.Link {
		switch p.As {
		case obj.ATEXT:
			p.Mark |= LEAF

		case ADIV, ADIVU, AMOD, AMODU:
			cursym.Func().Text.Mark &^= LEAF

		case ABL,
			ABX,
			obj.ADUFFZERO,
			obj.ADUFFCOPY:
			cursym.Func().Text.Mark &^= LEAF
		}
	}

	var q2 *obj.Prog
	for p := cursym.Func().Text; p != nil; p = p.Link {
		o := p.As
		switch o {
		case obj.ATEXT:
			autosize = autoffset

			if p.Mark&LEAF != 0 && autosize == 0 {
				// A leaf function with no locals has no frame.
				p.From.Sym.Set(obj.AttrNoFrame, true)
			}

			if !p.From.Sym.NoFrame() {
				// If there is a stack frame at all, it includes
				// space to save the LR.
				autosize += 4
			}

			if autosize == 0 && cursym.Func().Text.Mark&LEAF == 0 {
				// A very few functions that do not return to their caller
				// are not identified as leaves but still have no frame.
				if ctxt.Debugvlog {
					ctxt.Logf("save suppressed in: %s\n", cursym.Name)
				}

				cursym.Func().Text.Mark |= LEAF
			}

			// FP offsets need an updated p.To.Offset.
			p.To.Offset = int64(autosize) - 4

			if cursym.Func().Text.Mark&LEAF != 0 {
				cursym.Set(obj.AttrLeaf, true)
				if p.From.Sym.NoFrame() {
					break
				}
			}

			if !p.From.Sym.NoSplit() {
				p = c.stacksplit(p, autosize) // emit split check
			}

			// MOVW.W		R14,$-autosize(SP)
			p = obj.Appendp(p, c.newprog)

			p.As = AMOVW
			p.Scond |= C_WBIT
			p.From.Type = obj.TYPE_REG
			p.From.Reg = REGLINK
			p.To.Type = obj.TYPE_MEM
			p.To.Offset = int64(-autosize)
			p.To.Reg = REGSP
			p.Spadj = autosize

			if cursym.Func().Text.From.Sym.Wrapper() {
				// if(g->panic != nil && g->panic->argp == FP) g->panic->argp = bottom-of-frame
				//
				//	MOVW g_panic(g), R1
				//	CMP  $0, R1
				//	B.NE checkargp
				// end:
				//	NOP
				// ... function ...
				// checkargp:
				//	MOVW panic_argp(R1), R2
				//	ADD  $(autosize+4), R13, R3
				//	CMP  R2, R3
				//	B.NE end
				//	ADD  $4, R13, R4
				//	MOVW R4, panic_argp(R1)
				//	B    end
				//
				// The NOP is needed to give the jumps somewhere to land.
				// It is a liblink NOP, not an ARM NOP: it encodes to 0 instruction bytes.

				p = obj.Appendp(p, newprog)
				p.As = AMOVW
				p.From.Type = obj.TYPE_MEM
				p.From.Reg = REGG
				p.From.Offset = 4 * int64(ctxt.Arch.PtrSize) // G.panic
				p.To.Type = obj.TYPE_REG
				p.To.Reg = REG_R1

				p = obj.Appendp(p, newprog)
				p.As = ACMP
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = 0
				p.Reg = REG_R1

				// B.NE checkargp
				bne := obj.Appendp(p, newprog)
				bne.As = ABNE
				bne.To.Type = obj.TYPE_BRANCH

				// end: NOP
				end := obj.Appendp(bne, newprog)
				end.As = obj.ANOP

				// find end of function
				var last *obj.Prog
				for last = end; last.Link != nil; last = last.Link {
				}

				// MOVW panic_argp(R1), R2
				mov := obj.Appendp(last, newprog)
				mov.As = AMOVW
				mov.From.Type = obj.TYPE_MEM
				mov.From.Reg = REG_R1
				mov.From.Offset = 0 // Panic.argp
				mov.To.Type = obj.TYPE_REG
				mov.To.Reg = REG_R2

				// B.NE branch target is MOVW above
				bne.To.SetTarget(mov)

				// ADD $(autosize+4), R13, R3
				p = obj.Appendp(mov, newprog)
				p.As = AADD
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = int64(autosize) + 4
				p.Reg = REG_R13
				p.To.Type = obj.TYPE_REG
				p.To.Reg = REG_R3

				// CMP R2, R3
				p = obj.Appendp(p, newprog)
				p.As = ACMP
				p.From.Type = obj.TYPE_REG
				p.From.Reg = REG_R2
				p.Reg = REG_R3

				// B.NE end
				p = obj.Appendp(p, newprog)
				p.As = ABNE
				p.To.Type = obj.TYPE_BRANCH
				p.To.SetTarget(end)

				// ADD $4, R13, R4
				p = obj.Appendp(p, newprog)
				p.As = AADD
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = 4
				p.Reg = REG_R13
				p.To.Type = obj.TYPE_REG
				p.To.Reg = REG_R4

				// MOVW R4, panic_argp(R1)
				p = obj.Appendp(p, newprog)
				p.As = AMOVW
				p.From.Type = obj.TYPE_REG
				p.From.Reg = REG_R4
				p.To.Type = obj.TYPE_MEM
				p.To.Reg = REG_R1
				p.To.Offset = 0 // Panic.argp

				// B end
				p = obj.Appendp(p, newprog)
				p.As = AB
				p.To.Type = obj.TYPE_BRANCH
				p.To.SetTarget(end)

				// reset for subsequent passes
				p = end
			}

		case obj.ARET:
			nocache(p)
			if cursym.Func().Text.Mark&LEAF != 0 {
				if autosize == 0 {
					p.As = AB
					p.From = obj.Addr{}
					if p.To.Sym != nil { // retjmp
						p.To.Type = obj.TYPE_BRANCH
					} else {
						p.To.Type = obj.TYPE_MEM
						p.To.Offset = 0
						p.To.Reg = REGLINK
					}

					break
				}
			}

			p.As = AMOVW
			p.Scond |= C_PBIT
			p.From.Type = obj.TYPE_MEM
			p.From.Offset = int64(autosize)
			p.From.Reg = REGSP
			p.To.Type = obj.TYPE_REG
			p.To.Reg = REGPC

			// If there are instructions following
			// this ARET, they come from a branch
			// with the same stackframe, so no spadj.

			if p.To.Sym != nil { // retjmp
				p.To.Reg = REGLINK
				q2 = obj.Appendp(p, newprog)
				q2.As = AB
				q2.To.Type = obj.TYPE_BRANCH
				q2.To.Sym = p.To.Sym
				p.To.Sym = nil
				p.To.Name = obj.NAME_NONE
				p = q2
			}

		case AADD:
			if p.From.Type == obj.TYPE_CONST && p.From.Reg == 0 && p.To.Type == obj.TYPE_REG && p.To.Reg == REGSP {
				p.Spadj = int32(-p.From.Offset)
			}

		case ASUB:
			if p.From.Type == obj.TYPE_CONST && p.From.Reg == 0 && p.To.Type == obj.TYPE_REG && p.To.Reg == REGSP {
				p.Spadj = int32(p.From.Offset)
			}

		case ADIV, ADIVU, AMOD, AMODU:
			if cursym.Func().Text.From.Sym.NoSplit() {
				ctxt.Diag("cannot divide in NOSPLIT function")
			}
			const debugdivmod = false
			if debugdivmod {
				break
			}
			if p.From.Type != obj.TYPE_REG {
				break
			}
			if p.To.Type != obj.TYPE_REG {
				break
			}

			// Make copy because we overwrite p below.
			q1 := *p
			if q1.Reg == REGTMP || q1.Reg == 0 && q1.To.Reg == REGTMP {
				ctxt.Diag("div already using REGTMP: %v", p)
			}

			/* MOV m(g),REGTMP */
			p.As = AMOVW
			p.Pos = q1.Pos
			p.From.Type = obj.TYPE_MEM
			p.From.Reg = REGG
			p.From.Offset = 6 * 4 // offset of g.m
			p.Reg = 0
			p.To.Type = obj.TYPE_REG
			p.To.Reg = REGTMP

			/* MOV a,m_divmod(REGTMP) */
			p = obj.Appendp(p, newprog)
			p.As = AMOVW
			p.Pos = q1.Pos
			p.From.Type = obj.TYPE_REG
			p.From.Reg = q1.From.Reg
			p.To.Type = obj.TYPE_MEM
			p.To.Reg = REGTMP
			p.To.Offset = 7 * 4 // offset of m.divmod

			/* MOV b, R8 */
			p = obj.Appendp(p, newprog)
			p.As = AMOVW
			p.Pos = q1.Pos
			p.From.Type = obj.TYPE_REG
			p.From.Reg = q1.Reg
			if q1.Reg == 0 {
				p.From.Reg = q1.To.Reg
			}
			p.To.Type = obj.TYPE_REG
			p.To.Reg = REG_R8
			p.To.Offset = 0

			/* CALL appropriate */
			p = obj.Appendp(p, newprog)
			p.As = ABL
			p.Pos = q1.Pos
			p.To.Type = obj.TYPE_BRANCH
			switch o {
			case ADIV:
				p.To.Sym = symdiv
			case ADIVU:
				p.To.Sym = symdivu
			case AMOD:
				p.To.Sym = symmod
			case AMODU:
				p.To.Sym = symmodu
			}

			/* MOV REGTMP, b */
			p = obj.Appendp(p, newprog)
			p.As = AMOVW
			p.Pos = q1.Pos
			p.From.Type = obj.TYPE_REG
			p.From.Reg = REGTMP
			p.From.Offset = 0
			p.To.Type = obj.TYPE_REG
			p.To.Reg = q1.To.Reg

		case AMOVW:
			if (p.Scond&C_WBIT != 0) && p.To.Type == obj.TYPE_MEM && p.To.Reg == REGSP {
				p.Spadj = int32(-p.To.Offset)
			}
			if (p.Scond&C_PBIT != 0) && p.From.Type == obj.TYPE_MEM && p.From.Reg == REGSP && p.To.Reg != REGPC {
				p.Spadj = int32(-p.From.Offset)
			}
			if p.From.Type == obj.TYPE_ADDR && p.From.Reg == REGSP && p.To.Type == obj.TYPE_REG && p.To.Reg == REGSP {
				p.Spadj = int32(-p.From.Offset)
			}

		case obj.AGETCALLERPC:
			if cursym.Leaf() {
				/* MOVW LR, Rd */
				p.As = AMOVW
				p.From.Type = obj.TYPE_REG
				p.From.Reg = REGLINK
			} else {
				/* MOVW (RSP), Rd */
				p.As = AMOVW
				p.From.Type = obj.TYPE_MEM
				p.From.Reg = REGSP
			}
		}

		if p.To.Type == obj.TYPE_REG && p.To.Reg == REGSP && p.Spadj == 0 {
			f := c.cursym.Func()
			if f.FuncFlag&abi.FuncFlagSPWrite == 0 {
				c.cursym.Func().FuncFlag |= abi.FuncFlagSPWrite
				if ctxt.Debugvlog || !ctxt.IsAsm {
					ctxt.Logf("auto-SPWRITE: %s %v\n", c.cursym.Name, p)
					if !ctxt.IsAsm {
						ctxt.Diag("invalid auto-SPWRITE in non-assembly")
						ctxt.DiagFlush()
						log.Fatalf("bad SPWRITE")
					}
				}
			}
		}
	}
}

func (c *ctxt5) stacksplit(p *obj.Prog, framesize int32) *obj.Prog {
	if c.ctxt.Flag_maymorestack != "" {
		// Save LR and make room for REGCTXT.
		const frameSize = 8
		// MOVW.W R14,$-8(SP)
		p = obj.Appendp(p, c.newprog)
		p.As = AMOVW
		p.Scond |= C_WBIT
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REGLINK
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = -frameSize
		p.To.Reg = REGSP
		p.Spadj = frameSize

		// MOVW REGCTXT, 4(SP)
		p = obj.Appendp(p, c.newprog)
		p.As = AMOVW
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REGCTXT
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = 4
		p.To.Reg = REGSP

		// CALL maymorestack
		p = obj.Appendp(p, c.newprog)
		p.As = obj.ACALL
		p.To.Type = obj.TYPE_BRANCH
		// See ../x86/obj6.go
		p.To.Sym = c.ctxt.LookupABI(c.ctxt.Flag_maymorestack, c.cursym.ABI())

		// Restore REGCTXT and LR.

		// MOVW 4(SP), REGCTXT
		p = obj.Appendp(p, c.newprog)
		p.As = AMOVW
		p.From.Type = obj.TYPE_MEM
		p.From.Offset = 4
		p.From.Reg = REGSP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REGCTXT

		// MOVW.P 8(SP), R14
		p.As = AMOVW
		p.Scond |= C_PBIT
		p.From.Type = obj.TYPE_MEM
		p.From.Offset = frameSize
		p.From.Reg = REGSP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REGLINK
		p.Spadj = -frameSize
	}

	// Jump back to here after morestack returns.
	startPred := p

	// MOVW g_stackguard(g), R1
	p = obj.Appendp(p, c.newprog)

	p.As = AMOVW
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = REGG
	p.From.Offset = 2 * int64(c.ctxt.Arch.PtrSize) // G.stackguard0
	if c.cursym.CFunc() {
		p.From.Offset = 3 * int64(c.ctxt.Arch.PtrSize) // G.stackguard1
	}
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_R1

	if framesize <= abi.StackSmall {
		// small stack: SP < stackguard
		//	CMP	stackguard, SP
		p = obj.Appendp(p, c.newprog)

		p.As = ACMP
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R1
		p.Reg = REGSP
	} else if framesize <= abi.StackBig {
		// large stack: SP-framesize < stackguard-StackSmall
		//	MOVW $-(framesize-StackSmall)(SP), R2
		//	CMP stackguard, R2
		p = obj.Appendp(p, c.newprog)

		p.As = AMOVW
		p.From.Type = obj.TYPE_ADDR
		p.From.Reg = REGSP
		p.From.Offset = -(int64(framesize) - abi.StackSmall)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R2

		p = obj.Appendp(p, c.newprog)
		p.As = ACMP
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R1
		p.Reg = REG_R2
	} else {
		// Such a large stack we need to protect against underflow.
		// The runtime guarantees SP > objabi.StackBig, but
		// framesize is large enough that SP-framesize may
		// underflow, causing a direct comparison with the
		// stack guard to incorrectly succeed. We explicitly
		// guard against underflow.
		//
		//	// Try subtracting from SP and check for underflow.
		//	// If this underflows, it sets C to 0.
		//	SUB.S $(framesize-StackSmall), SP, R2
		//	// If C is 1 (unsigned >=), compare with guard.
		//	CMP.HS stackguard, R2

		p = obj.Appendp(p, c.newprog)
		p.As = ASUB
		p.Scond = C_SBIT
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(framesize) - abi.StackSmall
		p.Reg = REGSP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R2

		p = obj.Appendp(p, c.newprog)
		p.As = ACMP
		p.Scond = C_SCOND_HS
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R1
		p.Reg = REG_R2
	}

	// BLS call-to-morestack (C is 0 or Z is 1)
	bls := obj.Appendp(p, c.newprog)
	bls.As = ABLS
	bls.To.Type = obj.TYPE_BRANCH

	var last *obj.Prog
	for last = c.cursym.Func().Text; last.Link != nil; last = last.Link {
	}

	// Now we are at the end of the function, but logically
	// we are still in function prologue. We need to fix the
	// SP data and PCDATA.
	spfix := obj.Appendp(last, c.newprog)
	spfix.As = obj.ANOP
	spfix.Spadj = -framesize

	pcdata := c.ctxt.EmitEntryStackMap(c.cursym, spfix, c.newprog)

	// MOVW	LR, R3
	movw := obj.Appendp(pcdata, c.newprog)
	movw.As = AMOVW
	movw.From.Type = obj.TYPE_REG
	movw.From.Reg = REGLINK
	movw.To.Type = obj.TYPE_REG
	movw.To.Reg = REG_R3

	bls.To.SetTarget(movw)

	// BL runtime.morestack
	call := obj.Appendp(movw, c.newprog)
	call.As = obj.ACALL
	call.To.Type = obj.TYPE_BRANCH
	morestack := "runtime.morestack"
	switch {
	case c.cursym.CFunc():
		morestack = "runtime.morestackc"
	case !c.cursym.Func().Text.From.Sym.NeedCtxt():
		morestack = "runtime.morestack_noctxt"
	}
	call.To.Sym = c.ctxt.Lookup(morestack)

	// B start
	b := obj.Appendp(call, c.newprog)
	b.As = obj.AJMP
	b.To.Type = obj.TYPE_BRANCH
	b.To.SetTarget(startPred.Link)
	b.Spadj = +framesize

	return bls
}

var unaryDst = map[obj.As]bool{
	ASWI:  true,
	AWORD: true,
}

var Linkarm = obj.LinkArch{
	Arch:           sys.ArchARM,
	Init:           buildop,
	Preprocess:     preprocess,
	Assemble:       span5,
	Progedit:       progedit,
	UnaryDst:       unaryDst,
	DWARFRegisters: ARMDWARFRegisters,
}
