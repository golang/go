// cmd/7l/noop.c, cmd/7l/obj.c, cmd/ld/pass.c from Vita Nuova.
// https://code.google.com/p/ken-cc/source/browse/
//
// 	Copyright © 1994-1999 Lucent Technologies Inc. All rights reserved.
// 	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
// 	Portions Copyright © 1997-1999 Vita Nuova Limited
// 	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
// 	Portions Copyright © 2004,2006 Bruce Ellis
// 	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
// 	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
// 	Portions Copyright © 2009 The Go Authors. All rights reserved.
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

package arm64

import (
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"math"
)

var complements = []obj.As{
	AADD:  ASUB,
	AADDW: ASUBW,
	ASUB:  AADD,
	ASUBW: AADDW,
	ACMP:  ACMN,
	ACMPW: ACMNW,
	ACMN:  ACMP,
	ACMNW: ACMPW,
}

func (c *ctxt7) stacksplit(p *obj.Prog, framesize int32) *obj.Prog {
	// MOV	g_stackguard(g), R1
	p = obj.Appendp(p, c.newprog)

	p.As = AMOVD
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = REGG
	p.From.Offset = 2 * int64(c.ctxt.Arch.PtrSize) // G.stackguard0
	if c.cursym.CFunc() {
		p.From.Offset = 3 * int64(c.ctxt.Arch.PtrSize) // G.stackguard1
	}
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_R1

	// Mark the stack bound check and morestack call async nonpreemptible.
	// If we get preempted here, when resumed the preemption request is
	// cleared, but we'll still call morestack, which will double the stack
	// unnecessarily. See issue #35470.
	p = c.ctxt.StartUnsafePoint(p, c.newprog)

	q := (*obj.Prog)(nil)
	if framesize <= objabi.StackSmall {
		// small stack: SP < stackguard
		//	MOV	SP, R2
		//	CMP	stackguard, R2
		p = obj.Appendp(p, c.newprog)

		p.As = AMOVD
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REGSP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R2

		p = obj.Appendp(p, c.newprog)
		p.As = ACMP
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R1
		p.Reg = REG_R2
	} else if framesize <= objabi.StackBig {
		// large stack: SP-framesize < stackguard-StackSmall
		//	SUB	$(framesize-StackSmall), SP, R2
		//	CMP	stackguard, R2
		p = obj.Appendp(p, c.newprog)

		p.As = ASUB
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(framesize) - objabi.StackSmall
		p.Reg = REGSP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R2

		p = obj.Appendp(p, c.newprog)
		p.As = ACMP
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R1
		p.Reg = REG_R2
	} else {
		// Such a large stack we need to protect against wraparound
		// if SP is close to zero.
		//	SP-stackguard+StackGuard < framesize + (StackGuard-StackSmall)
		// The +StackGuard on both sides is required to keep the left side positive:
		// SP is allowed to be slightly below stackguard. See stack.h.
		//	CMP	$StackPreempt, R1
		//	BEQ	label_of_call_to_morestack
		//	ADD	$StackGuard, SP, R2
		//	SUB	R1, R2
		//	MOV	$(framesize+(StackGuard-StackSmall)), R3
		//	CMP	R3, R2
		p = obj.Appendp(p, c.newprog)

		p.As = ACMP
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = objabi.StackPreempt
		p.Reg = REG_R1

		p = obj.Appendp(p, c.newprog)
		q = p
		p.As = ABEQ
		p.To.Type = obj.TYPE_BRANCH

		p = obj.Appendp(p, c.newprog)
		p.As = AADD
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(objabi.StackGuard)
		p.Reg = REGSP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R2

		p = obj.Appendp(p, c.newprog)
		p.As = ASUB
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R2

		p = obj.Appendp(p, c.newprog)
		p.As = AMOVD
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(framesize) + (int64(objabi.StackGuard) - objabi.StackSmall)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R3

		p = obj.Appendp(p, c.newprog)
		p.As = ACMP
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R3
		p.Reg = REG_R2
	}

	// BLS	do-morestack
	bls := obj.Appendp(p, c.newprog)
	bls.As = ABLS
	bls.To.Type = obj.TYPE_BRANCH

	end := c.ctxt.EndUnsafePoint(bls, c.newprog, -1)

	var last *obj.Prog
	for last = c.cursym.Func.Text; last.Link != nil; last = last.Link {
	}

	// Now we are at the end of the function, but logically
	// we are still in function prologue. We need to fix the
	// SP data and PCDATA.
	spfix := obj.Appendp(last, c.newprog)
	spfix.As = obj.ANOP
	spfix.Spadj = -framesize

	pcdata := c.ctxt.EmitEntryStackMap(c.cursym, spfix, c.newprog)
	pcdata = c.ctxt.StartUnsafePoint(pcdata, c.newprog)

	// MOV	LR, R3
	movlr := obj.Appendp(pcdata, c.newprog)
	movlr.As = AMOVD
	movlr.From.Type = obj.TYPE_REG
	movlr.From.Reg = REGLINK
	movlr.To.Type = obj.TYPE_REG
	movlr.To.Reg = REG_R3
	if q != nil {
		q.Pcond = movlr
	}
	bls.Pcond = movlr

	debug := movlr
	if false {
		debug = obj.Appendp(debug, c.newprog)
		debug.As = AMOVD
		debug.From.Type = obj.TYPE_CONST
		debug.From.Offset = int64(framesize)
		debug.To.Type = obj.TYPE_REG
		debug.To.Reg = REGTMP
	}

	// BL	runtime.morestack(SB)
	call := obj.Appendp(debug, c.newprog)
	call.As = ABL
	call.To.Type = obj.TYPE_BRANCH
	morestack := "runtime.morestack"
	switch {
	case c.cursym.CFunc():
		morestack = "runtime.morestackc"
	case !c.cursym.Func.Text.From.Sym.NeedCtxt():
		morestack = "runtime.morestack_noctxt"
	}
	call.To.Sym = c.ctxt.Lookup(morestack)

	pcdata = c.ctxt.EndUnsafePoint(call, c.newprog, -1)

	// B	start
	jmp := obj.Appendp(pcdata, c.newprog)
	jmp.As = AB
	jmp.To.Type = obj.TYPE_BRANCH
	jmp.Pcond = c.cursym.Func.Text.Link
	jmp.Spadj = +framesize

	return end
}

func progedit(ctxt *obj.Link, p *obj.Prog, newprog obj.ProgAlloc) {
	c := ctxt7{ctxt: ctxt, newprog: newprog}

	p.From.Class = 0
	p.To.Class = 0

	// $0 results in C_ZCON, which matches both C_REG and various
	// C_xCON, however the C_REG cases in asmout don't expect a
	// constant, so they will use the register fields and assemble
	// a R0. To prevent that, rewrite $0 as ZR.
	if p.From.Type == obj.TYPE_CONST && p.From.Offset == 0 {
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REGZERO
	}
	if p.To.Type == obj.TYPE_CONST && p.To.Offset == 0 {
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REGZERO
	}

	// Rewrite BR/BL to symbol as TYPE_BRANCH.
	switch p.As {
	case AB,
		ABL,
		obj.ARET,
		obj.ADUFFZERO,
		obj.ADUFFCOPY:
		if p.To.Sym != nil {
			p.To.Type = obj.TYPE_BRANCH
		}
		break
	}

	// Rewrite float constants to values stored in memory.
	switch p.As {
	case AFMOVS:
		if p.From.Type == obj.TYPE_FCONST {
			f64 := p.From.Val.(float64)
			f32 := float32(f64)
			if c.chipfloat7(f64) > 0 {
				break
			}
			if math.Float32bits(f32) == 0 {
				p.From.Type = obj.TYPE_REG
				p.From.Reg = REGZERO
				break
			}
			p.From.Type = obj.TYPE_MEM
			p.From.Sym = c.ctxt.Float32Sym(f32)
			p.From.Name = obj.NAME_EXTERN
			p.From.Offset = 0
		}

	case AFMOVD:
		if p.From.Type == obj.TYPE_FCONST {
			f64 := p.From.Val.(float64)
			if c.chipfloat7(f64) > 0 {
				break
			}
			if math.Float64bits(f64) == 0 {
				p.From.Type = obj.TYPE_REG
				p.From.Reg = REGZERO
				break
			}
			p.From.Type = obj.TYPE_MEM
			p.From.Sym = c.ctxt.Float64Sym(f64)
			p.From.Name = obj.NAME_EXTERN
			p.From.Offset = 0
		}

		break
	}

	// Rewrite negative immediates as positive immediates with
	// complementary instruction.
	switch p.As {
	case AADD, ASUB, ACMP, ACMN:
		if p.From.Type == obj.TYPE_CONST && p.From.Offset < 0 && p.From.Offset != -1<<63 {
			p.From.Offset = -p.From.Offset
			p.As = complements[p.As]
		}
	case AADDW, ASUBW, ACMPW, ACMNW:
		if p.From.Type == obj.TYPE_CONST && p.From.Offset < 0 && int32(p.From.Offset) != -1<<31 {
			p.From.Offset = -p.From.Offset
			p.As = complements[p.As]
		}
	}

	// For 32-bit logical instruction with constant,
	// rewrite the high 32-bit to be a repetition of
	// the low 32-bit, so that the BITCON test can be
	// shared for both 32-bit and 64-bit. 32-bit ops
	// will zero the high 32-bit of the destination
	// register anyway.
	if isANDWop(p.As) && p.From.Type == obj.TYPE_CONST {
		v := p.From.Offset & 0xffffffff
		p.From.Offset = v | v<<32
	}

	if c.ctxt.Flag_dynlink {
		c.rewriteToUseGot(p)
	}
}

// Rewrite p, if necessary, to access global data via the global offset table.
func (c *ctxt7) rewriteToUseGot(p *obj.Prog) {
	if p.As == obj.ADUFFCOPY || p.As == obj.ADUFFZERO {
		//     ADUFFxxx $offset
		// becomes
		//     MOVD runtime.duffxxx@GOT, REGTMP
		//     ADD $offset, REGTMP
		//     CALL REGTMP
		var sym *obj.LSym
		if p.As == obj.ADUFFZERO {
			sym = c.ctxt.Lookup("runtime.duffzero")
		} else {
			sym = c.ctxt.Lookup("runtime.duffcopy")
		}
		offset := p.To.Offset
		p.As = AMOVD
		p.From.Type = obj.TYPE_MEM
		p.From.Name = obj.NAME_GOTREF
		p.From.Sym = sym
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REGTMP
		p.To.Name = obj.NAME_NONE
		p.To.Offset = 0
		p.To.Sym = nil
		p1 := obj.Appendp(p, c.newprog)
		p1.As = AADD
		p1.From.Type = obj.TYPE_CONST
		p1.From.Offset = offset
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = REGTMP
		p2 := obj.Appendp(p1, c.newprog)
		p2.As = obj.ACALL
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = REGTMP
	}

	// We only care about global data: NAME_EXTERN means a global
	// symbol in the Go sense, and p.Sym.Local is true for a few
	// internally defined symbols.
	if p.From.Type == obj.TYPE_ADDR && p.From.Name == obj.NAME_EXTERN && !p.From.Sym.Local() {
		// MOVD $sym, Rx becomes MOVD sym@GOT, Rx
		// MOVD $sym+<off>, Rx becomes MOVD sym@GOT, Rx; ADD <off>, Rx
		if p.As != AMOVD {
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
	// MOVx sym, Ry becomes MOVD sym@GOT, REGTMP; MOVx (REGTMP), Ry
	// MOVx Ry, sym becomes MOVD sym@GOT, REGTMP; MOVD Ry, (REGTMP)
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
	p1.As = AMOVD
	p1.From.Type = obj.TYPE_MEM
	p1.From.Sym = source.Sym
	p1.From.Name = obj.NAME_GOTREF
	p1.To.Type = obj.TYPE_REG
	p1.To.Reg = REGTMP

	p2.As = p.As
	p2.From = p.From
	p2.To = p.To
	if p.From.Name == obj.NAME_EXTERN {
		p2.From.Reg = REGTMP
		p2.From.Name = obj.NAME_NONE
		p2.From.Sym = nil
	} else if p.To.Name == obj.NAME_EXTERN {
		p2.To.Reg = REGTMP
		p2.To.Name = obj.NAME_NONE
		p2.To.Sym = nil
	} else {
		return
	}
	obj.Nopout(p)
}

func preprocess(ctxt *obj.Link, cursym *obj.LSym, newprog obj.ProgAlloc) {
	if cursym.Func.Text == nil || cursym.Func.Text.Link == nil {
		return
	}

	c := ctxt7{ctxt: ctxt, newprog: newprog, cursym: cursym}

	p := c.cursym.Func.Text
	textstksiz := p.To.Offset
	if textstksiz == -8 {
		// Historical way to mark NOFRAME.
		p.From.Sym.Set(obj.AttrNoFrame, true)
		textstksiz = 0
	}
	if textstksiz < 0 {
		c.ctxt.Diag("negative frame size %d - did you mean NOFRAME?", textstksiz)
	}
	if p.From.Sym.NoFrame() {
		if textstksiz != 0 {
			c.ctxt.Diag("NOFRAME functions must have a frame size of 0, not %d", textstksiz)
		}
	}

	c.cursym.Func.Args = p.To.Val.(int32)
	c.cursym.Func.Locals = int32(textstksiz)

	/*
	 * find leaf subroutines
	 * strip NOPs
	 * expand RET
	 */
	q := (*obj.Prog)(nil)
	var q1 *obj.Prog
	for p := c.cursym.Func.Text; p != nil; p = p.Link {
		switch p.As {
		case obj.ATEXT:
			p.Mark |= LEAF

		case obj.ARET:
			break

		case obj.ANOP:
			if p.Link != nil {
				q1 = p.Link
				q.Link = q1 /* q is non-nop */
				q1.Mark |= p.Mark
			}
			continue

		case ABL,
			obj.ADUFFZERO,
			obj.ADUFFCOPY:
			c.cursym.Func.Text.Mark &^= LEAF
			fallthrough

		case ACBNZ,
			ACBZ,
			ACBNZW,
			ACBZW,
			ATBZ,
			ATBNZ,
			AB,
			ABEQ,
			ABNE,
			ABCS,
			ABHS,
			ABCC,
			ABLO,
			ABMI,
			ABPL,
			ABVS,
			ABVC,
			ABHI,
			ABLS,
			ABGE,
			ABLT,
			ABGT,
			ABLE,
			AADR, /* strange */
			AADRP:
			q1 = p.Pcond

			if q1 != nil {
				for q1.As == obj.ANOP {
					q1 = q1.Link
					p.Pcond = q1
				}
			}

			break
		}

		q = p
	}

	var retjmp *obj.LSym
	for p := c.cursym.Func.Text; p != nil; p = p.Link {
		o := p.As
		switch o {
		case obj.ATEXT:
			c.cursym.Func.Text = p
			c.autosize = int32(textstksiz)

			if p.Mark&LEAF != 0 && c.autosize == 0 {
				// A leaf function with no locals has no frame.
				p.From.Sym.Set(obj.AttrNoFrame, true)
			}

			if !p.From.Sym.NoFrame() {
				// If there is a stack frame at all, it includes
				// space to save the LR.
				c.autosize += 8
			}

			if c.autosize != 0 {
				extrasize := int32(0)
				if c.autosize%16 == 8 {
					// Allocate extra 8 bytes on the frame top to save FP
					extrasize = 8
				} else if c.autosize&(16-1) == 0 {
					// Allocate extra 16 bytes to save FP for the old frame whose size is 8 mod 16
					extrasize = 16
				} else {
					c.ctxt.Diag("%v: unaligned frame size %d - must be 16 aligned", p, c.autosize-8)
				}
				c.autosize += extrasize
				c.cursym.Func.Locals += extrasize

				// low 32 bits for autosize
				// high 32 bits for extrasize
				p.To.Offset = int64(c.autosize) | int64(extrasize)<<32
			} else {
				// NOFRAME
				p.To.Offset = 0
			}

			if c.autosize == 0 && c.cursym.Func.Text.Mark&LEAF == 0 {
				if c.ctxt.Debugvlog {
					c.ctxt.Logf("save suppressed in: %s\n", c.cursym.Func.Text.From.Sym.Name)
				}
				c.cursym.Func.Text.Mark |= LEAF
			}

			if cursym.Func.Text.Mark&LEAF != 0 {
				cursym.Set(obj.AttrLeaf, true)
				if p.From.Sym.NoFrame() {
					break
				}
			}

			if !p.From.Sym.NoSplit() {
				p = c.stacksplit(p, c.autosize) // emit split check
			}

			aoffset := c.autosize
			if aoffset > 0xF0 {
				aoffset = 0xF0
			}

			// Frame is non-empty. Make sure to save link register, even if
			// it is a leaf function, so that traceback works.
			q = p
			if c.autosize > aoffset {
				// Frame size is too large for a MOVD.W instruction.
				// Store link register before decrementing SP, so if a signal comes
				// during the execution of the function prologue, the traceback
				// code will not see a half-updated stack frame.
				// This sequence is not async preemptible, as if we open a frame
				// at the current SP, it will clobber the saved LR.
				q = c.ctxt.StartUnsafePoint(q, c.newprog)

				q = obj.Appendp(q, c.newprog)
				q.Pos = p.Pos
				q.As = ASUB
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = int64(c.autosize)
				q.Reg = REGSP
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REGTMP

				q = obj.Appendp(q, c.newprog)
				q.Pos = p.Pos
				q.As = AMOVD
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REGLINK
				q.To.Type = obj.TYPE_MEM
				q.To.Reg = REGTMP

				q1 = obj.Appendp(q, c.newprog)
				q1.Pos = p.Pos
				q1.As = AMOVD
				q1.From.Type = obj.TYPE_REG
				q1.From.Reg = REGTMP
				q1.To.Type = obj.TYPE_REG
				q1.To.Reg = REGSP
				q1.Spadj = c.autosize

				if c.ctxt.Headtype == objabi.Hdarwin {
					// iOS does not support SA_ONSTACK. We will run the signal handler
					// on the G stack. If we write below SP, it may be clobbered by
					// the signal handler. So we save LR after decrementing SP.
					q1 = obj.Appendp(q1, c.newprog)
					q1.Pos = p.Pos
					q1.As = AMOVD
					q1.From.Type = obj.TYPE_REG
					q1.From.Reg = REGLINK
					q1.To.Type = obj.TYPE_MEM
					q1.To.Reg = REGSP
				}

				q1 = c.ctxt.EndUnsafePoint(q1, c.newprog, -1)
			} else {
				// small frame, update SP and save LR in a single MOVD.W instruction
				q1 = obj.Appendp(q, c.newprog)
				q1.As = AMOVD
				q1.Pos = p.Pos
				q1.From.Type = obj.TYPE_REG
				q1.From.Reg = REGLINK
				q1.To.Type = obj.TYPE_MEM
				q1.Scond = C_XPRE
				q1.To.Offset = int64(-aoffset)
				q1.To.Reg = REGSP
				q1.Spadj = aoffset
			}

			if objabi.Framepointer_enabled(objabi.GOOS, objabi.GOARCH) {
				q1 = obj.Appendp(q1, c.newprog)
				q1.Pos = p.Pos
				q1.As = AMOVD
				q1.From.Type = obj.TYPE_REG
				q1.From.Reg = REGFP
				q1.To.Type = obj.TYPE_MEM
				q1.To.Reg = REGSP
				q1.To.Offset = -8

				q1 = obj.Appendp(q1, c.newprog)
				q1.Pos = p.Pos
				q1.As = ASUB
				q1.From.Type = obj.TYPE_CONST
				q1.From.Offset = 8
				q1.Reg = REGSP
				q1.To.Type = obj.TYPE_REG
				q1.To.Reg = REGFP
			}

			if c.cursym.Func.Text.From.Sym.Wrapper() {
				// if(g->panic != nil && g->panic->argp == FP) g->panic->argp = bottom-of-frame
				//
				//	MOV  g_panic(g), R1
				//	CBNZ checkargp
				// end:
				//	NOP
				// ... function body ...
				// checkargp:
				//	MOV  panic_argp(R1), R2
				//	ADD  $(autosize+8), RSP, R3
				//	CMP  R2, R3
				//	BNE  end
				//	ADD  $8, RSP, R4
				//	MOVD R4, panic_argp(R1)
				//	B    end
				//
				// The NOP is needed to give the jumps somewhere to land.
				// It is a liblink NOP, not an ARM64 NOP: it encodes to 0 instruction bytes.
				q = q1

				// MOV g_panic(g), R1
				q = obj.Appendp(q, c.newprog)
				q.As = AMOVD
				q.From.Type = obj.TYPE_MEM
				q.From.Reg = REGG
				q.From.Offset = 4 * int64(c.ctxt.Arch.PtrSize) // G.panic
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R1

				// CBNZ R1, checkargp
				cbnz := obj.Appendp(q, c.newprog)
				cbnz.As = ACBNZ
				cbnz.From.Type = obj.TYPE_REG
				cbnz.From.Reg = REG_R1
				cbnz.To.Type = obj.TYPE_BRANCH

				// Empty branch target at the top of the function body
				end := obj.Appendp(cbnz, c.newprog)
				end.As = obj.ANOP

				// find the end of the function
				var last *obj.Prog
				for last = end; last.Link != nil; last = last.Link {
				}

				// MOV panic_argp(R1), R2
				mov := obj.Appendp(last, c.newprog)
				mov.As = AMOVD
				mov.From.Type = obj.TYPE_MEM
				mov.From.Reg = REG_R1
				mov.From.Offset = 0 // Panic.argp
				mov.To.Type = obj.TYPE_REG
				mov.To.Reg = REG_R2

				// CBNZ branches to the MOV above
				cbnz.Pcond = mov

				// ADD $(autosize+8), SP, R3
				q = obj.Appendp(mov, c.newprog)
				q.As = AADD
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = int64(c.autosize) + 8
				q.Reg = REGSP
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R3

				// CMP R2, R3
				q = obj.Appendp(q, c.newprog)
				q.As = ACMP
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R2
				q.Reg = REG_R3

				// BNE end
				q = obj.Appendp(q, c.newprog)
				q.As = ABNE
				q.To.Type = obj.TYPE_BRANCH
				q.Pcond = end

				// ADD $8, SP, R4
				q = obj.Appendp(q, c.newprog)
				q.As = AADD
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = 8
				q.Reg = REGSP
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R4

				// MOV R4, panic_argp(R1)
				q = obj.Appendp(q, c.newprog)
				q.As = AMOVD
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R4
				q.To.Type = obj.TYPE_MEM
				q.To.Reg = REG_R1
				q.To.Offset = 0 // Panic.argp

				// B end
				q = obj.Appendp(q, c.newprog)
				q.As = AB
				q.To.Type = obj.TYPE_BRANCH
				q.Pcond = end
			}

		case obj.ARET:
			nocache(p)
			if p.From.Type == obj.TYPE_CONST {
				c.ctxt.Diag("using BECOME (%v) is not supported!", p)
				break
			}

			retjmp = p.To.Sym
			p.To = obj.Addr{}
			if c.cursym.Func.Text.Mark&LEAF != 0 {
				if c.autosize != 0 {
					p.As = AADD
					p.From.Type = obj.TYPE_CONST
					p.From.Offset = int64(c.autosize)
					p.To.Type = obj.TYPE_REG
					p.To.Reg = REGSP
					p.Spadj = -c.autosize

					if objabi.Framepointer_enabled(objabi.GOOS, objabi.GOARCH) {
						p = obj.Appendp(p, c.newprog)
						p.As = ASUB
						p.From.Type = obj.TYPE_CONST
						p.From.Offset = 8
						p.Reg = REGSP
						p.To.Type = obj.TYPE_REG
						p.To.Reg = REGFP
					}
				}
			} else {
				/* want write-back pre-indexed SP+autosize -> SP, loading REGLINK*/

				if objabi.Framepointer_enabled(objabi.GOOS, objabi.GOARCH) {
					p.As = AMOVD
					p.From.Type = obj.TYPE_MEM
					p.From.Reg = REGSP
					p.From.Offset = -8
					p.To.Type = obj.TYPE_REG
					p.To.Reg = REGFP
					p = obj.Appendp(p, c.newprog)
				}

				aoffset := c.autosize

				if aoffset <= 0xF0 {
					p.As = AMOVD
					p.From.Type = obj.TYPE_MEM
					p.Scond = C_XPOST
					p.From.Offset = int64(aoffset)
					p.From.Reg = REGSP
					p.To.Type = obj.TYPE_REG
					p.To.Reg = REGLINK
					p.Spadj = -aoffset
				} else {
					p.As = AMOVD
					p.From.Type = obj.TYPE_MEM
					p.From.Offset = 0
					p.From.Reg = REGSP
					p.To.Type = obj.TYPE_REG
					p.To.Reg = REGLINK

					q = newprog()
					q.As = AADD
					q.From.Type = obj.TYPE_CONST
					q.From.Offset = int64(aoffset)
					q.To.Type = obj.TYPE_REG
					q.To.Reg = REGSP
					q.Link = p.Link
					q.Spadj = int32(-q.From.Offset)
					q.Pos = p.Pos
					p.Link = q
					p = q
				}
			}

			if p.As != obj.ARET {
				q = newprog()
				q.Pos = p.Pos
				q.Link = p.Link
				p.Link = q
				p = q
			}

			if retjmp != nil { // retjmp
				p.As = AB
				p.To.Type = obj.TYPE_BRANCH
				p.To.Sym = retjmp
				p.Spadj = +c.autosize
				break
			}

			p.As = obj.ARET
			p.To.Type = obj.TYPE_MEM
			p.To.Offset = 0
			p.To.Reg = REGLINK
			p.Spadj = +c.autosize

		case AADD, ASUB:
			if p.To.Type == obj.TYPE_REG && p.To.Reg == REGSP && p.From.Type == obj.TYPE_CONST {
				if p.As == AADD {
					p.Spadj = int32(-p.From.Offset)
				} else {
					p.Spadj = int32(+p.From.Offset)
				}
			}

		case obj.AGETCALLERPC:
			if cursym.Leaf() {
				/* MOVD LR, Rd */
				p.As = AMOVD
				p.From.Type = obj.TYPE_REG
				p.From.Reg = REGLINK
			} else {
				/* MOVD (RSP), Rd */
				p.As = AMOVD
				p.From.Type = obj.TYPE_MEM
				p.From.Reg = REGSP
			}

		case obj.ADUFFCOPY:
			if objabi.Framepointer_enabled(objabi.GOOS, objabi.GOARCH) {
				//  ADR	ret_addr, R27
				//  STP	(FP, R27), -24(SP)
				//  SUB	24, SP, FP
				//  DUFFCOPY
				// ret_addr:
				//  SUB	8, SP, FP

				q1 := p
				// copy DUFFCOPY from q1 to q4
				q4 := obj.Appendp(p, c.newprog)
				q4.Pos = p.Pos
				q4.As = obj.ADUFFCOPY
				q4.To = p.To

				q1.As = AADR
				q1.From.Type = obj.TYPE_BRANCH
				q1.To.Type = obj.TYPE_REG
				q1.To.Reg = REG_R27

				q2 := obj.Appendp(q1, c.newprog)
				q2.Pos = p.Pos
				q2.As = ASTP
				q2.From.Type = obj.TYPE_REGREG
				q2.From.Reg = REGFP
				q2.From.Offset = int64(REG_R27)
				q2.To.Type = obj.TYPE_MEM
				q2.To.Reg = REGSP
				q2.To.Offset = -24

				// maintaine FP for DUFFCOPY
				q3 := obj.Appendp(q2, c.newprog)
				q3.Pos = p.Pos
				q3.As = ASUB
				q3.From.Type = obj.TYPE_CONST
				q3.From.Offset = 24
				q3.Reg = REGSP
				q3.To.Type = obj.TYPE_REG
				q3.To.Reg = REGFP

				q5 := obj.Appendp(q4, c.newprog)
				q5.Pos = p.Pos
				q5.As = ASUB
				q5.From.Type = obj.TYPE_CONST
				q5.From.Offset = 8
				q5.Reg = REGSP
				q5.To.Type = obj.TYPE_REG
				q5.To.Reg = REGFP
				q1.Pcond = q5
				p = q5
			}

		case obj.ADUFFZERO:
			if objabi.Framepointer_enabled(objabi.GOOS, objabi.GOARCH) {
				//  ADR	ret_addr, R27
				//  STP	(FP, R27), -24(SP)
				//  SUB	24, SP, FP
				//  DUFFZERO
				// ret_addr:
				//  SUB	8, SP, FP

				q1 := p
				// copy DUFFZERO from q1 to q4
				q4 := obj.Appendp(p, c.newprog)
				q4.Pos = p.Pos
				q4.As = obj.ADUFFZERO
				q4.To = p.To

				q1.As = AADR
				q1.From.Type = obj.TYPE_BRANCH
				q1.To.Type = obj.TYPE_REG
				q1.To.Reg = REG_R27

				q2 := obj.Appendp(q1, c.newprog)
				q2.Pos = p.Pos
				q2.As = ASTP
				q2.From.Type = obj.TYPE_REGREG
				q2.From.Reg = REGFP
				q2.From.Offset = int64(REG_R27)
				q2.To.Type = obj.TYPE_MEM
				q2.To.Reg = REGSP
				q2.To.Offset = -24

				// maintaine FP for DUFFZERO
				q3 := obj.Appendp(q2, c.newprog)
				q3.Pos = p.Pos
				q3.As = ASUB
				q3.From.Type = obj.TYPE_CONST
				q3.From.Offset = 24
				q3.Reg = REGSP
				q3.To.Type = obj.TYPE_REG
				q3.To.Reg = REGFP

				q5 := obj.Appendp(q4, c.newprog)
				q5.Pos = p.Pos
				q5.As = ASUB
				q5.From.Type = obj.TYPE_CONST
				q5.From.Offset = 8
				q5.Reg = REGSP
				q5.To.Type = obj.TYPE_REG
				q5.To.Reg = REGFP
				q1.Pcond = q5
				p = q5
			}
		}
	}
}

func nocache(p *obj.Prog) {
	p.Optab = 0
	p.From.Class = 0
	p.To.Class = 0
}

var unaryDst = map[obj.As]bool{
	AWORD:  true,
	ADWORD: true,
	ABL:    true,
	AB:     true,
	ACLREX: true,
}

var Linkarm64 = obj.LinkArch{
	Arch:           sys.ArchARM64,
	Init:           buildop,
	Preprocess:     preprocess,
	Assemble:       span7,
	Progedit:       progedit,
	UnaryDst:       unaryDst,
	DWARFRegisters: ARM64DWARFRegisters,
}
