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
	"cmd/internal/sys"
	"fmt"
	"log"
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

func stacksplit(ctxt *obj.Link, p *obj.Prog, framesize int32) *obj.Prog {
	// MOV	g_stackguard(g), R1
	p = obj.Appendp(ctxt, p)

	p.As = AMOVD
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = REGG
	p.From.Offset = 2 * int64(ctxt.Arch.PtrSize) // G.stackguard0
	if ctxt.Cursym.CFunc() {
		p.From.Offset = 3 * int64(ctxt.Arch.PtrSize) // G.stackguard1
	}
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_R1

	q := (*obj.Prog)(nil)
	if framesize <= obj.StackSmall {
		// small stack: SP < stackguard
		//	MOV	SP, R2
		//	CMP	stackguard, R2
		p = obj.Appendp(ctxt, p)

		p.As = AMOVD
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REGSP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R2

		p = obj.Appendp(ctxt, p)
		p.As = ACMP
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R1
		p.Reg = REG_R2
	} else if framesize <= obj.StackBig {
		// large stack: SP-framesize < stackguard-StackSmall
		//	SUB	$framesize, SP, R2
		//	CMP	stackguard, R2
		p = obj.Appendp(ctxt, p)

		p.As = ASUB
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(framesize)
		p.Reg = REGSP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R2

		p = obj.Appendp(ctxt, p)
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
		p = obj.Appendp(ctxt, p)

		p.As = ACMP
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = obj.StackPreempt
		p.Reg = REG_R1

		p = obj.Appendp(ctxt, p)
		q = p
		p.As = ABEQ
		p.To.Type = obj.TYPE_BRANCH

		p = obj.Appendp(ctxt, p)
		p.As = AADD
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = obj.StackGuard
		p.Reg = REGSP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R2

		p = obj.Appendp(ctxt, p)
		p.As = ASUB
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R2

		p = obj.Appendp(ctxt, p)
		p.As = AMOVD
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(framesize) + (obj.StackGuard - obj.StackSmall)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R3

		p = obj.Appendp(ctxt, p)
		p.As = ACMP
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R3
		p.Reg = REG_R2
	}

	// BLS	do-morestack
	bls := obj.Appendp(ctxt, p)
	bls.As = ABLS
	bls.To.Type = obj.TYPE_BRANCH

	var last *obj.Prog
	for last = ctxt.Cursym.Text; last.Link != nil; last = last.Link {
	}

	// Now we are at the end of the function, but logically
	// we are still in function prologue. We need to fix the
	// SP data and PCDATA.
	spfix := obj.Appendp(ctxt, last)
	spfix.As = obj.ANOP
	spfix.Spadj = -framesize

	pcdata := obj.Appendp(ctxt, spfix)
	pcdata.Lineno = ctxt.Cursym.Text.Lineno
	pcdata.Mode = ctxt.Cursym.Text.Mode
	pcdata.As = obj.APCDATA
	pcdata.From.Type = obj.TYPE_CONST
	pcdata.From.Offset = obj.PCDATA_StackMapIndex
	pcdata.To.Type = obj.TYPE_CONST
	pcdata.To.Offset = -1 // pcdata starts at -1 at function entry

	// MOV	LR, R3
	movlr := obj.Appendp(ctxt, pcdata)
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
		debug = obj.Appendp(ctxt, debug)
		debug.As = AMOVD
		debug.From.Type = obj.TYPE_CONST
		debug.From.Offset = int64(framesize)
		debug.To.Type = obj.TYPE_REG
		debug.To.Reg = REGTMP
	}

	// BL	runtime.morestack(SB)
	call := obj.Appendp(ctxt, debug)
	call.As = ABL
	call.To.Type = obj.TYPE_BRANCH
	morestack := "runtime.morestack"
	switch {
	case ctxt.Cursym.CFunc():
		morestack = "runtime.morestackc"
	case ctxt.Cursym.Text.From3.Offset&obj.NEEDCTXT == 0:
		morestack = "runtime.morestack_noctxt"
	}
	call.To.Sym = obj.Linklookup(ctxt, morestack, 0)

	// B	start
	jmp := obj.Appendp(ctxt, call)
	jmp.As = AB
	jmp.To.Type = obj.TYPE_BRANCH
	jmp.Pcond = ctxt.Cursym.Text.Link
	jmp.Spadj = +framesize

	// placeholder for bls's jump target
	// p = obj.Appendp(ctxt, p)
	// p.As = obj.ANOP

	return bls
}

func progedit(ctxt *obj.Link, p *obj.Prog) {
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
			f32 := float32(p.From.Val.(float64))
			i32 := math.Float32bits(f32)
			if i32 == 0 {
				p.From.Type = obj.TYPE_REG
				p.From.Reg = REGZERO
				break
			}
			literal := fmt.Sprintf("$f32.%08x", i32)
			s := obj.Linklookup(ctxt, literal, 0)
			s.Size = 4
			p.From.Type = obj.TYPE_MEM
			p.From.Sym = s
			p.From.Sym.Set(obj.AttrLocal, true)
			p.From.Name = obj.NAME_EXTERN
			p.From.Offset = 0
		}

	case AFMOVD:
		if p.From.Type == obj.TYPE_FCONST {
			i64 := math.Float64bits(p.From.Val.(float64))
			if i64 == 0 {
				p.From.Type = obj.TYPE_REG
				p.From.Reg = REGZERO
				break
			}
			literal := fmt.Sprintf("$f64.%016x", i64)
			s := obj.Linklookup(ctxt, literal, 0)
			s.Size = 8
			p.From.Type = obj.TYPE_MEM
			p.From.Sym = s
			p.From.Sym.Set(obj.AttrLocal, true)
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
	switch p.As {
	case AANDW, AORRW, AEORW, AANDSW:
		if p.From.Type == obj.TYPE_CONST {
			v := p.From.Offset & 0xffffffff
			p.From.Offset = v | v<<32
		}
	}

	if ctxt.Flag_dynlink {
		rewriteToUseGot(ctxt, p)
	}
}

// Rewrite p, if necessary, to access global data via the global offset table.
func rewriteToUseGot(ctxt *obj.Link, p *obj.Prog) {
	if p.As == obj.ADUFFCOPY || p.As == obj.ADUFFZERO {
		//     ADUFFxxx $offset
		// becomes
		//     MOVD runtime.duffxxx@GOT, REGTMP
		//     ADD $offset, REGTMP
		//     CALL REGTMP
		var sym *obj.LSym
		if p.As == obj.ADUFFZERO {
			sym = obj.Linklookup(ctxt, "runtime.duffzero", 0)
		} else {
			sym = obj.Linklookup(ctxt, "runtime.duffcopy", 0)
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
		p1 := obj.Appendp(ctxt, p)
		p1.As = AADD
		p1.From.Type = obj.TYPE_CONST
		p1.From.Offset = offset
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = REGTMP
		p2 := obj.Appendp(ctxt, p1)
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
			ctxt.Diag("do not know how to handle TYPE_ADDR in %v with -dynlink", p)
		}
		if p.To.Type != obj.TYPE_REG {
			ctxt.Diag("do not know how to handle LEAQ-type insn to non-register in %v with -dynlink", p)
		}
		p.From.Type = obj.TYPE_MEM
		p.From.Name = obj.NAME_GOTREF
		if p.From.Offset != 0 {
			q := obj.Appendp(ctxt, p)
			q.As = AADD
			q.From.Type = obj.TYPE_CONST
			q.From.Offset = p.From.Offset
			q.To = p.To
			p.From.Offset = 0
		}
	}
	if p.From3 != nil && p.From3.Name == obj.NAME_EXTERN {
		ctxt.Diag("don't know how to handle %v with -dynlink", p)
	}
	var source *obj.Addr
	// MOVx sym, Ry becomes MOVD sym@GOT, REGTMP; MOVx (REGTMP), Ry
	// MOVx Ry, sym becomes MOVD sym@GOT, REGTMP; MOVD Ry, (REGTMP)
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
	if source.Sym.Type == obj.STLSBSS {
		return
	}
	if source.Type != obj.TYPE_MEM {
		ctxt.Diag("don't know how to handle %v with -dynlink", p)
	}
	p1 := obj.Appendp(ctxt, p)
	p2 := obj.Appendp(ctxt, p1)
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

func follow(ctxt *obj.Link, s *obj.LSym) {
	ctxt.Cursym = s

	firstp := ctxt.NewProg()
	lastp := firstp
	xfol(ctxt, s.Text, &lastp)
	lastp.Link = nil
	s.Text = firstp.Link
}

func relinv(a obj.As) obj.As {
	switch a {
	case ABEQ:
		return ABNE
	case ABNE:
		return ABEQ
	case ABCS:
		return ABCC
	case ABHS:
		return ABLO
	case ABCC:
		return ABCS
	case ABLO:
		return ABHS
	case ABMI:
		return ABPL
	case ABPL:
		return ABMI
	case ABVS:
		return ABVC
	case ABVC:
		return ABVS
	case ABHI:
		return ABLS
	case ABLS:
		return ABHI
	case ABGE:
		return ABLT
	case ABLT:
		return ABGE
	case ABGT:
		return ABLE
	case ABLE:
		return ABGT
	case ACBZ:
		return ACBNZ
	case ACBNZ:
		return ACBZ
	case ACBZW:
		return ACBNZW
	case ACBNZW:
		return ACBZW
	}

	log.Fatalf("unknown relation: %s", Anames[a-obj.ABaseARM64])
	return 0
}

func xfol(ctxt *obj.Link, p *obj.Prog, last **obj.Prog) {
	var q *obj.Prog
	var r *obj.Prog
	var i int

loop:
	if p == nil {
		return
	}
	a := p.As
	if a == AB {
		q = p.Pcond
		if q != nil {
			p.Mark |= FOLL
			p = q
			if !(p.Mark&FOLL != 0) {
				goto loop
			}
		}
	}

	if p.Mark&FOLL != 0 {
		i = 0
		q = p
		for ; i < 4; i, q = i+1, q.Link {
			if q == *last || q == nil {
				break
			}
			a = q.As
			if a == obj.ANOP {
				i--
				continue
			}

			if a == AB || a == obj.ARET || a == AERET {
				goto copy
			}
			if q.Pcond == nil || (q.Pcond.Mark&FOLL != 0) {
				continue
			}
			if a != ABEQ && a != ABNE {
				continue
			}

		copy:
			for {
				r = ctxt.NewProg()
				*r = *p
				if !(r.Mark&FOLL != 0) {
					fmt.Printf("can't happen 1\n")
				}
				r.Mark |= FOLL
				if p != q {
					p = p.Link
					(*last).Link = r
					*last = r
					continue
				}

				(*last).Link = r
				*last = r
				if a == AB || a == obj.ARET || a == AERET {
					return
				}
				if a == ABNE {
					r.As = ABEQ
				} else {
					r.As = ABNE
				}
				r.Pcond = p.Link
				r.Link = p.Pcond
				if !(r.Link.Mark&FOLL != 0) {
					xfol(ctxt, r.Link, last)
				}
				if !(r.Pcond.Mark&FOLL != 0) {
					fmt.Printf("can't happen 2\n")
				}
				return
			}
		}

		a = AB
		q = ctxt.NewProg()
		q.As = a
		q.Lineno = p.Lineno
		q.To.Type = obj.TYPE_BRANCH
		q.To.Offset = p.Pc
		q.Pcond = p
		p = q
	}

	p.Mark |= FOLL
	(*last).Link = p
	*last = p
	if a == AB || a == obj.ARET || a == AERET {
		return
	}
	if p.Pcond != nil {
		if a != ABL && p.Link != nil {
			q = obj.Brchain(ctxt, p.Link)
			if a != obj.ATEXT {
				if q != nil && (q.Mark&FOLL != 0) {
					p.As = relinv(a)
					p.Link = p.Pcond
					p.Pcond = q
				}
			}

			xfol(ctxt, p.Link, last)
			q = obj.Brchain(ctxt, p.Pcond)
			if q == nil {
				q = p.Pcond
			}
			if q.Mark&FOLL != 0 {
				p.Pcond = q
				return
			}

			p = q
			goto loop
		}
	}

	p = p.Link
	goto loop
}

func preprocess(ctxt *obj.Link, cursym *obj.LSym) {
	ctxt.Cursym = cursym

	if cursym.Text == nil || cursym.Text.Link == nil {
		return
	}

	p := cursym.Text
	textstksiz := p.To.Offset
	aoffset := int32(textstksiz)

	cursym.Args = p.To.Val.(int32)
	cursym.Locals = int32(textstksiz)

	/*
	 * find leaf subroutines
	 * strip NOPs
	 * expand RET
	 */
	q := (*obj.Prog)(nil)
	var q1 *obj.Prog
	for p := cursym.Text; p != nil; p = p.Link {
		switch p.As {
		case obj.ATEXT:
			p.Mark |= LEAF

		case obj.ARET:
			break

		case obj.ANOP:
			q1 = p.Link
			q.Link = q1 /* q is non-nop */
			q1.Mark |= p.Mark
			continue

		case ABL,
			obj.ADUFFZERO,
			obj.ADUFFCOPY:
			cursym.Text.Mark &^= LEAF
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

	var q2 *obj.Prog
	var retjmp *obj.LSym
	for p := cursym.Text; p != nil; p = p.Link {
		o := p.As
		switch o {
		case obj.ATEXT:
			cursym.Text = p
			if textstksiz < 0 {
				ctxt.Autosize = 0
			} else {
				ctxt.Autosize = int32(textstksiz + 8)
			}
			if (cursym.Text.Mark&LEAF != 0) && ctxt.Autosize <= 8 {
				ctxt.Autosize = 0
			} else if ctxt.Autosize&(16-1) != 0 {
				// The frame includes an LR.
				// If the frame size is 8, it's only an LR,
				// so there's no potential for breaking references to
				// local variables by growing the frame size,
				// because there are no local variables.
				// But otherwise, if there is a non-empty locals section,
				// the author of the code is responsible for making sure
				// that the frame size is 8 mod 16.
				if ctxt.Autosize == 8 {
					ctxt.Autosize += 8
					cursym.Locals += 8
				} else {
					ctxt.Diag("%v: unaligned frame size %d - must be 8 mod 16 (or 0)", p, ctxt.Autosize-8)
				}
			}
			p.To.Offset = int64(ctxt.Autosize) - 8
			if ctxt.Autosize == 0 && !(cursym.Text.Mark&LEAF != 0) {
				if ctxt.Debugvlog != 0 {
					ctxt.Logf("save suppressed in: %s\n", cursym.Text.From.Sym.Name)
				}
				cursym.Text.Mark |= LEAF
			}

			if !(p.From3.Offset&obj.NOSPLIT != 0) {
				p = stacksplit(ctxt, p, ctxt.Autosize) // emit split check
			}

			aoffset = ctxt.Autosize
			if aoffset > 0xF0 {
				aoffset = 0xF0
			}
			if cursym.Text.Mark&LEAF != 0 {
				cursym.Set(obj.AttrLeaf, true)
				if ctxt.Autosize == 0 {
					break
				}
			}

			// Frame is non-empty. Make sure to save link register, even if
			// it is a leaf function, so that traceback works.
			q = p
			if ctxt.Autosize > aoffset {
				// Frame size is too large for a MOVD.W instruction.
				// Store link register before decrementing SP, so if a signal comes
				// during the execution of the function prologue, the traceback
				// code will not see a half-updated stack frame.
				q = obj.Appendp(ctxt, q)
				q.Lineno = p.Lineno
				q.As = ASUB
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = int64(ctxt.Autosize)
				q.Reg = REGSP
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REGTMP

				q = obj.Appendp(ctxt, q)
				q.Lineno = p.Lineno
				q.As = AMOVD
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REGLINK
				q.To.Type = obj.TYPE_MEM
				q.To.Reg = REGTMP

				q1 = obj.Appendp(ctxt, q)
				q1.Lineno = p.Lineno
				q1.As = AMOVD
				q1.From.Type = obj.TYPE_REG
				q1.From.Reg = REGTMP
				q1.To.Type = obj.TYPE_REG
				q1.To.Reg = REGSP
				q1.Spadj = ctxt.Autosize
			} else {
				// small frame, update SP and save LR in a single MOVD.W instruction
				q1 = obj.Appendp(ctxt, q)
				q1.As = AMOVD
				q1.Lineno = p.Lineno
				q1.From.Type = obj.TYPE_REG
				q1.From.Reg = REGLINK
				q1.To.Type = obj.TYPE_MEM
				q1.Scond = C_XPRE
				q1.To.Offset = int64(-aoffset)
				q1.To.Reg = REGSP
				q1.Spadj = aoffset
			}

			if cursym.Text.From3.Offset&obj.WRAPPER != 0 {
				// if(g->panic != nil && g->panic->argp == FP) g->panic->argp = bottom-of-frame
				//
				//	MOV g_panic(g), R1
				//	CMP ZR, R1
				//	BEQ end
				//	MOV panic_argp(R1), R2
				//	ADD $(autosize+8), RSP, R3
				//	CMP R2, R3
				//	BNE end
				//	ADD $8, RSP, R4
				//	MOVD R4, panic_argp(R1)
				// end:
				//	NOP
				//
				// The NOP is needed to give the jumps somewhere to land.
				// It is a liblink NOP, not a ARM64 NOP: it encodes to 0 instruction bytes.
				q = q1

				q = obj.Appendp(ctxt, q)
				q.As = AMOVD
				q.From.Type = obj.TYPE_MEM
				q.From.Reg = REGG
				q.From.Offset = 4 * int64(ctxt.Arch.PtrSize) // G.panic
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R1

				q = obj.Appendp(ctxt, q)
				q.As = ACMP
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REGZERO
				q.Reg = REG_R1

				q = obj.Appendp(ctxt, q)
				q.As = ABEQ
				q.To.Type = obj.TYPE_BRANCH
				q1 = q

				q = obj.Appendp(ctxt, q)
				q.As = AMOVD
				q.From.Type = obj.TYPE_MEM
				q.From.Reg = REG_R1
				q.From.Offset = 0 // Panic.argp
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R2

				q = obj.Appendp(ctxt, q)
				q.As = AADD
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = int64(ctxt.Autosize) + 8
				q.Reg = REGSP
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R3

				q = obj.Appendp(ctxt, q)
				q.As = ACMP
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R2
				q.Reg = REG_R3

				q = obj.Appendp(ctxt, q)
				q.As = ABNE
				q.To.Type = obj.TYPE_BRANCH
				q2 = q

				q = obj.Appendp(ctxt, q)
				q.As = AADD
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = 8
				q.Reg = REGSP
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R4

				q = obj.Appendp(ctxt, q)
				q.As = AMOVD
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R4
				q.To.Type = obj.TYPE_MEM
				q.To.Reg = REG_R1
				q.To.Offset = 0 // Panic.argp

				q = obj.Appendp(ctxt, q)

				q.As = obj.ANOP
				q1.Pcond = q
				q2.Pcond = q
			}

		case obj.ARET:
			nocache(p)
			if p.From.Type == obj.TYPE_CONST {
				ctxt.Diag("using BECOME (%v) is not supported!", p)
				break
			}

			retjmp = p.To.Sym
			p.To = obj.Addr{}
			if cursym.Text.Mark&LEAF != 0 {
				if ctxt.Autosize != 0 {
					p.As = AADD
					p.From.Type = obj.TYPE_CONST
					p.From.Offset = int64(ctxt.Autosize)
					p.To.Type = obj.TYPE_REG
					p.To.Reg = REGSP
					p.Spadj = -ctxt.Autosize
				}
			} else {
				/* want write-back pre-indexed SP+autosize -> SP, loading REGLINK*/
				aoffset = ctxt.Autosize

				if aoffset > 0xF0 {
					aoffset = 0xF0
				}
				p.As = AMOVD
				p.From.Type = obj.TYPE_MEM
				p.Scond = C_XPOST
				p.From.Offset = int64(aoffset)
				p.From.Reg = REGSP
				p.To.Type = obj.TYPE_REG
				p.To.Reg = REGLINK
				p.Spadj = -aoffset
				if ctxt.Autosize > aoffset {
					q = ctxt.NewProg()
					q.As = AADD
					q.From.Type = obj.TYPE_CONST
					q.From.Offset = int64(ctxt.Autosize) - int64(aoffset)
					q.To.Type = obj.TYPE_REG
					q.To.Reg = REGSP
					q.Link = p.Link
					q.Spadj = int32(-q.From.Offset)
					q.Lineno = p.Lineno
					p.Link = q
					p = q
				}
			}

			if p.As != obj.ARET {
				q = ctxt.NewProg()
				q.Lineno = p.Lineno
				q.Link = p.Link
				p.Link = q
				p = q
			}

			if retjmp != nil { // retjmp
				p.As = AB
				p.To.Type = obj.TYPE_BRANCH
				p.To.Sym = retjmp
				p.Spadj = +ctxt.Autosize
				break
			}

			p.As = obj.ARET
			p.To.Type = obj.TYPE_MEM
			p.To.Offset = 0
			p.To.Reg = REGLINK
			p.Spadj = +ctxt.Autosize

		case AADD, ASUB:
			if p.To.Type == obj.TYPE_REG && p.To.Reg == REGSP && p.From.Type == obj.TYPE_CONST {
				if p.As == AADD {
					p.Spadj = int32(-p.From.Offset)
				} else {
					p.Spadj = int32(+p.From.Offset)
				}
			}
			break
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
	ASVC:   true,
}

var Linkarm64 = obj.LinkArch{
	Arch:       sys.ArchARM64,
	Preprocess: preprocess,
	Assemble:   span7,
	Follow:     follow,
	Progedit:   progedit,
	UnaryDst:   unaryDst,
}
