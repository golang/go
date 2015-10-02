// cmd/9l/noop.c, cmd/9l/pass.c, cmd/9l/span.c from Vita Nuova.
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2008 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2008 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
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

package ppc64

import (
	"cmd/internal/obj"
	"encoding/binary"
	"fmt"
	"math"
)

func progedit(ctxt *obj.Link, p *obj.Prog) {
	p.From.Class = 0
	p.To.Class = 0

	// Rewrite BR/BL to symbol as TYPE_BRANCH.
	switch p.As {
	case ABR,
		ABL,
		obj.ARET,
		obj.ADUFFZERO,
		obj.ADUFFCOPY:
		if p.To.Sym != nil {
			p.To.Type = obj.TYPE_BRANCH
		}
	}

	// Rewrite float constants to values stored in memory.
	switch p.As {
	case AFMOVS:
		if p.From.Type == obj.TYPE_FCONST {
			f32 := float32(p.From.Val.(float64))
			i32 := math.Float32bits(f32)
			literal := fmt.Sprintf("$f32.%08x", i32)
			s := obj.Linklookup(ctxt, literal, 0)
			s.Size = 4
			p.From.Type = obj.TYPE_MEM
			p.From.Sym = s
			p.From.Name = obj.NAME_EXTERN
			p.From.Offset = 0
		}

	case AFMOVD:
		if p.From.Type == obj.TYPE_FCONST {
			i64 := math.Float64bits(p.From.Val.(float64))
			literal := fmt.Sprintf("$f64.%016x", i64)
			s := obj.Linklookup(ctxt, literal, 0)
			s.Size = 8
			p.From.Type = obj.TYPE_MEM
			p.From.Sym = s
			p.From.Name = obj.NAME_EXTERN
			p.From.Offset = 0
		}

		// Put >32-bit constants in memory and load them
	case AMOVD:
		if p.From.Type == obj.TYPE_CONST && p.From.Name == obj.NAME_NONE && p.From.Reg == 0 && int64(int32(p.From.Offset)) != p.From.Offset {
			literal := fmt.Sprintf("$i64.%016x", uint64(p.From.Offset))
			s := obj.Linklookup(ctxt, literal, 0)
			s.Size = 8
			p.From.Type = obj.TYPE_MEM
			p.From.Sym = s
			p.From.Name = obj.NAME_EXTERN
			p.From.Offset = 0
		}
	}

	// Rewrite SUB constants into ADD.
	switch p.As {
	case ASUBC:
		if p.From.Type == obj.TYPE_CONST {
			p.From.Offset = -p.From.Offset
			p.As = AADDC
		}

	case ASUBCCC:
		if p.From.Type == obj.TYPE_CONST {
			p.From.Offset = -p.From.Offset
			p.As = AADDCCC
		}

	case ASUB:
		if p.From.Type == obj.TYPE_CONST {
			p.From.Offset = -p.From.Offset
			p.As = AADD
		}
	}
}

func preprocess(ctxt *obj.Link, cursym *obj.LSym) {
	// TODO(minux): add morestack short-cuts with small fixed frame-size.
	ctxt.Cursym = cursym

	if cursym.Text == nil || cursym.Text.Link == nil {
		return
	}

	p := cursym.Text
	textstksiz := p.To.Offset

	cursym.Args = p.To.Val.(int32)
	cursym.Locals = int32(textstksiz)

	/*
	 * find leaf subroutines
	 * strip NOPs
	 * expand RET
	 * expand BECOME pseudo
	 */
	if ctxt.Debugvlog != 0 {
		fmt.Fprintf(ctxt.Bso, "%5.2f noops\n", obj.Cputime())
	}
	ctxt.Bso.Flush()

	var q *obj.Prog
	var q1 *obj.Prog
	for p := cursym.Text; p != nil; p = p.Link {
		switch p.As {
		/* too hard, just leave alone */
		case obj.ATEXT:
			q = p

			p.Mark |= LABEL | LEAF | SYNC
			if p.Link != nil {
				p.Link.Mark |= LABEL
			}

		case ANOR:
			q = p
			if p.To.Type == obj.TYPE_REG {
				if p.To.Reg == REGZERO {
					p.Mark |= LABEL | SYNC
				}
			}

		case ALWAR,
			ASTWCCC,
			AECIWX,
			AECOWX,
			AEIEIO,
			AICBI,
			AISYNC,
			ATLBIE,
			ATLBIEL,
			ASLBIA,
			ASLBIE,
			ASLBMFEE,
			ASLBMFEV,
			ASLBMTE,
			ADCBF,
			ADCBI,
			ADCBST,
			ADCBT,
			ADCBTST,
			ADCBZ,
			ASYNC,
			ATLBSYNC,
			APTESYNC,
			ATW,
			AWORD,
			ARFI,
			ARFCI,
			ARFID,
			AHRFID:
			q = p
			p.Mark |= LABEL | SYNC
			continue

		case AMOVW, AMOVWZ, AMOVD:
			q = p
			if p.From.Reg >= REG_SPECIAL || p.To.Reg >= REG_SPECIAL {
				p.Mark |= LABEL | SYNC
			}
			continue

		case AFABS,
			AFABSCC,
			AFADD,
			AFADDCC,
			AFCTIW,
			AFCTIWCC,
			AFCTIWZ,
			AFCTIWZCC,
			AFDIV,
			AFDIVCC,
			AFMADD,
			AFMADDCC,
			AFMOVD,
			AFMOVDU,
			/* case AFMOVDS: */
			AFMOVS,
			AFMOVSU,

			/* case AFMOVSD: */
			AFMSUB,
			AFMSUBCC,
			AFMUL,
			AFMULCC,
			AFNABS,
			AFNABSCC,
			AFNEG,
			AFNEGCC,
			AFNMADD,
			AFNMADDCC,
			AFNMSUB,
			AFNMSUBCC,
			AFRSP,
			AFRSPCC,
			AFSUB,
			AFSUBCC:
			q = p

			p.Mark |= FLOAT
			continue

		case ABL,
			ABCL,
			obj.ADUFFZERO,
			obj.ADUFFCOPY:
			cursym.Text.Mark &^= LEAF
			fallthrough

		case ABC,
			ABEQ,
			ABGE,
			ABGT,
			ABLE,
			ABLT,
			ABNE,
			ABR,
			ABVC,
			ABVS:
			p.Mark |= BRANCH
			q = p
			q1 = p.Pcond
			if q1 != nil {
				for q1.As == obj.ANOP {
					q1 = q1.Link
					p.Pcond = q1
				}

				if q1.Mark&LEAF == 0 {
					q1.Mark |= LABEL
				}
			} else {
				p.Mark |= LABEL
			}
			q1 = p.Link
			if q1 != nil {
				q1.Mark |= LABEL
			}
			continue

		case AFCMPO, AFCMPU:
			q = p
			p.Mark |= FCMP | FLOAT
			continue

		case obj.ARET:
			q = p
			if p.Link != nil {
				p.Link.Mark |= LABEL
			}
			continue

		case obj.ANOP:
			q1 = p.Link
			q.Link = q1 /* q is non-nop */
			q1.Mark |= p.Mark
			continue

		default:
			q = p
			continue
		}
	}

	autosize := int32(0)
	var aoffset int
	var mov int
	var o int
	var p1 *obj.Prog
	var p2 *obj.Prog
	for p := cursym.Text; p != nil; p = p.Link {
		o = int(p.As)
		switch o {
		case obj.ATEXT:
			mov = AMOVD
			aoffset = 0
			autosize = int32(textstksiz + 8)
			if (p.Mark&LEAF != 0) && autosize <= 8 {
				autosize = 0
			} else if autosize&4 != 0 {
				autosize += 4
			}
			p.To.Offset = int64(autosize) - 8

			if p.From3.Offset&obj.NOSPLIT == 0 {
				p = stacksplit(ctxt, p, autosize) // emit split check
			}

			q = p

			if autosize != 0 {
				/* use MOVDU to adjust R1 when saving R31, if autosize is small */
				if cursym.Text.Mark&LEAF == 0 && autosize >= -BIG && autosize <= BIG {
					mov = AMOVDU
					aoffset = int(-autosize)
				} else {
					q = obj.Appendp(ctxt, p)
					q.As = AADD
					q.Lineno = p.Lineno
					q.From.Type = obj.TYPE_CONST
					q.From.Offset = int64(-autosize)
					q.To.Type = obj.TYPE_REG
					q.To.Reg = REGSP
					q.Spadj = +autosize
				}
			} else if cursym.Text.Mark&LEAF == 0 {
				if ctxt.Debugvlog != 0 {
					fmt.Fprintf(ctxt.Bso, "save suppressed in: %s\n", cursym.Name)
					ctxt.Bso.Flush()
				}

				cursym.Text.Mark |= LEAF
			}

			if cursym.Text.Mark&LEAF != 0 {
				cursym.Leaf = 1
				break
			}

			q = obj.Appendp(ctxt, q)
			q.As = AMOVD
			q.Lineno = p.Lineno
			q.From.Type = obj.TYPE_REG
			q.From.Reg = REG_LR
			q.To.Type = obj.TYPE_REG
			q.To.Reg = REGTMP

			q = obj.Appendp(ctxt, q)
			q.As = int16(mov)
			q.Lineno = p.Lineno
			q.From.Type = obj.TYPE_REG
			q.From.Reg = REGTMP
			q.To.Type = obj.TYPE_MEM
			q.To.Offset = int64(aoffset)
			q.To.Reg = REGSP
			if q.As == AMOVDU {
				q.Spadj = int32(-aoffset)
			}

			if cursym.Text.From3.Offset&obj.WRAPPER != 0 {
				// if(g->panic != nil && g->panic->argp == FP) g->panic->argp = bottom-of-frame
				//
				//	MOVD g_panic(g), R3
				//	CMP R0, R3
				//	BEQ end
				//	MOVD panic_argp(R3), R4
				//	ADD $(autosize+8), R1, R5
				//	CMP R4, R5
				//	BNE end
				//	ADD $8, R1, R6
				//	MOVD R6, panic_argp(R3)
				// end:
				//	NOP
				//
				// The NOP is needed to give the jumps somewhere to land.
				// It is a liblink NOP, not a ppc64 NOP: it encodes to 0 instruction bytes.

				q = obj.Appendp(ctxt, q)

				q.As = AMOVD
				q.From.Type = obj.TYPE_MEM
				q.From.Reg = REGG
				q.From.Offset = 4 * int64(ctxt.Arch.Ptrsize) // G.panic
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R3

				q = obj.Appendp(ctxt, q)
				q.As = ACMP
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R0
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R3

				q = obj.Appendp(ctxt, q)
				q.As = ABEQ
				q.To.Type = obj.TYPE_BRANCH
				p1 = q

				q = obj.Appendp(ctxt, q)
				q.As = AMOVD
				q.From.Type = obj.TYPE_MEM
				q.From.Reg = REG_R3
				q.From.Offset = 0 // Panic.argp
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R4

				q = obj.Appendp(ctxt, q)
				q.As = AADD
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = int64(autosize) + 8
				q.Reg = REGSP
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R5

				q = obj.Appendp(ctxt, q)
				q.As = ACMP
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R4
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R5

				q = obj.Appendp(ctxt, q)
				q.As = ABNE
				q.To.Type = obj.TYPE_BRANCH
				p2 = q

				q = obj.Appendp(ctxt, q)
				q.As = AADD
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = 8
				q.Reg = REGSP
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R6

				q = obj.Appendp(ctxt, q)
				q.As = AMOVD
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R6
				q.To.Type = obj.TYPE_MEM
				q.To.Reg = REG_R3
				q.To.Offset = 0 // Panic.argp

				q = obj.Appendp(ctxt, q)

				q.As = obj.ANOP
				p1.Pcond = q
				p2.Pcond = q
			}

		case obj.ARET:
			if p.From.Type == obj.TYPE_CONST {
				ctxt.Diag("using BECOME (%v) is not supported!", p)
				break
			}

			if p.To.Sym != nil { // retjmp
				p.As = ABR
				p.To.Type = obj.TYPE_BRANCH
				break
			}

			if cursym.Text.Mark&LEAF != 0 {
				if autosize == 0 {
					p.As = ABR
					p.From = obj.Addr{}
					p.To.Type = obj.TYPE_REG
					p.To.Reg = REG_LR
					p.Mark |= BRANCH
					break
				}

				p.As = AADD
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = int64(autosize)
				p.To.Type = obj.TYPE_REG
				p.To.Reg = REGSP
				p.Spadj = -autosize

				q = ctxt.NewProg()
				q.As = ABR
				q.Lineno = p.Lineno
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_LR
				q.Mark |= BRANCH
				q.Spadj = +autosize

				q.Link = p.Link
				p.Link = q
				break
			}

			p.As = AMOVD
			p.From.Type = obj.TYPE_MEM
			p.From.Offset = 0
			p.From.Reg = REGSP
			p.To.Type = obj.TYPE_REG
			p.To.Reg = REGTMP

			q = ctxt.NewProg()
			q.As = AMOVD
			q.Lineno = p.Lineno
			q.From.Type = obj.TYPE_REG
			q.From.Reg = REGTMP
			q.To.Type = obj.TYPE_REG
			q.To.Reg = REG_LR

			q.Link = p.Link
			p.Link = q
			p = q

			if false {
				// Debug bad returns
				q = ctxt.NewProg()

				q.As = AMOVD
				q.Lineno = p.Lineno
				q.From.Type = obj.TYPE_MEM
				q.From.Offset = 0
				q.From.Reg = REGTMP
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REGTMP

				q.Link = p.Link
				p.Link = q
				p = q
			}

			if autosize != 0 {
				q = ctxt.NewProg()
				q.As = AADD
				q.Lineno = p.Lineno
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = int64(autosize)
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REGSP
				q.Spadj = -autosize

				q.Link = p.Link
				p.Link = q
			}

			q1 = ctxt.NewProg()
			q1.As = ABR
			q1.Lineno = p.Lineno
			q1.To.Type = obj.TYPE_REG
			q1.To.Reg = REG_LR
			q1.Mark |= BRANCH
			q1.Spadj = +autosize

			q1.Link = q.Link
			q.Link = q1

		case AADD:
			if p.To.Type == obj.TYPE_REG && p.To.Reg == REGSP && p.From.Type == obj.TYPE_CONST {
				p.Spadj = int32(-p.From.Offset)
			}
		}
	}
}

/*
// instruction scheduling
	if(debug['Q'] == 0)
		return;

	curtext = nil;
	q = nil;	// p - 1
	q1 = firstp;	// top of block
	o = 0;		// count of instructions
	for(p = firstp; p != nil; p = p1) {
		p1 = p->link;
		o++;
		if(p->mark & NOSCHED){
			if(q1 != p){
				sched(q1, q);
			}
			for(; p != nil; p = p->link){
				if(!(p->mark & NOSCHED))
					break;
				q = p;
			}
			p1 = p;
			q1 = p;
			o = 0;
			continue;
		}
		if(p->mark & (LABEL|SYNC)) {
			if(q1 != p)
				sched(q1, q);
			q1 = p;
			o = 1;
		}
		if(p->mark & (BRANCH|SYNC)) {
			sched(q1, p);
			q1 = p1;
			o = 0;
		}
		if(o >= NSCHED) {
			sched(q1, p);
			q1 = p1;
			o = 0;
		}
		q = p;
	}
*/
func stacksplit(ctxt *obj.Link, p *obj.Prog, framesize int32) *obj.Prog {
	// MOVD	g_stackguard(g), R3
	p = obj.Appendp(ctxt, p)

	p.As = AMOVD
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = REGG
	p.From.Offset = 2 * int64(ctxt.Arch.Ptrsize) // G.stackguard0
	if ctxt.Cursym.Cfunc != 0 {
		p.From.Offset = 3 * int64(ctxt.Arch.Ptrsize) // G.stackguard1
	}
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_R3

	var q *obj.Prog
	if framesize <= obj.StackSmall {
		// small stack: SP < stackguard
		//	CMP	stackguard, SP
		p = obj.Appendp(ctxt, p)

		p.As = ACMPU
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R3
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REGSP
	} else if framesize <= obj.StackBig {
		// large stack: SP-framesize < stackguard-StackSmall
		//	ADD $-framesize, SP, R4
		//	CMP stackguard, R4
		p = obj.Appendp(ctxt, p)

		p.As = AADD
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(-framesize)
		p.Reg = REGSP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R4

		p = obj.Appendp(ctxt, p)
		p.As = ACMPU
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R3
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R4
	} else {
		// Such a large stack we need to protect against wraparound.
		// If SP is close to zero:
		//	SP-stackguard+StackGuard <= framesize + (StackGuard-StackSmall)
		// The +StackGuard on both sides is required to keep the left side positive:
		// SP is allowed to be slightly below stackguard. See stack.h.
		//
		// Preemption sets stackguard to StackPreempt, a very large value.
		// That breaks the math above, so we have to check for that explicitly.
		//	// stackguard is R3
		//	CMP	R3, $StackPreempt
		//	BEQ	label-of-call-to-morestack
		//	ADD	$StackGuard, SP, R4
		//	SUB	R3, R4
		//	MOVD	$(framesize+(StackGuard-StackSmall)), R31
		//	CMPU	R31, R4
		p = obj.Appendp(ctxt, p)

		p.As = ACMP
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R3
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = obj.StackPreempt

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
		p.To.Reg = REG_R4

		p = obj.Appendp(ctxt, p)
		p.As = ASUB
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R3
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R4

		p = obj.Appendp(ctxt, p)
		p.As = AMOVD
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(framesize) + obj.StackGuard - obj.StackSmall
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REGTMP

		p = obj.Appendp(ctxt, p)
		p.As = ACMPU
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REGTMP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R4
	}

	// q1: BLT	done
	p = obj.Appendp(ctxt, p)
	q1 := p

	p.As = ABLT
	p.To.Type = obj.TYPE_BRANCH

	// MOVD	LR, R5
	p = obj.Appendp(ctxt, p)

	p.As = AMOVD
	p.From.Type = obj.TYPE_REG
	p.From.Reg = REG_LR
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_R5
	if q != nil {
		q.Pcond = p
	}

	// BL	runtime.morestack(SB)
	p = obj.Appendp(ctxt, p)

	p.As = ABL
	p.To.Type = obj.TYPE_BRANCH
	if ctxt.Cursym.Cfunc != 0 {
		p.To.Sym = obj.Linklookup(ctxt, "runtime.morestackc", 0)
	} else if ctxt.Cursym.Text.From3.Offset&obj.NEEDCTXT == 0 {
		p.To.Sym = obj.Linklookup(ctxt, "runtime.morestack_noctxt", 0)
	} else {
		p.To.Sym = obj.Linklookup(ctxt, "runtime.morestack", 0)
	}

	// BR	start
	p = obj.Appendp(ctxt, p)

	p.As = ABR
	p.To.Type = obj.TYPE_BRANCH
	p.Pcond = ctxt.Cursym.Text.Link

	// placeholder for q1's jump target
	p = obj.Appendp(ctxt, p)

	p.As = obj.ANOP // zero-width place holder
	q1.Pcond = p

	return p
}

func follow(ctxt *obj.Link, s *obj.LSym) {
	ctxt.Cursym = s

	firstp := ctxt.NewProg()
	lastp := firstp
	xfol(ctxt, s.Text, &lastp)
	lastp.Link = nil
	s.Text = firstp.Link
}

func relinv(a int) int {
	switch a {
	case ABEQ:
		return ABNE
	case ABNE:
		return ABEQ

	case ABGE:
		return ABLT
	case ABLT:
		return ABGE

	case ABGT:
		return ABLE
	case ABLE:
		return ABGT

	case ABVC:
		return ABVS
	case ABVS:
		return ABVC
	}

	return 0
}

func xfol(ctxt *obj.Link, p *obj.Prog, last **obj.Prog) {
	var q *obj.Prog
	var r *obj.Prog
	var a int
	var b int
	var i int

loop:
	if p == nil {
		return
	}
	a = int(p.As)
	if a == ABR {
		q = p.Pcond
		if (p.Mark&NOSCHED != 0) || q != nil && (q.Mark&NOSCHED != 0) {
			p.Mark |= FOLL
			(*last).Link = p
			*last = p
			p = p.Link
			xfol(ctxt, p, last)
			p = q
			if p != nil && p.Mark&FOLL == 0 {
				goto loop
			}
			return
		}

		if q != nil {
			p.Mark |= FOLL
			p = q
			if p.Mark&FOLL == 0 {
				goto loop
			}
		}
	}

	if p.Mark&FOLL != 0 {
		i = 0
		q = p
		for ; i < 4; i, q = i+1, q.Link {
			if q == *last || (q.Mark&NOSCHED != 0) {
				break
			}
			b = 0 /* set */
			a = int(q.As)
			if a == obj.ANOP {
				i--
				continue
			}

			if a == ABR || a == obj.ARET || a == ARFI || a == ARFCI || a == ARFID || a == AHRFID {
				goto copy
			}
			if q.Pcond == nil || (q.Pcond.Mark&FOLL != 0) {
				continue
			}
			b = relinv(a)
			if b == 0 {
				continue
			}

		copy:
			for {
				r = ctxt.NewProg()
				*r = *p
				if r.Mark&FOLL == 0 {
					fmt.Printf("cant happen 1\n")
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
				if a == ABR || a == obj.ARET || a == ARFI || a == ARFCI || a == ARFID || a == AHRFID {
					return
				}
				r.As = int16(b)
				r.Pcond = p.Link
				r.Link = p.Pcond
				if r.Link.Mark&FOLL == 0 {
					xfol(ctxt, r.Link, last)
				}
				if r.Pcond.Mark&FOLL == 0 {
					fmt.Printf("cant happen 2\n")
				}
				return
			}
		}

		a = ABR
		q = ctxt.NewProg()
		q.As = int16(a)
		q.Lineno = p.Lineno
		q.To.Type = obj.TYPE_BRANCH
		q.To.Offset = p.Pc
		q.Pcond = p
		p = q
	}

	p.Mark |= FOLL
	(*last).Link = p
	*last = p
	if a == ABR || a == obj.ARET || a == ARFI || a == ARFCI || a == ARFID || a == AHRFID {
		if p.Mark&NOSCHED != 0 {
			p = p.Link
			goto loop
		}

		return
	}

	if p.Pcond != nil {
		if a != ABL && p.Link != nil {
			xfol(ctxt, p.Link, last)
			p = p.Pcond
			if p == nil || (p.Mark&FOLL != 0) {
				return
			}
			goto loop
		}
	}

	p = p.Link
	goto loop
}

var Linkppc64 = obj.LinkArch{
	ByteOrder:  binary.BigEndian,
	Name:       "ppc64",
	Thechar:    '9',
	Preprocess: preprocess,
	Assemble:   span9,
	Follow:     follow,
	Progedit:   progedit,
	Minlc:      4,
	Ptrsize:    8,
	Regsize:    8,
}

var Linkppc64le = obj.LinkArch{
	ByteOrder:  binary.LittleEndian,
	Name:       "ppc64le",
	Thechar:    '9',
	Preprocess: preprocess,
	Assemble:   span9,
	Follow:     follow,
	Progedit:   progedit,
	Minlc:      4,
	Ptrsize:    8,
	Regsize:    8,
}
