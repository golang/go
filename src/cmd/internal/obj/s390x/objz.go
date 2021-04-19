// Based on cmd/internal/obj/ppc64/obj9.go.
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2008 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2008 Lucent Technologies Inc. and others
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

package s390x

import (
	"cmd/internal/obj"
	"cmd/internal/sys"
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

	// Rewrite float constants to values stored in memory unless they are +0.
	switch p.As {
	case AFMOVS:
		if p.From.Type == obj.TYPE_FCONST {
			f32 := float32(p.From.Val.(float64))
			i32 := math.Float32bits(f32)
			if i32 == 0 { // +0
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
			if i64 == 0 { // +0
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

		// put constants not loadable by LOAD IMMEDIATE into memory
	case AMOVD:
		if p.From.Type == obj.TYPE_CONST {
			val := p.From.Offset
			if int64(int32(val)) != val &&
				int64(uint32(val)) != val &&
				int64(uint64(val)&(0xffffffff<<32)) != val {
				literal := fmt.Sprintf("$i64.%016x", uint64(p.From.Offset))
				s := obj.Linklookup(ctxt, literal, 0)
				s.Size = 8
				p.From.Type = obj.TYPE_MEM
				p.From.Sym = s
				p.From.Sym.Set(obj.AttrLocal, true)
				p.From.Name = obj.NAME_EXTERN
				p.From.Offset = 0
			}
		}
	}

	// Rewrite SUB constants into ADD.
	switch p.As {
	case ASUBC:
		if p.From.Type == obj.TYPE_CONST && isint32(-p.From.Offset) {
			p.From.Offset = -p.From.Offset
			p.As = AADDC
		}

	case ASUB:
		if p.From.Type == obj.TYPE_CONST && isint32(-p.From.Offset) {
			p.From.Offset = -p.From.Offset
			p.As = AADD
		}
	}

	if ctxt.Flag_dynlink {
		rewriteToUseGot(ctxt, p)
	}
}

// Rewrite p, if necessary, to access global data via the global offset table.
func rewriteToUseGot(ctxt *obj.Link, p *obj.Prog) {
	// At the moment EXRL instructions are not emitted by the compiler and only reference local symbols in
	// assembly code.
	if p.As == AEXRL {
		return
	}

	// We only care about global data: NAME_EXTERN means a global
	// symbol in the Go sense, and p.Sym.Local is true for a few
	// internally defined symbols.
	if p.From.Type == obj.TYPE_ADDR && p.From.Name == obj.NAME_EXTERN && !p.From.Sym.Local() {
		// MOVD $sym, Rx becomes MOVD sym@GOT, Rx
		// MOVD $sym+<off>, Rx becomes MOVD sym@GOT, Rx; ADD <off>, Rx
		if p.To.Type != obj.TYPE_REG || p.As != AMOVD {
			ctxt.Diag("do not know how to handle LEA-type insn to non-register in %v with -dynlink", p)
		}
		p.From.Type = obj.TYPE_MEM
		p.From.Name = obj.NAME_GOTREF
		q := p
		if p.From.Offset != 0 {
			q = obj.Appendp(ctxt, p)
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
	// MOVD sym, Ry becomes MOVD sym@GOT, REGTMP; MOVD (REGTMP), Ry
	// MOVD Ry, sym becomes MOVD sym@GOT, REGTMP; MOVD Ry, (REGTMP)
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

func preprocess(ctxt *obj.Link, cursym *obj.LSym) {
	// TODO(minux): add morestack short-cuts with small fixed frame-size.
	ctxt.Cursym = cursym

	if cursym.Text == nil || cursym.Text.Link == nil {
		return
	}

	p := cursym.Text
	textstksiz := p.To.Offset
	if textstksiz == -8 {
		// Compatibility hack.
		p.From3.Offset |= obj.NOFRAME
		textstksiz = 0
	}
	if textstksiz%8 != 0 {
		ctxt.Diag("frame size %d not a multiple of 8", textstksiz)
	}
	if p.From3.Offset&obj.NOFRAME != 0 {
		if textstksiz != 0 {
			ctxt.Diag("NOFRAME functions must have a frame size of 0, not %d", textstksiz)
		}
	}

	cursym.Args = p.To.Val.(int32)
	cursym.Locals = int32(textstksiz)

	/*
	 * find leaf subroutines
	 * strip NOPs
	 * expand RET
	 * expand BECOME pseudo
	 */
	if ctxt.Debugvlog != 0 {
		ctxt.Logf("%5.2f noops\n", obj.Cputime())
	}

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

		case ASYNC,
			AWORD:
			q = p
			p.Mark |= LABEL | SYNC
			continue

		case AMOVW, AMOVWZ, AMOVD:
			q = p
			if p.From.Reg >= REG_RESERVED || p.To.Reg >= REG_RESERVED {
				p.Mark |= LABEL | SYNC
			}
			continue

		case AFABS,
			AFADD,
			AFDIV,
			AFMADD,
			AFMOVD,
			AFMOVS,
			AFMSUB,
			AFMUL,
			AFNABS,
			AFNEG,
			AFNMADD,
			AFNMSUB,
			ALEDBR,
			ALDEBR,
			AFSUB:
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
			ABLEU,
			ABLTU,
			ABNE,
			ABR,
			ABVC,
			ABVS,
			ACMPBEQ,
			ACMPBGE,
			ACMPBGT,
			ACMPBLE,
			ACMPBLT,
			ACMPBNE,
			ACMPUBEQ,
			ACMPUBGE,
			ACMPUBGT,
			ACMPUBLE,
			ACMPUBLT,
			ACMPUBNE:
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
	var p1 *obj.Prog
	var p2 *obj.Prog
	var pLast *obj.Prog
	var pPre *obj.Prog
	var pPreempt *obj.Prog
	wasSplit := false
	for p := cursym.Text; p != nil; p = p.Link {
		pLast = p
		switch p.As {
		case obj.ATEXT:
			autosize = int32(textstksiz)

			if p.Mark&LEAF != 0 && autosize == 0 {
				// A leaf function with no locals has no frame.
				p.From3.Offset |= obj.NOFRAME
			}

			if p.From3.Offset&obj.NOFRAME == 0 {
				// If there is a stack frame at all, it includes
				// space to save the LR.
				autosize += int32(ctxt.FixedFrameSize())
			}

			if p.Mark&LEAF != 0 && autosize < obj.StackSmall {
				// A leaf function with a small stack can be marked
				// NOSPLIT, avoiding a stack check.
				p.From3.Offset |= obj.NOSPLIT
			}

			p.To.Offset = int64(autosize)

			q = p

			if p.From3.Offset&obj.NOSPLIT == 0 {
				p, pPreempt = stacksplitPre(ctxt, p, autosize) // emit pre part of split check
				pPre = p
				wasSplit = true //need post part of split
			}

			if autosize != 0 {
				// Make sure to save link register for non-empty frame, even if
				// it is a leaf function, so that traceback works.
				// Store link register before decrementing SP, so if a signal comes
				// during the execution of the function prologue, the traceback
				// code will not see a half-updated stack frame.
				q = obj.Appendp(ctxt, p)
				q.As = AMOVD
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_LR
				q.To.Type = obj.TYPE_MEM
				q.To.Reg = REGSP
				q.To.Offset = int64(-autosize)

				q = obj.Appendp(ctxt, q)
				q.As = AMOVD
				q.From.Type = obj.TYPE_ADDR
				q.From.Offset = int64(-autosize)
				q.From.Reg = REGSP // not actually needed - REGSP is assumed if no reg is provided
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REGSP
				q.Spadj = autosize
			} else if cursym.Text.Mark&LEAF == 0 {
				// A very few functions that do not return to their caller
				// (e.g. gogo) are not identified as leaves but still have
				// no frame.
				cursym.Text.Mark |= LEAF
			}

			if cursym.Text.Mark&LEAF != 0 {
				cursym.Set(obj.AttrLeaf, true)
				break
			}

			if cursym.Text.From3.Offset&obj.WRAPPER != 0 {
				// if(g->panic != nil && g->panic->argp == FP) g->panic->argp = bottom-of-frame
				//
				//	MOVD g_panic(g), R3
				//	CMP R3, $0
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
				// It is a liblink NOP, not a s390x NOP: it encodes to 0 instruction bytes.

				q = obj.Appendp(ctxt, q)

				q.As = AMOVD
				q.From.Type = obj.TYPE_MEM
				q.From.Reg = REGG
				q.From.Offset = 4 * int64(ctxt.Arch.PtrSize) // G.panic
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R3

				q = obj.Appendp(ctxt, q)
				q.As = ACMP
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R3
				q.To.Type = obj.TYPE_CONST
				q.To.Offset = 0

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
				q.From.Offset = int64(autosize) + ctxt.FixedFrameSize()
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
				q.From.Offset = ctxt.FixedFrameSize()
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

			retTarget := p.To.Sym

			if cursym.Text.Mark&LEAF != 0 {
				if autosize == 0 {
					p.As = ABR
					p.From = obj.Addr{}
					if retTarget == nil {
						p.To.Type = obj.TYPE_REG
						p.To.Reg = REG_LR
					} else {
						p.To.Type = obj.TYPE_BRANCH
						p.To.Sym = retTarget
					}
					p.Mark |= BRANCH
					break
				}

				p.As = AADD
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = int64(autosize)
				p.To.Type = obj.TYPE_REG
				p.To.Reg = REGSP
				p.Spadj = -autosize

				q = obj.Appendp(ctxt, p)
				q.As = ABR
				q.From = obj.Addr{}
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_LR
				q.Mark |= BRANCH
				q.Spadj = autosize
				break
			}

			p.As = AMOVD
			p.From.Type = obj.TYPE_MEM
			p.From.Reg = REGSP
			p.From.Offset = 0
			p.To.Type = obj.TYPE_REG
			p.To.Reg = REG_LR

			q = p

			if autosize != 0 {
				q = obj.Appendp(ctxt, q)
				q.As = AADD
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = int64(autosize)
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REGSP
				q.Spadj = -autosize
			}

			q = obj.Appendp(ctxt, q)
			q.As = ABR
			q.From = obj.Addr{}
			if retTarget == nil {
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_LR
			} else {
				q.To.Type = obj.TYPE_BRANCH
				q.To.Sym = retTarget
			}
			q.Mark |= BRANCH
			q.Spadj = autosize

		case AADD:
			if p.To.Type == obj.TYPE_REG && p.To.Reg == REGSP && p.From.Type == obj.TYPE_CONST {
				p.Spadj = int32(-p.From.Offset)
			}
		}
	}
	if wasSplit {
		pLast = stacksplitPost(ctxt, pLast, pPre, pPreempt, autosize) // emit post part of split check
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
func stacksplitPre(ctxt *obj.Link, p *obj.Prog, framesize int32) (*obj.Prog, *obj.Prog) {
	var q *obj.Prog

	// MOVD	g_stackguard(g), R3
	p = obj.Appendp(ctxt, p)

	p.As = AMOVD
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = REGG
	p.From.Offset = 2 * int64(ctxt.Arch.PtrSize) // G.stackguard0
	if ctxt.Cursym.CFunc() {
		p.From.Offset = 3 * int64(ctxt.Arch.PtrSize) // G.stackguard1
	}
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_R3

	q = nil
	if framesize <= obj.StackSmall {
		// small stack: SP < stackguard
		//	CMP	stackguard, SP

		//p.To.Type = obj.TYPE_REG
		//p.To.Reg = REGSP

		// q1: BLT	done

		p = obj.Appendp(ctxt, p)
		//q1 = p
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R3
		p.Reg = REGSP
		p.As = ACMPUBGE
		p.To.Type = obj.TYPE_BRANCH
		//p = obj.Appendp(ctxt, p)

		//p.As = ACMPU
		//p.From.Type = obj.TYPE_REG
		//p.From.Reg = REG_R3
		//p.To.Type = obj.TYPE_REG
		//p.To.Reg = REGSP

		//p = obj.Appendp(ctxt, p)
		//p.As = ABGE
		//p.To.Type = obj.TYPE_BRANCH

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
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R3
		p.Reg = REG_R4
		p.As = ACMPUBGE
		p.To.Type = obj.TYPE_BRANCH

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
		//	MOVD	$(framesize+(StackGuard-StackSmall)), TEMP
		//	CMPUBGE	TEMP, R4
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
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REGTMP
		p.Reg = REG_R4
		p.As = ACMPUBGE
		p.To.Type = obj.TYPE_BRANCH
	}

	return p, q
}

func stacksplitPost(ctxt *obj.Link, p *obj.Prog, pPre *obj.Prog, pPreempt *obj.Prog, framesize int32) *obj.Prog {
	// Now we are at the end of the function, but logically
	// we are still in function prologue. We need to fix the
	// SP data and PCDATA.
	spfix := obj.Appendp(ctxt, p)
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

	// MOVD	LR, R5
	p = obj.Appendp(ctxt, pcdata)
	pPre.Pcond = p
	p.As = AMOVD
	p.From.Type = obj.TYPE_REG
	p.From.Reg = REG_LR
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_R5
	if pPreempt != nil {
		pPreempt.Pcond = p
	}

	// BL	runtime.morestack(SB)
	p = obj.Appendp(ctxt, p)

	p.As = ABL
	p.To.Type = obj.TYPE_BRANCH
	if ctxt.Cursym.CFunc() {
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
	return p
}

var pc_cnt int64

func follow(ctxt *obj.Link, s *obj.LSym) {
	ctxt.Cursym = s

	pc_cnt = 0
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
	var b obj.As

	for p != nil {
		a := p.As
		if a == ABR {
			q = p.Pcond
			if (p.Mark&NOSCHED != 0) || q != nil && (q.Mark&NOSCHED != 0) {
				p.Mark |= FOLL
				(*last).Link = p
				*last = p
				(*last).Pc = pc_cnt
				pc_cnt += 1
				p = p.Link
				xfol(ctxt, p, last)
				p = q
				if p != nil && p.Mark&FOLL == 0 {
					continue
				}
				return
			}

			if q != nil {
				p.Mark |= FOLL
				p = q
				if p.Mark&FOLL == 0 {
					continue
				}
			}
		}

		if p.Mark&FOLL != 0 {
			q = p
			for i := 0; i < 4; i, q = i+1, q.Link {
				if q == *last || (q.Mark&NOSCHED != 0) {
					break
				}
				b = 0 /* set */
				a = q.As
				if a == obj.ANOP {
					i--
					continue
				}
				if a != ABR && a != obj.ARET {
					if q.Pcond == nil || (q.Pcond.Mark&FOLL != 0) {
						continue
					}
					b = relinv(a)
					if b == 0 {
						continue
					}
				}

				for {
					r = ctxt.NewProg()
					*r = *p
					if r.Mark&FOLL == 0 {
						fmt.Printf("can't happen 1\n")
					}
					r.Mark |= FOLL
					if p != q {
						p = p.Link
						(*last).Link = r
						*last = r
						(*last).Pc = pc_cnt
						pc_cnt += 1
						continue
					}

					(*last).Link = r
					*last = r
					(*last).Pc = pc_cnt
					pc_cnt += 1
					if a == ABR || a == obj.ARET {
						return
					}
					r.As = b
					r.Pcond = p.Link
					r.Link = p.Pcond
					if r.Link.Mark&FOLL == 0 {
						xfol(ctxt, r.Link, last)
					}
					if r.Pcond.Mark&FOLL == 0 {
						fmt.Printf("can't happen 2\n")
					}
					return
				}
			}

			a = ABR
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
		(*last).Pc = pc_cnt
		pc_cnt += 1

		if a == ABR || a == obj.ARET {
			if p.Mark&NOSCHED != 0 {
				p = p.Link
				continue
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
				continue
			}
		}

		p = p.Link
	}
}

var unaryDst = map[obj.As]bool{
	ASTCK:  true,
	ASTCKC: true,
	ASTCKE: true,
	ASTCKF: true,
	ANEG:   true,
	ANEGW:  true,
	AVONE:  true,
	AVZERO: true,
}

var Links390x = obj.LinkArch{
	Arch:       sys.ArchS390X,
	Preprocess: preprocess,
	Assemble:   spanz,
	Follow:     follow,
	Progedit:   progedit,
	UnaryDst:   unaryDst,
}
