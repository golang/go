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

package mips

import (
	"cmd/internal/obj"
	"encoding/binary"
	"fmt"
	"math"
)

func progedit(ctxt *obj.Link, p *obj.Prog) {
	p.From.Class = 0
	p.To.Class = 0

	// Rewrite JMP/JAL to symbol as TYPE_BRANCH.
	switch p.As {
	case AJMP,
		AJAL,
		ARET,
		obj.ADUFFZERO,
		obj.ADUFFCOPY:
		if p.To.Sym != nil {
			p.To.Type = obj.TYPE_BRANCH
		}
	}

	// Rewrite float constants to values stored in memory.
	switch p.As {
	case AMOVF:
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

	case AMOVD:
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
	case AMOVV:
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
	case ASUB:
		if p.From.Type == obj.TYPE_CONST {
			p.From.Offset = -p.From.Offset
			p.As = AADD
		}

	case ASUBU:
		if p.From.Type == obj.TYPE_CONST {
			p.From.Offset = -p.From.Offset
			p.As = AADDU
		}

	case ASUBV:
		if p.From.Type == obj.TYPE_CONST {
			p.From.Offset = -p.From.Offset
			p.As = AADDV
		}

	case ASUBVU:
		if p.From.Type == obj.TYPE_CONST {
			p.From.Offset = -p.From.Offset
			p.As = AADDVU
		}
	}
}

func preprocess(ctxt *obj.Link, cursym *obj.LSym) {
	// TODO(minux): add morestack short-cuts with small fixed frame-size.
	ctxt.Cursym = cursym

	// a switch for enabling/disabling instruction scheduling
	nosched := true

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

		/* too hard, just leave alone */
		case AMOVW,
			AMOVV:
			q = p
			if p.To.Type == obj.TYPE_REG && p.To.Reg >= REG_SPECIAL {
				p.Mark |= LABEL | SYNC
				break
			}
			if p.From.Type == obj.TYPE_REG && p.From.Reg >= REG_SPECIAL {
				p.Mark |= LABEL | SYNC
			}

		/* too hard, just leave alone */
		case ASYSCALL,
			AWORD,
			ATLBWR,
			ATLBWI,
			ATLBP,
			ATLBR:
			q = p
			p.Mark |= LABEL | SYNC

		case ANOR:
			q = p
			if p.To.Type == obj.TYPE_REG {
				if p.To.Reg == REGZERO {
					p.Mark |= LABEL | SYNC
				}
			}

		case ABGEZAL,
			ABLTZAL,
			AJAL,
			obj.ADUFFZERO,
			obj.ADUFFCOPY:
			cursym.Text.Mark &^= LEAF
			fallthrough

		case AJMP,
			ABEQ,
			ABGEZ,
			ABGTZ,
			ABLEZ,
			ABLTZ,
			ABNE,
			ABFPT, ABFPF:
			if p.As == ABFPT || p.As == ABFPF {
				// We don't treat ABFPT and ABFPF as branches here,
				// so that we will always fill nop (0x0) in their
				// delay slot during assembly.
				// This is to workaround a kernel FPU emulator bug
				// where it uses the user stack to simulate the
				// instruction in the delay slot if it's not 0x0,
				// and somehow that leads to SIGSEGV when the kernel
				// jump to the stack.
				p.Mark |= SYNC
			} else {
				p.Mark |= BRANCH
			}
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
			}
			//else {
			//	p.Mark |= LABEL
			//}
			q1 = p.Link
			if q1 != nil {
				q1.Mark |= LABEL
			}
			continue

		case ARET:
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
	var o int
	var p1 *obj.Prog
	var p2 *obj.Prog
	for p := cursym.Text; p != nil; p = p.Link {
		o = int(p.As)
		switch o {
		case obj.ATEXT:
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
				q = obj.Appendp(ctxt, p)
				q.As = AADDV
				q.Lineno = p.Lineno
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = int64(-autosize)
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REGSP
				q.Spadj = +autosize
			} else if cursym.Text.Mark&LEAF == 0 {
				if cursym.Text.From3.Offset&obj.NOSPLIT != 0 {
					if ctxt.Debugvlog != 0 {
						fmt.Fprintf(ctxt.Bso, "save suppressed in: %s\n", cursym.Name)
						ctxt.Bso.Flush()
					}

					cursym.Text.Mark |= LEAF
				}
			}

			if cursym.Text.Mark&LEAF != 0 {
				cursym.Leaf = 1
				break
			}

			q = obj.Appendp(ctxt, q)
			q.As = AMOVV
			q.Lineno = p.Lineno
			q.From.Type = obj.TYPE_REG
			q.From.Reg = REGLINK
			q.To.Type = obj.TYPE_MEM
			q.To.Offset = int64(0)
			q.To.Reg = REGSP

			if cursym.Text.From3.Offset&obj.WRAPPER != 0 {
				// if(g->panic != nil && g->panic->argp == FP) g->panic->argp = bottom-of-frame
				//
				//	MOVV	g_panic(g), R1
				//	BEQ		R1, end
				//	MOVV	panic_argp(R1), R2
				//	ADDV	$(autosize+8), R29, R3
				//	BNE		R2, R3, end
				//	ADDV	$8, R29, R2
				//	MOVV	R2, panic_argp(R1)
				// end:
				//	NOP
				//
				// The NOP is needed to give the jumps somewhere to land.
				// It is a liblink NOP, not an mips NOP: it encodes to 0 instruction bytes.

				q = obj.Appendp(ctxt, q)

				q.As = AMOVV
				q.From.Type = obj.TYPE_MEM
				q.From.Reg = REGG
				q.From.Offset = 4 * int64(ctxt.Arch.Ptrsize) // G.panic
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R1

				q = obj.Appendp(ctxt, q)
				q.As = ABEQ
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R1
				q.To.Type = obj.TYPE_BRANCH
				q.Mark |= BRANCH
				p1 = q

				q = obj.Appendp(ctxt, q)
				q.As = AMOVV
				q.From.Type = obj.TYPE_MEM
				q.From.Reg = REG_R1
				q.From.Offset = 0 // Panic.argp
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R2

				q = obj.Appendp(ctxt, q)
				q.As = AADDV
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = int64(autosize) + 8
				q.Reg = REGSP
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R3

				q = obj.Appendp(ctxt, q)
				q.As = ABNE
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R2
				q.Reg = REG_R3
				q.To.Type = obj.TYPE_BRANCH
				q.Mark |= BRANCH
				p2 = q

				q = obj.Appendp(ctxt, q)
				q.As = AADDV
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = 8
				q.Reg = REGSP
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R2

				q = obj.Appendp(ctxt, q)
				q.As = AMOVV
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R2
				q.To.Type = obj.TYPE_MEM
				q.To.Reg = REG_R1
				q.To.Offset = 0 // Panic.argp

				q = obj.Appendp(ctxt, q)

				q.As = obj.ANOP
				p1.Pcond = q
				p2.Pcond = q
			}

		case ARET:
			if p.From.Type == obj.TYPE_CONST {
				ctxt.Diag("using BECOME (%v) is not supported!", p)
				break
			}

			if p.To.Sym != nil { // retjmp
				p.As = AJMP
				p.To.Type = obj.TYPE_BRANCH
				break
			}

			if cursym.Text.Mark&LEAF != 0 {
				if autosize == 0 {
					p.As = AJMP
					p.From = obj.Addr{}
					p.To.Type = obj.TYPE_MEM
					p.To.Offset = 0
					p.To.Reg = REGLINK
					p.Mark |= BRANCH
					break
				}

				p.As = AADDV
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = int64(autosize)
				p.To.Type = obj.TYPE_REG
				p.To.Reg = REGSP
				p.Spadj = -autosize

				q = ctxt.NewProg()
				q.As = AJMP
				q.Lineno = p.Lineno
				q.To.Type = obj.TYPE_MEM
				q.To.Offset = 0
				q.To.Reg = REGLINK
				q.Mark |= BRANCH
				q.Spadj = +autosize

				q.Link = p.Link
				p.Link = q
				break
			}

			p.As = AMOVV
			p.From.Type = obj.TYPE_MEM
			p.From.Offset = 0
			p.From.Reg = REGSP
			p.To.Type = obj.TYPE_REG
			p.To.Reg = REG_R4

			if false {
				// Debug bad returns
				q = ctxt.NewProg()

				q.As = AMOVV
				q.Lineno = p.Lineno
				q.From.Type = obj.TYPE_MEM
				q.From.Offset = 0
				q.From.Reg = REG_R4
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REGTMP

				q.Link = p.Link
				p.Link = q
				p = q
			}

			if autosize != 0 {
				q = ctxt.NewProg()
				q.As = AADDV
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
			q1.As = AJMP
			q1.Lineno = p.Lineno
			q1.To.Type = obj.TYPE_MEM
			q1.To.Offset = 0
			q1.To.Reg = REG_R4
			q1.Mark |= BRANCH
			q1.Spadj = +autosize

			q1.Link = q.Link
			q.Link = q1

		case AADDV,
			AADDVU:
			if p.To.Type == obj.TYPE_REG && p.To.Reg == REGSP && p.From.Type == obj.TYPE_CONST {
				p.Spadj = int32(-p.From.Offset)
			}
		}
	}

	if nosched {
		// if we don't do instruction scheduling, simply add
		// NOP after each branch instruction.
		for p = cursym.Text; p != nil; p = p.Link {
			if p.Mark&BRANCH != 0 {
				addnop(ctxt, p)
			}
		}
		return
	}

	// instruction scheduling
	q = nil          // p - 1
	q1 = cursym.Text // top of block
	o = 0            // count of instructions
	for p = cursym.Text; p != nil; p = p1 {
		p1 = p.Link
		o++
		if p.Mark&NOSCHED != 0 {
			if q1 != p {
				sched(ctxt, q1, q)
			}
			for ; p != nil; p = p.Link {
				if p.Mark&NOSCHED == 0 {
					break
				}
				q = p
			}
			p1 = p
			q1 = p
			o = 0
			continue
		}
		if p.Mark&(LABEL|SYNC) != 0 {
			if q1 != p {
				sched(ctxt, q1, q)
			}
			q1 = p
			o = 1
		}
		if p.Mark&(BRANCH|SYNC) != 0 {
			sched(ctxt, q1, p)
			q1 = p1
			o = 0
		}
		if o >= NSCHED {
			sched(ctxt, q1, p)
			q1 = p1
			o = 0
		}
		q = p
	}
}

func stacksplit(ctxt *obj.Link, p *obj.Prog, framesize int32) *obj.Prog {
	// MOVV	g_stackguard(g), R1
	p = obj.Appendp(ctxt, p)

	p.As = AMOVV
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = REGG
	p.From.Offset = 2 * int64(ctxt.Arch.Ptrsize) // G.stackguard0
	if ctxt.Cursym.Cfunc != 0 {
		p.From.Offset = 3 * int64(ctxt.Arch.Ptrsize) // G.stackguard1
	}
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_R1

	var q *obj.Prog
	if framesize <= obj.StackSmall {
		// small stack: SP < stackguard
		//	AGTU	SP, stackguard, R1
		p = obj.Appendp(ctxt, p)

		p.As = ASGTU
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REGSP
		p.Reg = REG_R1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R1
	} else if framesize <= obj.StackBig {
		// large stack: SP-framesize < stackguard-StackSmall
		//	ADDV	$-framesize, SP, R2
		//	SGTU	R2, stackguard, R1
		p = obj.Appendp(ctxt, p)

		p.As = AADDV
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(-framesize)
		p.Reg = REGSP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R2

		p = obj.Appendp(ctxt, p)
		p.As = ASGTU
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R2
		p.Reg = REG_R1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R1
	} else {
		// Such a large stack we need to protect against wraparound.
		// If SP is close to zero:
		//	SP-stackguard+StackGuard <= framesize + (StackGuard-StackSmall)
		// The +StackGuard on both sides is required to keep the left side positive:
		// SP is allowed to be slightly below stackguard. See stack.h.
		//
		// Preemption sets stackguard to StackPreempt, a very large value.
		// That breaks the math above, so we have to check for that explicitly.
		//	// stackguard is R1
		//	MOVV	$StackPreempt, R2
		//	BEQ	R1, R2, label-of-call-to-morestack
		//	ADDV	$StackGuard, SP, R2
		//	SUBVU	R1, R2
		//	MOVV	$(framesize+(StackGuard-StackSmall)), R1
		//	SGTU	R2, R1, R1
		p = obj.Appendp(ctxt, p)

		p.As = AMOVV
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = obj.StackPreempt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R2

		p = obj.Appendp(ctxt, p)
		q = p
		p.As = ABEQ
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R1
		p.Reg = REG_R2
		p.To.Type = obj.TYPE_BRANCH
		p.Mark |= BRANCH

		p = obj.Appendp(ctxt, p)
		p.As = AADDV
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = obj.StackGuard
		p.Reg = REGSP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R2

		p = obj.Appendp(ctxt, p)
		p.As = ASUBVU
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R2

		p = obj.Appendp(ctxt, p)
		p.As = AMOVV
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(framesize) + obj.StackGuard - obj.StackSmall
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R1

		p = obj.Appendp(ctxt, p)
		p.As = ASGTU
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R2
		p.Reg = REG_R1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R1
	}

	// q1: BNE	R1, done
	p = obj.Appendp(ctxt, p)
	q1 := p

	p.As = ABNE
	p.From.Type = obj.TYPE_REG
	p.From.Reg = REG_R1
	p.To.Type = obj.TYPE_BRANCH
	p.Mark |= BRANCH

	// MOVV	LINK, R3
	p = obj.Appendp(ctxt, p)

	p.As = AMOVV
	p.From.Type = obj.TYPE_REG
	p.From.Reg = REGLINK
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_R3
	if q != nil {
		q.Pcond = p
		p.Mark |= LABEL
	}

	// JAL	runtime.morestack(SB)
	p = obj.Appendp(ctxt, p)

	p.As = AJAL
	p.To.Type = obj.TYPE_BRANCH
	if ctxt.Cursym.Cfunc != 0 {
		p.To.Sym = obj.Linklookup(ctxt, "runtime.morestackc", 0)
	} else if ctxt.Cursym.Text.From3.Offset&obj.NEEDCTXT == 0 {
		p.To.Sym = obj.Linklookup(ctxt, "runtime.morestack_noctxt", 0)
	} else {
		p.To.Sym = obj.Linklookup(ctxt, "runtime.morestack", 0)
	}
	p.Mark |= BRANCH

	// JMP	start
	p = obj.Appendp(ctxt, p)

	p.As = AJMP
	p.To.Type = obj.TYPE_BRANCH
	p.Pcond = ctxt.Cursym.Text.Link
	p.Mark |= BRANCH

	// placeholder for q1's jump target
	p = obj.Appendp(ctxt, p)

	p.As = obj.ANOP // zero-width place holder
	q1.Pcond = p

	return p
}

func addnop(ctxt *obj.Link, p *obj.Prog) {
	q := ctxt.NewProg()
	// we want to use the canonical NOP (SLL $0,R0,R0) here,
	// however, as the assembler will always replace $0
	// as R0, we have to resort to manually encode the SLL
	// instruction as WORD $0.
	q.As = AWORD
	q.Lineno = p.Lineno
	q.From.Type = obj.TYPE_CONST
	q.From.Name = obj.NAME_NONE
	q.From.Offset = 0

	q.Link = p.Link
	p.Link = q
}

const (
	E_HILO  = 1 << 0
	E_FCR   = 1 << 1
	E_MCR   = 1 << 2
	E_MEM   = 1 << 3
	E_MEMSP = 1 << 4 /* uses offset and size */
	E_MEMSB = 1 << 5 /* uses offset and size */
	ANYMEM  = E_MEM | E_MEMSP | E_MEMSB
	//DELAY = LOAD|BRANCH|FCMP
	DELAY = BRANCH /* only schedule branch */
)

type Dep struct {
	ireg uint32
	freg uint32
	cc   uint32
}

type Sch struct {
	p       obj.Prog
	set     Dep
	used    Dep
	soffset int32
	size    uint8
	nop     uint8
	comp    bool
}

func sched(ctxt *obj.Link, p0, pe *obj.Prog) {
	var sch [NSCHED]Sch

	/*
	 * build side structure
	 */
	s := sch[:]
	for p := p0; ; p = p.Link {
		s[0].p = *p
		markregused(ctxt, &s[0])
		if p == pe {
			break
		}
		s = s[1:]
	}
	se := s

	for i := cap(sch) - cap(se); i >= 0; i-- {
		s = sch[i:]
		if s[0].p.Mark&DELAY == 0 {
			continue
		}
		if -cap(s) < -cap(se) {
			if !conflict(&s[0], &s[1]) {
				continue
			}
		}

		var t []Sch
		var j int
		for j = cap(sch) - cap(s) - 1; j >= 0; j-- {
			t = sch[j:]
			if t[0].comp {
				if s[0].p.Mark&BRANCH != 0 {
					goto no2
				}
			}
			if t[0].p.Mark&DELAY != 0 {
				if -cap(s) >= -cap(se) || conflict(&t[0], &s[1]) {
					goto no2
				}
			}
			for u := t[1:]; -cap(u) <= -cap(s); u = u[1:] {
				if depend(ctxt, &u[0], &t[0]) {
					goto no2
				}
			}
			goto out2
		no2:
		}

		if s[0].p.Mark&BRANCH != 0 {
			s[0].nop = 1
		}
		continue

	out2:
		// t[0] is the instruction being moved to fill the delay
		stmp := t[0]
		copy(t[:i-j], t[1:i-j+1])
		s[0] = stmp

		if t[i-j-1].p.Mark&BRANCH != 0 {
			// t[i-j] is being put into a branch delay slot
			// combine its Spadj with the branch instruction
			t[i-j-1].p.Spadj += t[i-j].p.Spadj
			t[i-j].p.Spadj = 0
		}

		i--
	}

	/*
	 * put it all back
	 */
	var p *obj.Prog
	var q *obj.Prog
	for s, p = sch[:], p0; -cap(s) <= -cap(se); s, p = s[1:], q {
		q = p.Link
		if q != s[0].p.Link {
			*p = s[0].p
			p.Link = q
		}
		for s[0].nop != 0 {
			s[0].nop--
			addnop(ctxt, p)
		}
	}
}

func markregused(ctxt *obj.Link, s *Sch) {
	p := &s.p
	s.comp = compound(ctxt, p)
	s.nop = 0
	if s.comp {
		s.set.ireg |= 1 << (REGTMP - REG_R0)
		s.used.ireg |= 1 << (REGTMP - REG_R0)
	}

	ar := 0  /* dest is really reference */
	ad := 0  /* source/dest is really address */
	ld := 0  /* opcode is load instruction */
	sz := 20 /* size of load/store for overlap computation */

	/*
	 * flags based on opcode
	 */
	switch p.As {
	case obj.ATEXT:
		ctxt.Autosize = int32(p.To.Offset + 8)
		ad = 1

	case AJAL:
		c := p.Reg
		if c == 0 {
			c = REGLINK
		}
		s.set.ireg |= 1 << uint(c-REG_R0)
		ar = 1
		ad = 1

	case ABGEZAL,
		ABLTZAL:
		s.set.ireg |= 1 << (REGLINK - REG_R0)
		fallthrough
	case ABEQ,
		ABGEZ,
		ABGTZ,
		ABLEZ,
		ABLTZ,
		ABNE:
		ar = 1
		ad = 1

	case ABFPT,
		ABFPF:
		ad = 1
		s.used.cc |= E_FCR

	case ACMPEQD,
		ACMPEQF,
		ACMPGED,
		ACMPGEF,
		ACMPGTD,
		ACMPGTF:
		ar = 1
		s.set.cc |= E_FCR
		p.Mark |= FCMP

	case AJMP:
		ar = 1
		ad = 1

	case AMOVB,
		AMOVBU:
		sz = 1
		ld = 1

	case AMOVH,
		AMOVHU:
		sz = 2
		ld = 1

	case AMOVF,
		AMOVW,
		AMOVWL,
		AMOVWR:
		sz = 4
		ld = 1

	case AMOVD,
		AMOVV,
		AMOVVL,
		AMOVVR:
		sz = 8
		ld = 1

	case ADIV,
		ADIVU,
		AMUL,
		AMULU,
		AREM,
		AREMU,
		ADIVV,
		ADIVVU,
		AMULV,
		AMULVU,
		AREMV,
		AREMVU:
		s.set.cc = E_HILO
		fallthrough
	case AADD,
		AADDU,
		AADDV,
		AADDVU,
		AAND,
		ANOR,
		AOR,
		ASGT,
		ASGTU,
		ASLL,
		ASRA,
		ASRL,
		ASLLV,
		ASRAV,
		ASRLV,
		ASUB,
		ASUBU,
		ASUBV,
		ASUBVU,
		AXOR,

		AADDD,
		AADDF,
		AADDW,
		ASUBD,
		ASUBF,
		ASUBW,
		AMULF,
		AMULD,
		AMULW,
		ADIVF,
		ADIVD,
		ADIVW:
		if p.Reg == 0 {
			if p.To.Type == obj.TYPE_REG {
				p.Reg = p.To.Reg
			}
			//if(p->reg == NREG)
			//	print("botch %P\n", p);
		}
	}

	/*
	 * flags based on 'to' field
	 */
	c := int(p.To.Class)
	if c == 0 {
		c = aclass(ctxt, &p.To) + 1
		p.To.Class = int8(c)
	}
	c--
	switch c {
	default:
		fmt.Printf("unknown class %d %v\n", c, p)

	case C_ZCON,
		C_SCON,
		C_ADD0CON,
		C_AND0CON,
		C_ADDCON,
		C_ANDCON,
		C_UCON,
		C_LCON,
		C_NONE,
		C_SBRA,
		C_LBRA,
		C_ADDR,
		C_TEXTSIZE:
		break

	case C_HI,
		C_LO:
		s.set.cc |= E_HILO

	case C_FCREG:
		s.set.cc |= E_FCR

	case C_MREG:
		s.set.cc |= E_MCR

	case C_ZOREG,
		C_SOREG,
		C_LOREG:
		c = int(p.To.Reg)
		s.used.ireg |= 1 << uint(c-REG_R0)
		if ad != 0 {
			break
		}
		s.size = uint8(sz)
		s.soffset = regoff(ctxt, &p.To)

		m := uint32(ANYMEM)
		if c == REGSB {
			m = E_MEMSB
		}
		if c == REGSP {
			m = E_MEMSP
		}

		if ar != 0 {
			s.used.cc |= m
		} else {
			s.set.cc |= m
		}

	case C_SACON,
		C_LACON:
		s.used.ireg |= 1 << (REGSP - REG_R0)

	case C_SECON,
		C_LECON:
		s.used.ireg |= 1 << (REGSB - REG_R0)

	case C_REG:
		if ar != 0 {
			s.used.ireg |= 1 << uint(p.To.Reg-REG_R0)
		} else {
			s.set.ireg |= 1 << uint(p.To.Reg-REG_R0)
		}

	case C_FREG:
		if ar != 0 {
			s.used.freg |= 1 << uint(p.To.Reg-REG_F0)
		} else {
			s.set.freg |= 1 << uint(p.To.Reg-REG_F0)
		}
		if ld != 0 && p.From.Type == obj.TYPE_REG {
			p.Mark |= LOAD
		}

	case C_SAUTO,
		C_LAUTO:
		s.used.ireg |= 1 << (REGSP - REG_R0)
		if ad != 0 {
			break
		}
		s.size = uint8(sz)
		s.soffset = regoff(ctxt, &p.To)

		if ar != 0 {
			s.used.cc |= E_MEMSP
		} else {
			s.set.cc |= E_MEMSP
		}

	case C_SEXT,
		C_LEXT:
		s.used.ireg |= 1 << (REGSB - REG_R0)
		if ad != 0 {
			break
		}
		s.size = uint8(sz)
		s.soffset = regoff(ctxt, &p.To)

		if ar != 0 {
			s.used.cc |= E_MEMSB
		} else {
			s.set.cc |= E_MEMSB
		}
	}

	/*
	 * flags based on 'from' field
	 */
	c = int(p.From.Class)
	if c == 0 {
		c = aclass(ctxt, &p.From) + 1
		p.From.Class = int8(c)
	}
	c--
	switch c {
	default:
		fmt.Printf("unknown class %d %v\n", c, p)

	case C_ZCON,
		C_SCON,
		C_ADD0CON,
		C_AND0CON,
		C_ADDCON,
		C_ANDCON,
		C_UCON,
		C_LCON,
		C_NONE,
		C_SBRA,
		C_LBRA,
		C_ADDR,
		C_TEXTSIZE:
		break

	case C_HI,
		C_LO:
		s.used.cc |= E_HILO

	case C_FCREG:
		s.used.cc |= E_FCR

	case C_MREG:
		s.used.cc |= E_MCR

	case C_ZOREG,
		C_SOREG,
		C_LOREG:
		c = int(p.From.Reg)
		s.used.ireg |= 1 << uint(c-REG_R0)
		if ld != 0 {
			p.Mark |= LOAD
		}
		s.size = uint8(sz)
		s.soffset = regoff(ctxt, &p.From)

		m := uint32(ANYMEM)
		if c == REGSB {
			m = E_MEMSB
		}
		if c == REGSP {
			m = E_MEMSP
		}

		s.used.cc |= m

	case C_SACON,
		C_LACON:
		c = int(p.From.Reg)
		if c == 0 {
			c = REGSP
		}
		s.used.ireg |= 1 << uint(c-REG_R0)

	case C_SECON,
		C_LECON:
		s.used.ireg |= 1 << (REGSB - REG_R0)

	case C_REG:
		s.used.ireg |= 1 << uint(p.From.Reg-REG_R0)

	case C_FREG:
		s.used.freg |= 1 << uint(p.From.Reg-REG_F0)
		if ld != 0 && p.To.Type == obj.TYPE_REG {
			p.Mark |= LOAD
		}

	case C_SAUTO,
		C_LAUTO:
		s.used.ireg |= 1 << (REGSP - REG_R0)
		if ld != 0 {
			p.Mark |= LOAD
		}
		if ad != 0 {
			break
		}
		s.size = uint8(sz)
		s.soffset = regoff(ctxt, &p.From)

		s.used.cc |= E_MEMSP

	case C_SEXT:
	case C_LEXT:
		s.used.ireg |= 1 << (REGSB - REG_R0)
		if ld != 0 {
			p.Mark |= LOAD
		}
		if ad != 0 {
			break
		}
		s.size = uint8(sz)
		s.soffset = regoff(ctxt, &p.From)

		s.used.cc |= E_MEMSB
	}

	c = int(p.Reg)
	if c != 0 {
		if REG_F0 <= c && c <= REG_F31 {
			s.used.freg |= 1 << uint(c-REG_F0)
		} else {
			s.used.ireg |= 1 << uint(c-REG_R0)
		}
	}
	s.set.ireg &^= (1 << (REGZERO - REG_R0)) /* R0 cant be set */
}

/*
 * test to see if 2 instrictions can be
 * interchanged without changing semantics
 */
func depend(ctxt *obj.Link, sa, sb *Sch) bool {
	if sa.set.ireg&(sb.set.ireg|sb.used.ireg) != 0 {
		return true
	}
	if sb.set.ireg&sa.used.ireg != 0 {
		return true
	}

	if sa.set.freg&(sb.set.freg|sb.used.freg) != 0 {
		return true
	}
	if sb.set.freg&sa.used.freg != 0 {
		return true
	}

	/*
	 * special case.
	 * loads from same address cannot pass.
	 * this is for hardware fifo's and the like
	 */
	if sa.used.cc&sb.used.cc&E_MEM != 0 {
		if sa.p.Reg == sb.p.Reg {
			if regoff(ctxt, &sa.p.From) == regoff(ctxt, &sb.p.From) {
				return true
			}
		}
	}

	x := (sa.set.cc & (sb.set.cc | sb.used.cc)) | (sb.set.cc & sa.used.cc)
	if x != 0 {
		/*
		 * allow SB and SP to pass each other.
		 * allow SB to pass SB iff doffsets are ok
		 * anything else conflicts
		 */
		if x != E_MEMSP && x != E_MEMSB {
			return true
		}
		x = sa.set.cc | sb.set.cc | sa.used.cc | sb.used.cc
		if x&E_MEM != 0 {
			return true
		}
		if offoverlap(sa, sb) {
			return true
		}
	}

	return false
}

func offoverlap(sa, sb *Sch) bool {
	if sa.soffset < sb.soffset {
		if sa.soffset+int32(sa.size) > sb.soffset {
			return true
		}
		return false
	}
	if sb.soffset+int32(sb.size) > sa.soffset {
		return true
	}
	return false
}

/*
 * test 2 adjacent instructions
 * and find out if inserted instructions
 * are desired to prevent stalls.
 */
func conflict(sa, sb *Sch) bool {
	if sa.set.ireg&sb.used.ireg != 0 {
		return true
	}
	if sa.set.freg&sb.used.freg != 0 {
		return true
	}
	if sa.set.cc&sb.used.cc != 0 {
		return true
	}
	return false
}

func compound(ctxt *obj.Link, p *obj.Prog) bool {
	o := oplook(ctxt, p)
	if o.size != 4 {
		return true
	}
	if p.To.Type == obj.TYPE_REG && p.To.Reg == REGSB {
		return true
	}
	return false
}

func follow(ctxt *obj.Link, s *obj.LSym) {
	ctxt.Cursym = s

	firstp := ctxt.NewProg()
	lastp := firstp
	xfol(ctxt, s.Text, &lastp)
	lastp.Link = nil
	s.Text = firstp.Link
}

func xfol(ctxt *obj.Link, p *obj.Prog, last **obj.Prog) {
	var q *obj.Prog
	var r *obj.Prog
	var a int
	var i int

loop:
	if p == nil {
		return
	}
	a = int(p.As)
	if a == AJMP {
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
			a = int(q.As)
			if a == obj.ANOP {
				i--
				continue
			}

			if a == AJMP || a == ARET || a == ARFE {
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
				if a == AJMP || a == ARET || a == ARFE {
					return
				}
				r.As = ABNE
				if a == ABNE {
					r.As = ABEQ
				}
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

		a = AJMP
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
	if a == AJMP || a == ARET || a == ARFE {
		if p.Mark&NOSCHED != 0 {
			p = p.Link
			goto loop
		}

		return
	}

	if p.Pcond != nil {
		if a != AJAL && p.Link != nil {
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

var Linkmips64 = obj.LinkArch{
	ByteOrder:  binary.BigEndian,
	Name:       "mips64",
	Thechar:    '0',
	Preprocess: preprocess,
	Assemble:   span0,
	Follow:     follow,
	Progedit:   progedit,
	Minlc:      4,
	Ptrsize:    8,
	Regsize:    8,
}

var Linkmips64le = obj.LinkArch{
	ByteOrder:  binary.LittleEndian,
	Name:       "mips64le",
	Thechar:    '0',
	Preprocess: preprocess,
	Assemble:   span0,
	Follow:     follow,
	Progedit:   progedit,
	Minlc:      4,
	Ptrsize:    8,
	Regsize:    8,
}
