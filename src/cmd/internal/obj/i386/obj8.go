// Inferno utils/8l/pass.c
// http://code.google.com/p/inferno-os/source/browse/utils/8l/pass.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
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

package i386

import (
	"cmd/internal/obj"
	"encoding/binary"
	"fmt"
	"log"
	"math"
)

func canuselocaltls(ctxt *obj.Link) bool {
	switch ctxt.Headtype {
	case obj.Hlinux,
		obj.Hnacl,
		obj.Hplan9,
		obj.Hwindows:
		return false
	}

	return true
}

func progedit(ctxt *obj.Link, p *obj.Prog) {
	var literal string
	var s *obj.LSym
	var q *obj.Prog

	// See obj6.c for discussion of TLS.
	if canuselocaltls(ctxt) {
		// Reduce TLS initial exec model to TLS local exec model.
		// Sequences like
		//	MOVL TLS, BX
		//	... off(BX)(TLS*1) ...
		// become
		//	NOP
		//	... off(TLS) ...
		if p.As == AMOVL && p.From.Type == obj.TYPE_REG && p.From.Reg == REG_TLS && p.To.Type == obj.TYPE_REG && REG_AX <= p.To.Reg && p.To.Reg <= REG_DI {
			p.As = obj.ANOP
			p.From.Type = obj.TYPE_NONE
			p.To.Type = obj.TYPE_NONE
		}

		if p.From.Type == obj.TYPE_MEM && p.From.Index == REG_TLS && REG_AX <= p.From.Reg && p.From.Reg <= REG_DI {
			p.From.Type = obj.TYPE_MEM
			p.From.Reg = REG_TLS
			p.From.Scale = 0
			p.From.Index = REG_NONE
		}

		if p.To.Type == obj.TYPE_MEM && p.To.Index == REG_TLS && REG_AX <= p.To.Reg && p.To.Reg <= REG_DI {
			p.To.Type = obj.TYPE_MEM
			p.To.Reg = REG_TLS
			p.To.Scale = 0
			p.To.Index = REG_NONE
		}
	} else {
		// As a courtesy to the C compilers, rewrite TLS local exec load as TLS initial exec load.
		// The instruction
		//	MOVL off(TLS), BX
		// becomes the sequence
		//	MOVL TLS, BX
		//	MOVL off(BX)(TLS*1), BX
		// This allows the C compilers to emit references to m and g using the direct off(TLS) form.
		if p.As == AMOVL && p.From.Type == obj.TYPE_MEM && p.From.Reg == REG_TLS && p.To.Type == obj.TYPE_REG && REG_AX <= p.To.Reg && p.To.Reg <= REG_DI {
			q = obj.Appendp(ctxt, p)
			q.As = p.As
			q.From.Type = obj.TYPE_MEM
			q.From.Reg = p.To.Reg
			q.From.Index = REG_TLS
			q.From.Scale = 2 // TODO: use 1
			q.To = p.To
			p.From.Type = obj.TYPE_REG
			p.From.Reg = REG_TLS
			p.From.Index = REG_NONE
			p.From.Offset = 0
		}
	}

	// TODO: Remove.
	if ctxt.Headtype == obj.Hplan9 {
		if p.From.Scale == 1 && p.From.Index == REG_TLS {
			p.From.Scale = 2
		}
		if p.To.Scale == 1 && p.To.Index == REG_TLS {
			p.To.Scale = 2
		}
	}

	// Rewrite CALL/JMP/RET to symbol as TYPE_BRANCH.
	switch p.As {
	case obj.ACALL,
		obj.AJMP,
		obj.ARET:
		if p.To.Type == obj.TYPE_MEM && (p.To.Name == obj.NAME_EXTERN || p.To.Name == obj.NAME_STATIC) && p.To.Sym != nil {
			p.To.Type = obj.TYPE_BRANCH
		}
	}

	// Rewrite float constants to values stored in memory.
	switch p.As {
	// Convert AMOVSS $(0), Xx to AXORPS Xx, Xx
	case AMOVSS:
		if p.From.Type == obj.TYPE_FCONST {
			if p.From.U.Dval == 0 {
				if p.To.Type == obj.TYPE_REG && REG_X0 <= p.To.Reg && p.To.Reg <= REG_X7 {
					p.As = AXORPS
					p.From = p.To
					break
				}
			}
		}
		fallthrough

		// fallthrough

	case AFMOVF,
		AFADDF,
		AFSUBF,
		AFSUBRF,
		AFMULF,
		AFDIVF,
		AFDIVRF,
		AFCOMF,
		AFCOMFP,
		AADDSS,
		ASUBSS,
		AMULSS,
		ADIVSS,
		ACOMISS,
		AUCOMISS:
		if p.From.Type == obj.TYPE_FCONST {
			var i32 uint32
			var f32 float32
			f32 = float32(p.From.U.Dval)
			i32 = math.Float32bits(f32)
			literal = fmt.Sprintf("$f32.%08x", i32)
			s = obj.Linklookup(ctxt, literal, 0)
			if s.Type == 0 {
				s.Type = obj.SRODATA
				obj.Adduint32(ctxt, s, i32)
				s.Reachable = 0
			}

			p.From.Type = obj.TYPE_MEM
			p.From.Name = obj.NAME_EXTERN
			p.From.Sym = s
			p.From.Offset = 0
		}

		// Convert AMOVSD $(0), Xx to AXORPS Xx, Xx
	case AMOVSD:
		if p.From.Type == obj.TYPE_FCONST {
			if p.From.U.Dval == 0 {
				if p.To.Type == obj.TYPE_REG && REG_X0 <= p.To.Reg && p.To.Reg <= REG_X7 {
					p.As = AXORPS
					p.From = p.To
					break
				}
			}
		}
		fallthrough

		// fallthrough

	case AFMOVD,
		AFADDD,
		AFSUBD,
		AFSUBRD,
		AFMULD,
		AFDIVD,
		AFDIVRD,
		AFCOMD,
		AFCOMDP,
		AADDSD,
		ASUBSD,
		AMULSD,
		ADIVSD,
		ACOMISD,
		AUCOMISD:
		if p.From.Type == obj.TYPE_FCONST {
			var i64 uint64
			i64 = math.Float64bits(p.From.U.Dval)
			literal = fmt.Sprintf("$f64.%016x", i64)
			s = obj.Linklookup(ctxt, literal, 0)
			if s.Type == 0 {
				s.Type = obj.SRODATA
				obj.Adduint64(ctxt, s, i64)
				s.Reachable = 0
			}

			p.From.Type = obj.TYPE_MEM
			p.From.Name = obj.NAME_EXTERN
			p.From.Sym = s
			p.From.Offset = 0
		}
	}
}

func preprocess(ctxt *obj.Link, cursym *obj.LSym) {
	var p *obj.Prog
	var q *obj.Prog
	var p1 *obj.Prog
	var p2 *obj.Prog
	var autoffset int32
	var deltasp int32
	var a int

	if ctxt.Symmorestack[0] == nil {
		ctxt.Symmorestack[0] = obj.Linklookup(ctxt, "runtime.morestack", 0)
		ctxt.Symmorestack[1] = obj.Linklookup(ctxt, "runtime.morestack_noctxt", 0)
	}

	if ctxt.Headtype == obj.Hplan9 && ctxt.Plan9privates == nil {
		ctxt.Plan9privates = obj.Linklookup(ctxt, "_privates", 0)
	}

	ctxt.Cursym = cursym

	if cursym.Text == nil || cursym.Text.Link == nil {
		return
	}

	p = cursym.Text
	autoffset = int32(p.To.Offset)
	if autoffset < 0 {
		autoffset = 0
	}

	cursym.Locals = autoffset
	cursym.Args = p.To.U.Argsize

	q = nil

	if p.From3.Offset&obj.NOSPLIT == 0 || (p.From3.Offset&obj.WRAPPER != 0) {
		p = obj.Appendp(ctxt, p)
		p = load_g_cx(ctxt, p) // load g into CX
	}

	if cursym.Text.From3.Offset&obj.NOSPLIT == 0 {
		p = stacksplit(ctxt, p, autoffset, cursym.Text.From3.Offset&obj.NEEDCTXT == 0, &q) // emit split check
	}

	if autoffset != 0 {
		p = obj.Appendp(ctxt, p)
		p.As = AADJSP
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(autoffset)
		p.Spadj = autoffset
	} else {
		// zero-byte stack adjustment.
		// Insert a fake non-zero adjustment so that stkcheck can
		// recognize the end of the stack-splitting prolog.
		p = obj.Appendp(ctxt, p)

		p.As = obj.ANOP
		p.Spadj = int32(-ctxt.Arch.Ptrsize)
		p = obj.Appendp(ctxt, p)
		p.As = obj.ANOP
		p.Spadj = int32(ctxt.Arch.Ptrsize)
	}

	if q != nil {
		q.Pcond = p
	}
	deltasp = autoffset

	if cursym.Text.From3.Offset&obj.WRAPPER != 0 {
		// if(g->panic != nil && g->panic->argp == FP) g->panic->argp = bottom-of-frame
		//
		//	MOVL g_panic(CX), BX
		//	TESTL BX, BX
		//	JEQ end
		//	LEAL (autoffset+4)(SP), DI
		//	CMPL panic_argp(BX), DI
		//	JNE end
		//	MOVL SP, panic_argp(BX)
		// end:
		//	NOP
		//
		// The NOP is needed to give the jumps somewhere to land.
		// It is a liblink NOP, not an x86 NOP: it encodes to 0 instruction bytes.

		p = obj.Appendp(ctxt, p)

		p.As = AMOVL
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = REG_CX
		p.From.Offset = 4 * int64(ctxt.Arch.Ptrsize) // G.panic
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_BX

		p = obj.Appendp(ctxt, p)
		p.As = ATESTL
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_BX
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_BX

		p = obj.Appendp(ctxt, p)
		p.As = AJEQ
		p.To.Type = obj.TYPE_BRANCH
		p1 = p

		p = obj.Appendp(ctxt, p)
		p.As = ALEAL
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = REG_SP
		p.From.Offset = int64(autoffset) + 4
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_DI

		p = obj.Appendp(ctxt, p)
		p.As = ACMPL
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = REG_BX
		p.From.Offset = 0 // Panic.argp
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_DI

		p = obj.Appendp(ctxt, p)
		p.As = AJNE
		p.To.Type = obj.TYPE_BRANCH
		p2 = p

		p = obj.Appendp(ctxt, p)
		p.As = AMOVL
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_SP
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = REG_BX
		p.To.Offset = 0 // Panic.argp

		p = obj.Appendp(ctxt, p)

		p.As = obj.ANOP
		p1.Pcond = p
		p2.Pcond = p
	}

	if ctxt.Debugzerostack != 0 && autoffset != 0 && cursym.Text.From3.Offset&obj.NOSPLIT == 0 {
		// 8l -Z means zero the stack frame on entry.
		// This slows down function calls but can help avoid
		// false positives in garbage collection.
		p = obj.Appendp(ctxt, p)

		p.As = AMOVL
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_SP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_DI

		p = obj.Appendp(ctxt, p)
		p.As = AMOVL
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(autoffset) / 4
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_CX

		p = obj.Appendp(ctxt, p)
		p.As = AMOVL
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_AX

		p = obj.Appendp(ctxt, p)
		p.As = AREP

		p = obj.Appendp(ctxt, p)
		p.As = ASTOSL
	}

	for ; p != nil; p = p.Link {
		a = int(p.From.Name)
		if a == obj.NAME_AUTO {
			p.From.Offset += int64(deltasp)
		}
		if a == obj.NAME_PARAM {
			p.From.Offset += int64(deltasp) + 4
		}
		a = int(p.To.Name)
		if a == obj.NAME_AUTO {
			p.To.Offset += int64(deltasp)
		}
		if a == obj.NAME_PARAM {
			p.To.Offset += int64(deltasp) + 4
		}

		switch p.As {
		default:
			continue

		case APUSHL,
			APUSHFL:
			deltasp += 4
			p.Spadj = 4
			continue

		case APUSHW,
			APUSHFW:
			deltasp += 2
			p.Spadj = 2
			continue

		case APOPL,
			APOPFL:
			deltasp -= 4
			p.Spadj = -4
			continue

		case APOPW,
			APOPFW:
			deltasp -= 2
			p.Spadj = -2
			continue

		case obj.ARET:
			break
		}

		if autoffset != deltasp {
			ctxt.Diag("unbalanced PUSH/POP")
		}

		if autoffset != 0 {
			p.As = AADJSP
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = int64(-autoffset)
			p.Spadj = -autoffset
			p = obj.Appendp(ctxt, p)
			p.As = obj.ARET

			// If there are instructions following
			// this ARET, they come from a branch
			// with the same stackframe, so undo
			// the cleanup.
			p.Spadj = +autoffset
		}

		if p.To.Sym != nil { // retjmp
			p.As = obj.AJMP
		}
	}
}

// Append code to p to load g into cx.
// Overwrites p with the first instruction (no first appendp).
// Overwriting p is unusual but it lets use this in both the
// prologue (caller must call appendp first) and in the epilogue.
// Returns last new instruction.
func load_g_cx(ctxt *obj.Link, p *obj.Prog) *obj.Prog {
	var next *obj.Prog

	p.As = AMOVL
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = REG_TLS
	p.From.Offset = 0
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_CX

	next = p.Link
	progedit(ctxt, p)
	for p.Link != next {
		p = p.Link
	}

	if p.From.Index == REG_TLS {
		p.From.Scale = 2
	}

	return p
}

// Append code to p to check for stack split.
// Appends to (does not overwrite) p.
// Assumes g is in CX.
// Returns last new instruction.
// On return, *jmpok is the instruction that should jump
// to the stack frame allocation if no split is needed.
func stacksplit(ctxt *obj.Link, p *obj.Prog, framesize int32, noctxt bool, jmpok **obj.Prog) *obj.Prog {
	var q *obj.Prog
	var q1 *obj.Prog

	if ctxt.Debugstack != 0 {
		// 8l -K means check not only for stack
		// overflow but stack underflow.
		// On underflow, INT 3 (breakpoint).
		// Underflow itself is rare but this also
		// catches out-of-sync stack guard info.
		p = obj.Appendp(ctxt, p)

		p.As = ACMPL
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = REG_CX
		p.From.Offset = 4
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_SP

		p = obj.Appendp(ctxt, p)
		p.As = AJCC
		p.To.Type = obj.TYPE_BRANCH
		p.To.Offset = 4
		q1 = p

		p = obj.Appendp(ctxt, p)
		p.As = AINT
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 3

		p = obj.Appendp(ctxt, p)
		p.As = obj.ANOP
		q1.Pcond = p
	}

	q1 = nil

	if framesize <= obj.StackSmall {
		// small stack: SP <= stackguard
		//	CMPL SP, stackguard
		p = obj.Appendp(ctxt, p)

		p.As = ACMPL
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_SP
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = REG_CX
		p.To.Offset = 2 * int64(ctxt.Arch.Ptrsize) // G.stackguard0
		if ctxt.Cursym.Cfunc != 0 {
			p.To.Offset = 3 * int64(ctxt.Arch.Ptrsize) // G.stackguard1
		}
	} else if framesize <= obj.StackBig {
		// large stack: SP-framesize <= stackguard-StackSmall
		//	LEAL -(framesize-StackSmall)(SP), AX
		//	CMPL AX, stackguard
		p = obj.Appendp(ctxt, p)

		p.As = ALEAL
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = REG_SP
		p.From.Offset = -(int64(framesize) - obj.StackSmall)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_AX

		p = obj.Appendp(ctxt, p)
		p.As = ACMPL
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_AX
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = REG_CX
		p.To.Offset = 2 * int64(ctxt.Arch.Ptrsize) // G.stackguard0
		if ctxt.Cursym.Cfunc != 0 {
			p.To.Offset = 3 * int64(ctxt.Arch.Ptrsize) // G.stackguard1
		}
	} else {
		// Such a large stack we need to protect against wraparound
		// if SP is close to zero.
		//	SP-stackguard+StackGuard <= framesize + (StackGuard-StackSmall)
		// The +StackGuard on both sides is required to keep the left side positive:
		// SP is allowed to be slightly below stackguard. See stack.h.
		//
		// Preemption sets stackguard to StackPreempt, a very large value.
		// That breaks the math above, so we have to check for that explicitly.
		//	MOVL	stackguard, CX
		//	CMPL	CX, $StackPreempt
		//	JEQ	label-of-call-to-morestack
		//	LEAL	StackGuard(SP), AX
		//	SUBL	stackguard, AX
		//	CMPL	AX, $(framesize+(StackGuard-StackSmall))
		p = obj.Appendp(ctxt, p)

		p.As = AMOVL
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = REG_CX
		p.From.Offset = 0
		p.From.Offset = 2 * int64(ctxt.Arch.Ptrsize) // G.stackguard0
		if ctxt.Cursym.Cfunc != 0 {
			p.From.Offset = 3 * int64(ctxt.Arch.Ptrsize) // G.stackguard1
		}
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_SI

		p = obj.Appendp(ctxt, p)
		p.As = ACMPL
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_SI
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = int64(uint32(obj.StackPreempt & (1<<32 - 1)))

		p = obj.Appendp(ctxt, p)
		p.As = AJEQ
		p.To.Type = obj.TYPE_BRANCH
		q1 = p

		p = obj.Appendp(ctxt, p)
		p.As = ALEAL
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = REG_SP
		p.From.Offset = obj.StackGuard
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_AX

		p = obj.Appendp(ctxt, p)
		p.As = ASUBL
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_SI
		p.From.Offset = 0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_AX

		p = obj.Appendp(ctxt, p)
		p.As = ACMPL
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_AX
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = int64(framesize) + (obj.StackGuard - obj.StackSmall)
	}

	// common
	p = obj.Appendp(ctxt, p)

	p.As = AJHI
	p.To.Type = obj.TYPE_BRANCH
	p.To.Offset = 4
	q = p

	p = obj.Appendp(ctxt, p)
	p.As = obj.ACALL
	p.To.Type = obj.TYPE_BRANCH
	if ctxt.Cursym.Cfunc != 0 {
		p.To.Sym = obj.Linklookup(ctxt, "runtime.morestackc", 0)
	} else {
		p.To.Sym = ctxt.Symmorestack[bool2int(noctxt)]
	}

	p = obj.Appendp(ctxt, p)
	p.As = obj.AJMP
	p.To.Type = obj.TYPE_BRANCH
	p.Pcond = ctxt.Cursym.Text.Link

	if q != nil {
		q.Pcond = p.Link
	}
	if q1 != nil {
		q1.Pcond = q.Link
	}

	*jmpok = q
	return p
}

func follow(ctxt *obj.Link, s *obj.LSym) {
	var firstp *obj.Prog
	var lastp *obj.Prog

	ctxt.Cursym = s

	firstp = ctxt.NewProg()
	lastp = firstp
	xfol(ctxt, s.Text, &lastp)
	lastp.Link = nil
	s.Text = firstp.Link
}

func nofollow(a int) bool {
	switch a {
	case obj.AJMP,
		obj.ARET,
		AIRETL,
		AIRETW,
		obj.AUNDEF:
		return true
	}

	return false
}

func pushpop(a int) bool {
	switch a {
	case APUSHL,
		APUSHFL,
		APUSHW,
		APUSHFW,
		APOPL,
		APOPFL,
		APOPW,
		APOPFW:
		return true
	}

	return false
}

func relinv(a int) int {
	switch a {
	case AJEQ:
		return AJNE
	case AJNE:
		return AJEQ
	case AJLE:
		return AJGT
	case AJLS:
		return AJHI
	case AJLT:
		return AJGE
	case AJMI:
		return AJPL
	case AJGE:
		return AJLT
	case AJPL:
		return AJMI
	case AJGT:
		return AJLE
	case AJHI:
		return AJLS
	case AJCS:
		return AJCC
	case AJCC:
		return AJCS
	case AJPS:
		return AJPC
	case AJPC:
		return AJPS
	case AJOS:
		return AJOC
	case AJOC:
		return AJOS
	}

	log.Fatalf("unknown relation: %s", Anames[a])
	return 0
}

func xfol(ctxt *obj.Link, p *obj.Prog, last **obj.Prog) {
	var q *obj.Prog
	var i int
	var a int

loop:
	if p == nil {
		return
	}
	if p.As == obj.AJMP {
		q = p.Pcond
		if q != nil && q.As != obj.ATEXT {
			/* mark instruction as done and continue layout at target of jump */
			p.Mark = 1

			p = q
			if p.Mark == 0 {
				goto loop
			}
		}
	}

	if p.Mark != 0 {
		/*
		 * p goes here, but already used it elsewhere.
		 * copy up to 4 instructions or else branch to other copy.
		 */
		i = 0
		q = p
		for ; i < 4; (func() { i++; q = q.Link })() {
			if q == nil {
				break
			}
			if q == *last {
				break
			}
			a = int(q.As)
			if a == obj.ANOP {
				i--
				continue
			}

			if nofollow(a) || pushpop(a) {
				break // NOTE(rsc): arm does goto copy
			}
			if q.Pcond == nil || q.Pcond.Mark != 0 {
				continue
			}
			if a == obj.ACALL || a == ALOOP {
				continue
			}
			for {
				if p.As == obj.ANOP {
					p = p.Link
					continue
				}

				q = obj.Copyp(ctxt, p)
				p = p.Link
				q.Mark = 1
				(*last).Link = q
				*last = q
				if int(q.As) != a || q.Pcond == nil || q.Pcond.Mark != 0 {
					continue
				}

				q.As = int16(relinv(int(q.As)))
				p = q.Pcond
				q.Pcond = q.Link
				q.Link = p
				xfol(ctxt, q.Link, last)
				p = q.Link
				if p.Mark != 0 {
					return
				}
				goto loop
				/* */
			}
		}
		q = ctxt.NewProg()
		q.As = obj.AJMP
		q.Lineno = p.Lineno
		q.To.Type = obj.TYPE_BRANCH
		q.To.Offset = p.Pc
		q.Pcond = p
		p = q
	}

	/* emit p */
	p.Mark = 1

	(*last).Link = p
	*last = p
	a = int(p.As)

	/* continue loop with what comes after p */
	if nofollow(a) {
		return
	}
	if p.Pcond != nil && a != obj.ACALL {
		/*
		 * some kind of conditional branch.
		 * recurse to follow one path.
		 * continue loop on the other.
		 */
		q = obj.Brchain(ctxt, p.Pcond)
		if q != nil {
			p.Pcond = q
		}
		q = obj.Brchain(ctxt, p.Link)
		if q != nil {
			p.Link = q
		}
		if p.From.Type == obj.TYPE_CONST {
			if p.From.Offset == 1 {
				/*
				 * expect conditional jump to be taken.
				 * rewrite so that's the fall-through case.
				 */
				p.As = int16(relinv(a))

				q = p.Link
				p.Link = p.Pcond
				p.Pcond = q
			}
		} else {
			q = p.Link
			if q.Mark != 0 {
				if a != ALOOP {
					p.As = int16(relinv(a))
					p.Link = p.Pcond
					p.Pcond = q
				}
			}
		}

		xfol(ctxt, p.Link, last)
		if p.Pcond.Mark != 0 {
			return
		}
		p = p.Pcond
		goto loop
	}

	p = p.Link
	goto loop
}

var Link386 = obj.LinkArch{
	Dconv:      Dconv,
	Rconv:      Rconv,
	ByteOrder:  binary.LittleEndian,
	Pconv:      Pconv,
	Name:       "386",
	Thechar:    '8',
	Endian:     obj.LittleEndian,
	Preprocess: preprocess,
	Assemble:   span8,
	Follow:     follow,
	Progedit:   progedit,
	Minlc:      1,
	Ptrsize:    4,
	Regsize:    4,
}
