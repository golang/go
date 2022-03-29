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
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"log"
	"math"
)

func progedit(ctxt *obj.Link, p *obj.Prog, newprog obj.ProgAlloc) {
	p.From.Class = 0
	p.To.Class = 0

	c := ctxtz{ctxt: ctxt, newprog: newprog}

	// Rewrite BR/BL to symbol as TYPE_BRANCH.
	switch p.As {
	case ABR, ABL, obj.ARET, obj.ADUFFZERO, obj.ADUFFCOPY:
		if p.To.Sym != nil {
			p.To.Type = obj.TYPE_BRANCH
		}
	}

	// Rewrite float constants to values stored in memory unless they are +0.
	switch p.As {
	case AFMOVS:
		if p.From.Type == obj.TYPE_FCONST {
			f32 := float32(p.From.Val.(float64))
			if math.Float32bits(f32) == 0 { // +0
				break
			}
			p.From.Type = obj.TYPE_MEM
			p.From.Sym = ctxt.Float32Sym(f32)
			p.From.Name = obj.NAME_EXTERN
			p.From.Offset = 0
		}

	case AFMOVD:
		if p.From.Type == obj.TYPE_FCONST {
			f64 := p.From.Val.(float64)
			if math.Float64bits(f64) == 0 { // +0
				break
			}
			p.From.Type = obj.TYPE_MEM
			p.From.Sym = ctxt.Float64Sym(f64)
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
				p.From.Type = obj.TYPE_MEM
				p.From.Sym = ctxt.Int64Sym(p.From.Offset)
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

	if c.ctxt.Flag_dynlink {
		c.rewriteToUseGot(p)
	}
}

// Rewrite p, if necessary, to access global data via the global offset table.
func (c *ctxtz) rewriteToUseGot(p *obj.Prog) {
	// At the moment EXRL instructions are not emitted by the compiler and only reference local symbols in
	// assembly code.
	if p.As == AEXRL {
		return
	}

	// We only care about global data: NAME_EXTERN means a global
	// symbol in the Go sense, and p.Sym.Local is true for a few
	// internally defined symbols.
	// Rewrites must not clobber flags and therefore cannot use the
	// ADD instruction.
	if p.From.Type == obj.TYPE_ADDR && p.From.Name == obj.NAME_EXTERN && !p.From.Sym.Local() {
		// MOVD $sym, Rx becomes MOVD sym@GOT, Rx
		// MOVD $sym+<off>, Rx becomes MOVD sym@GOT, Rx or REGTMP2; MOVD $<off>(Rx or REGTMP2), Rx
		if p.To.Type != obj.TYPE_REG || p.As != AMOVD {
			c.ctxt.Diag("do not know how to handle LEA-type insn to non-register in %v with -dynlink", p)
		}
		p.From.Type = obj.TYPE_MEM
		p.From.Name = obj.NAME_GOTREF
		q := p
		if p.From.Offset != 0 {
			target := p.To.Reg
			if target == REG_R0 {
				// Cannot use R0 as input to address calculation.
				// REGTMP might be used by the assembler.
				p.To.Reg = REGTMP2
			}
			q = obj.Appendp(q, c.newprog)
			q.As = AMOVD
			q.From.Type = obj.TYPE_ADDR
			q.From.Offset = p.From.Offset
			q.From.Reg = p.To.Reg
			q.To.Type = obj.TYPE_REG
			q.To.Reg = target
			p.From.Offset = 0
		}
	}
	if p.GetFrom3() != nil && p.GetFrom3().Name == obj.NAME_EXTERN {
		c.ctxt.Diag("don't know how to handle %v with -dynlink", p)
	}
	var source *obj.Addr
	// MOVD sym, Ry becomes MOVD sym@GOT, REGTMP2; MOVD (REGTMP2), Ry
	// MOVD Ry, sym becomes MOVD sym@GOT, REGTMP2; MOVD Ry, (REGTMP2)
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
	p1.To.Reg = REGTMP2

	p2.As = p.As
	p2.From = p.From
	p2.To = p.To
	if p.From.Name == obj.NAME_EXTERN {
		p2.From.Reg = REGTMP2
		p2.From.Name = obj.NAME_NONE
		p2.From.Sym = nil
	} else if p.To.Name == obj.NAME_EXTERN {
		p2.To.Reg = REGTMP2
		p2.To.Name = obj.NAME_NONE
		p2.To.Sym = nil
	} else {
		return
	}
	obj.Nopout(p)
}

func preprocess(ctxt *obj.Link, cursym *obj.LSym, newprog obj.ProgAlloc) {
	// TODO(minux): add morestack short-cuts with small fixed frame-size.
	if cursym.Func().Text == nil || cursym.Func().Text.Link == nil {
		return
	}

	c := ctxtz{ctxt: ctxt, cursym: cursym, newprog: newprog}

	p := c.cursym.Func().Text
	textstksiz := p.To.Offset
	if textstksiz == -8 {
		// Compatibility hack.
		p.From.Sym.Set(obj.AttrNoFrame, true)
		textstksiz = 0
	}
	if textstksiz%8 != 0 {
		c.ctxt.Diag("frame size %d not a multiple of 8", textstksiz)
	}
	if p.From.Sym.NoFrame() {
		if textstksiz != 0 {
			c.ctxt.Diag("NOFRAME functions must have a frame size of 0, not %d", textstksiz)
		}
	}

	c.cursym.Func().Args = p.To.Val.(int32)
	c.cursym.Func().Locals = int32(textstksiz)

	/*
	 * find leaf subroutines
	 * strip NOPs
	 * expand RET
	 */

	var q *obj.Prog
	for p := c.cursym.Func().Text; p != nil; p = p.Link {
		switch p.As {
		case obj.ATEXT:
			q = p
			p.Mark |= LEAF

		case ABL, ABCL:
			q = p
			c.cursym.Func().Text.Mark &^= LEAF
			fallthrough

		case ABC,
			ABRC,
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
			ACRJ,
			ACGRJ,
			ACLRJ,
			ACLGRJ,
			ACIJ,
			ACGIJ,
			ACLIJ,
			ACLGIJ,
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
			q = p
			p.Mark |= BRANCH

		default:
			q = p
		}
	}

	autosize := int32(0)
	var pLast *obj.Prog
	var pPre *obj.Prog
	var pPreempt *obj.Prog
	var pCheck *obj.Prog
	wasSplit := false
	for p := c.cursym.Func().Text; p != nil; p = p.Link {
		pLast = p
		switch p.As {
		case obj.ATEXT:
			autosize = int32(textstksiz)

			if p.Mark&LEAF != 0 && autosize == 0 {
				// A leaf function with no locals has no frame.
				p.From.Sym.Set(obj.AttrNoFrame, true)
			}

			if !p.From.Sym.NoFrame() {
				// If there is a stack frame at all, it includes
				// space to save the LR.
				autosize += int32(c.ctxt.FixedFrameSize())
			}

			if p.Mark&LEAF != 0 && autosize < objabi.StackSmall {
				// A leaf function with a small stack can be marked
				// NOSPLIT, avoiding a stack check.
				p.From.Sym.Set(obj.AttrNoSplit, true)
			}

			p.To.Offset = int64(autosize)

			q := p

			if !p.From.Sym.NoSplit() {
				p, pPreempt, pCheck = c.stacksplitPre(p, autosize) // emit pre part of split check
				pPre = p
				p = c.ctxt.EndUnsafePoint(p, c.newprog, -1)
				wasSplit = true //need post part of split
			}

			if autosize != 0 {
				// Make sure to save link register for non-empty frame, even if
				// it is a leaf function, so that traceback works.
				// Store link register before decrementing SP, so if a signal comes
				// during the execution of the function prologue, the traceback
				// code will not see a half-updated stack frame.
				// This sequence is not async preemptible, as if we open a frame
				// at the current SP, it will clobber the saved LR.
				q = c.ctxt.StartUnsafePoint(p, c.newprog)

				q = obj.Appendp(q, c.newprog)
				q.As = AMOVD
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_LR
				q.To.Type = obj.TYPE_MEM
				q.To.Reg = REGSP
				q.To.Offset = int64(-autosize)

				q = obj.Appendp(q, c.newprog)
				q.As = AMOVD
				q.From.Type = obj.TYPE_ADDR
				q.From.Offset = int64(-autosize)
				q.From.Reg = REGSP // not actually needed - REGSP is assumed if no reg is provided
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REGSP
				q.Spadj = autosize

				q = c.ctxt.EndUnsafePoint(q, c.newprog, -1)
			} else if c.cursym.Func().Text.Mark&LEAF == 0 {
				// A very few functions that do not return to their caller
				// (e.g. gogo) are not identified as leaves but still have
				// no frame.
				c.cursym.Func().Text.Mark |= LEAF
			}

			if c.cursym.Func().Text.Mark&LEAF != 0 {
				c.cursym.Set(obj.AttrLeaf, true)
				break
			}

			if c.cursym.Func().Text.From.Sym.Wrapper() {
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

				q = obj.Appendp(q, c.newprog)

				q.As = AMOVD
				q.From.Type = obj.TYPE_MEM
				q.From.Reg = REGG
				q.From.Offset = 4 * int64(c.ctxt.Arch.PtrSize) // G.panic
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R3

				q = obj.Appendp(q, c.newprog)
				q.As = ACMP
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R3
				q.To.Type = obj.TYPE_CONST
				q.To.Offset = 0

				q = obj.Appendp(q, c.newprog)
				q.As = ABEQ
				q.To.Type = obj.TYPE_BRANCH
				p1 := q

				q = obj.Appendp(q, c.newprog)
				q.As = AMOVD
				q.From.Type = obj.TYPE_MEM
				q.From.Reg = REG_R3
				q.From.Offset = 0 // Panic.argp
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R4

				q = obj.Appendp(q, c.newprog)
				q.As = AADD
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = int64(autosize) + c.ctxt.FixedFrameSize()
				q.Reg = REGSP
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R5

				q = obj.Appendp(q, c.newprog)
				q.As = ACMP
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R4
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R5

				q = obj.Appendp(q, c.newprog)
				q.As = ABNE
				q.To.Type = obj.TYPE_BRANCH
				p2 := q

				q = obj.Appendp(q, c.newprog)
				q.As = AADD
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = c.ctxt.FixedFrameSize()
				q.Reg = REGSP
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R6

				q = obj.Appendp(q, c.newprog)
				q.As = AMOVD
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R6
				q.To.Type = obj.TYPE_MEM
				q.To.Reg = REG_R3
				q.To.Offset = 0 // Panic.argp

				q = obj.Appendp(q, c.newprog)

				q.As = obj.ANOP
				p1.To.SetTarget(q)
				p2.To.SetTarget(q)
			}

		case obj.ARET:
			retTarget := p.To.Sym

			if c.cursym.Func().Text.Mark&LEAF != 0 {
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

				q = obj.Appendp(p, c.newprog)
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
				break
			}

			p.As = AMOVD
			p.From.Type = obj.TYPE_MEM
			p.From.Reg = REGSP
			p.From.Offset = 0
			p.To = obj.Addr{
				Type: obj.TYPE_REG,
				Reg:  REG_LR,
			}

			q = p

			if autosize != 0 {
				q = obj.Appendp(q, c.newprog)
				q.As = AADD
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = int64(autosize)
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REGSP
				q.Spadj = -autosize
			}

			q = obj.Appendp(q, c.newprog)
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

		case obj.AGETCALLERPC:
			if cursym.Leaf() {
				/* MOVD LR, Rd */
				p.As = AMOVD
				p.From.Type = obj.TYPE_REG
				p.From.Reg = REG_LR
			} else {
				/* MOVD (RSP), Rd */
				p.As = AMOVD
				p.From.Type = obj.TYPE_MEM
				p.From.Reg = REGSP
			}
		}

		if p.To.Type == obj.TYPE_REG && p.To.Reg == REGSP && p.Spadj == 0 {
			f := c.cursym.Func()
			if f.FuncFlag&objabi.FuncFlag_SPWRITE == 0 {
				c.cursym.Func().FuncFlag |= objabi.FuncFlag_SPWRITE
				if ctxt.Debugvlog || !ctxt.IsAsm {
					ctxt.Logf("auto-SPWRITE: %s\n", c.cursym.Name)
					if !ctxt.IsAsm {
						ctxt.Diag("invalid auto-SPWRITE in non-assembly")
						ctxt.DiagFlush()
						log.Fatalf("bad SPWRITE")
					}
				}
			}
		}
	}
	if wasSplit {
		c.stacksplitPost(pLast, pPre, pPreempt, pCheck, autosize) // emit post part of split check
	}
}

// stacksplitPre generates the function stack check prologue following
// Prog p (which should be the TEXT Prog). It returns one or two
// branch Progs that must be patched to jump to the morestack epilogue,
// and the Prog that starts the morestack check.
func (c *ctxtz) stacksplitPre(p *obj.Prog, framesize int32) (pPre, pPreempt, pCheck *obj.Prog) {
	if c.ctxt.Flag_maymorestack != "" {
		// Save LR and REGCTXT
		const frameSize = 16
		p = c.ctxt.StartUnsafePoint(p, c.newprog)
		// MOVD LR, -16(SP)
		p = obj.Appendp(p, c.newprog)
		p.As = AMOVD
		p.From = obj.Addr{Type: obj.TYPE_REG, Reg: REG_LR}
		p.To = obj.Addr{Type: obj.TYPE_MEM, Reg: REGSP, Offset: -frameSize}
		// MOVD $-16(SP), SP
		p = obj.Appendp(p, c.newprog)
		p.As = AMOVD
		p.From = obj.Addr{Type: obj.TYPE_ADDR, Offset: -frameSize, Reg: REGSP}
		p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REGSP}
		p.Spadj = frameSize
		// MOVD REGCTXT, 8(SP)
		p = obj.Appendp(p, c.newprog)
		p.As = AMOVD
		p.From = obj.Addr{Type: obj.TYPE_REG, Reg: REGCTXT}
		p.To = obj.Addr{Type: obj.TYPE_MEM, Reg: REGSP, Offset: 8}

		// BL maymorestack
		p = obj.Appendp(p, c.newprog)
		p.As = ABL
		// See ../x86/obj6.go
		sym := c.ctxt.LookupABI(c.ctxt.Flag_maymorestack, c.cursym.ABI())
		p.To = obj.Addr{Type: obj.TYPE_BRANCH, Sym: sym}

		// Restore LR and REGCTXT

		// MOVD REGCTXT, 8(SP)
		p = obj.Appendp(p, c.newprog)
		p.As = AMOVD
		p.From = obj.Addr{Type: obj.TYPE_MEM, Reg: REGSP, Offset: 8}
		p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REGCTXT}
		// MOVD (SP), LR
		p = obj.Appendp(p, c.newprog)
		p.As = AMOVD
		p.From = obj.Addr{Type: obj.TYPE_MEM, Reg: REGSP, Offset: 0}
		p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REG_LR}
		// MOVD $16(SP), SP
		p = obj.Appendp(p, c.newprog)
		p.As = AMOVD
		p.From = obj.Addr{Type: obj.TYPE_CONST, Reg: REGSP, Offset: frameSize}
		p.To = obj.Addr{Type: obj.TYPE_REG, Reg: REGSP}
		p.Spadj = -frameSize

		p = c.ctxt.EndUnsafePoint(p, c.newprog, -1)
	}

	// MOVD	g_stackguard(g), R3
	p = obj.Appendp(p, c.newprog)
	// Jump back to here after morestack returns.
	pCheck = p

	p.As = AMOVD
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = REGG
	p.From.Offset = 2 * int64(c.ctxt.Arch.PtrSize) // G.stackguard0
	if c.cursym.CFunc() {
		p.From.Offset = 3 * int64(c.ctxt.Arch.PtrSize) // G.stackguard1
	}
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_R3

	// Mark the stack bound check and morestack call async nonpreemptible.
	// If we get preempted here, when resumed the preemption request is
	// cleared, but we'll still call morestack, which will double the stack
	// unnecessarily. See issue #35470.
	p = c.ctxt.StartUnsafePoint(p, c.newprog)

	if framesize <= objabi.StackSmall {
		// small stack: SP < stackguard
		//	CMPUBGE	stackguard, SP, label-of-call-to-morestack

		p = obj.Appendp(p, c.newprog)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R3
		p.Reg = REGSP
		p.As = ACMPUBGE
		p.To.Type = obj.TYPE_BRANCH

		return p, nil, pCheck
	}

	// large stack: SP-framesize < stackguard-StackSmall

	offset := int64(framesize) - objabi.StackSmall
	if framesize > objabi.StackBig {
		// Such a large stack we need to protect against underflow.
		// The runtime guarantees SP > objabi.StackBig, but
		// framesize is large enough that SP-framesize may
		// underflow, causing a direct comparison with the
		// stack guard to incorrectly succeed. We explicitly
		// guard against underflow.
		//
		//	MOVD	$(framesize-StackSmall), R4
		//	CMPUBLT	SP, R4, label-of-call-to-morestack

		p = obj.Appendp(p, c.newprog)
		p.As = AMOVD
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = offset
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R4

		p = obj.Appendp(p, c.newprog)
		pPreempt = p
		p.As = ACMPUBLT
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REGSP
		p.Reg = REG_R4
		p.To.Type = obj.TYPE_BRANCH
	}

	// Check against the stack guard. We've ensured this won't underflow.
	//	ADD $-(framesize-StackSmall), SP, R4
	//	CMPUBGE stackguard, R4, label-of-call-to-morestack
	p = obj.Appendp(p, c.newprog)
	p.As = AADD
	p.From.Type = obj.TYPE_CONST
	p.From.Offset = -offset
	p.Reg = REGSP
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_R4

	p = obj.Appendp(p, c.newprog)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = REG_R3
	p.Reg = REG_R4
	p.As = ACMPUBGE
	p.To.Type = obj.TYPE_BRANCH

	return p, pPreempt, pCheck
}

// stacksplitPost generates the function epilogue that calls morestack
// and returns the new last instruction in the function.
//
// p is the last Prog in the function. pPre and pPreempt, if non-nil,
// are the instructions that branch to the epilogue. This will fill in
// their branch targets. pCheck is the Prog that begins the stack check.
func (c *ctxtz) stacksplitPost(p *obj.Prog, pPre, pPreempt, pCheck *obj.Prog, framesize int32) *obj.Prog {
	// Now we are at the end of the function, but logically
	// we are still in function prologue. We need to fix the
	// SP data and PCDATA.
	spfix := obj.Appendp(p, c.newprog)
	spfix.As = obj.ANOP
	spfix.Spadj = -framesize

	pcdata := c.ctxt.EmitEntryStackMap(c.cursym, spfix, c.newprog)
	pcdata = c.ctxt.StartUnsafePoint(pcdata, c.newprog)

	// MOVD	LR, R5
	p = obj.Appendp(pcdata, c.newprog)
	pPre.To.SetTarget(p)
	p.As = AMOVD
	p.From.Type = obj.TYPE_REG
	p.From.Reg = REG_LR
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_R5
	if pPreempt != nil {
		pPreempt.To.SetTarget(p)
	}

	// BL	runtime.morestack(SB)
	p = obj.Appendp(p, c.newprog)

	p.As = ABL
	p.To.Type = obj.TYPE_BRANCH
	if c.cursym.CFunc() {
		p.To.Sym = c.ctxt.Lookup("runtime.morestackc")
	} else if !c.cursym.Func().Text.From.Sym.NeedCtxt() {
		p.To.Sym = c.ctxt.Lookup("runtime.morestack_noctxt")
	} else {
		p.To.Sym = c.ctxt.Lookup("runtime.morestack")
	}

	p = c.ctxt.EndUnsafePoint(p, c.newprog, -1)

	// BR	pCheck
	p = obj.Appendp(p, c.newprog)

	p.As = ABR
	p.To.Type = obj.TYPE_BRANCH
	p.To.SetTarget(pCheck)
	return p
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
	Arch:           sys.ArchS390X,
	Init:           buildop,
	Preprocess:     preprocess,
	Assemble:       spanz,
	Progedit:       progedit,
	UnaryDst:       unaryDst,
	DWARFRegisters: S390XDWARFRegisters,
}
