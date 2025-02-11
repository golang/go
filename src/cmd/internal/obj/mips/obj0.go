// cmd/9l/noop.c, cmd/9l/pass.c, cmd/9l/span.c from Vita Nuova.
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

package mips

import (
	"cmd/internal/obj"
	"cmd/internal/sys"
	"encoding/binary"
	"fmt"
	"internal/abi"
	"log"
	"math"
)

func progedit(ctxt *obj.Link, p *obj.Prog, newprog obj.ProgAlloc) {
	c := ctxt0{ctxt: ctxt}

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
			if math.Float32bits(f32) == 0 {
				p.As = AMOVW
				p.From.Type = obj.TYPE_REG
				p.From.Reg = REGZERO
				break
			}
			p.From.Type = obj.TYPE_MEM
			p.From.Sym = ctxt.Float32Sym(f32)
			p.From.Name = obj.NAME_EXTERN
			p.From.Offset = 0
		}

	case AMOVD:
		if p.From.Type == obj.TYPE_FCONST {
			f64 := p.From.Val.(float64)
			if math.Float64bits(f64) == 0 && c.ctxt.Arch.Family == sys.MIPS64 {
				p.As = AMOVV
				p.From.Type = obj.TYPE_REG
				p.From.Reg = REGZERO
				break
			}
			p.From.Type = obj.TYPE_MEM
			p.From.Sym = ctxt.Float64Sym(f64)
			p.From.Name = obj.NAME_EXTERN
			p.From.Offset = 0
		}

		// Put >32-bit constants in memory and load them
	case AMOVV:
		if p.From.Type == obj.TYPE_CONST && p.From.Name == obj.NAME_NONE && p.From.Reg == 0 && int64(int32(p.From.Offset)) != p.From.Offset {
			p.From.Type = obj.TYPE_MEM
			p.From.Sym = ctxt.Int64Sym(p.From.Offset)
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

func preprocess(ctxt *obj.Link, cursym *obj.LSym, newprog obj.ProgAlloc) {
	// TODO(minux): add morestack short-cuts with small fixed frame-size.
	c := ctxt0{ctxt: ctxt, newprog: newprog, cursym: cursym}

	// a switch for enabling/disabling instruction scheduling
	nosched := true

	if c.cursym.Func().Text == nil || c.cursym.Func().Text.Link == nil {
		return
	}

	p := c.cursym.Func().Text
	textstksiz := p.To.Offset
	if textstksiz == -ctxt.Arch.FixedFrameSize {
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

	c.cursym.Func().Args = p.To.Val.(int32)
	c.cursym.Func().Locals = int32(textstksiz)

	/*
	 * find leaf subroutines
	 * expand RET
	 * expand BECOME pseudo
	 */

	for p := c.cursym.Func().Text; p != nil; p = p.Link {
		switch p.As {
		/* too hard, just leave alone */
		case obj.ATEXT:
			p.Mark |= LABEL | LEAF | SYNC
			if p.Link != nil {
				p.Link.Mark |= LABEL
			}

		/* too hard, just leave alone */
		case AMOVW,
			AMOVV:
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
			p.Mark |= LABEL | SYNC

		case ANOR:
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
			c.cursym.Func().Text.Mark &^= LEAF
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
			q1 := p.To.Target()
			if q1 != nil {
				for q1.As == obj.ANOP {
					q1 = q1.Link
					p.To.SetTarget(q1)
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

		case ARET:
			if p.Link != nil {
				p.Link.Mark |= LABEL
			}
		}
	}

	var mov, add obj.As
	if c.ctxt.Arch.Family == sys.MIPS64 {
		add = AADDV
		mov = AMOVV
	} else {
		add = AADDU
		mov = AMOVW
	}

	var q *obj.Prog
	var q1 *obj.Prog
	autosize := int32(0)
	var p1 *obj.Prog
	var p2 *obj.Prog
	for p := c.cursym.Func().Text; p != nil; p = p.Link {
		o := p.As
		switch o {
		case obj.ATEXT:
			autosize = int32(textstksiz)

			if p.Mark&LEAF != 0 && autosize == 0 {
				// A leaf function with no locals has no frame.
				p.From.Sym.Set(obj.AttrNoFrame, true)
			}

			if !p.From.Sym.NoFrame() {
				// If there is a stack frame at all, it includes
				// space to save the LR.
				autosize += int32(c.ctxt.Arch.FixedFrameSize)
			}

			if autosize&4 != 0 && c.ctxt.Arch.Family == sys.MIPS64 {
				autosize += 4
			}

			if autosize == 0 && c.cursym.Func().Text.Mark&LEAF == 0 {
				if c.cursym.Func().Text.From.Sym.NoSplit() {
					if ctxt.Debugvlog {
						ctxt.Logf("save suppressed in: %s\n", c.cursym.Name)
					}

					c.cursym.Func().Text.Mark |= LEAF
				}
			}

			p.To.Offset = int64(autosize) - ctxt.Arch.FixedFrameSize

			if c.cursym.Func().Text.Mark&LEAF != 0 {
				c.cursym.Set(obj.AttrLeaf, true)
				if p.From.Sym.NoFrame() {
					break
				}
			}

			if !p.From.Sym.NoSplit() {
				p = c.stacksplit(p, autosize) // emit split check
			}

			q = p

			if autosize != 0 {
				// Make sure to save link register for non-empty frame, even if
				// it is a leaf function, so that traceback works.
				// Store link register before decrement SP, so if a signal comes
				// during the execution of the function prologue, the traceback
				// code will not see a half-updated stack frame.
				// This sequence is not async preemptible, as if we open a frame
				// at the current SP, it will clobber the saved LR.
				q = c.ctxt.StartUnsafePoint(q, c.newprog)

				q = obj.Appendp(q, newprog)
				q.As = mov
				q.Pos = p.Pos
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REGLINK
				q.To.Type = obj.TYPE_MEM
				q.To.Offset = int64(-autosize)
				q.To.Reg = REGSP

				q = obj.Appendp(q, newprog)
				q.As = add
				q.Pos = p.Pos
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = int64(-autosize)
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REGSP
				q.Spadj = +autosize

				q = c.ctxt.EndUnsafePoint(q, c.newprog, -1)

				// On Linux, in a cgo binary we may get a SIGSETXID signal early on
				// before the signal stack is set, as glibc doesn't allow us to block
				// SIGSETXID. So a signal may land on the current stack and clobber
				// the content below the SP. We store the LR again after the SP is
				// decremented.
				q = obj.Appendp(q, newprog)
				q.As = mov
				q.Pos = p.Pos
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REGLINK
				q.To.Type = obj.TYPE_MEM
				q.To.Offset = 0
				q.To.Reg = REGSP
			}

			if c.cursym.Func().Text.From.Sym.Wrapper() && c.cursym.Func().Text.Mark&LEAF == 0 {
				// if(g->panic != nil && g->panic->argp == FP) g->panic->argp = bottom-of-frame
				//
				//	MOV	g_panic(g), R1
				//	BEQ	R1, end
				//	MOV	panic_argp(R1), R2
				//	ADD	$(autosize+FIXED_FRAME), R29, R3
				//	BNE	R2, R3, end
				//	ADD	$FIXED_FRAME, R29, R2
				//	MOV	R2, panic_argp(R1)
				// end:
				//	NOP
				//
				// The NOP is needed to give the jumps somewhere to land.
				// It is a liblink NOP, not an mips NOP: it encodes to 0 instruction bytes.
				//
				// We don't generate this for leafs because that means the wrapped
				// function was inlined into the wrapper.

				q = obj.Appendp(q, newprog)

				q.As = mov
				q.From.Type = obj.TYPE_MEM
				q.From.Reg = REGG
				q.From.Offset = 4 * int64(c.ctxt.Arch.PtrSize) // G.panic
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R1

				q = obj.Appendp(q, newprog)
				q.As = ABEQ
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R1
				q.To.Type = obj.TYPE_BRANCH
				q.Mark |= BRANCH
				p1 = q

				q = obj.Appendp(q, newprog)
				q.As = mov
				q.From.Type = obj.TYPE_MEM
				q.From.Reg = REG_R1
				q.From.Offset = 0 // Panic.argp
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R2

				q = obj.Appendp(q, newprog)
				q.As = add
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = int64(autosize) + ctxt.Arch.FixedFrameSize
				q.Reg = REGSP
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R3

				q = obj.Appendp(q, newprog)
				q.As = ABNE
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R2
				q.Reg = REG_R3
				q.To.Type = obj.TYPE_BRANCH
				q.Mark |= BRANCH
				p2 = q

				q = obj.Appendp(q, newprog)
				q.As = add
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = ctxt.Arch.FixedFrameSize
				q.Reg = REGSP
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R2

				q = obj.Appendp(q, newprog)
				q.As = mov
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R2
				q.To.Type = obj.TYPE_MEM
				q.To.Reg = REG_R1
				q.To.Offset = 0 // Panic.argp

				q = obj.Appendp(q, newprog)

				q.As = obj.ANOP
				p1.To.SetTarget(q)
				p2.To.SetTarget(q)
			}

		case ARET:
			if p.From.Type == obj.TYPE_CONST {
				ctxt.Diag("using BECOME (%v) is not supported!", p)
				break
			}

			retSym := p.To.Sym
			p.To.Name = obj.NAME_NONE // clear fields as we may modify p to other instruction
			p.To.Sym = nil

			if c.cursym.Func().Text.Mark&LEAF != 0 {
				if autosize == 0 {
					p.As = AJMP
					p.From = obj.Addr{}
					if retSym != nil { // retjmp
						p.To.Type = obj.TYPE_BRANCH
						p.To.Name = obj.NAME_EXTERN
						p.To.Sym = retSym
					} else {
						p.To.Type = obj.TYPE_MEM
						p.To.Reg = REGLINK
						p.To.Offset = 0
					}
					p.Mark |= BRANCH
					break
				}

				p.As = add
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = int64(autosize)
				p.To.Type = obj.TYPE_REG
				p.To.Reg = REGSP
				p.Spadj = -autosize

				q = c.newprog()
				q.As = AJMP
				q.Pos = p.Pos
				if retSym != nil { // retjmp
					q.To.Type = obj.TYPE_BRANCH
					q.To.Name = obj.NAME_EXTERN
					q.To.Sym = retSym
				} else {
					q.To.Type = obj.TYPE_MEM
					q.To.Reg = REGLINK
					q.To.Offset = 0
				}
				q.Mark |= BRANCH
				q.Spadj = +autosize

				q.Link = p.Link
				p.Link = q
				break
			}

			p.As = mov
			p.From.Type = obj.TYPE_MEM
			p.From.Offset = 0
			p.From.Reg = REGSP
			p.To.Type = obj.TYPE_REG
			p.To.Reg = REGLINK

			if autosize != 0 {
				q = c.newprog()
				q.As = add
				q.Pos = p.Pos
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = int64(autosize)
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REGSP
				q.Spadj = -autosize

				q.Link = p.Link
				p.Link = q
			}

			q1 = c.newprog()
			q1.As = AJMP
			q1.Pos = p.Pos
			if retSym != nil { // retjmp
				q1.To.Type = obj.TYPE_BRANCH
				q1.To.Name = obj.NAME_EXTERN
				q1.To.Sym = retSym
			} else {
				q1.To.Type = obj.TYPE_MEM
				q1.To.Offset = 0
				q1.To.Reg = REGLINK
			}
			q1.Mark |= BRANCH
			q1.Spadj = +autosize

			q1.Link = q.Link
			q.Link = q1

		case AADD,
			AADDU,
			AADDV,
			AADDVU:
			if p.To.Type == obj.TYPE_REG && p.To.Reg == REGSP && p.From.Type == obj.TYPE_CONST {
				p.Spadj = int32(-p.From.Offset)
			}

		case obj.AGETCALLERPC:
			if cursym.Leaf() {
				/* MOV LR, Rd */
				p.As = mov
				p.From.Type = obj.TYPE_REG
				p.From.Reg = REGLINK
			} else {
				/* MOV (RSP), Rd */
				p.As = mov
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

	if c.ctxt.Arch.Family == sys.MIPS {
		// rewrite MOVD into two MOVF in 32-bit mode to avoid unaligned memory access
		for p = c.cursym.Func().Text; p != nil; p = p1 {
			p1 = p.Link

			if p.As != AMOVD {
				continue
			}
			if p.From.Type != obj.TYPE_MEM && p.To.Type != obj.TYPE_MEM {
				continue
			}

			p.As = AMOVF
			q = c.newprog()
			*q = *p
			q.Link = p.Link
			p.Link = q
			p1 = q.Link

			var addrOff int64
			if c.ctxt.Arch.ByteOrder == binary.BigEndian {
				addrOff = 4 // swap load/save order
			}
			if p.From.Type == obj.TYPE_MEM {
				reg := REG_F0 + (p.To.Reg-REG_F0)&^1
				p.To.Reg = reg
				q.To.Reg = reg + 1
				p.From.Offset += addrOff
				q.From.Offset += 4 - addrOff
			} else if p.To.Type == obj.TYPE_MEM {
				reg := REG_F0 + (p.From.Reg-REG_F0)&^1
				p.From.Reg = reg
				q.From.Reg = reg + 1
				p.To.Offset += addrOff
				q.To.Offset += 4 - addrOff
			}
		}
	}

	if nosched {
		// if we don't do instruction scheduling, simply add
		// NOP after each branch instruction.
		for p = c.cursym.Func().Text; p != nil; p = p.Link {
			if p.Mark&BRANCH != 0 {
				c.addnop(p)
			}
		}
		return
	}

	// instruction scheduling
	q = nil                   // p - 1
	q1 = c.cursym.Func().Text // top of block
	o := 0                    // count of instructions
	for p = c.cursym.Func().Text; p != nil; p = p1 {
		p1 = p.Link
		o++
		if p.Mark&NOSCHED != 0 {
			if q1 != p {
				c.sched(q1, q)
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
				c.sched(q1, q)
			}
			q1 = p
			o = 1
		}
		if p.Mark&(BRANCH|SYNC) != 0 {
			c.sched(q1, p)
			q1 = p1
			o = 0
		}
		if o >= NSCHED {
			c.sched(q1, p)
			q1 = p1
			o = 0
		}
		q = p
	}
}

func (c *ctxt0) stacksplit(p *obj.Prog, framesize int32) *obj.Prog {
	var mov, add obj.As

	if c.ctxt.Arch.Family == sys.MIPS64 {
		add = AADDV
		mov = AMOVV
	} else {
		add = AADDU
		mov = AMOVW
	}

	if c.ctxt.Flag_maymorestack != "" {
		// Save LR and REGCTXT.
		frameSize := 2 * c.ctxt.Arch.PtrSize

		p = c.ctxt.StartUnsafePoint(p, c.newprog)

		// MOV	REGLINK, -8/-16(SP)
		p = obj.Appendp(p, c.newprog)
		p.As = mov
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REGLINK
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = int64(-frameSize)
		p.To.Reg = REGSP

		// MOV	REGCTXT, -4/-8(SP)
		p = obj.Appendp(p, c.newprog)
		p.As = mov
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REGCTXT
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = -int64(c.ctxt.Arch.PtrSize)
		p.To.Reg = REGSP

		// ADD	$-8/$-16, SP
		p = obj.Appendp(p, c.newprog)
		p.As = add
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(-frameSize)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REGSP
		p.Spadj = int32(frameSize)

		// JAL	maymorestack
		p = obj.Appendp(p, c.newprog)
		p.As = AJAL
		p.To.Type = obj.TYPE_BRANCH
		// See ../x86/obj6.go
		p.To.Sym = c.ctxt.LookupABI(c.ctxt.Flag_maymorestack, c.cursym.ABI())
		p.Mark |= BRANCH

		// Restore LR and REGCTXT.

		// MOV	0(SP), REGLINK
		p = obj.Appendp(p, c.newprog)
		p.As = mov
		p.From.Type = obj.TYPE_MEM
		p.From.Offset = 0
		p.From.Reg = REGSP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REGLINK

		// MOV	4/8(SP), REGCTXT
		p = obj.Appendp(p, c.newprog)
		p.As = mov
		p.From.Type = obj.TYPE_MEM
		p.From.Offset = int64(c.ctxt.Arch.PtrSize)
		p.From.Reg = REGSP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REGCTXT

		// ADD	$8/$16, SP
		p = obj.Appendp(p, c.newprog)
		p.As = add
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(frameSize)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REGSP
		p.Spadj = int32(-frameSize)

		p = c.ctxt.EndUnsafePoint(p, c.newprog, -1)
	}

	// Jump back to here after morestack returns.
	startPred := p

	// MOV	g_stackguard(g), R1
	p = obj.Appendp(p, c.newprog)

	p.As = mov
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = REGG
	p.From.Offset = 2 * int64(c.ctxt.Arch.PtrSize) // G.stackguard0
	if c.cursym.CFunc() {
		p.From.Offset = 3 * int64(c.ctxt.Arch.PtrSize) // G.stackguard1
	}
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_R1

	var q *obj.Prog
	if framesize <= abi.StackSmall {
		// small stack: SP < stackguard
		//	AGTU	SP, stackguard, R1
		p = obj.Appendp(p, c.newprog)

		p.As = ASGTU
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REGSP
		p.Reg = REG_R1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R1
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
			//	SGTU	$(framesize-StackSmall), SP, R2
			//	BNE	R2, label-of-call-to-morestack

			p = obj.Appendp(p, c.newprog)
			p.As = ASGTU
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = offset
			p.Reg = REGSP
			p.To.Type = obj.TYPE_REG
			p.To.Reg = REG_R2

			p = obj.Appendp(p, c.newprog)
			q = p
			p.As = ABNE
			p.From.Type = obj.TYPE_REG
			p.From.Reg = REG_R2
			p.To.Type = obj.TYPE_BRANCH
			p.Mark |= BRANCH
		}

		// Check against the stack guard. We've ensured this won't underflow.
		//	ADD	$-(framesize-StackSmall), SP, R2
		//	SGTU	R2, stackguard, R1
		p = obj.Appendp(p, c.newprog)

		p.As = add
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = -offset
		p.Reg = REGSP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R2

		p = obj.Appendp(p, c.newprog)
		p.As = ASGTU
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R2
		p.Reg = REG_R1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R1
	}

	// q1: BNE	R1, done
	p = obj.Appendp(p, c.newprog)
	q1 := p

	p.As = ABNE
	p.From.Type = obj.TYPE_REG
	p.From.Reg = REG_R1
	p.To.Type = obj.TYPE_BRANCH
	p.Mark |= BRANCH

	// MOV	LINK, R3
	p = obj.Appendp(p, c.newprog)

	p.As = mov
	p.From.Type = obj.TYPE_REG
	p.From.Reg = REGLINK
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_R3
	if q != nil {
		q.To.SetTarget(p)
		p.Mark |= LABEL
	}

	p = c.ctxt.EmitEntryStackMap(c.cursym, p, c.newprog)

	// JAL	runtime.morestack(SB)
	p = obj.Appendp(p, c.newprog)

	p.As = AJAL
	p.To.Type = obj.TYPE_BRANCH
	if c.cursym.CFunc() {
		p.To.Sym = c.ctxt.Lookup("runtime.morestackc")
	} else if !c.cursym.Func().Text.From.Sym.NeedCtxt() {
		p.To.Sym = c.ctxt.Lookup("runtime.morestack_noctxt")
	} else {
		p.To.Sym = c.ctxt.Lookup("runtime.morestack")
	}
	p.Mark |= BRANCH

	// JMP	start
	p = obj.Appendp(p, c.newprog)

	p.As = AJMP
	p.To.Type = obj.TYPE_BRANCH
	p.To.SetTarget(startPred.Link)
	startPred.Link.Mark |= LABEL
	p.Mark |= BRANCH

	// placeholder for q1's jump target
	p = obj.Appendp(p, c.newprog)

	p.As = obj.ANOP // zero-width place holder
	q1.To.SetTarget(p)

	return p
}

func (c *ctxt0) addnop(p *obj.Prog) {
	q := c.newprog()
	q.As = ANOOP
	q.Pos = p.Pos
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

func (c *ctxt0) sched(p0, pe *obj.Prog) {
	var sch [NSCHED]Sch

	/*
	 * build side structure
	 */
	s := sch[:]
	for p := p0; ; p = p.Link {
		s[0].p = *p
		c.markregused(&s[0])
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
					continue
				}
			}
			if t[0].p.Mark&DELAY != 0 {
				if -cap(s) >= -cap(se) || conflict(&t[0], &s[1]) {
					continue
				}
			}
			for u := t[1:]; -cap(u) <= -cap(s); u = u[1:] {
				if c.depend(&u[0], &t[0]) {
					continue
				}
			}
			goto out2
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
			c.addnop(p)
		}
	}
}

func (c *ctxt0) markregused(s *Sch) {
	p := &s.p
	s.comp = c.compound(p)
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
		c.autosize = int32(p.To.Offset + 8)
		ad = 1

	case AJAL:
		r := p.Reg
		if r == 0 {
			r = REGLINK
		}
		s.set.ireg |= 1 << uint(r-REG_R0)
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
	cls := int(p.To.Class)
	if cls == 0 {
		cls = c.aclass(&p.To) + 1
		p.To.Class = int8(cls)
	}
	cls--
	switch cls {
	default:
		fmt.Printf("unknown class %d %v\n", cls, p)

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
		cls = int(p.To.Reg)
		s.used.ireg |= 1 << uint(cls-REG_R0)
		if ad != 0 {
			break
		}
		s.size = uint8(sz)
		s.soffset = c.regoff(&p.To)

		m := uint32(ANYMEM)
		if cls == REGSB {
			m = E_MEMSB
		}
		if cls == REGSP {
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
		s.soffset = c.regoff(&p.To)

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
		s.soffset = c.regoff(&p.To)

		if ar != 0 {
			s.used.cc |= E_MEMSB
		} else {
			s.set.cc |= E_MEMSB
		}
	}

	/*
	 * flags based on 'from' field
	 */
	cls = int(p.From.Class)
	if cls == 0 {
		cls = c.aclass(&p.From) + 1
		p.From.Class = int8(cls)
	}
	cls--
	switch cls {
	default:
		fmt.Printf("unknown class %d %v\n", cls, p)

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
		cls = int(p.From.Reg)
		s.used.ireg |= 1 << uint(cls-REG_R0)
		if ld != 0 {
			p.Mark |= LOAD
		}
		s.size = uint8(sz)
		s.soffset = c.regoff(&p.From)

		m := uint32(ANYMEM)
		if cls == REGSB {
			m = E_MEMSB
		}
		if cls == REGSP {
			m = E_MEMSP
		}

		s.used.cc |= m

	case C_SACON,
		C_LACON:
		cls = int(p.From.Reg)
		if cls == 0 {
			cls = REGSP
		}
		s.used.ireg |= 1 << uint(cls-REG_R0)

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
		s.soffset = c.regoff(&p.From)

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
		s.soffset = c.regoff(&p.From)

		s.used.cc |= E_MEMSB
	}

	cls = int(p.Reg)
	if cls != 0 {
		if REG_F0 <= cls && cls <= REG_F31 {
			s.used.freg |= 1 << uint(cls-REG_F0)
		} else {
			s.used.ireg |= 1 << uint(cls-REG_R0)
		}
	}
	s.set.ireg &^= (1 << (REGZERO - REG_R0)) /* R0 can't be set */
}

/*
 * test to see if two instructions can be
 * interchanged without changing semantics
 */
func (c *ctxt0) depend(sa, sb *Sch) bool {
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
			if c.regoff(&sa.p.From) == c.regoff(&sb.p.From) {
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

func (c *ctxt0) compound(p *obj.Prog) bool {
	o := c.oplook(p)
	if o.size != 4 {
		return true
	}
	if p.To.Type == obj.TYPE_REG && p.To.Reg == REGSB {
		return true
	}
	return false
}

var Linkmips64 = obj.LinkArch{
	Arch:           sys.ArchMIPS64,
	Init:           buildop,
	Preprocess:     preprocess,
	Assemble:       span0,
	Progedit:       progedit,
	DWARFRegisters: MIPSDWARFRegisters,
}

var Linkmips64le = obj.LinkArch{
	Arch:           sys.ArchMIPS64LE,
	Init:           buildop,
	Preprocess:     preprocess,
	Assemble:       span0,
	Progedit:       progedit,
	DWARFRegisters: MIPSDWARFRegisters,
}

var Linkmips = obj.LinkArch{
	Arch:           sys.ArchMIPS,
	Init:           buildop,
	Preprocess:     preprocess,
	Assemble:       span0,
	Progedit:       progedit,
	DWARFRegisters: MIPSDWARFRegisters,
}

var Linkmipsle = obj.LinkArch{
	Arch:           sys.ArchMIPSLE,
	Init:           buildop,
	Preprocess:     preprocess,
	Assemble:       span0,
	Progedit:       progedit,
	DWARFRegisters: MIPSDWARFRegisters,
}
