// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loong64

import (
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"cmd/internal/sys"
	"internal/abi"
	"log"
	"math"
)

func progedit(ctxt *obj.Link, p *obj.Prog, newprog obj.ProgAlloc) {
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
			if math.Float64bits(f64) == 0 {
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

	if ctxt.Flag_dynlink {
		rewriteToUseGot(ctxt, p, newprog)
	}
}

func rewriteToUseGot(ctxt *obj.Link, p *obj.Prog, newprog obj.ProgAlloc) {
	//     ADUFFxxx $offset
	// becomes
	//     MOVV runtime.duffxxx@GOT, REGTMP
	//     ADD $offset, REGTMP
	//     JAL REGTMP
	if p.As == obj.ADUFFCOPY || p.As == obj.ADUFFZERO {
		var sym *obj.LSym
		if p.As == obj.ADUFFZERO {
			sym = ctxt.Lookup("runtime.duffzero")
		} else {
			sym = ctxt.Lookup("runtime.duffcopy")
		}
		offset := p.To.Offset
		p.As = AMOVV
		p.From.Type = obj.TYPE_MEM
		p.From.Sym = sym
		p.From.Name = obj.NAME_GOTREF
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REGTMP
		p.To.Name = obj.NAME_NONE
		p.To.Offset = 0
		p.To.Sym = nil
		p1 := obj.Appendp(p, newprog)
		p1.As = AADDV
		p1.From.Type = obj.TYPE_CONST
		p1.From.Offset = offset
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = REGTMP
		p2 := obj.Appendp(p1, newprog)
		p2.As = AJAL
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = REGTMP
	}

	// We only care about global data: NAME_EXTERN means a global
	// symbol in the Go sense, and p.Sym.Local is true for a few
	// internally defined symbols.
	if p.From.Type == obj.TYPE_ADDR && p.From.Name == obj.NAME_EXTERN && !p.From.Sym.Local() {
		// MOVV $sym, Rx becomes MOVV sym@GOT, Rx
		// MOVV $sym+<off>, Rx becomes MOVV sym@GOT, Rx; ADD <off>, Rx
		if p.As != AMOVV {
			ctxt.Diag("do not know how to handle TYPE_ADDR in %v with -shared", p)
		}
		if p.To.Type != obj.TYPE_REG {
			ctxt.Diag("do not know how to handle LEAQ-type insn to non-register in %v with -shared", p)
		}
		p.From.Type = obj.TYPE_MEM
		p.From.Name = obj.NAME_GOTREF
		if p.From.Offset != 0 {
			q := obj.Appendp(p, newprog)
			q.As = AADDV
			q.From.Type = obj.TYPE_CONST
			q.From.Offset = p.From.Offset
			q.To = p.To
			p.From.Offset = 0
		}
	}
	if p.GetFrom3() != nil && p.GetFrom3().Name == obj.NAME_EXTERN {
		ctxt.Diag("don't know how to handle %v with -shared", p)
	}

	var source *obj.Addr
	// MOVx sym, Ry becomes MOVV sym@GOT, REGTMP; MOVx (REGTMP), Ry
	// MOVx Ry, sym becomes MOVV sym@GOT, REGTMP; MOVx Ry, (REGTMP)
	// An addition may be inserted between the two MOVs if there is an offset.
	if p.From.Name == obj.NAME_EXTERN && !p.From.Sym.Local() {
		if p.To.Name == obj.NAME_EXTERN && !p.To.Sym.Local() {
			ctxt.Diag("cannot handle NAME_EXTERN on both sides in %v with -shared", p)
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
		ctxt.Diag("don't know how to handle %v with -shared", p)
	}
	p1 := obj.Appendp(p, newprog)
	p2 := obj.Appendp(p1, newprog)
	p1.As = AMOVV
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
	c := ctxt0{ctxt: ctxt, newprog: newprog, cursym: cursym}

	p := c.cursym.Func().Text
	textstksiz := p.To.Offset

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
	 */

	for p := c.cursym.Func().Text; p != nil; p = p.Link {
		switch p.As {
		case obj.ATEXT:
			p.Mark |= LABEL | LEAF | SYNC
			if p.Link != nil {
				p.Link.Mark |= LABEL
			}

		case AMOVW,
			AMOVV:
			if p.To.Type == obj.TYPE_REG && p.To.Reg >= REG_SPECIAL {
				p.Mark |= LABEL | SYNC
				break
			}
			if p.From.Type == obj.TYPE_REG && p.From.Reg >= REG_SPECIAL {
				p.Mark |= LABEL | SYNC
			}

		case ASYSCALL,
			AWORD:
			p.Mark |= LABEL | SYNC

		case ANOR:
			if p.To.Type == obj.TYPE_REG {
				if p.To.Reg == REGZERO {
					p.Mark |= LABEL | SYNC
				}
			}

		case AJAL,
			obj.ADUFFZERO,
			obj.ADUFFCOPY:
			c.cursym.Func().Text.Mark &^= LEAF
			fallthrough

		case AJMP,
			ABEQ,
			ABGEU,
			ABLTU,
			ABLTZ,
			ABNE,
			ABFPT, ABFPF:
			p.Mark |= BRANCH
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

	add = AADDV
	mov = AMOVV

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

			if autosize&4 != 0 {
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
				q.Pos = q.Pos.WithXlogue(src.PosPrologueEnd)
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
				//	MOV	g_panic(g), R20
				//	BEQ	R20, end
				//	MOV	panic_argp(R20), R24
				//	ADD	$(autosize+FIXED_FRAME), R3, R30
				//	BNE	R24, R30, end
				//	ADD	$FIXED_FRAME, R3, R24
				//	MOV	R24, panic_argp(R20)
				// end:
				//	NOP
				//
				// The NOP is needed to give the jumps somewhere to land.
				// It is a liblink NOP, not a hardware NOP: it encodes to 0 instruction bytes.
				//
				// We don't generate this for leaves because that means the wrapped
				// function was inlined into the wrapper.

				q = obj.Appendp(q, newprog)

				q.As = mov
				q.From.Type = obj.TYPE_MEM
				q.From.Reg = REGG
				q.From.Offset = 4 * int64(c.ctxt.Arch.PtrSize) // G.panic
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R20

				q = obj.Appendp(q, newprog)
				q.As = ABEQ
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R20
				q.To.Type = obj.TYPE_BRANCH
				q.Mark |= BRANCH
				p1 = q

				q = obj.Appendp(q, newprog)
				q.As = mov
				q.From.Type = obj.TYPE_MEM
				q.From.Reg = REG_R20
				q.From.Offset = 0 // Panic.argp
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R24

				q = obj.Appendp(q, newprog)
				q.As = add
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = int64(autosize) + ctxt.Arch.FixedFrameSize
				q.Reg = REGSP
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R30

				q = obj.Appendp(q, newprog)
				q.As = ABNE
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R24
				q.Reg = REG_R30
				q.To.Type = obj.TYPE_BRANCH
				q.Mark |= BRANCH
				p2 = q

				q = obj.Appendp(q, newprog)
				q.As = add
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = ctxt.Arch.FixedFrameSize
				q.Reg = REGSP
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R24

				q = obj.Appendp(q, newprog)
				q.As = mov
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R24
				q.To.Type = obj.TYPE_MEM
				q.To.Reg = REG_R20
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
					q.To.Offset = 0
					q.To.Reg = REGLINK
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
				// MOV LR, Rd
				p.As = mov
				p.From.Type = obj.TYPE_REG
				p.From.Reg = REGLINK
			} else {
				// MOV (RSP), Rd
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
}

func (c *ctxt0) stacksplit(p *obj.Prog, framesize int32) *obj.Prog {
	var mov, add obj.As

	add = AADDV
	mov = AMOVV
	if c.ctxt.Flag_maymorestack != "" {
		// Save LR and REGCTXT.
		frameSize := 2 * c.ctxt.Arch.PtrSize

		p = c.ctxt.StartUnsafePoint(p, c.newprog)

		// Spill Arguments. This has to happen before we open
		// any more frame space.
		p = c.cursym.Func().SpillRegisterArgs(p, c.newprog)

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

		// Unspill arguments
		p = c.cursym.Func().UnspillRegisterArgs(p, c.newprog)
		p = c.ctxt.EndUnsafePoint(p, c.newprog, -1)
	}

	// Jump back to here after morestack returns.
	startPred := p

	// MOV	g_stackguard(g), R20
	p = obj.Appendp(p, c.newprog)

	p.As = mov
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = REGG
	p.From.Offset = 2 * int64(c.ctxt.Arch.PtrSize) // G.stackguard0
	if c.cursym.CFunc() {
		p.From.Offset = 3 * int64(c.ctxt.Arch.PtrSize) // G.stackguard1
	}
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_R20

	// Mark the stack bound check and morestack call async nonpreemptible.
	// If we get preempted here, when resumed the preemption request is
	// cleared, but we'll still call morestack, which will double the stack
	// unnecessarily. See issue #35470.
	p = c.ctxt.StartUnsafePoint(p, c.newprog)

	var q *obj.Prog
	if framesize <= abi.StackSmall {
		// small stack: SP < stackguard
		//	AGTU	SP, stackguard, R20
		p = obj.Appendp(p, c.newprog)

		p.As = ASGTU
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REGSP
		p.Reg = REG_R20
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R20
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
			//      SGTU    $(framesize-StackSmall), SP, R24
			//      BNE     R24, label-of-call-to-morestack

			p = obj.Appendp(p, c.newprog)
			p.As = ASGTU
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = offset
			p.Reg = REGSP
			p.To.Type = obj.TYPE_REG
			p.To.Reg = REG_R24

			p = obj.Appendp(p, c.newprog)
			q = p
			p.As = ABNE
			p.From.Type = obj.TYPE_REG
			p.From.Reg = REG_R24
			p.To.Type = obj.TYPE_BRANCH
			p.Mark |= BRANCH
		}

		p = obj.Appendp(p, c.newprog)

		p.As = add
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = -offset
		p.Reg = REGSP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R24

		p = obj.Appendp(p, c.newprog)
		p.As = ASGTU
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R24
		p.Reg = REG_R20
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R20
	}

	// q1: BNE	R20, done
	p = obj.Appendp(p, c.newprog)
	q1 := p

	p.As = ABNE
	p.From.Type = obj.TYPE_REG
	p.From.Reg = REG_R20
	p.To.Type = obj.TYPE_BRANCH
	p.Mark |= BRANCH

	// MOV	LINK, R31
	p = obj.Appendp(p, c.newprog)

	p.As = mov
	p.From.Type = obj.TYPE_REG
	p.From.Reg = REGLINK
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_R31
	if q != nil {
		q.To.SetTarget(p)
		p.Mark |= LABEL
	}

	p = c.ctxt.EmitEntryStackMap(c.cursym, p, c.newprog)

	// Spill the register args that could be clobbered by the
	// morestack code
	p = c.cursym.Func().SpillRegisterArgs(p, c.newprog)

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

	p = c.cursym.Func().UnspillRegisterArgs(p, c.newprog)
	p = c.ctxt.EndUnsafePoint(p, c.newprog, -1)

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

var Linkloong64 = obj.LinkArch{
	Arch:           sys.ArchLoong64,
	Init:           buildop,
	Preprocess:     preprocess,
	Assemble:       span0,
	Progedit:       progedit,
	DWARFRegisters: LOONG64DWARFRegisters,
}
