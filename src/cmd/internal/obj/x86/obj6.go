// Inferno utils/6l/pass.c
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/6l/pass.c
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

package x86

import (
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"cmd/internal/sys"
	"math"
	"strings"
)

func CanUse1InsnTLS(ctxt *obj.Link) bool {
	if isAndroid {
		// Android uses a global variable for the tls offset.
		return false
	}

	if ctxt.Arch.Family == sys.I386 {
		switch ctxt.Headtype {
		case objabi.Hlinux,
			objabi.Hnacl,
			objabi.Hplan9,
			objabi.Hwindows:
			return false
		}

		return true
	}

	switch ctxt.Headtype {
	case objabi.Hplan9, objabi.Hwindows:
		return false
	case objabi.Hlinux, objabi.Hfreebsd:
		return !ctxt.Flag_shared
	}

	return true
}

func progedit(ctxt *obj.Link, p *obj.Prog, newprog obj.ProgAlloc) {
	// Thread-local storage references use the TLS pseudo-register.
	// As a register, TLS refers to the thread-local storage base, and it
	// can only be loaded into another register:
	//
	//         MOVQ TLS, AX
	//
	// An offset from the thread-local storage base is written off(reg)(TLS*1).
	// Semantically it is off(reg), but the (TLS*1) annotation marks this as
	// indexing from the loaded TLS base. This emits a relocation so that
	// if the linker needs to adjust the offset, it can. For example:
	//
	//         MOVQ TLS, AX
	//         MOVQ 0(AX)(TLS*1), CX // load g into CX
	//
	// On systems that support direct access to the TLS memory, this
	// pair of instructions can be reduced to a direct TLS memory reference:
	//
	//         MOVQ 0(TLS), CX // load g into CX
	//
	// The 2-instruction and 1-instruction forms correspond to the two code
	// sequences for loading a TLS variable in the local exec model given in "ELF
	// Handling For Thread-Local Storage".
	//
	// We apply this rewrite on systems that support the 1-instruction form.
	// The decision is made using only the operating system and the -shared flag,
	// not the link mode. If some link modes on a particular operating system
	// require the 2-instruction form, then all builds for that operating system
	// will use the 2-instruction form, so that the link mode decision can be
	// delayed to link time.
	//
	// In this way, all supported systems use identical instructions to
	// access TLS, and they are rewritten appropriately first here in
	// liblink and then finally using relocations in the linker.
	//
	// When -shared is passed, we leave the code in the 2-instruction form but
	// assemble (and relocate) them in different ways to generate the initial
	// exec code sequence. It's a bit of a fluke that this is possible without
	// rewriting the instructions more comprehensively, and it only does because
	// we only support a single TLS variable (g).

	if CanUse1InsnTLS(ctxt) {
		// Reduce 2-instruction sequence to 1-instruction sequence.
		// Sequences like
		//	MOVQ TLS, BX
		//	... off(BX)(TLS*1) ...
		// become
		//	NOP
		//	... off(TLS) ...
		//
		// TODO(rsc): Remove the Hsolaris special case. It exists only to
		// guarantee we are producing byte-identical binaries as before this code.
		// But it should be unnecessary.
		if (p.As == AMOVQ || p.As == AMOVL) && p.From.Type == obj.TYPE_REG && p.From.Reg == REG_TLS && p.To.Type == obj.TYPE_REG && REG_AX <= p.To.Reg && p.To.Reg <= REG_R15 && ctxt.Headtype != objabi.Hsolaris {
			obj.Nopout(p)
		}
		if p.From.Type == obj.TYPE_MEM && p.From.Index == REG_TLS && REG_AX <= p.From.Reg && p.From.Reg <= REG_R15 {
			p.From.Reg = REG_TLS
			p.From.Scale = 0
			p.From.Index = REG_NONE
		}

		if p.To.Type == obj.TYPE_MEM && p.To.Index == REG_TLS && REG_AX <= p.To.Reg && p.To.Reg <= REG_R15 {
			p.To.Reg = REG_TLS
			p.To.Scale = 0
			p.To.Index = REG_NONE
		}
	} else {
		// load_g_cx, below, always inserts the 1-instruction sequence. Rewrite it
		// as the 2-instruction sequence if necessary.
		//	MOVQ 0(TLS), BX
		// becomes
		//	MOVQ TLS, BX
		//	MOVQ 0(BX)(TLS*1), BX
		if (p.As == AMOVQ || p.As == AMOVL) && p.From.Type == obj.TYPE_MEM && p.From.Reg == REG_TLS && p.To.Type == obj.TYPE_REG && REG_AX <= p.To.Reg && p.To.Reg <= REG_R15 {
			q := obj.Appendp(p, newprog)
			q.As = p.As
			q.From = p.From
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

	// Android uses a tls offset determined at runtime. Rewrite
	//	MOVQ TLS, BX
	// to
	//	MOVQ runtime.tls_g(SB), BX
	if isAndroid && (p.As == AMOVQ || p.As == AMOVL) && p.From.Type == obj.TYPE_REG && p.From.Reg == REG_TLS && p.To.Type == obj.TYPE_REG && REG_AX <= p.To.Reg && p.To.Reg <= REG_R15 {
		p.From.Type = obj.TYPE_MEM
		p.From.Name = obj.NAME_EXTERN
		p.From.Reg = REG_NONE
		p.From.Sym = ctxt.Lookup("runtime.tls_g")
		p.From.Index = REG_NONE
	}

	// TODO: Remove.
	if ctxt.Headtype == objabi.Hwindows && ctxt.Arch.Family == sys.AMD64 || ctxt.Headtype == objabi.Hplan9 {
		if p.From.Scale == 1 && p.From.Index == REG_TLS {
			p.From.Scale = 2
		}
		if p.To.Scale == 1 && p.To.Index == REG_TLS {
			p.To.Scale = 2
		}
	}

	// Rewrite 0 to $0 in 3rd argument to CMPPS etc.
	// That's what the tables expect.
	switch p.As {
	case ACMPPD, ACMPPS, ACMPSD, ACMPSS:
		if p.To.Type == obj.TYPE_MEM && p.To.Name == obj.NAME_NONE && p.To.Reg == REG_NONE && p.To.Index == REG_NONE && p.To.Sym == nil {
			p.To.Type = obj.TYPE_CONST
		}
	}

	// Rewrite CALL/JMP/RET to symbol as TYPE_BRANCH.
	switch p.As {
	case obj.ACALL, obj.AJMP, obj.ARET:
		if p.To.Type == obj.TYPE_MEM && (p.To.Name == obj.NAME_EXTERN || p.To.Name == obj.NAME_STATIC) && p.To.Sym != nil {
			p.To.Type = obj.TYPE_BRANCH
		}
	}

	// Rewrite MOVL/MOVQ $XXX(FP/SP) as LEAL/LEAQ.
	if p.From.Type == obj.TYPE_ADDR && (ctxt.Arch.Family == sys.AMD64 || p.From.Name != obj.NAME_EXTERN && p.From.Name != obj.NAME_STATIC) {
		switch p.As {
		case AMOVL:
			p.As = ALEAL
			p.From.Type = obj.TYPE_MEM
		case AMOVQ:
			p.As = ALEAQ
			p.From.Type = obj.TYPE_MEM
		}
	}

	if ctxt.Headtype == objabi.Hnacl && ctxt.Arch.Family == sys.AMD64 {
		if p.GetFrom3() != nil {
			nacladdr(ctxt, p, p.GetFrom3())
		}
		nacladdr(ctxt, p, &p.From)
		nacladdr(ctxt, p, &p.To)
	}

	// Rewrite float constants to values stored in memory.
	switch p.As {
	// Convert AMOVSS $(0), Xx to AXORPS Xx, Xx
	case AMOVSS:
		if p.From.Type == obj.TYPE_FCONST {
			//  f == 0 can't be used here due to -0, so use Float64bits
			if f := p.From.Val.(float64); math.Float64bits(f) == 0 {
				if p.To.Type == obj.TYPE_REG && REG_X0 <= p.To.Reg && p.To.Reg <= REG_X15 {
					p.As = AXORPS
					p.From = p.To
					break
				}
			}
		}
		fallthrough

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
			f32 := float32(p.From.Val.(float64))
			p.From.Type = obj.TYPE_MEM
			p.From.Name = obj.NAME_EXTERN
			p.From.Sym = ctxt.Float32Sym(f32)
			p.From.Offset = 0
		}

	case AMOVSD:
		// Convert AMOVSD $(0), Xx to AXORPS Xx, Xx
		if p.From.Type == obj.TYPE_FCONST {
			//  f == 0 can't be used here due to -0, so use Float64bits
			if f := p.From.Val.(float64); math.Float64bits(f) == 0 {
				if p.To.Type == obj.TYPE_REG && REG_X0 <= p.To.Reg && p.To.Reg <= REG_X15 {
					p.As = AXORPS
					p.From = p.To
					break
				}
			}
		}
		fallthrough

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
			f64 := p.From.Val.(float64)
			p.From.Type = obj.TYPE_MEM
			p.From.Name = obj.NAME_EXTERN
			p.From.Sym = ctxt.Float64Sym(f64)
			p.From.Offset = 0
		}
	}

	if ctxt.Flag_dynlink {
		rewriteToUseGot(ctxt, p, newprog)
	}

	if ctxt.Flag_shared && ctxt.Arch.Family == sys.I386 {
		rewriteToPcrel(ctxt, p, newprog)
	}
}

// Rewrite p, if necessary, to access global data via the global offset table.
func rewriteToUseGot(ctxt *obj.Link, p *obj.Prog, newprog obj.ProgAlloc) {
	var lea, mov obj.As
	var reg int16
	if ctxt.Arch.Family == sys.AMD64 {
		lea = ALEAQ
		mov = AMOVQ
		reg = REG_R15
	} else {
		lea = ALEAL
		mov = AMOVL
		reg = REG_CX
		if p.As == ALEAL && p.To.Reg != p.From.Reg && p.To.Reg != p.From.Index {
			// Special case: clobber the destination register with
			// the PC so we don't have to clobber CX.
			// The SSA backend depends on CX not being clobbered across LEAL.
			// See cmd/compile/internal/ssa/gen/386.rules (search for Flag_shared).
			reg = p.To.Reg
		}
	}

	if p.As == obj.ADUFFCOPY || p.As == obj.ADUFFZERO {
		//     ADUFFxxx $offset
		// becomes
		//     $MOV runtime.duffxxx@GOT, $reg
		//     $LEA $offset($reg), $reg
		//     CALL $reg
		// (we use LEAx rather than ADDx because ADDx clobbers
		// flags and duffzero on 386 does not otherwise do so).
		var sym *obj.LSym
		if p.As == obj.ADUFFZERO {
			sym = ctxt.Lookup("runtime.duffzero")
		} else {
			sym = ctxt.Lookup("runtime.duffcopy")
		}
		offset := p.To.Offset
		p.As = mov
		p.From.Type = obj.TYPE_MEM
		p.From.Name = obj.NAME_GOTREF
		p.From.Sym = sym
		p.To.Type = obj.TYPE_REG
		p.To.Reg = reg
		p.To.Offset = 0
		p.To.Sym = nil
		p1 := obj.Appendp(p, newprog)
		p1.As = lea
		p1.From.Type = obj.TYPE_MEM
		p1.From.Offset = offset
		p1.From.Reg = reg
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = reg
		p2 := obj.Appendp(p1, newprog)
		p2.As = obj.ACALL
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = reg
	}

	// We only care about global data: NAME_EXTERN means a global
	// symbol in the Go sense, and p.Sym.Local is true for a few
	// internally defined symbols.
	if p.As == lea && p.From.Type == obj.TYPE_MEM && p.From.Name == obj.NAME_EXTERN && !p.From.Sym.Local() {
		// $LEA sym, Rx becomes $MOV $sym, Rx which will be rewritten below
		p.As = mov
		p.From.Type = obj.TYPE_ADDR
	}
	if p.From.Type == obj.TYPE_ADDR && p.From.Name == obj.NAME_EXTERN && !p.From.Sym.Local() {
		// $MOV $sym, Rx becomes $MOV sym@GOT, Rx
		// $MOV $sym+<off>, Rx becomes $MOV sym@GOT, Rx; $LEA <off>(Rx), Rx
		// On 386 only, more complicated things like PUSHL $sym become $MOV sym@GOT, CX; PUSHL CX
		cmplxdest := false
		pAs := p.As
		var dest obj.Addr
		if p.To.Type != obj.TYPE_REG || pAs != mov {
			if ctxt.Arch.Family == sys.AMD64 {
				ctxt.Diag("do not know how to handle LEA-type insn to non-register in %v with -dynlink", p)
			}
			cmplxdest = true
			dest = p.To
			p.As = mov
			p.To.Type = obj.TYPE_REG
			p.To.Reg = reg
			p.To.Sym = nil
			p.To.Name = obj.NAME_NONE
		}
		p.From.Type = obj.TYPE_MEM
		p.From.Name = obj.NAME_GOTREF
		q := p
		if p.From.Offset != 0 {
			q = obj.Appendp(p, newprog)
			q.As = lea
			q.From.Type = obj.TYPE_MEM
			q.From.Reg = p.To.Reg
			q.From.Offset = p.From.Offset
			q.To = p.To
			p.From.Offset = 0
		}
		if cmplxdest {
			q = obj.Appendp(q, newprog)
			q.As = pAs
			q.To = dest
			q.From.Type = obj.TYPE_REG
			q.From.Reg = reg
		}
	}
	if p.GetFrom3() != nil && p.GetFrom3().Name == obj.NAME_EXTERN {
		ctxt.Diag("don't know how to handle %v with -dynlink", p)
	}
	var source *obj.Addr
	// MOVx sym, Ry becomes $MOV sym@GOT, R15; MOVx (R15), Ry
	// MOVx Ry, sym becomes $MOV sym@GOT, R15; MOVx Ry, (R15)
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
	if p.As == obj.ACALL {
		// When dynlinking on 386, almost any call might end up being a call
		// to a PLT, so make sure the GOT pointer is loaded into BX.
		// RegTo2 is set on the replacement call insn to stop it being
		// processed when it is in turn passed to progedit.
		if ctxt.Arch.Family == sys.AMD64 || (p.To.Sym != nil && p.To.Sym.Local()) || p.RegTo2 != 0 {
			return
		}
		p1 := obj.Appendp(p, newprog)
		p2 := obj.Appendp(p1, newprog)

		p1.As = ALEAL
		p1.From.Type = obj.TYPE_MEM
		p1.From.Name = obj.NAME_STATIC
		p1.From.Sym = ctxt.Lookup("_GLOBAL_OFFSET_TABLE_")
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = REG_BX

		p2.As = p.As
		p2.Scond = p.Scond
		p2.From = p.From
		if p.RestArgs != nil {
			p2.RestArgs = append(p2.RestArgs, p.RestArgs...)
		}
		p2.Reg = p.Reg
		p2.To = p.To
		// p.To.Type was set to TYPE_BRANCH above, but that makes checkaddr
		// in ../pass.go complain, so set it back to TYPE_MEM here, until p2
		// itself gets passed to progedit.
		p2.To.Type = obj.TYPE_MEM
		p2.RegTo2 = 1

		obj.Nopout(p)
		return

	}
	if p.As == obj.ATEXT || p.As == obj.AFUNCDATA || p.As == obj.ARET || p.As == obj.AJMP {
		return
	}
	if source.Type != obj.TYPE_MEM {
		ctxt.Diag("don't know how to handle %v with -dynlink", p)
	}
	p1 := obj.Appendp(p, newprog)
	p2 := obj.Appendp(p1, newprog)

	p1.As = mov
	p1.From.Type = obj.TYPE_MEM
	p1.From.Sym = source.Sym
	p1.From.Name = obj.NAME_GOTREF
	p1.To.Type = obj.TYPE_REG
	p1.To.Reg = reg

	p2.As = p.As
	p2.From = p.From
	p2.To = p.To
	if p.From.Name == obj.NAME_EXTERN {
		p2.From.Reg = reg
		p2.From.Name = obj.NAME_NONE
		p2.From.Sym = nil
	} else if p.To.Name == obj.NAME_EXTERN {
		p2.To.Reg = reg
		p2.To.Name = obj.NAME_NONE
		p2.To.Sym = nil
	} else {
		return
	}
	obj.Nopout(p)
}

func rewriteToPcrel(ctxt *obj.Link, p *obj.Prog, newprog obj.ProgAlloc) {
	// RegTo2 is set on the instructions we insert here so they don't get
	// processed twice.
	if p.RegTo2 != 0 {
		return
	}
	if p.As == obj.ATEXT || p.As == obj.AFUNCDATA || p.As == obj.ACALL || p.As == obj.ARET || p.As == obj.AJMP {
		return
	}
	// Any Prog (aside from the above special cases) with an Addr with Name ==
	// NAME_EXTERN, NAME_STATIC or NAME_GOTREF has a CALL __x86.get_pc_thunk.XX
	// inserted before it.
	isName := func(a *obj.Addr) bool {
		if a.Sym == nil || (a.Type != obj.TYPE_MEM && a.Type != obj.TYPE_ADDR) || a.Reg != 0 {
			return false
		}
		if a.Sym.Type == objabi.STLSBSS {
			return false
		}
		return a.Name == obj.NAME_EXTERN || a.Name == obj.NAME_STATIC || a.Name == obj.NAME_GOTREF
	}

	if isName(&p.From) && p.From.Type == obj.TYPE_ADDR {
		// Handle things like "MOVL $sym, (SP)" or "PUSHL $sym" by rewriting
		// to "MOVL $sym, CX; MOVL CX, (SP)" or "MOVL $sym, CX; PUSHL CX"
		// respectively.
		if p.To.Type != obj.TYPE_REG {
			q := obj.Appendp(p, newprog)
			q.As = p.As
			q.From.Type = obj.TYPE_REG
			q.From.Reg = REG_CX
			q.To = p.To
			p.As = AMOVL
			p.To.Type = obj.TYPE_REG
			p.To.Reg = REG_CX
			p.To.Sym = nil
			p.To.Name = obj.NAME_NONE
		}
	}

	if !isName(&p.From) && !isName(&p.To) && (p.GetFrom3() == nil || !isName(p.GetFrom3())) {
		return
	}
	var dst int16 = REG_CX
	if (p.As == ALEAL || p.As == AMOVL) && p.To.Reg != p.From.Reg && p.To.Reg != p.From.Index {
		dst = p.To.Reg
		// Why? See the comment near the top of rewriteToUseGot above.
		// AMOVLs might be introduced by the GOT rewrites.
	}
	q := obj.Appendp(p, newprog)
	q.RegTo2 = 1
	r := obj.Appendp(q, newprog)
	r.RegTo2 = 1
	q.As = obj.ACALL
	thunkname := "__x86.get_pc_thunk." + strings.ToLower(rconv(int(dst)))
	q.To.Sym = ctxt.LookupInit(thunkname, func(s *obj.LSym) { s.Set(obj.AttrLocal, true) })
	q.To.Type = obj.TYPE_MEM
	q.To.Name = obj.NAME_EXTERN
	r.As = p.As
	r.Scond = p.Scond
	r.From = p.From
	r.RestArgs = p.RestArgs
	r.Reg = p.Reg
	r.To = p.To
	if isName(&p.From) {
		r.From.Reg = dst
	}
	if isName(&p.To) {
		r.To.Reg = dst
	}
	if p.GetFrom3() != nil && isName(p.GetFrom3()) {
		r.GetFrom3().Reg = dst
	}
	obj.Nopout(p)
}

func nacladdr(ctxt *obj.Link, p *obj.Prog, a *obj.Addr) {
	if p.As == ALEAL || p.As == ALEAQ {
		return
	}

	if a.Reg == REG_BP {
		ctxt.Diag("invalid address: %v", p)
		return
	}

	if a.Reg == REG_TLS {
		a.Reg = REG_BP
	}
	if a.Type == obj.TYPE_MEM && a.Name == obj.NAME_NONE {
		switch a.Reg {
		// all ok
		case REG_BP, REG_SP, REG_R15:
			break

		default:
			if a.Index != REG_NONE {
				ctxt.Diag("invalid address %v", p)
			}
			a.Index = a.Reg
			if a.Index != REG_NONE {
				a.Scale = 1
			}
			a.Reg = REG_R15
		}
	}
}

func preprocess(ctxt *obj.Link, cursym *obj.LSym, newprog obj.ProgAlloc) {
	if cursym.Func.Text == nil || cursym.Func.Text.Link == nil {
		return
	}

	p := cursym.Func.Text
	autoffset := int32(p.To.Offset)
	if autoffset < 0 {
		autoffset = 0
	}

	hasCall := false
	for q := p; q != nil; q = q.Link {
		if q.As == obj.ACALL || q.As == obj.ADUFFCOPY || q.As == obj.ADUFFZERO {
			hasCall = true
			break
		}
	}

	var bpsize int
	if ctxt.Arch.Family == sys.AMD64 && ctxt.Framepointer_enabled &&
		!p.From.Sym.NoFrame() && // (1) below
		!(autoffset == 0 && p.From.Sym.NoSplit()) && // (2) below
		!(autoffset == 0 && !hasCall) { // (3) below
		// Make room to save a base pointer.
		// There are 2 cases we must avoid:
		// 1) If noframe is set (which we do for functions which tail call).
		// 2) Scary runtime internals which would be all messed up by frame pointers.
		//    We detect these using a heuristic: frameless nosplit functions.
		//    TODO: Maybe someday we label them all with NOFRAME and get rid of this heuristic.
		// For performance, we also want to avoid:
		// 3) Frameless leaf functions
		bpsize = ctxt.Arch.PtrSize
		autoffset += int32(bpsize)
		p.To.Offset += int64(bpsize)
	} else {
		bpsize = 0
	}

	textarg := int64(p.To.Val.(int32))
	cursym.Func.Args = int32(textarg)
	cursym.Func.Locals = int32(p.To.Offset)

	// TODO(rsc): Remove.
	if ctxt.Arch.Family == sys.I386 && cursym.Func.Locals < 0 {
		cursym.Func.Locals = 0
	}

	// TODO(rsc): Remove 'ctxt.Arch.Family == sys.AMD64 &&'.
	if ctxt.Arch.Family == sys.AMD64 && autoffset < objabi.StackSmall && !p.From.Sym.NoSplit() {
		leaf := true
	LeafSearch:
		for q := p; q != nil; q = q.Link {
			switch q.As {
			case obj.ACALL:
				// Treat common runtime calls that take no arguments
				// the same as duffcopy and duffzero.
				if !isZeroArgRuntimeCall(q.To.Sym) {
					leaf = false
					break LeafSearch
				}
				fallthrough
			case obj.ADUFFCOPY, obj.ADUFFZERO:
				if autoffset >= objabi.StackSmall-8 {
					leaf = false
					break LeafSearch
				}
			}
		}

		if leaf {
			p.From.Sym.Set(obj.AttrNoSplit, true)
		}
	}

	if !p.From.Sym.NoSplit() || p.From.Sym.Wrapper() {
		p = obj.Appendp(p, newprog)
		p = load_g_cx(ctxt, p, newprog) // load g into CX
	}

	if !cursym.Func.Text.From.Sym.NoSplit() {
		p = stacksplit(ctxt, cursym, p, newprog, autoffset, int32(textarg)) // emit split check
	}

	// Delve debugger would like the next instruction to be noted as the end of the function prologue.
	// TODO: are there other cases (e.g., wrapper functions) that need marking?
	markedPrologue := false

	if autoffset != 0 {
		if autoffset%int32(ctxt.Arch.RegSize) != 0 {
			ctxt.Diag("unaligned stack size %d", autoffset)
		}
		p = obj.Appendp(p, newprog)
		p.As = AADJSP
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(autoffset)
		p.Spadj = autoffset
		p.Pos = p.Pos.WithXlogue(src.PosPrologueEnd)
		markedPrologue = true
	}

	deltasp := autoffset

	if bpsize > 0 {
		// Save caller's BP
		p = obj.Appendp(p, newprog)

		p.As = AMOVQ
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_BP
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = REG_SP
		p.To.Scale = 1
		p.To.Offset = int64(autoffset) - int64(bpsize)
		if !markedPrologue {
			p.Pos = p.Pos.WithXlogue(src.PosPrologueEnd)
		}

		// Move current frame to BP
		p = obj.Appendp(p, newprog)

		p.As = ALEAQ
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = REG_SP
		p.From.Scale = 1
		p.From.Offset = int64(autoffset) - int64(bpsize)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_BP
	}

	if cursym.Func.Text.From.Sym.Wrapper() {
		// if g._panic != nil && g._panic.argp == FP {
		//   g._panic.argp = bottom-of-frame
		// }
		//
		//	MOVQ g_panic(CX), BX
		//	TESTQ BX, BX
		//	JNE checkargp
		// end:
		//	NOP
		//  ... rest of function ...
		// checkargp:
		//	LEAQ (autoffset+8)(SP), DI
		//	CMPQ panic_argp(BX), DI
		//	JNE end
		//  MOVQ SP, panic_argp(BX)
		//  JMP end
		//
		// The NOP is needed to give the jumps somewhere to land.
		// It is a liblink NOP, not an x86 NOP: it encodes to 0 instruction bytes.
		//
		// The layout is chosen to help static branch prediction:
		// Both conditional jumps are unlikely, so they are arranged to be forward jumps.

		// MOVQ g_panic(CX), BX
		p = obj.Appendp(p, newprog)
		p.As = AMOVQ
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = REG_CX
		p.From.Offset = 4 * int64(ctxt.Arch.PtrSize) // g_panic
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_BX
		if ctxt.Headtype == objabi.Hnacl && ctxt.Arch.Family == sys.AMD64 {
			p.As = AMOVL
			p.From.Type = obj.TYPE_MEM
			p.From.Reg = REG_R15
			p.From.Scale = 1
			p.From.Index = REG_CX
		}
		if ctxt.Arch.Family == sys.I386 {
			p.As = AMOVL
		}

		// TESTQ BX, BX
		p = obj.Appendp(p, newprog)
		p.As = ATESTQ
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_BX
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_BX
		if ctxt.Headtype == objabi.Hnacl || ctxt.Arch.Family == sys.I386 {
			p.As = ATESTL
		}

		// JNE checkargp (checkargp to be resolved later)
		jne := obj.Appendp(p, newprog)
		jne.As = AJNE
		jne.To.Type = obj.TYPE_BRANCH

		// end:
		//  NOP
		end := obj.Appendp(jne, newprog)
		end.As = obj.ANOP

		// Fast forward to end of function.
		var last *obj.Prog
		for last = end; last.Link != nil; last = last.Link {
		}

		// LEAQ (autoffset+8)(SP), DI
		p = obj.Appendp(last, newprog)
		p.As = ALEAQ
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = REG_SP
		p.From.Offset = int64(autoffset) + int64(ctxt.Arch.RegSize)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_DI
		if ctxt.Headtype == objabi.Hnacl || ctxt.Arch.Family == sys.I386 {
			p.As = ALEAL
		}

		// Set jne branch target.
		jne.Pcond = p

		// CMPQ panic_argp(BX), DI
		p = obj.Appendp(p, newprog)
		p.As = ACMPQ
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = REG_BX
		p.From.Offset = 0 // Panic.argp
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_DI
		if ctxt.Headtype == objabi.Hnacl && ctxt.Arch.Family == sys.AMD64 {
			p.As = ACMPL
			p.From.Type = obj.TYPE_MEM
			p.From.Reg = REG_R15
			p.From.Scale = 1
			p.From.Index = REG_BX
		}
		if ctxt.Arch.Family == sys.I386 {
			p.As = ACMPL
		}

		// JNE end
		p = obj.Appendp(p, newprog)
		p.As = AJNE
		p.To.Type = obj.TYPE_BRANCH
		p.Pcond = end

		// MOVQ SP, panic_argp(BX)
		p = obj.Appendp(p, newprog)
		p.As = AMOVQ
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_SP
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = REG_BX
		p.To.Offset = 0 // Panic.argp
		if ctxt.Headtype == objabi.Hnacl && ctxt.Arch.Family == sys.AMD64 {
			p.As = AMOVL
			p.To.Type = obj.TYPE_MEM
			p.To.Reg = REG_R15
			p.To.Scale = 1
			p.To.Index = REG_BX
		}
		if ctxt.Arch.Family == sys.I386 {
			p.As = AMOVL
		}

		// JMP end
		p = obj.Appendp(p, newprog)
		p.As = obj.AJMP
		p.To.Type = obj.TYPE_BRANCH
		p.Pcond = end

		// Reset p for following code.
		p = end
	}

	for ; p != nil; p = p.Link {
		pcsize := ctxt.Arch.RegSize
		switch p.From.Name {
		case obj.NAME_AUTO:
			p.From.Offset += int64(deltasp) - int64(bpsize)
		case obj.NAME_PARAM:
			p.From.Offset += int64(deltasp) + int64(pcsize)
		}
		if p.GetFrom3() != nil {
			switch p.GetFrom3().Name {
			case obj.NAME_AUTO:
				p.GetFrom3().Offset += int64(deltasp) - int64(bpsize)
			case obj.NAME_PARAM:
				p.GetFrom3().Offset += int64(deltasp) + int64(pcsize)
			}
		}
		switch p.To.Name {
		case obj.NAME_AUTO:
			p.To.Offset += int64(deltasp) - int64(bpsize)
		case obj.NAME_PARAM:
			p.To.Offset += int64(deltasp) + int64(pcsize)
		}

		switch p.As {
		default:
			continue

		case APUSHL, APUSHFL:
			deltasp += 4
			p.Spadj = 4
			continue

		case APUSHQ, APUSHFQ:
			deltasp += 8
			p.Spadj = 8
			continue

		case APUSHW, APUSHFW:
			deltasp += 2
			p.Spadj = 2
			continue

		case APOPL, APOPFL:
			deltasp -= 4
			p.Spadj = -4
			continue

		case APOPQ, APOPFQ:
			deltasp -= 8
			p.Spadj = -8
			continue

		case APOPW, APOPFW:
			deltasp -= 2
			p.Spadj = -2
			continue

		case obj.ARET:
			// do nothing
		}

		if autoffset != deltasp {
			ctxt.Diag("unbalanced PUSH/POP")
		}

		if autoffset != 0 {
			to := p.To // Keep To attached to RET for retjmp below
			p.To = obj.Addr{}
			if bpsize > 0 {
				// Restore caller's BP
				p.As = AMOVQ

				p.From.Type = obj.TYPE_MEM
				p.From.Reg = REG_SP
				p.From.Scale = 1
				p.From.Offset = int64(autoffset) - int64(bpsize)
				p.To.Type = obj.TYPE_REG
				p.To.Reg = REG_BP
				p = obj.Appendp(p, newprog)
			}

			p.As = AADJSP
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = int64(-autoffset)
			p.Spadj = -autoffset
			p = obj.Appendp(p, newprog)
			p.As = obj.ARET
			p.To = to

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

func isZeroArgRuntimeCall(s *obj.LSym) bool {
	if s == nil {
		return false
	}
	switch s.Name {
	case "runtime.panicdivide", "runtime.panicwrap", "runtime.panicshift":
		return true
	}
	if strings.HasPrefix(s.Name, "runtime.panicIndex") || strings.HasPrefix(s.Name, "runtime.panicSlice") {
		// These functions do take arguments (in registers),
		// but use no stack before they do a stack check. We
		// should include them. See issue 31219.
		return true
	}
	return false
}

func indir_cx(ctxt *obj.Link, a *obj.Addr) {
	if ctxt.Headtype == objabi.Hnacl && ctxt.Arch.Family == sys.AMD64 {
		a.Type = obj.TYPE_MEM
		a.Reg = REG_R15
		a.Index = REG_CX
		a.Scale = 1
		return
	}

	a.Type = obj.TYPE_MEM
	a.Reg = REG_CX
}

// Append code to p to load g into cx.
// Overwrites p with the first instruction (no first appendp).
// Overwriting p is unusual but it lets use this in both the
// prologue (caller must call appendp first) and in the epilogue.
// Returns last new instruction.
func load_g_cx(ctxt *obj.Link, p *obj.Prog, newprog obj.ProgAlloc) *obj.Prog {
	p.As = AMOVQ
	if ctxt.Arch.PtrSize == 4 {
		p.As = AMOVL
	}
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = REG_TLS
	p.From.Offset = 0
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_CX

	next := p.Link
	progedit(ctxt, p, newprog)
	for p.Link != next {
		p = p.Link
		progedit(ctxt, p, newprog)
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
func stacksplit(ctxt *obj.Link, cursym *obj.LSym, p *obj.Prog, newprog obj.ProgAlloc, framesize int32, textarg int32) *obj.Prog {
	cmp := ACMPQ
	lea := ALEAQ
	mov := AMOVQ
	sub := ASUBQ

	if ctxt.Headtype == objabi.Hnacl || ctxt.Arch.Family == sys.I386 {
		cmp = ACMPL
		lea = ALEAL
		mov = AMOVL
		sub = ASUBL
	}

	var q1 *obj.Prog
	if framesize <= objabi.StackSmall {
		// small stack: SP <= stackguard
		//	CMPQ SP, stackguard
		p = obj.Appendp(p, newprog)

		p.As = cmp
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_SP
		indir_cx(ctxt, &p.To)
		p.To.Offset = 2 * int64(ctxt.Arch.PtrSize) // G.stackguard0
		if cursym.CFunc() {
			p.To.Offset = 3 * int64(ctxt.Arch.PtrSize) // G.stackguard1
		}
	} else if framesize <= objabi.StackBig {
		// large stack: SP-framesize <= stackguard-StackSmall
		//	LEAQ -xxx(SP), AX
		//	CMPQ AX, stackguard
		p = obj.Appendp(p, newprog)

		p.As = lea
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = REG_SP
		p.From.Offset = -(int64(framesize) - objabi.StackSmall)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_AX

		p = obj.Appendp(p, newprog)
		p.As = cmp
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_AX
		indir_cx(ctxt, &p.To)
		p.To.Offset = 2 * int64(ctxt.Arch.PtrSize) // G.stackguard0
		if cursym.CFunc() {
			p.To.Offset = 3 * int64(ctxt.Arch.PtrSize) // G.stackguard1
		}
	} else {
		// Such a large stack we need to protect against wraparound.
		// If SP is close to zero:
		//	SP-stackguard+StackGuard <= framesize + (StackGuard-StackSmall)
		// The +StackGuard on both sides is required to keep the left side positive:
		// SP is allowed to be slightly below stackguard. See stack.h.
		//
		// Preemption sets stackguard to StackPreempt, a very large value.
		// That breaks the math above, so we have to check for that explicitly.
		//	MOVQ	stackguard, CX
		//	CMPQ	CX, $StackPreempt
		//	JEQ	label-of-call-to-morestack
		//	LEAQ	StackGuard(SP), AX
		//	SUBQ	CX, AX
		//	CMPQ	AX, $(framesize+(StackGuard-StackSmall))

		p = obj.Appendp(p, newprog)

		p.As = mov
		indir_cx(ctxt, &p.From)
		p.From.Offset = 2 * int64(ctxt.Arch.PtrSize) // G.stackguard0
		if cursym.CFunc() {
			p.From.Offset = 3 * int64(ctxt.Arch.PtrSize) // G.stackguard1
		}
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_SI

		p = obj.Appendp(p, newprog)
		p.As = cmp
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_SI
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = objabi.StackPreempt
		if ctxt.Arch.Family == sys.I386 {
			p.To.Offset = int64(uint32(objabi.StackPreempt & (1<<32 - 1)))
		}

		p = obj.Appendp(p, newprog)
		p.As = AJEQ
		p.To.Type = obj.TYPE_BRANCH
		q1 = p

		p = obj.Appendp(p, newprog)
		p.As = lea
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = REG_SP
		p.From.Offset = int64(objabi.StackGuard)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_AX

		p = obj.Appendp(p, newprog)
		p.As = sub
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_SI
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_AX

		p = obj.Appendp(p, newprog)
		p.As = cmp
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_AX
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = int64(framesize) + (int64(objabi.StackGuard) - objabi.StackSmall)
	}

	// common
	jls := obj.Appendp(p, newprog)
	jls.As = AJLS
	jls.To.Type = obj.TYPE_BRANCH

	var last *obj.Prog
	for last = cursym.Func.Text; last.Link != nil; last = last.Link {
	}

	// Now we are at the end of the function, but logically
	// we are still in function prologue. We need to fix the
	// SP data and PCDATA.
	spfix := obj.Appendp(last, newprog)
	spfix.As = obj.ANOP
	spfix.Spadj = -framesize

	pcdata := ctxt.EmitEntryLiveness(cursym, spfix, newprog)

	call := obj.Appendp(pcdata, newprog)
	call.Pos = cursym.Func.Text.Pos
	call.As = obj.ACALL
	call.To.Type = obj.TYPE_BRANCH
	call.To.Name = obj.NAME_EXTERN
	morestack := "runtime.morestack"
	switch {
	case cursym.CFunc():
		morestack = "runtime.morestackc"
	case !cursym.Func.Text.From.Sym.NeedCtxt():
		morestack = "runtime.morestack_noctxt"
	}
	call.To.Sym = ctxt.Lookup(morestack)
	// When compiling 386 code for dynamic linking, the call needs to be adjusted
	// to follow PIC rules. This in turn can insert more instructions, so we need
	// to keep track of the start of the call (where the jump will be to) and the
	// end (which following instructions are appended to).
	callend := call
	progedit(ctxt, callend, newprog)
	for ; callend.Link != nil; callend = callend.Link {
		progedit(ctxt, callend.Link, newprog)
	}

	jmp := obj.Appendp(callend, newprog)
	jmp.As = obj.AJMP
	jmp.To.Type = obj.TYPE_BRANCH
	jmp.Pcond = cursym.Func.Text.Link
	jmp.Spadj = +framesize

	jls.Pcond = call
	if q1 != nil {
		q1.Pcond = call
	}

	return jls
}

var unaryDst = map[obj.As]bool{
	ABSWAPL:     true,
	ABSWAPQ:     true,
	ACLFLUSH:    true,
	ACLFLUSHOPT: true,
	ACMPXCHG16B: true,
	ACMPXCHG8B:  true,
	ADECB:       true,
	ADECL:       true,
	ADECQ:       true,
	ADECW:       true,
	AFBSTP:      true,
	AFFREE:      true,
	AFLDENV:     true,
	AFSAVE:      true,
	AFSTCW:      true,
	AFSTENV:     true,
	AFSTSW:      true,
	AFXSAVE64:   true,
	AFXSAVE:     true,
	AINCB:       true,
	AINCL:       true,
	AINCQ:       true,
	AINCW:       true,
	ANEGB:       true,
	ANEGL:       true,
	ANEGQ:       true,
	ANEGW:       true,
	ANOTB:       true,
	ANOTL:       true,
	ANOTQ:       true,
	ANOTW:       true,
	APOPL:       true,
	APOPQ:       true,
	APOPW:       true,
	ARDFSBASEL:  true,
	ARDFSBASEQ:  true,
	ARDGSBASEL:  true,
	ARDGSBASEQ:  true,
	ARDRANDL:    true,
	ARDRANDQ:    true,
	ARDRANDW:    true,
	ARDSEEDL:    true,
	ARDSEEDQ:    true,
	ARDSEEDW:    true,
	ASETCC:      true,
	ASETCS:      true,
	ASETEQ:      true,
	ASETGE:      true,
	ASETGT:      true,
	ASETHI:      true,
	ASETLE:      true,
	ASETLS:      true,
	ASETLT:      true,
	ASETMI:      true,
	ASETNE:      true,
	ASETOC:      true,
	ASETOS:      true,
	ASETPC:      true,
	ASETPL:      true,
	ASETPS:      true,
	ASGDT:       true,
	ASIDT:       true,
	ASLDTL:      true,
	ASLDTQ:      true,
	ASLDTW:      true,
	ASMSWL:      true,
	ASMSWQ:      true,
	ASMSWW:      true,
	ASTMXCSR:    true,
	ASTRL:       true,
	ASTRQ:       true,
	ASTRW:       true,
	AXSAVE64:    true,
	AXSAVE:      true,
	AXSAVEC64:   true,
	AXSAVEC:     true,
	AXSAVEOPT64: true,
	AXSAVEOPT:   true,
	AXSAVES64:   true,
	AXSAVES:     true,
}

var Linkamd64 = obj.LinkArch{
	Arch:           sys.ArchAMD64,
	Init:           instinit,
	Preprocess:     preprocess,
	Assemble:       span6,
	Progedit:       progedit,
	UnaryDst:       unaryDst,
	DWARFRegisters: AMD64DWARFRegisters,
}

var Linkamd64p32 = obj.LinkArch{
	Arch:           sys.ArchAMD64P32,
	Init:           instinit,
	Preprocess:     preprocess,
	Assemble:       span6,
	Progedit:       progedit,
	UnaryDst:       unaryDst,
	DWARFRegisters: AMD64DWARFRegisters,
}

var Link386 = obj.LinkArch{
	Arch:           sys.Arch386,
	Init:           instinit,
	Preprocess:     preprocess,
	Assemble:       span6,
	Progedit:       progedit,
	UnaryDst:       unaryDst,
	DWARFRegisters: X86DWARFRegisters,
}
