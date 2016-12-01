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
	"cmd/internal/sys"
	"fmt"
	"log"
	"math"
	"strings"
)

func CanUse1InsnTLS(ctxt *obj.Link) bool {
	if isAndroid {
		// For android, we use a disgusting hack that assumes
		// the thread-local storage slot for g is allocated
		// using pthread_key_create with a fixed offset
		// (see src/runtime/cgo/gcc_android_amd64.c).
		// This makes access to the TLS storage (for g) doable
		// with 1 instruction.
		return true
	}

	if ctxt.Arch.RegSize == 4 {
		switch ctxt.Headtype {
		case obj.Hlinux,
			obj.Hnacl,
			obj.Hplan9,
			obj.Hwindows,
			obj.Hwindowsgui:
			return false
		}

		return true
	}

	switch ctxt.Headtype {
	case obj.Hplan9, obj.Hwindows, obj.Hwindowsgui:
		return false
	case obj.Hlinux:
		return !ctxt.Flag_shared
	}

	return true
}

func progedit(ctxt *obj.Link, p *obj.Prog) {
	// Maintain information about code generation mode.
	if ctxt.Mode == 0 {
		ctxt.Mode = ctxt.Arch.RegSize * 8
	}
	p.Mode = int8(ctxt.Mode)

	switch p.As {
	case AMODE:
		if p.From.Type == obj.TYPE_CONST || (p.From.Type == obj.TYPE_MEM && p.From.Reg == REG_NONE) {
			switch int(p.From.Offset) {
			case 16, 32, 64:
				ctxt.Mode = int(p.From.Offset)
			}
		}
		obj.Nopout(p)
	}

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
		if (p.As == AMOVQ || p.As == AMOVL) && p.From.Type == obj.TYPE_REG && p.From.Reg == REG_TLS && p.To.Type == obj.TYPE_REG && REG_AX <= p.To.Reg && p.To.Reg <= REG_R15 && ctxt.Headtype != obj.Hsolaris {
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
			q := obj.Appendp(ctxt, p)
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

	// TODO: Remove.
	if (ctxt.Headtype == obj.Hwindows || ctxt.Headtype == obj.Hwindowsgui) && p.Mode == 64 || ctxt.Headtype == obj.Hplan9 {
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

	if ctxt.Headtype == obj.Hnacl && p.Mode == 64 {
		if p.From3 != nil {
			nacladdr(ctxt, p, p.From3)
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
			i32 := math.Float32bits(f32)
			literal := fmt.Sprintf("$f32.%08x", i32)
			s := obj.Linklookup(ctxt, literal, 0)
			p.From.Type = obj.TYPE_MEM
			p.From.Name = obj.NAME_EXTERN
			p.From.Sym = s
			p.From.Sym.Set(obj.AttrLocal, true)
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
			i64 := math.Float64bits(p.From.Val.(float64))
			literal := fmt.Sprintf("$f64.%016x", i64)
			s := obj.Linklookup(ctxt, literal, 0)
			p.From.Type = obj.TYPE_MEM
			p.From.Name = obj.NAME_EXTERN
			p.From.Sym = s
			p.From.Sym.Set(obj.AttrLocal, true)
			p.From.Offset = 0
		}
	}

	if ctxt.Flag_dynlink {
		rewriteToUseGot(ctxt, p)
	}

	if ctxt.Flag_shared && p.Mode == 32 {
		rewriteToPcrel(ctxt, p)
	}
}

// Rewrite p, if necessary, to access global data via the global offset table.
func rewriteToUseGot(ctxt *obj.Link, p *obj.Prog) {
	var add, lea, mov obj.As
	var reg int16
	if p.Mode == 64 {
		add = AADDQ
		lea = ALEAQ
		mov = AMOVQ
		reg = REG_R15
	} else {
		add = AADDL
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
		//     $ADD $offset, $reg
		//     CALL $reg
		var sym *obj.LSym
		if p.As == obj.ADUFFZERO {
			sym = obj.Linklookup(ctxt, "runtime.duffzero", 0)
		} else {
			sym = obj.Linklookup(ctxt, "runtime.duffcopy", 0)
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
		p1 := obj.Appendp(ctxt, p)
		p1.As = add
		p1.From.Type = obj.TYPE_CONST
		p1.From.Offset = offset
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = reg
		p2 := obj.Appendp(ctxt, p1)
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
			if p.Mode == 64 {
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
			q = obj.Appendp(ctxt, p)
			q.As = lea
			q.From.Type = obj.TYPE_MEM
			q.From.Reg = p.To.Reg
			q.From.Offset = p.From.Offset
			q.To = p.To
			p.From.Offset = 0
		}
		if cmplxdest {
			q = obj.Appendp(ctxt, q)
			q.As = pAs
			q.To = dest
			q.From.Type = obj.TYPE_REG
			q.From.Reg = reg
		}
	}
	if p.From3 != nil && p.From3.Name == obj.NAME_EXTERN {
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
		if p.Mode == 64 || (p.To.Sym != nil && p.To.Sym.Local()) || p.RegTo2 != 0 {
			return
		}
		p1 := obj.Appendp(ctxt, p)
		p2 := obj.Appendp(ctxt, p1)

		p1.As = ALEAL
		p1.From.Type = obj.TYPE_MEM
		p1.From.Name = obj.NAME_STATIC
		p1.From.Sym = obj.Linklookup(ctxt, "_GLOBAL_OFFSET_TABLE_", 0)
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = REG_BX

		p2.As = p.As
		p2.Scond = p.Scond
		p2.From = p.From
		p2.From3 = p.From3
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
	p1 := obj.Appendp(ctxt, p)
	p2 := obj.Appendp(ctxt, p1)

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

func rewriteToPcrel(ctxt *obj.Link, p *obj.Prog) {
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
		if a.Sym.Type == obj.STLSBSS {
			return false
		}
		return a.Name == obj.NAME_EXTERN || a.Name == obj.NAME_STATIC || a.Name == obj.NAME_GOTREF
	}

	if isName(&p.From) && p.From.Type == obj.TYPE_ADDR {
		// Handle things like "MOVL $sym, (SP)" or "PUSHL $sym" by rewriting
		// to "MOVL $sym, CX; MOVL CX, (SP)" or "MOVL $sym, CX; PUSHL CX"
		// respectively.
		if p.To.Type != obj.TYPE_REG {
			q := obj.Appendp(ctxt, p)
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

	if !isName(&p.From) && !isName(&p.To) && (p.From3 == nil || !isName(p.From3)) {
		return
	}
	var dst int16 = REG_CX
	if (p.As == ALEAL || p.As == AMOVL) && p.To.Reg != p.From.Reg && p.To.Reg != p.From.Index {
		dst = p.To.Reg
		// Why?  See the comment near the top of rewriteToUseGot above.
		// AMOVLs might be introduced by the GOT rewrites.
	}
	q := obj.Appendp(ctxt, p)
	q.RegTo2 = 1
	r := obj.Appendp(ctxt, q)
	r.RegTo2 = 1
	q.As = obj.ACALL
	q.To.Sym = obj.Linklookup(ctxt, "__x86.get_pc_thunk."+strings.ToLower(Rconv(int(dst))), 0)
	q.To.Type = obj.TYPE_MEM
	q.To.Name = obj.NAME_EXTERN
	q.To.Sym.Set(obj.AttrLocal, true)
	r.As = p.As
	r.Scond = p.Scond
	r.From = p.From
	r.From3 = p.From3
	r.Reg = p.Reg
	r.To = p.To
	if isName(&p.From) {
		r.From.Reg = dst
	}
	if isName(&p.To) {
		r.To.Reg = dst
	}
	if p.From3 != nil && isName(p.From3) {
		r.From3.Reg = dst
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

func preprocess(ctxt *obj.Link, cursym *obj.LSym) {
	if ctxt.Headtype == obj.Hplan9 && ctxt.Plan9privates == nil {
		ctxt.Plan9privates = obj.Linklookup(ctxt, "_privates", 0)
	}

	ctxt.Cursym = cursym

	if cursym.Text == nil || cursym.Text.Link == nil {
		return
	}

	p := cursym.Text
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
	if p.Mode == 64 && ctxt.Framepointer_enabled &&
		p.From3.Offset&obj.NOFRAME == 0 && // (1) below
		!(autoffset == 0 && p.From3.Offset&obj.NOSPLIT != 0) && // (2) below
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
	cursym.Args = int32(textarg)
	cursym.Locals = int32(p.To.Offset)

	// TODO(rsc): Remove.
	if p.Mode == 32 && cursym.Locals < 0 {
		cursym.Locals = 0
	}

	// TODO(rsc): Remove 'p.Mode == 64 &&'.
	if p.Mode == 64 && autoffset < obj.StackSmall && p.From3Offset()&obj.NOSPLIT == 0 {
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
				if autoffset >= obj.StackSmall-8 {
					leaf = false
					break LeafSearch
				}
			}
		}

		if leaf {
			p.From3.Offset |= obj.NOSPLIT
		}
	}

	if p.From3Offset()&obj.NOSPLIT == 0 || p.From3Offset()&obj.WRAPPER != 0 {
		p = obj.Appendp(ctxt, p)
		p = load_g_cx(ctxt, p) // load g into CX
	}

	if cursym.Text.From3Offset()&obj.NOSPLIT == 0 {
		p = stacksplit(ctxt, p, autoffset, int32(textarg)) // emit split check
	}

	if autoffset != 0 {
		if autoffset%int32(ctxt.Arch.RegSize) != 0 {
			ctxt.Diag("unaligned stack size %d", autoffset)
		}
		p = obj.Appendp(ctxt, p)
		p.As = AADJSP
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(autoffset)
		p.Spadj = autoffset
	}

	deltasp := autoffset

	if bpsize > 0 {
		// Save caller's BP
		p = obj.Appendp(ctxt, p)

		p.As = AMOVQ
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_BP
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = REG_SP
		p.To.Scale = 1
		p.To.Offset = int64(autoffset) - int64(bpsize)

		// Move current frame to BP
		p = obj.Appendp(ctxt, p)

		p.As = ALEAQ
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = REG_SP
		p.From.Scale = 1
		p.From.Offset = int64(autoffset) - int64(bpsize)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_BP
	}

	if cursym.Text.From3Offset()&obj.WRAPPER != 0 {
		// if(g->panic != nil && g->panic->argp == FP) g->panic->argp = bottom-of-frame
		//
		//	MOVQ g_panic(CX), BX
		//	TESTQ BX, BX
		//	JEQ end
		//	LEAQ (autoffset+8)(SP), DI
		//	CMPQ panic_argp(BX), DI
		//	JNE end
		//	MOVQ SP, panic_argp(BX)
		// end:
		//	NOP
		//
		// The NOP is needed to give the jumps somewhere to land.
		// It is a liblink NOP, not an x86 NOP: it encodes to 0 instruction bytes.

		p = obj.Appendp(ctxt, p)

		p.As = AMOVQ
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = REG_CX
		p.From.Offset = 4 * int64(ctxt.Arch.PtrSize) // G.panic
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_BX
		if ctxt.Headtype == obj.Hnacl && p.Mode == 64 {
			p.As = AMOVL
			p.From.Type = obj.TYPE_MEM
			p.From.Reg = REG_R15
			p.From.Scale = 1
			p.From.Index = REG_CX
		}
		if p.Mode == 32 {
			p.As = AMOVL
		}

		p = obj.Appendp(ctxt, p)
		p.As = ATESTQ
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_BX
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_BX
		if ctxt.Headtype == obj.Hnacl || p.Mode == 32 {
			p.As = ATESTL
		}

		p = obj.Appendp(ctxt, p)
		p.As = AJEQ
		p.To.Type = obj.TYPE_BRANCH
		p1 := p

		p = obj.Appendp(ctxt, p)
		p.As = ALEAQ
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = REG_SP
		p.From.Offset = int64(autoffset) + int64(ctxt.Arch.RegSize)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_DI
		if ctxt.Headtype == obj.Hnacl || p.Mode == 32 {
			p.As = ALEAL
		}

		p = obj.Appendp(ctxt, p)
		p.As = ACMPQ
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = REG_BX
		p.From.Offset = 0 // Panic.argp
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_DI
		if ctxt.Headtype == obj.Hnacl && p.Mode == 64 {
			p.As = ACMPL
			p.From.Type = obj.TYPE_MEM
			p.From.Reg = REG_R15
			p.From.Scale = 1
			p.From.Index = REG_BX
		}
		if p.Mode == 32 {
			p.As = ACMPL
		}

		p = obj.Appendp(ctxt, p)
		p.As = AJNE
		p.To.Type = obj.TYPE_BRANCH
		p2 := p

		p = obj.Appendp(ctxt, p)
		p.As = AMOVQ
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_SP
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = REG_BX
		p.To.Offset = 0 // Panic.argp
		if ctxt.Headtype == obj.Hnacl && p.Mode == 64 {
			p.As = AMOVL
			p.To.Type = obj.TYPE_MEM
			p.To.Reg = REG_R15
			p.To.Scale = 1
			p.To.Index = REG_BX
		}
		if p.Mode == 32 {
			p.As = AMOVL
		}

		p = obj.Appendp(ctxt, p)
		p.As = obj.ANOP
		p1.Pcond = p
		p2.Pcond = p
	}

	for ; p != nil; p = p.Link {
		pcsize := int(p.Mode) / 8
		switch p.From.Name {
		case obj.NAME_AUTO:
			p.From.Offset += int64(deltasp) - int64(bpsize)
		case obj.NAME_PARAM:
			p.From.Offset += int64(deltasp) + int64(pcsize)
		}
		if p.From3 != nil {
			switch p.From3.Name {
			case obj.NAME_AUTO:
				p.From3.Offset += int64(deltasp) - int64(bpsize)
			case obj.NAME_PARAM:
				p.From3.Offset += int64(deltasp) + int64(pcsize)
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
			if bpsize > 0 {
				// Restore caller's BP
				p.As = AMOVQ

				p.From.Type = obj.TYPE_MEM
				p.From.Reg = REG_SP
				p.From.Scale = 1
				p.From.Offset = int64(autoffset) - int64(bpsize)
				p.To.Type = obj.TYPE_REG
				p.To.Reg = REG_BP
				p = obj.Appendp(ctxt, p)
			}

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

func isZeroArgRuntimeCall(s *obj.LSym) bool {
	if s == nil {
		return false
	}
	switch s.Name {
	case "runtime.panicindex", "runtime.panicslice", "runtime.panicdivide":
		return true
	}
	return false
}

func indir_cx(ctxt *obj.Link, p *obj.Prog, a *obj.Addr) {
	if ctxt.Headtype == obj.Hnacl && p.Mode == 64 {
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
func load_g_cx(ctxt *obj.Link, p *obj.Prog) *obj.Prog {
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
func stacksplit(ctxt *obj.Link, p *obj.Prog, framesize int32, textarg int32) *obj.Prog {
	cmp := ACMPQ
	lea := ALEAQ
	mov := AMOVQ
	sub := ASUBQ

	if ctxt.Headtype == obj.Hnacl || p.Mode == 32 {
		cmp = ACMPL
		lea = ALEAL
		mov = AMOVL
		sub = ASUBL
	}

	var q1 *obj.Prog
	if framesize <= obj.StackSmall {
		// small stack: SP <= stackguard
		//	CMPQ SP, stackguard
		p = obj.Appendp(ctxt, p)

		p.As = cmp
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_SP
		indir_cx(ctxt, p, &p.To)
		p.To.Offset = 2 * int64(ctxt.Arch.PtrSize) // G.stackguard0
		if ctxt.Cursym.CFunc() {
			p.To.Offset = 3 * int64(ctxt.Arch.PtrSize) // G.stackguard1
		}
	} else if framesize <= obj.StackBig {
		// large stack: SP-framesize <= stackguard-StackSmall
		//	LEAQ -xxx(SP), AX
		//	CMPQ AX, stackguard
		p = obj.Appendp(ctxt, p)

		p.As = lea
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = REG_SP
		p.From.Offset = -(int64(framesize) - obj.StackSmall)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_AX

		p = obj.Appendp(ctxt, p)
		p.As = cmp
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_AX
		indir_cx(ctxt, p, &p.To)
		p.To.Offset = 2 * int64(ctxt.Arch.PtrSize) // G.stackguard0
		if ctxt.Cursym.CFunc() {
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

		p = obj.Appendp(ctxt, p)

		p.As = mov
		indir_cx(ctxt, p, &p.From)
		p.From.Offset = 2 * int64(ctxt.Arch.PtrSize) // G.stackguard0
		if ctxt.Cursym.CFunc() {
			p.From.Offset = 3 * int64(ctxt.Arch.PtrSize) // G.stackguard1
		}
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_SI

		p = obj.Appendp(ctxt, p)
		p.As = cmp
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_SI
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = obj.StackPreempt
		if p.Mode == 32 {
			p.To.Offset = int64(uint32(obj.StackPreempt & (1<<32 - 1)))
		}

		p = obj.Appendp(ctxt, p)
		p.As = AJEQ
		p.To.Type = obj.TYPE_BRANCH
		q1 = p

		p = obj.Appendp(ctxt, p)
		p.As = lea
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = REG_SP
		p.From.Offset = obj.StackGuard
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_AX

		p = obj.Appendp(ctxt, p)
		p.As = sub
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_SI
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_AX

		p = obj.Appendp(ctxt, p)
		p.As = cmp
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_AX
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = int64(framesize) + (obj.StackGuard - obj.StackSmall)
	}

	// common
	jls := obj.Appendp(ctxt, p)
	jls.As = AJLS
	jls.To.Type = obj.TYPE_BRANCH

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

	call := obj.Appendp(ctxt, pcdata)
	call.Lineno = ctxt.Cursym.Text.Lineno
	call.Mode = ctxt.Cursym.Text.Mode
	call.As = obj.ACALL
	call.To.Type = obj.TYPE_BRANCH
	call.To.Name = obj.NAME_EXTERN
	morestack := "runtime.morestack"
	switch {
	case ctxt.Cursym.CFunc():
		morestack = "runtime.morestackc"
	case ctxt.Cursym.Text.From3Offset()&obj.NEEDCTXT == 0:
		morestack = "runtime.morestack_noctxt"
	}
	call.To.Sym = obj.Linklookup(ctxt, morestack, 0)
	// When compiling 386 code for dynamic linking, the call needs to be adjusted
	// to follow PIC rules. This in turn can insert more instructions, so we need
	// to keep track of the start of the call (where the jump will be to) and the
	// end (which following instructions are appended to).
	callend := call
	progedit(ctxt, callend)
	for ; callend.Link != nil; callend = callend.Link {
		progedit(ctxt, callend.Link)
	}

	jmp := obj.Appendp(ctxt, callend)
	jmp.As = obj.AJMP
	jmp.To.Type = obj.TYPE_BRANCH
	jmp.Pcond = ctxt.Cursym.Text.Link
	jmp.Spadj = +framesize

	jls.Pcond = call
	if q1 != nil {
		q1.Pcond = call
	}

	return jls
}

func follow(ctxt *obj.Link, s *obj.LSym) {
	ctxt.Cursym = s

	firstp := ctxt.NewProg()
	lastp := firstp
	xfol(ctxt, s.Text, &lastp)
	lastp.Link = nil
	s.Text = firstp.Link
}

func nofollow(a obj.As) bool {
	switch a {
	case obj.AJMP,
		obj.ARET,
		AIRETL,
		AIRETQ,
		AIRETW,
		ARETFL,
		ARETFQ,
		ARETFW,
		obj.AUNDEF:
		return true
	}

	return false
}

func pushpop(a obj.As) bool {
	switch a {
	case APUSHL,
		APUSHFL,
		APUSHQ,
		APUSHFQ,
		APUSHW,
		APUSHFW,
		APOPL,
		APOPFL,
		APOPQ,
		APOPFQ,
		APOPW,
		APOPFW:
		return true
	}

	return false
}

func relinv(a obj.As) obj.As {
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

	log.Fatalf("unknown relation: %s", a)
	return 0
}

func xfol(ctxt *obj.Link, p *obj.Prog, last **obj.Prog) {
	var q *obj.Prog
	var i int
	var a obj.As

loop:
	if p == nil {
		return
	}
	if p.As == obj.AJMP {
		q = p.Pcond
		if q != nil && q.As != obj.ATEXT {
			/* mark instruction as done and continue layout at target of jump */
			p.Mark |= DONE

			p = q
			if p.Mark&DONE == 0 {
				goto loop
			}
		}
	}

	if p.Mark&DONE != 0 {
		/*
		 * p goes here, but already used it elsewhere.
		 * copy up to 4 instructions or else branch to other copy.
		 */
		i = 0
		q = p
		for ; i < 4; i, q = i+1, q.Link {
			if q == nil {
				break
			}
			if q == *last {
				break
			}
			a = q.As
			if a == obj.ANOP {
				i--
				continue
			}

			if nofollow(a) || pushpop(a) {
				break // NOTE(rsc): arm does goto copy
			}
			if q.Pcond == nil || q.Pcond.Mark&DONE != 0 {
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
				q.Mark |= DONE
				(*last).Link = q
				*last = q
				if q.As != a || q.Pcond == nil || q.Pcond.Mark&DONE != 0 {
					continue
				}

				q.As = relinv(q.As)
				p = q.Pcond
				q.Pcond = q.Link
				q.Link = p
				xfol(ctxt, q.Link, last)
				p = q.Link
				if p.Mark&DONE != 0 {
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
	p.Mark |= DONE

	(*last).Link = p
	*last = p
	a = p.As

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
				p.As = relinv(a)

				q = p.Link
				p.Link = p.Pcond
				p.Pcond = q
			}
		} else {
			q = p.Link
			if q.Mark&DONE != 0 {
				if a != ALOOP {
					p.As = relinv(a)
					p.Link = p.Pcond
					p.Pcond = q
				}
			}
		}

		xfol(ctxt, p.Link, last)
		if p.Pcond.Mark&DONE != 0 {
			return
		}
		p = p.Pcond
		goto loop
	}

	p = p.Link
	goto loop
}

var unaryDst = map[obj.As]bool{
	ABSWAPL:    true,
	ABSWAPQ:    true,
	ACMPXCHG8B: true,
	ADECB:      true,
	ADECL:      true,
	ADECQ:      true,
	ADECW:      true,
	AINCB:      true,
	AINCL:      true,
	AINCQ:      true,
	AINCW:      true,
	ANEGB:      true,
	ANEGL:      true,
	ANEGQ:      true,
	ANEGW:      true,
	ANOTB:      true,
	ANOTL:      true,
	ANOTQ:      true,
	ANOTW:      true,
	APOPL:      true,
	APOPQ:      true,
	APOPW:      true,
	ASETCC:     true,
	ASETCS:     true,
	ASETEQ:     true,
	ASETGE:     true,
	ASETGT:     true,
	ASETHI:     true,
	ASETLE:     true,
	ASETLS:     true,
	ASETLT:     true,
	ASETMI:     true,
	ASETNE:     true,
	ASETOC:     true,
	ASETOS:     true,
	ASETPC:     true,
	ASETPL:     true,
	ASETPS:     true,
	AFFREE:     true,
	AFLDENV:    true,
	AFSAVE:     true,
	AFSTCW:     true,
	AFSTENV:    true,
	AFSTSW:     true,
	AFXSAVE:    true,
	AFXSAVE64:  true,
	ASTMXCSR:   true,
}

var Linkamd64 = obj.LinkArch{
	Arch:       sys.ArchAMD64,
	Preprocess: preprocess,
	Assemble:   span6,
	Follow:     follow,
	Progedit:   progedit,
	UnaryDst:   unaryDst,
}

var Linkamd64p32 = obj.LinkArch{
	Arch:       sys.ArchAMD64P32,
	Preprocess: preprocess,
	Assemble:   span6,
	Follow:     follow,
	Progedit:   progedit,
	UnaryDst:   unaryDst,
}

var Link386 = obj.LinkArch{
	Arch:       sys.Arch386,
	Preprocess: preprocess,
	Assemble:   span6,
	Follow:     follow,
	Progedit:   progedit,
	UnaryDst:   unaryDst,
}
