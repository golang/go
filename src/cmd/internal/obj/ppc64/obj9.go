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

package ppc64

import (
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"cmd/internal/sys"
	"internal/abi"
	"internal/buildcfg"
	"log"
	"math/bits"
	"strings"
)

// Test if this value can encoded as a mask for
// li -1, rx; rlic rx,rx,sh,mb.
// Masks can also extend from the msb and wrap to
// the lsb too. That is, the valid masks are 32 bit strings
// of the form: 0..01..10..0 or 1..10..01..1 or 1...1
func isPPC64DoublewordRotateMask(v64 int64) bool {
	// Isolate rightmost 1 (if none 0) and add.
	v := uint64(v64)
	vp := (v & -v) + v
	// Likewise, for the wrapping case.
	vn := ^v
	vpn := (vn & -vn) + vn
	return (v&vp == 0 || vn&vpn == 0) && v != 0
}

// Encode a doubleword rotate mask into mb (mask begin) and
// me (mask end, inclusive). Note, POWER ISA labels bits in
// big endian order.
func encodePPC64RLDCMask(mask int64) (mb, me int) {
	// Determine boundaries and then decode them
	mb = bits.LeadingZeros64(uint64(mask))
	me = 64 - bits.TrailingZeros64(uint64(mask))
	mbn := bits.LeadingZeros64(^uint64(mask))
	men := 64 - bits.TrailingZeros64(^uint64(mask))
	// Check for a wrapping mask (e.g bits at 0 and 63)
	if mb == 0 && me == 64 {
		// swap the inverted values
		mb, me = men, mbn
	}
	// Note, me is inclusive.
	return mb, me - 1
}

// Is this a symbol which should never have a TOC prologue generated?
// These are special functions which should not have a TOC regeneration
// prologue.
func isNOTOCfunc(name string) bool {
	switch {
	case name == "runtime.duffzero":
		return true
	case name == "runtime.duffcopy":
		return true
	case strings.HasPrefix(name, "runtime.elf_"):
		return true
	default:
		return false
	}
}

func progedit(ctxt *obj.Link, p *obj.Prog, newprog obj.ProgAlloc) {
	p.From.Class = 0
	p.To.Class = 0

	c := ctxt9{ctxt: ctxt, newprog: newprog}

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
			p.From.Type = obj.TYPE_MEM
			p.From.Sym = ctxt.Float32Sym(f32)
			p.From.Name = obj.NAME_EXTERN
			p.From.Offset = 0
		}

	case AFMOVD:
		if p.From.Type == obj.TYPE_FCONST {
			f64 := p.From.Val.(float64)
			// Constant not needed in memory for float +/- 0
			if f64 != 0 {
				p.From.Type = obj.TYPE_MEM
				p.From.Sym = ctxt.Float64Sym(f64)
				p.From.Name = obj.NAME_EXTERN
				p.From.Offset = 0
			}
		}

	case AMOVW, AMOVWZ:
		// Note, for backwards compatibility, MOVW $const, Rx and MOVWZ $const, Rx are identical.
		if p.From.Type == obj.TYPE_CONST && p.From.Offset != 0 && p.From.Offset&0xFFFF == 0 {
			// This is a constant shifted 16 bits to the left, convert it to ADDIS/ORIS $const,...
			p.As = AADDIS
			// Use ORIS for large constants which should not be sign extended.
			if p.From.Offset >= 0x80000000 {
				p.As = AORIS
			}
			p.Reg = REG_R0
			p.From.Offset >>= 16
		}

	case AMOVD:
		// Skip this opcode if it is not a constant load.
		if p.From.Type != obj.TYPE_CONST || p.From.Name != obj.NAME_NONE || p.From.Reg != 0 {
			break
		}

		// 32b constants (signed and unsigned) can be generated via 1 or 2 instructions. They can be assembled directly.
		isS32 := int64(int32(p.From.Offset)) == p.From.Offset
		isU32 := uint64(uint32(p.From.Offset)) == uint64(p.From.Offset)
		// If prefixed instructions are supported, a 34b signed constant can be generated by one pli instruction.
		isS34 := pfxEnabled && (p.From.Offset<<30)>>30 == p.From.Offset

		// Try converting MOVD $const,Rx into ADDIS/ORIS $s32>>16,R0,Rx
		switch {
		case isS32 && p.From.Offset&0xFFFF == 0 && p.From.Offset != 0:
			p.As = AADDIS
			p.From.Offset >>= 16
			p.Reg = REG_R0

		case isU32 && p.From.Offset&0xFFFF == 0 && p.From.Offset != 0:
			p.As = AORIS
			p.From.Offset >>= 16
			p.Reg = REG_R0

		case isS32 || isU32 || isS34:
			// The assembler can generate this opcode in 1 (on Power10) or 2 opcodes.

		// Otherwise, see if the large constant can be generated with 2 instructions. If not, load it from memory.
		default:
			// Is this a shifted 16b constant? If so, rewrite it to avoid a creating and loading a constant.
			val := p.From.Offset
			shift := bits.TrailingZeros64(uint64(val))
			mask := 0xFFFF << shift
			if val&int64(mask) == val || (val>>(shift+16) == -1 && (val>>shift)<<shift == val) {
				// Rewrite this value into MOVD $const>>shift, Rto; SLD $shift, Rto
				q := obj.Appendp(p, c.newprog)
				q.As = ASLD
				q.From.SetConst(int64(shift))
				q.To = p.To
				p.From.Offset >>= shift
				p = q
			} else if isPPC64DoublewordRotateMask(val) {
				// This constant is a mask value, generate MOVD $-1, Rto; RLDIC Rto, ^me, mb, Rto
				mb, me := encodePPC64RLDCMask(val)
				q := obj.Appendp(p, c.newprog)
				q.As = ARLDC
				q.AddRestSourceConst((^int64(me)) & 0x3F)
				q.AddRestSourceConst(int64(mb))
				q.From = p.To
				q.To = p.To
				p.From.Offset = -1
				p = q
			} else {
				// Load the constant from memory.
				p.From.Type = obj.TYPE_MEM
				p.From.Sym = ctxt.Int64Sym(p.From.Offset)
				p.From.Name = obj.NAME_EXTERN
				p.From.Offset = 0
			}
		}
	}

	switch p.As {
	// Rewrite SUB constants into ADD.
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
		if p.From.Type != obj.TYPE_CONST {
			break
		}
		// Rewrite SUB $const,... into ADD $-const,...
		p.From.Offset = -p.From.Offset
		p.As = AADD
		// This is now an ADD opcode, try simplifying it below.
		fallthrough

	// Rewrite ADD/OR/XOR/ANDCC $const,... forms into ADDIS/ORIS/XORIS/ANDISCC
	case AADD:
		// Don't rewrite if this is not adding a constant value, or is not an int32
		if p.From.Type != obj.TYPE_CONST || p.From.Offset == 0 || int64(int32(p.From.Offset)) != p.From.Offset {
			break
		}
		if p.From.Offset&0xFFFF == 0 {
			// The constant can be added using ADDIS
			p.As = AADDIS
			p.From.Offset >>= 16
		} else if buildcfg.GOPPC64 >= 10 {
			// Let the assembler generate paddi for large constants.
			break
		} else if (p.From.Offset < -0x8000 && int64(int32(p.From.Offset)) == p.From.Offset) || (p.From.Offset > 0xFFFF && p.From.Offset < 0x7FFF8000) {
			// For a constant x, 0xFFFF (UINT16_MAX) < x < 0x7FFF8000 or -0x80000000 (INT32_MIN) <= x < -0x8000 (INT16_MIN)
			// This is not done for 0x7FFF < x < 0x10000; the assembler will generate a slightly faster instruction sequence.
			//
			// The constant x can be rewritten as ADDIS + ADD as follows:
			//     ADDIS $x>>16 + (x>>15)&1, rX, rY
			//     ADD   $int64(int16(x)), rY, rY
			// The range is slightly asymmetric as 0x7FFF8000 and above overflow the sign bit, whereas for
			// negative values, this would happen with constant values between -1 and -32768 which can
			// assemble into a single addi.
			is := p.From.Offset>>16 + (p.From.Offset>>15)&1
			i := int64(int16(p.From.Offset))
			p.As = AADDIS
			p.From.Offset = is
			q := obj.Appendp(p, c.newprog)
			q.As = AADD
			q.From.SetConst(i)
			q.Reg = p.To.Reg
			q.To = p.To
			p = q
		}
	case AOR:
		if p.From.Type == obj.TYPE_CONST && uint64(p.From.Offset)&0xFFFFFFFF0000FFFF == 0 && p.From.Offset != 0 {
			p.As = AORIS
			p.From.Offset >>= 16
		}
	case AXOR:
		if p.From.Type == obj.TYPE_CONST && uint64(p.From.Offset)&0xFFFFFFFF0000FFFF == 0 && p.From.Offset != 0 {
			p.As = AXORIS
			p.From.Offset >>= 16
		}
	case AANDCC:
		if p.From.Type == obj.TYPE_CONST && uint64(p.From.Offset)&0xFFFFFFFF0000FFFF == 0 && p.From.Offset != 0 {
			p.As = AANDISCC
			p.From.Offset >>= 16
		}

	// To maintain backwards compatibility, we accept some 4 argument usage of
	// several opcodes which was likely not intended, but did work. These are not
	// added to optab to avoid the chance this behavior might be used with newer
	// instructions.
	//
	// Rewrite argument ordering like "ADDEX R3, $3, R4, R5" into
	//                                "ADDEX R3, R4, $3, R5"
	case AVSHASIGMAW, AVSHASIGMAD, AADDEX, AXXSLDWI, AXXPERMDI:
		if len(p.RestArgs) == 2 && p.Reg == 0 && p.RestArgs[0].Addr.Type == obj.TYPE_CONST && p.RestArgs[1].Addr.Type == obj.TYPE_REG {
			p.Reg = p.RestArgs[1].Addr.Reg
			p.RestArgs = p.RestArgs[:1]
		}
	}

	if c.ctxt.Headtype == objabi.Haix {
		c.rewriteToUseTOC(p)
	} else if c.ctxt.Flag_dynlink {
		c.rewriteToUseGot(p)
	}
}

// Rewrite p, if necessary, to access a symbol using its TOC anchor.
// This code is for AIX only.
func (c *ctxt9) rewriteToUseTOC(p *obj.Prog) {
	if p.As == obj.ATEXT || p.As == obj.AFUNCDATA || p.As == obj.ACALL || p.As == obj.ARET || p.As == obj.AJMP {
		return
	}

	if p.As == obj.ADUFFCOPY || p.As == obj.ADUFFZERO {
		// ADUFFZERO/ADUFFCOPY is considered as an ABL except in dynamic
		// link where it should be an indirect call.
		if !c.ctxt.Flag_dynlink {
			return
		}
		//     ADUFFxxx $offset
		// becomes
		//     MOVD runtime.duffxxx@TOC, R12
		//     ADD $offset, R12
		//     MOVD R12, LR
		//     BL (LR)
		var sym *obj.LSym
		if p.As == obj.ADUFFZERO {
			sym = c.ctxt.Lookup("runtime.duffzero")
		} else {
			sym = c.ctxt.Lookup("runtime.duffcopy")
		}
		// Retrieve or create the TOC anchor.
		symtoc := c.ctxt.LookupInit("TOC."+sym.Name, func(s *obj.LSym) {
			s.Type = objabi.SDATA
			s.Set(obj.AttrDuplicateOK, true)
			s.Set(obj.AttrStatic, true)
			c.ctxt.Data = append(c.ctxt.Data, s)
			s.WriteAddr(c.ctxt, 0, 8, sym, 0)
		})

		offset := p.To.Offset
		p.As = AMOVD
		p.From.Type = obj.TYPE_MEM
		p.From.Name = obj.NAME_TOCREF
		p.From.Sym = symtoc
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R12
		p.To.Name = obj.NAME_NONE
		p.To.Offset = 0
		p.To.Sym = nil
		p1 := obj.Appendp(p, c.newprog)
		p1.As = AADD
		p1.From.Type = obj.TYPE_CONST
		p1.From.Offset = offset
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = REG_R12
		p2 := obj.Appendp(p1, c.newprog)
		p2.As = AMOVD
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = REG_R12
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = REG_LR
		p3 := obj.Appendp(p2, c.newprog)
		p3.As = obj.ACALL
		p3.To.Type = obj.TYPE_REG
		p3.To.Reg = REG_LR
	}

	var source *obj.Addr
	if p.From.Name == obj.NAME_EXTERN || p.From.Name == obj.NAME_STATIC {
		if p.From.Type == obj.TYPE_ADDR {
			if p.As == ADWORD {
				// ADWORD $sym doesn't need TOC anchor
				return
			}
			if p.As != AMOVD {
				c.ctxt.Diag("do not know how to handle TYPE_ADDR in %v", p)
				return
			}
			if p.To.Type != obj.TYPE_REG {
				c.ctxt.Diag("do not know how to handle LEAQ-type insn to non-register in %v", p)
				return
			}
		} else if p.From.Type != obj.TYPE_MEM {
			c.ctxt.Diag("do not know how to handle %v without TYPE_MEM", p)
			return
		}
		source = &p.From

	} else if p.To.Name == obj.NAME_EXTERN || p.To.Name == obj.NAME_STATIC {
		if p.To.Type != obj.TYPE_MEM {
			c.ctxt.Diag("do not know how to handle %v without TYPE_MEM", p)
			return
		}
		if source != nil {
			c.ctxt.Diag("cannot handle symbols on both sides in %v", p)
			return
		}
		source = &p.To
	} else {
		return

	}

	if source.Sym == nil {
		c.ctxt.Diag("do not know how to handle nil symbol in %v", p)
		return
	}

	if source.Sym.Type == objabi.STLSBSS {
		return
	}

	// Retrieve or create the TOC anchor.
	symtoc := c.ctxt.LookupInit("TOC."+source.Sym.Name, func(s *obj.LSym) {
		s.Type = objabi.SDATA
		s.Set(obj.AttrDuplicateOK, true)
		s.Set(obj.AttrStatic, true)
		c.ctxt.Data = append(c.ctxt.Data, s)
		s.WriteAddr(c.ctxt, 0, 8, source.Sym, 0)
	})

	if source.Type == obj.TYPE_ADDR {
		// MOVD $sym, Rx becomes MOVD symtoc, Rx
		// MOVD $sym+<off>, Rx becomes MOVD symtoc, Rx; ADD <off>, Rx
		p.From.Type = obj.TYPE_MEM
		p.From.Sym = symtoc
		p.From.Name = obj.NAME_TOCREF

		if p.From.Offset != 0 {
			q := obj.Appendp(p, c.newprog)
			q.As = AADD
			q.From.Type = obj.TYPE_CONST
			q.From.Offset = p.From.Offset
			p.From.Offset = 0
			q.To = p.To
		}
		return

	}

	// MOVx sym, Ry becomes MOVD symtoc, REGTMP; MOVx (REGTMP), Ry
	// MOVx Ry, sym becomes MOVD symtoc, REGTMP; MOVx Ry, (REGTMP)
	// An addition may be inserted between the two MOVs if there is an offset.

	q := obj.Appendp(p, c.newprog)
	q.As = AMOVD
	q.From.Type = obj.TYPE_MEM
	q.From.Sym = symtoc
	q.From.Name = obj.NAME_TOCREF
	q.To.Type = obj.TYPE_REG
	q.To.Reg = REGTMP

	q = obj.Appendp(q, c.newprog)
	q.As = p.As
	q.From = p.From
	q.To = p.To
	if p.From.Name != obj.NAME_NONE {
		q.From.Type = obj.TYPE_MEM
		q.From.Reg = REGTMP
		q.From.Name = obj.NAME_NONE
		q.From.Sym = nil
	} else if p.To.Name != obj.NAME_NONE {
		q.To.Type = obj.TYPE_MEM
		q.To.Reg = REGTMP
		q.To.Name = obj.NAME_NONE
		q.To.Sym = nil
	} else {
		c.ctxt.Diag("unreachable case in rewriteToUseTOC with %v", p)
	}

	obj.Nopout(p)
}

// Rewrite p, if necessary, to access global data via the global offset table.
func (c *ctxt9) rewriteToUseGot(p *obj.Prog) {
	if p.As == obj.ADUFFCOPY || p.As == obj.ADUFFZERO {
		//     ADUFFxxx $offset
		// becomes
		//     MOVD runtime.duffxxx@GOT, R12
		//     ADD $offset, R12
		//     MOVD R12, LR
		//     BL (LR)
		var sym *obj.LSym
		if p.As == obj.ADUFFZERO {
			sym = c.ctxt.LookupABI("runtime.duffzero", obj.ABIInternal)
		} else {
			sym = c.ctxt.LookupABI("runtime.duffcopy", obj.ABIInternal)
		}
		offset := p.To.Offset
		p.As = AMOVD
		p.From.Type = obj.TYPE_MEM
		p.From.Name = obj.NAME_GOTREF
		p.From.Sym = sym
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R12
		p.To.Name = obj.NAME_NONE
		p.To.Offset = 0
		p.To.Sym = nil
		p1 := obj.Appendp(p, c.newprog)
		p1.As = AADD
		p1.From.Type = obj.TYPE_CONST
		p1.From.Offset = offset
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = REG_R12
		p2 := obj.Appendp(p1, c.newprog)
		p2.As = AMOVD
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = REG_R12
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = REG_LR
		p3 := obj.Appendp(p2, c.newprog)
		p3.As = obj.ACALL
		p3.To.Type = obj.TYPE_REG
		p3.To.Reg = REG_LR
	}

	// We only care about global data: NAME_EXTERN means a global
	// symbol in the Go sense, and p.Sym.Local is true for a few
	// internally defined symbols.
	if p.From.Type == obj.TYPE_ADDR && p.From.Name == obj.NAME_EXTERN && !p.From.Sym.Local() {
		// MOVD $sym, Rx becomes MOVD sym@GOT, Rx
		// MOVD $sym+<off>, Rx becomes MOVD sym@GOT, Rx; ADD <off>, Rx
		if p.As != AMOVD {
			c.ctxt.Diag("do not know how to handle TYPE_ADDR in %v with -dynlink", p)
		}
		if p.To.Type != obj.TYPE_REG {
			c.ctxt.Diag("do not know how to handle LEAQ-type insn to non-register in %v with -dynlink", p)
		}
		p.From.Type = obj.TYPE_MEM
		p.From.Name = obj.NAME_GOTREF
		if p.From.Offset != 0 {
			q := obj.Appendp(p, c.newprog)
			q.As = AADD
			q.From.Type = obj.TYPE_CONST
			q.From.Offset = p.From.Offset
			q.To = p.To
			p.From.Offset = 0
		}
	}
	if p.GetFrom3() != nil && p.GetFrom3().Name == obj.NAME_EXTERN {
		c.ctxt.Diag("don't know how to handle %v with -dynlink", p)
	}
	var source *obj.Addr
	// MOVx sym, Ry becomes MOVD sym@GOT, REGTMP; MOVx (REGTMP), Ry
	// MOVx Ry, sym becomes MOVD sym@GOT, REGTMP; MOVx Ry, (REGTMP)
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
	// TODO(minux): add morestack short-cuts with small fixed frame-size.
	if cursym.Func().Text == nil || cursym.Func().Text.Link == nil {
		return
	}

	c := ctxt9{ctxt: ctxt, cursym: cursym, newprog: newprog}

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
	 * expand RET
	 * expand BECOME pseudo
	 */

	var q *obj.Prog
	var q1 *obj.Prog
	for p := c.cursym.Func().Text; p != nil; p = p.Link {
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
			ALBAR,
			ASTBCCC,
			ASTWCCC,
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
			ALWSYNC,
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
			c.cursym.Func().Text.Mark &^= LEAF
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
			q1 = p.To.Target()
			if q1 != nil {
				// NOPs are not removed due to #40689.

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
			// NOPs are not removed due to
			// #40689
			continue

		default:
			q = p
			continue
		}
	}

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

			if p.Mark&LEAF != 0 && autosize < abi.StackSmall {
				// A leaf function with a small stack can be marked
				// NOSPLIT, avoiding a stack check.
				p.From.Sym.Set(obj.AttrNoSplit, true)
			}

			p.To.Offset = int64(autosize)

			q = p

			if NeedTOCpointer(c.ctxt) && !isNOTOCfunc(c.cursym.Name) {
				// When compiling Go into PIC, without PCrel support, all functions must start
				// with instructions to load the TOC pointer into r2:
				//
				//	addis r2, r12, .TOC.-func@ha
				//	addi r2, r2, .TOC.-func@l+4
				//
				// We could probably skip this prologue in some situations
				// but it's a bit subtle. However, it is both safe and
				// necessary to leave the prologue off duffzero and
				// duffcopy as we rely on being able to jump to a specific
				// instruction offset for them.
				//
				// These are AWORDS because there is no (afaict) way to
				// generate the addis instruction except as part of the
				// load of a large constant, and in that case there is no
				// way to use r12 as the source.
				//
				// Note that the same condition is tested in
				// putelfsym in cmd/link/internal/ld/symtab.go
				// where we set the st_other field to indicate
				// the presence of these instructions.
				q = obj.Appendp(q, c.newprog)
				q.As = AWORD
				q.Pos = p.Pos
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = 0x3c4c0000
				q = obj.Appendp(q, c.newprog)
				q.As = AWORD
				q.Pos = p.Pos
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = 0x38420000
				rel := obj.Addrel(c.cursym)
				rel.Off = 0
				rel.Siz = 8
				rel.Sym = c.ctxt.Lookup(".TOC.")
				rel.Type = objabi.R_ADDRPOWER_PCREL
			}

			if !c.cursym.Func().Text.From.Sym.NoSplit() {
				q = c.stacksplit(q, autosize) // emit split check
			}

			if autosize != 0 {
				var prologueEnd *obj.Prog
				// Save the link register and update the SP.  MOVDU is used unless
				// the frame size is too large.  The link register must be saved
				// even for non-empty leaf functions so that traceback works.
				if autosize >= -BIG && autosize <= BIG {
					// Use MOVDU to adjust R1 when saving R31, if autosize is small.
					q = obj.Appendp(q, c.newprog)
					q.As = AMOVD
					q.Pos = p.Pos
					q.From.Type = obj.TYPE_REG
					q.From.Reg = REG_LR
					q.To.Type = obj.TYPE_REG
					q.To.Reg = REGTMP
					prologueEnd = q

					q = obj.Appendp(q, c.newprog)
					q.As = AMOVDU
					q.Pos = p.Pos
					q.From.Type = obj.TYPE_REG
					q.From.Reg = REGTMP
					q.To.Type = obj.TYPE_MEM
					q.To.Offset = int64(-autosize)
					q.To.Reg = REGSP
					q.Spadj = autosize
				} else {
					// Frame size is too large for a MOVDU instruction.
					// Store link register before decrementing SP, so if a signal comes
					// during the execution of the function prologue, the traceback
					// code will not see a half-updated stack frame.
					// This sequence is not async preemptible, as if we open a frame
					// at the current SP, it will clobber the saved LR.
					q = obj.Appendp(q, c.newprog)
					q.As = AMOVD
					q.Pos = p.Pos
					q.From.Type = obj.TYPE_REG
					q.From.Reg = REG_LR
					q.To.Type = obj.TYPE_REG
					q.To.Reg = REG_R29 // REGTMP may be used to synthesize large offset in the next instruction

					q = c.ctxt.StartUnsafePoint(q, c.newprog)

					q = obj.Appendp(q, c.newprog)
					q.As = AMOVD
					q.Pos = p.Pos
					q.From.Type = obj.TYPE_REG
					q.From.Reg = REG_R29
					q.To.Type = obj.TYPE_MEM
					q.To.Offset = int64(-autosize)
					q.To.Reg = REGSP

					prologueEnd = q

					q = obj.Appendp(q, c.newprog)
					q.As = AADD
					q.Pos = p.Pos
					q.From.Type = obj.TYPE_CONST
					q.From.Offset = int64(-autosize)
					q.To.Type = obj.TYPE_REG
					q.To.Reg = REGSP
					q.Spadj = +autosize

					q = c.ctxt.EndUnsafePoint(q, c.newprog, -1)
				}
				prologueEnd.Pos = prologueEnd.Pos.WithXlogue(src.PosPrologueEnd)
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

			if NeedTOCpointer(c.ctxt) {
				q = obj.Appendp(q, c.newprog)
				q.As = AMOVD
				q.Pos = p.Pos
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R2
				q.To.Type = obj.TYPE_MEM
				q.To.Reg = REGSP
				q.To.Offset = 24
			}

			if c.cursym.Func().Text.From.Sym.Wrapper() {
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

				q = obj.Appendp(q, c.newprog)

				q.As = AMOVD
				q.From.Type = obj.TYPE_MEM
				q.From.Reg = REGG
				q.From.Offset = 4 * int64(c.ctxt.Arch.PtrSize) // G.panic
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R22

				q = obj.Appendp(q, c.newprog)
				q.As = ACMP
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R0
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R22

				q = obj.Appendp(q, c.newprog)
				q.As = ABEQ
				q.To.Type = obj.TYPE_BRANCH
				p1 = q

				q = obj.Appendp(q, c.newprog)
				q.As = AMOVD
				q.From.Type = obj.TYPE_MEM
				q.From.Reg = REG_R22
				q.From.Offset = 0 // Panic.argp
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R23

				q = obj.Appendp(q, c.newprog)
				q.As = AADD
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = int64(autosize) + c.ctxt.Arch.FixedFrameSize
				q.Reg = REGSP
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R24

				q = obj.Appendp(q, c.newprog)
				q.As = ACMP
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R23
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R24

				q = obj.Appendp(q, c.newprog)
				q.As = ABNE
				q.To.Type = obj.TYPE_BRANCH
				p2 = q

				q = obj.Appendp(q, c.newprog)
				q.As = AADD
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = c.ctxt.Arch.FixedFrameSize
				q.Reg = REGSP
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REG_R25

				q = obj.Appendp(q, c.newprog)
				q.As = AMOVD
				q.From.Type = obj.TYPE_REG
				q.From.Reg = REG_R25
				q.To.Type = obj.TYPE_MEM
				q.To.Reg = REG_R22
				q.To.Offset = 0 // Panic.argp

				q = obj.Appendp(q, c.newprog)

				q.As = obj.ANOP
				p1.To.SetTarget(q)
				p2.To.SetTarget(q)
			}

		case obj.ARET:
			if p.From.Type == obj.TYPE_CONST {
				c.ctxt.Diag("using BECOME (%v) is not supported!", p)
				break
			}

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

				q = c.newprog()
				q.As = ABR
				q.Pos = p.Pos
				if retTarget == nil {
					q.To.Type = obj.TYPE_REG
					q.To.Reg = REG_LR
				} else {
					q.To.Type = obj.TYPE_BRANCH
					q.To.Sym = retTarget
				}
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

			q = c.newprog()
			q.As = AMOVD
			q.Pos = p.Pos
			q.From.Type = obj.TYPE_REG
			q.From.Reg = REGTMP
			q.To.Type = obj.TYPE_REG
			q.To.Reg = REG_LR

			q.Link = p.Link
			p.Link = q
			p = q

			if false {
				// Debug bad returns
				q = c.newprog()

				q.As = AMOVD
				q.Pos = p.Pos
				q.From.Type = obj.TYPE_MEM
				q.From.Offset = 0
				q.From.Reg = REGTMP
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REGTMP

				q.Link = p.Link
				p.Link = q
				p = q
			}
			prev := p
			if autosize != 0 {
				q = c.newprog()
				q.As = AADD
				q.Pos = p.Pos
				q.From.Type = obj.TYPE_CONST
				q.From.Offset = int64(autosize)
				q.To.Type = obj.TYPE_REG
				q.To.Reg = REGSP
				q.Spadj = -autosize

				q.Link = p.Link
				prev.Link = q
				prev = q
			}

			q1 = c.newprog()
			q1.As = ABR
			q1.Pos = p.Pos
			if retTarget == nil {
				q1.To.Type = obj.TYPE_REG
				q1.To.Reg = REG_LR
			} else {
				q1.To.Type = obj.TYPE_BRANCH
				q1.To.Sym = retTarget
			}
			q1.Mark |= BRANCH
			q1.Spadj = +autosize

			q1.Link = q.Link
			prev.Link = q1
		case AADD:
			if p.To.Type == obj.TYPE_REG && p.To.Reg == REGSP && p.From.Type == obj.TYPE_CONST {
				p.Spadj = int32(-p.From.Offset)
			}
		case AMOVDU:
			if p.To.Type == obj.TYPE_MEM && p.To.Reg == REGSP {
				p.Spadj = int32(-p.To.Offset)
			}
			if p.From.Type == obj.TYPE_MEM && p.From.Reg == REGSP {
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

		if p.To.Type == obj.TYPE_REG && p.To.Reg == REGSP && p.Spadj == 0 && p.As != ACMPU {
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
func (c *ctxt9) stacksplit(p *obj.Prog, framesize int32) *obj.Prog {
	if c.ctxt.Flag_maymorestack != "" {
		if c.ctxt.Flag_shared || c.ctxt.Flag_dynlink {
			// See the call to morestack for why these are
			// complicated to support.
			c.ctxt.Diag("maymorestack with -shared or -dynlink is not supported")
		}

		// Spill arguments. This has to happen before we open
		// any more frame space.
		p = c.cursym.Func().SpillRegisterArgs(p, c.newprog)

		// Save LR and REGCTXT
		frameSize := 8 + c.ctxt.Arch.FixedFrameSize

		// MOVD LR, REGTMP
		p = obj.Appendp(p, c.newprog)
		p.As = AMOVD
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_LR
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REGTMP
		// MOVDU REGTMP, -16(SP)
		p = obj.Appendp(p, c.newprog)
		p.As = AMOVDU
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REGTMP
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = -frameSize
		p.To.Reg = REGSP
		p.Spadj = int32(frameSize)

		// MOVD REGCTXT, 8(SP)
		p = obj.Appendp(p, c.newprog)
		p.As = AMOVD
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REGCTXT
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = 8
		p.To.Reg = REGSP

		// BL maymorestack
		p = obj.Appendp(p, c.newprog)
		p.As = ABL
		p.To.Type = obj.TYPE_BRANCH
		// See ../x86/obj6.go
		p.To.Sym = c.ctxt.LookupABI(c.ctxt.Flag_maymorestack, c.cursym.ABI())

		// Restore LR and REGCTXT

		// MOVD 8(SP), REGCTXT
		p = obj.Appendp(p, c.newprog)
		p.As = AMOVD
		p.From.Type = obj.TYPE_MEM
		p.From.Offset = 8
		p.From.Reg = REGSP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REGCTXT

		// MOVD 0(SP), REGTMP
		p = obj.Appendp(p, c.newprog)
		p.As = AMOVD
		p.From.Type = obj.TYPE_MEM
		p.From.Offset = 0
		p.From.Reg = REGSP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REGTMP

		// MOVD REGTMP, LR
		p = obj.Appendp(p, c.newprog)
		p.As = AMOVD
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REGTMP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_LR

		// ADD $16, SP
		p = obj.Appendp(p, c.newprog)
		p.As = AADD
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = frameSize
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REGSP
		p.Spadj = -int32(frameSize)

		// Unspill arguments.
		p = c.cursym.Func().UnspillRegisterArgs(p, c.newprog)
	}

	// save entry point, but skipping the two instructions setting R2 in shared mode and maymorestack
	startPred := p

	// MOVD	g_stackguard(g), R22
	p = obj.Appendp(p, c.newprog)

	p.As = AMOVD
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = REGG
	p.From.Offset = 2 * int64(c.ctxt.Arch.PtrSize) // G.stackguard0
	if c.cursym.CFunc() {
		p.From.Offset = 3 * int64(c.ctxt.Arch.PtrSize) // G.stackguard1
	}
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_R22

	// Mark the stack bound check and morestack call async nonpreemptible.
	// If we get preempted here, when resumed the preemption request is
	// cleared, but we'll still call morestack, which will double the stack
	// unnecessarily. See issue #35470.
	p = c.ctxt.StartUnsafePoint(p, c.newprog)

	var q *obj.Prog
	if framesize <= abi.StackSmall {
		// small stack: SP < stackguard
		//	CMP	stackguard, SP
		p = obj.Appendp(p, c.newprog)

		p.As = ACMPU
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R22
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REGSP
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
			//	CMPU	SP, $(framesize-StackSmall)
			//	BLT	label-of-call-to-morestack
			if offset <= 0xffff {
				p = obj.Appendp(p, c.newprog)
				p.As = ACMPU
				p.From.Type = obj.TYPE_REG
				p.From.Reg = REGSP
				p.To.Type = obj.TYPE_CONST
				p.To.Offset = offset
			} else {
				// Constant is too big for CMPU.
				p = obj.Appendp(p, c.newprog)
				p.As = AMOVD
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = offset
				p.To.Type = obj.TYPE_REG
				p.To.Reg = REG_R23

				p = obj.Appendp(p, c.newprog)
				p.As = ACMPU
				p.From.Type = obj.TYPE_REG
				p.From.Reg = REGSP
				p.To.Type = obj.TYPE_REG
				p.To.Reg = REG_R23
			}

			p = obj.Appendp(p, c.newprog)
			q = p
			p.As = ABLT
			p.To.Type = obj.TYPE_BRANCH
		}

		// Check against the stack guard. We've ensured this won't underflow.
		//	ADD  $-(framesize-StackSmall), SP, R4
		//	CMPU stackguard, R4
		p = obj.Appendp(p, c.newprog)

		p.As = AADD
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = -offset
		p.Reg = REGSP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R23

		p = obj.Appendp(p, c.newprog)
		p.As = ACMPU
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R22
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R23
	}

	// q1: BLT	done
	p = obj.Appendp(p, c.newprog)
	q1 := p

	p.As = ABLT
	p.To.Type = obj.TYPE_BRANCH

	p = obj.Appendp(p, c.newprog)
	p.As = obj.ANOP // zero-width place holder

	if q != nil {
		q.To.SetTarget(p)
	}

	// Spill the register args that could be clobbered by the
	// morestack code.

	spill := c.cursym.Func().SpillRegisterArgs(p, c.newprog)

	// MOVD LR, R5
	p = obj.Appendp(spill, c.newprog)
	p.As = AMOVD
	p.From.Type = obj.TYPE_REG
	p.From.Reg = REG_LR
	p.To.Type = obj.TYPE_REG
	p.To.Reg = REG_R5

	p = c.ctxt.EmitEntryStackMap(c.cursym, p, c.newprog)

	var morestacksym *obj.LSym
	if c.cursym.CFunc() {
		morestacksym = c.ctxt.Lookup("runtime.morestackc")
	} else if !c.cursym.Func().Text.From.Sym.NeedCtxt() {
		morestacksym = c.ctxt.Lookup("runtime.morestack_noctxt")
	} else {
		morestacksym = c.ctxt.Lookup("runtime.morestack")
	}

	if NeedTOCpointer(c.ctxt) {
		// In PPC64 PIC code, R2 is used as TOC pointer derived from R12
		// which is the address of function entry point when entering
		// the function. We need to preserve R2 across call to morestack.
		// Fortunately, in shared mode, 8(SP) and 16(SP) are reserved in
		// the caller's frame, but not used (0(SP) is caller's saved LR,
		// 24(SP) is caller's saved R2). Use 8(SP) to save this function's R2.
		// MOVD R2, 8(SP)
		p = obj.Appendp(p, c.newprog)
		p.As = AMOVD
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R2
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = REGSP
		p.To.Offset = 8
	}

	if c.ctxt.Flag_dynlink {
		// Avoid calling morestack via a PLT when dynamically linking. The
		// PLT stubs generated by the system linker on ppc64le when "std r2,
		// 24(r1)" to save the TOC pointer in their callers stack
		// frame. Unfortunately (and necessarily) morestack is called before
		// the function that calls it sets up its frame and so the PLT ends
		// up smashing the saved TOC pointer for its caller's caller.
		//
		// According to the ABI documentation there is a mechanism to avoid
		// the TOC save that the PLT stub does (put a R_PPC64_TOCSAVE
		// relocation on the nop after the call to morestack) but at the time
		// of writing it is not supported at all by gold and my attempt to
		// use it with ld.bfd caused an internal linker error. So this hack
		// seems preferable.

		// MOVD $runtime.morestack(SB), R12
		p = obj.Appendp(p, c.newprog)
		p.As = AMOVD
		p.From.Type = obj.TYPE_MEM
		p.From.Sym = morestacksym
		p.From.Name = obj.NAME_GOTREF
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R12

		// MOVD R12, LR
		p = obj.Appendp(p, c.newprog)
		p.As = AMOVD
		p.From.Type = obj.TYPE_REG
		p.From.Reg = REG_R12
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_LR

		// BL LR
		p = obj.Appendp(p, c.newprog)
		p.As = obj.ACALL
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_LR
	} else {
		// BL	runtime.morestack(SB)
		p = obj.Appendp(p, c.newprog)

		p.As = ABL
		p.To.Type = obj.TYPE_BRANCH
		p.To.Sym = morestacksym
	}

	if NeedTOCpointer(c.ctxt) {
		// MOVD 8(SP), R2
		p = obj.Appendp(p, c.newprog)
		p.As = AMOVD
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = REGSP
		p.From.Offset = 8
		p.To.Type = obj.TYPE_REG
		p.To.Reg = REG_R2
	}

	// The instructions which unspill regs should be preemptible.
	p = c.ctxt.EndUnsafePoint(p, c.newprog, -1)
	unspill := c.cursym.Func().UnspillRegisterArgs(p, c.newprog)

	// BR	start
	p = obj.Appendp(unspill, c.newprog)
	p.As = ABR
	p.To.Type = obj.TYPE_BRANCH
	p.To.SetTarget(startPred.Link)

	// placeholder for q1's jump target
	p = obj.Appendp(p, c.newprog)

	p.As = obj.ANOP // zero-width place holder
	q1.To.SetTarget(p)

	return p
}

// MMA accumulator to/from instructions are slightly ambiguous since
// the argument represents both source and destination, specified as
// an accumulator. It is treated as a unary destination to simplify
// the code generation in ppc64map.
var unaryDst = map[obj.As]bool{
	AXXSETACCZ: true,
	AXXMTACC:   true,
	AXXMFACC:   true,
}

var Linkppc64 = obj.LinkArch{
	Arch:           sys.ArchPPC64,
	Init:           buildop,
	Preprocess:     preprocess,
	Assemble:       span9,
	Progedit:       progedit,
	UnaryDst:       unaryDst,
	DWARFRegisters: PPC64DWARFRegisters,
}

var Linkppc64le = obj.LinkArch{
	Arch:           sys.ArchPPC64LE,
	Init:           buildop,
	Preprocess:     preprocess,
	Assemble:       span9,
	Progedit:       progedit,
	UnaryDst:       unaryDst,
	DWARFRegisters: PPC64DWARFRegisters,
}
