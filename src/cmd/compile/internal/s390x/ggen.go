// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package s390x

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/s390x"
)

// clearLoopCutOff is the (somewhat arbitrary) value above which it is better
// to have a loop of clear instructions (e.g. XCs) rather than just generating
// multiple instructions (i.e. loop unrolling).
// Must be between 256 and 4096.
const clearLoopCutoff = 1024

// zerorange clears the stack in the given range.
func zerorange(pp *gc.Progs, p *obj.Prog, off, cnt int64, _ *uint32) *obj.Prog {
	if cnt == 0 {
		return p
	}

	// Adjust the frame to account for LR.
	off += gc.Ctxt.FixedFrameSize()
	reg := int16(s390x.REGSP)

	// If the off cannot fit in a 12-bit unsigned displacement then we
	// need to create a copy of the stack pointer that we can adjust.
	// We also need to do this if we are going to loop.
	if off < 0 || off > 4096-clearLoopCutoff || cnt > clearLoopCutoff {
		p = pp.Appendpp(p, s390x.AADD, obj.TYPE_CONST, 0, off, obj.TYPE_REG, s390x.REGRT1, 0)
		p.Reg = int16(s390x.REGSP)
		reg = s390x.REGRT1
		off = 0
	}

	// Generate a loop of large clears.
	if cnt > clearLoopCutoff {
		n := cnt - (cnt % 256)
		end := int16(s390x.REGRT2)
		p = pp.Appendpp(p, s390x.AADD, obj.TYPE_CONST, 0, off+n, obj.TYPE_REG, end, 0)
		p.Reg = reg
		p = pp.Appendpp(p, s390x.ACLEAR, obj.TYPE_CONST, 0, 256, obj.TYPE_MEM, reg, off)
		pl := p
		p = pp.Appendpp(p, s390x.AADD, obj.TYPE_CONST, 0, 256, obj.TYPE_REG, reg, 0)
		p = pp.Appendpp(p, s390x.ACMP, obj.TYPE_REG, reg, 0, obj.TYPE_REG, end, 0)
		p = pp.Appendpp(p, s390x.ABNE, obj.TYPE_NONE, 0, 0, obj.TYPE_BRANCH, 0, 0)
		gc.Patch(p, pl)

		cnt -= n
	}

	// Generate remaining clear instructions without a loop.
	for cnt > 0 {
		n := cnt

		// Can clear at most 256 bytes per instruction.
		if n > 256 {
			n = 256
		}

		switch n {
		// Handle very small clears with move instructions.
		case 8, 4, 2, 1:
			ins := s390x.AMOVB
			switch n {
			case 8:
				ins = s390x.AMOVD
			case 4:
				ins = s390x.AMOVW
			case 2:
				ins = s390x.AMOVH
			}
			p = pp.Appendpp(p, ins, obj.TYPE_CONST, 0, 0, obj.TYPE_MEM, reg, off)

		// Handle clears that would require multiple move instructions with CLEAR (assembled as XC).
		default:
			p = pp.Appendpp(p, s390x.ACLEAR, obj.TYPE_CONST, 0, n, obj.TYPE_MEM, reg, off)
		}

		cnt -= n
		off += n
	}

	return p
}

func zeroAuto(pp *gc.Progs, n *gc.Node) {
	// Note: this code must not clobber any registers or the
	// condition code.
	sym := n.Sym.Linksym()
	size := n.Type.Size()
	for i := int64(0); i < size; i += int64(gc.Widthptr) {
		p := pp.Prog(s390x.AMOVD)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 0
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_AUTO
		p.To.Reg = s390x.REGSP
		p.To.Offset = n.Xoffset + i
		p.To.Sym = sym
	}
}

func ginsnop(pp *gc.Progs) {
	p := pp.Prog(s390x.AOR)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = int16(s390x.REG_R0)
	p.To.Type = obj.TYPE_REG
	p.To.Reg = int16(s390x.REG_R0)
}
