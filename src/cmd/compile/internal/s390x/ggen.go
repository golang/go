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

func defframe(ptxt *obj.Prog) {
	// fill in argument size, stack size
	ptxt.To.Type = obj.TYPE_TEXTSIZE

	ptxt.To.Val = int32(gc.Rnd(gc.Curfn.Type.ArgWidth(), int64(gc.Widthptr)))
	frame := uint32(gc.Rnd(gc.Stksize+gc.Maxarg, int64(gc.Widthreg)))
	ptxt.To.Offset = int64(frame)

	// insert code to zero ambiguously live variables
	// so that the garbage collector only sees initialized values
	// when it looks for pointers.
	p := ptxt

	hi := int64(0)
	lo := hi

	// iterate through declarations - they are sorted in decreasing xoffset order.
	for _, n := range gc.Curfn.Func.Dcl {
		if !n.Name.Needzero {
			continue
		}
		if n.Class != gc.PAUTO {
			gc.Fatalf("needzero class %d", n.Class)
		}
		if n.Type.Width%int64(gc.Widthptr) != 0 || n.Xoffset%int64(gc.Widthptr) != 0 || n.Type.Width == 0 {
			gc.Fatalf("var %L has size %d offset %d", n, int(n.Type.Width), int(n.Xoffset))
		}

		if lo != hi && n.Xoffset+n.Type.Width >= lo-int64(2*gc.Widthreg) {
			// merge with range we already have
			lo = n.Xoffset

			continue
		}

		// zero old range
		p = zerorange(p, int64(frame), lo, hi)

		// set new range
		hi = n.Xoffset + n.Type.Width

		lo = n.Xoffset
	}

	// zero final range
	zerorange(p, int64(frame), lo, hi)
}

// zerorange clears the stack in the given range.
func zerorange(p *obj.Prog, frame int64, lo int64, hi int64) *obj.Prog {
	cnt := hi - lo
	if cnt == 0 {
		return p
	}

	// Adjust the frame to account for LR.
	frame += gc.Ctxt.FixedFrameSize()
	offset := frame + lo
	reg := int16(s390x.REGSP)

	// If the offset cannot fit in a 12-bit unsigned displacement then we
	// need to create a copy of the stack pointer that we can adjust.
	// We also need to do this if we are going to loop.
	if offset < 0 || offset > 4096-clearLoopCutoff || cnt > clearLoopCutoff {
		p = gc.Appendpp(p, s390x.AADD, obj.TYPE_CONST, 0, offset, obj.TYPE_REG, s390x.REGRT1, 0)
		p.Reg = int16(s390x.REGSP)
		reg = s390x.REGRT1
		offset = 0
	}

	// Generate a loop of large clears.
	if cnt > clearLoopCutoff {
		n := cnt - (cnt % 256)
		end := int16(s390x.REGRT2)
		p = gc.Appendpp(p, s390x.AADD, obj.TYPE_CONST, 0, offset+n, obj.TYPE_REG, end, 0)
		p.Reg = reg
		p = gc.Appendpp(p, s390x.AXC, obj.TYPE_MEM, reg, offset, obj.TYPE_MEM, reg, offset)
		p.From3 = new(obj.Addr)
		p.From3.Type = obj.TYPE_CONST
		p.From3.Offset = 256
		pl := p
		p = gc.Appendpp(p, s390x.AADD, obj.TYPE_CONST, 0, 256, obj.TYPE_REG, reg, 0)
		p = gc.Appendpp(p, s390x.ACMP, obj.TYPE_REG, reg, 0, obj.TYPE_REG, end, 0)
		p = gc.Appendpp(p, s390x.ABNE, obj.TYPE_NONE, 0, 0, obj.TYPE_BRANCH, 0, 0)
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
			p = gc.Appendpp(p, ins, obj.TYPE_CONST, 0, 0, obj.TYPE_MEM, reg, offset)

		// Handle clears that would require multiple move instructions with XC.
		default:
			p = gc.Appendpp(p, s390x.AXC, obj.TYPE_MEM, reg, offset, obj.TYPE_MEM, reg, offset)
			p.From3 = new(obj.Addr)
			p.From3.Type = obj.TYPE_CONST
			p.From3.Offset = n
		}

		cnt -= n
		offset += n
	}

	return p
}

func ginsnop() {
	p := gc.Prog(s390x.AOR)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = int16(s390x.REG_R0)
	p.To.Type = obj.TYPE_REG
	p.To.Reg = int16(s390x.REG_R0)
}
