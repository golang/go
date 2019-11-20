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
		ireg := int16(s390x.REGRT2) // register holds number of remaining loop iterations
		p = pp.Appendpp(p, s390x.AMOVD, obj.TYPE_CONST, 0, cnt/256, obj.TYPE_REG, ireg, 0)
		p = pp.Appendpp(p, s390x.ACLEAR, obj.TYPE_CONST, 0, 256, obj.TYPE_MEM, reg, off)
		pl := p
		p = pp.Appendpp(p, s390x.AADD, obj.TYPE_CONST, 0, 256, obj.TYPE_REG, reg, 0)
		p = pp.Appendpp(p, s390x.ABRCTG, obj.TYPE_REG, ireg, 0, obj.TYPE_BRANCH, 0, 0)
		gc.Patch(p, pl)
		cnt = cnt % 256
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

func ginsnop(pp *gc.Progs) *obj.Prog {
	return pp.Prog(s390x.ANOPH)
}
