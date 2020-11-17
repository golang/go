// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/arm64"
	"cmd/internal/objabi"
)

var darwin = objabi.GOOS == "darwin" || objabi.GOOS == "ios"

func padframe(frame int64) int64 {
	// arm64 requires that the frame size (not counting saved FP&LR)
	// be 16 bytes aligned. If not, pad it.
	if frame%16 != 0 {
		frame += 16 - (frame % 16)
	}
	return frame
}

func zerorange(pp *gc.Progs, p *obj.Prog, off, cnt int64, _ *uint32) *obj.Prog {
	if cnt == 0 {
		return p
	}
	if cnt < int64(4*gc.Widthptr) {
		for i := int64(0); i < cnt; i += int64(gc.Widthptr) {
			p = pp.Appendpp(p, arm64.AMOVD, obj.TYPE_REG, arm64.REGZERO, 0, obj.TYPE_MEM, arm64.REGSP, 8+off+i)
		}
	} else if cnt <= int64(128*gc.Widthptr) && !darwin { // darwin ld64 cannot handle BR26 reloc with non-zero addend
		if cnt%(2*int64(gc.Widthptr)) != 0 {
			p = pp.Appendpp(p, arm64.AMOVD, obj.TYPE_REG, arm64.REGZERO, 0, obj.TYPE_MEM, arm64.REGSP, 8+off)
			off += int64(gc.Widthptr)
			cnt -= int64(gc.Widthptr)
		}
		p = pp.Appendpp(p, arm64.AMOVD, obj.TYPE_REG, arm64.REGSP, 0, obj.TYPE_REG, arm64.REG_R20, 0)
		p = pp.Appendpp(p, arm64.AADD, obj.TYPE_CONST, 0, 8+off, obj.TYPE_REG, arm64.REG_R20, 0)
		p.Reg = arm64.REG_R20
		p = pp.Appendpp(p, obj.ADUFFZERO, obj.TYPE_NONE, 0, 0, obj.TYPE_MEM, 0, 0)
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = gc.Duffzero
		p.To.Offset = 4 * (64 - cnt/(2*int64(gc.Widthptr)))
	} else {
		// Not using REGTMP, so this is async preemptible (async preemption clobbers REGTMP).
		// We are at the function entry, where no register is live, so it is okay to clobber
		// other registers
		const rtmp = arm64.REG_R20
		p = pp.Appendpp(p, arm64.AMOVD, obj.TYPE_CONST, 0, 8+off-8, obj.TYPE_REG, rtmp, 0)
		p = pp.Appendpp(p, arm64.AMOVD, obj.TYPE_REG, arm64.REGSP, 0, obj.TYPE_REG, arm64.REGRT1, 0)
		p = pp.Appendpp(p, arm64.AADD, obj.TYPE_REG, rtmp, 0, obj.TYPE_REG, arm64.REGRT1, 0)
		p.Reg = arm64.REGRT1
		p = pp.Appendpp(p, arm64.AMOVD, obj.TYPE_CONST, 0, cnt, obj.TYPE_REG, rtmp, 0)
		p = pp.Appendpp(p, arm64.AADD, obj.TYPE_REG, rtmp, 0, obj.TYPE_REG, arm64.REGRT2, 0)
		p.Reg = arm64.REGRT1
		p = pp.Appendpp(p, arm64.AMOVD, obj.TYPE_REG, arm64.REGZERO, 0, obj.TYPE_MEM, arm64.REGRT1, int64(gc.Widthptr))
		p.Scond = arm64.C_XPRE
		p1 := p
		p = pp.Appendpp(p, arm64.ACMP, obj.TYPE_REG, arm64.REGRT1, 0, obj.TYPE_NONE, 0, 0)
		p.Reg = arm64.REGRT2
		p = pp.Appendpp(p, arm64.ABNE, obj.TYPE_NONE, 0, 0, obj.TYPE_BRANCH, 0, 0)
		gc.Patch(p, p1)
	}

	return p
}

func ginsnop(pp *gc.Progs) *obj.Prog {
	p := pp.Prog(arm64.AHINT)
	p.From.Type = obj.TYPE_CONST
	return p
}
