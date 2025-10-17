// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

import (
	"cmd/compile/internal/objw"
	"cmd/internal/obj"
	"cmd/internal/obj/arm64"
)

func padframe(frame int64) int64 {
	// arm64 requires that the frame size (not counting saved FP&LR)
	// be 16 bytes aligned. If not, pad it.
	if frame%16 != 0 {
		frame += 16 - (frame % 16)
	}
	return frame
}

func zerorange(pp *objw.Progs, p *obj.Prog, off, cnt int64, _ *uint32) *obj.Prog {
	if cnt%8 != 0 {
		panic("zeroed region not aligned")
	}
	off += 8 // return address was ignored in offset calculation
	for cnt >= 16 && off < 512 {
		p = pp.Append(p, arm64.ASTP, obj.TYPE_REGREG, arm64.REGZERO, arm64.REGZERO, obj.TYPE_MEM, arm64.REGSP, off)
		off += 16
		cnt -= 16
	}
	for cnt != 0 {
		p = pp.Append(p, arm64.AMOVD, obj.TYPE_REG, arm64.REGZERO, 0, obj.TYPE_MEM, arm64.REGSP, off)
		off += 8
		cnt -= 8
	}
	return p
}

func ginsnop(pp *objw.Progs) *obj.Prog {
	p := pp.Prog(arm64.AHINT)
	p.From.Type = obj.TYPE_CONST
	return p
}
