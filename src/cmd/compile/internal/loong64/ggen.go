// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loong64

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/objw"
	"cmd/internal/obj"
	"cmd/internal/obj/loong64"
)

func zerorange(pp *objw.Progs, p *obj.Prog, off, cnt int64, _ *uint32) *obj.Prog {
	if cnt%8 != 0 {
		panic("zeroed region not aligned")
	}

	// Adjust the frame to account for LR.
	off += base.Ctxt.Arch.FixedFrameSize

	for cnt != 0 {
		p = pp.Append(p, loong64.AMOVV, obj.TYPE_REG, loong64.REGZERO, 0, obj.TYPE_MEM, loong64.REGSP, off)
		off += 8
		cnt -= 8
	}

	return p
}

func ginsnop(pp *objw.Progs) *obj.Prog {
	p := pp.Prog(loong64.ANOOP)
	return p
}
