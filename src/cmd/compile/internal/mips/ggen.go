// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mips

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/mips"
)

// TODO(mips): implement DUFFZERO
func zerorange(pp *objw.Progs, p *obj.Prog, off, cnt int64, _ *uint32) *obj.Prog {

	if cnt == 0 {
		return p
	}
	if cnt < int64(4*types.PtrSize) {
		for i := int64(0); i < cnt; i += int64(types.PtrSize) {
			p = pp.Append(p, mips.AMOVW, obj.TYPE_REG, mips.REGZERO, 0, obj.TYPE_MEM, mips.REGSP, base.Ctxt.Arch.FixedFrameSize+off+i)
		}
	} else {
		//fmt.Printf("zerorange frame:%v, lo: %v, hi:%v \n", frame ,lo, hi)
		//	ADD 	$(FIXED_FRAME+frame+lo-4), SP, r1
		//	ADD 	$cnt, r1, r2
		// loop:
		//	MOVW	R0, (Widthptr)r1
		//	ADD 	$Widthptr, r1
		//	BNE		r1, r2, loop
		p = pp.Append(p, mips.AADD, obj.TYPE_CONST, 0, base.Ctxt.Arch.FixedFrameSize+off-4, obj.TYPE_REG, mips.REGRT1, 0)
		p.Reg = mips.REGSP
		p = pp.Append(p, mips.AADD, obj.TYPE_CONST, 0, cnt, obj.TYPE_REG, mips.REGRT2, 0)
		p.Reg = mips.REGRT1
		p = pp.Append(p, mips.AMOVW, obj.TYPE_REG, mips.REGZERO, 0, obj.TYPE_MEM, mips.REGRT1, int64(types.PtrSize))
		p1 := p
		p = pp.Append(p, mips.AADD, obj.TYPE_CONST, 0, int64(types.PtrSize), obj.TYPE_REG, mips.REGRT1, 0)
		p = pp.Append(p, mips.ABNE, obj.TYPE_REG, mips.REGRT1, 0, obj.TYPE_BRANCH, 0, 0)
		p.Reg = mips.REGRT2
		p.To.SetTarget(p1)
	}

	return p
}

func ginsnop(pp *objw.Progs) *obj.Prog {
	p := pp.Prog(mips.ANOOP)
	return p
}
