// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loong64

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/loong64"
)

func zerorange(pp *objw.Progs, p *obj.Prog, off, cnt int64, _ *uint32) *obj.Prog {
	if cnt == 0 {
		return p
	}

	// Adjust the frame to account for LR.
	off += base.Ctxt.Arch.FixedFrameSize

	if cnt < int64(4*types.PtrSize) {
		for i := int64(0); i < cnt; i += int64(types.PtrSize) {
			p = pp.Append(p, loong64.AMOVV, obj.TYPE_REG, loong64.REGZERO, 0, obj.TYPE_MEM, loong64.REGSP, off+i)
		}
	} else if cnt <= int64(128*types.PtrSize) {
		p = pp.Append(p, loong64.AADDV, obj.TYPE_CONST, 0, off, obj.TYPE_REG, loong64.REGRT1, 0)
		p.Reg = loong64.REGSP
		p = pp.Append(p, obj.ADUFFZERO, obj.TYPE_NONE, 0, 0, obj.TYPE_MEM, 0, 0)
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = ir.Syms.Duffzero
		p.To.Offset = 8 * (128 - cnt/int64(types.PtrSize))
	} else {
		//	ADDV	$(off), SP, r1
		//	ADDV	$cnt, r1, r2
		// loop:
		//	MOVV	R0, (r1)
		//	ADDV	$Widthptr, r1
		//	BNE	r1, r2, loop
		p = pp.Append(p, loong64.AADDV, obj.TYPE_CONST, 0, off, obj.TYPE_REG, loong64.REGRT1, 0)
		p.Reg = loong64.REGSP
		p = pp.Append(p, loong64.AADDV, obj.TYPE_CONST, 0, cnt, obj.TYPE_REG, loong64.REGRT2, 0)
		p.Reg = loong64.REGRT1
		p = pp.Append(p, loong64.AMOVV, obj.TYPE_REG, loong64.REGZERO, 0, obj.TYPE_MEM, loong64.REGRT1, 0)
		loop := p
		p = pp.Append(p, loong64.AADDV, obj.TYPE_CONST, 0, int64(types.PtrSize), obj.TYPE_REG, loong64.REGRT1, 0)
		p = pp.Append(p, loong64.ABNE, obj.TYPE_REG, loong64.REGRT1, 0, obj.TYPE_BRANCH, 0, 0)
		p.Reg = loong64.REGRT2
		p.To.SetTarget(loop)
	}

	return p
}

func ginsnop(pp *objw.Progs) *obj.Prog {
	p := pp.Prog(loong64.ANOOP)
	return p
}
