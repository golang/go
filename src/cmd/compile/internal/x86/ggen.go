// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x86

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
)

func zerorange(pp *objw.Progs, p *obj.Prog, off, cnt int64, ax *uint32) *obj.Prog {
	if cnt == 0 {
		return p
	}
	if *ax == 0 {
		p = pp.Append(p, x86.AMOVL, obj.TYPE_CONST, 0, 0, obj.TYPE_REG, x86.REG_AX, 0)
		*ax = 1
	}

	if cnt <= int64(4*types.RegSize) {
		for i := int64(0); i < cnt; i += int64(types.RegSize) {
			p = pp.Append(p, x86.AMOVL, obj.TYPE_REG, x86.REG_AX, 0, obj.TYPE_MEM, x86.REG_SP, off+i)
		}
	} else if cnt <= int64(128*types.RegSize) {
		p = pp.Append(p, x86.ALEAL, obj.TYPE_MEM, x86.REG_SP, off, obj.TYPE_REG, x86.REG_DI, 0)
		p = pp.Append(p, obj.ADUFFZERO, obj.TYPE_NONE, 0, 0, obj.TYPE_ADDR, 0, 1*(128-cnt/int64(types.RegSize)))
		p.To.Sym = ir.Syms.Duffzero
	} else {
		p = pp.Append(p, x86.AMOVL, obj.TYPE_CONST, 0, cnt/int64(types.RegSize), obj.TYPE_REG, x86.REG_CX, 0)
		p = pp.Append(p, x86.ALEAL, obj.TYPE_MEM, x86.REG_SP, off, obj.TYPE_REG, x86.REG_DI, 0)
		p = pp.Append(p, x86.AREP, obj.TYPE_NONE, 0, 0, obj.TYPE_NONE, 0, 0)
		p = pp.Append(p, x86.ASTOSL, obj.TYPE_NONE, 0, 0, obj.TYPE_NONE, 0, 0)
	}

	return p
}

func ginsnop(pp *objw.Progs) *obj.Prog {
	// See comment in ../amd64/ggen.go.
	p := pp.Prog(x86.AXCHGL)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = x86.REG_AX
	p.To.Type = obj.TYPE_REG
	p.To.Reg = x86.REG_AX
	return p
}
