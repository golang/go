// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package amd64

import (
	"cmd/compile/internal/objw"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
)

func zerorange(pp *objw.Progs, p *obj.Prog, off, cnt int64, state *uint32) *obj.Prog {
	if cnt%8 != 0 {
		panic("zeroed region not aligned")
	}
	for cnt >= 16 {
		p = pp.Append(p, x86.AMOVUPS, obj.TYPE_REG, x86.REG_X15, 0, obj.TYPE_MEM, x86.REG_SP, off)
		off += 16
		cnt -= 16
	}
	if cnt != 0 {
		p = pp.Append(p, x86.AMOVQ, obj.TYPE_REG, x86.REG_X15, 0, obj.TYPE_MEM, x86.REG_SP, off)
	}
	return p
}

func ginsnop(pp *objw.Progs) *obj.Prog {
	// This is a hardware nop (1-byte 0x90) instruction,
	// even though we describe it as an explicit XCHGL here.
	// Particularly, this does not zero the high 32 bits
	// like typical *L opcodes.
	// (gas assembles "xchg %eax,%eax" to 0x87 0xc0, which
	// does zero the high 32 bits.)
	p := pp.Prog(x86.AXCHGL)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = x86.REG_AX
	p.To.Type = obj.TYPE_REG
	p.To.Reg = x86.REG_AX
	return p
}
