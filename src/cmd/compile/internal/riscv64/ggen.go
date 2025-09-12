// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package riscv64

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/riscv"
)

func zeroRange(pp *objw.Progs, p *obj.Prog, off, cnt int64, _ *uint32) *obj.Prog {

	if cnt%int64(types.PtrSize) != 0 {
		panic("zeroed region not aligned")
	}

	// Adjust the frame to account for LR.
	off += base.Ctxt.Arch.FixedFrameSize

	for cnt != 0 {
		p = pp.Append(p, riscv.AMOV, obj.TYPE_REG, riscv.REG_ZERO, 0, obj.TYPE_MEM, riscv.REG_SP, off)
		cnt -= int64(types.PtrSize)
		off += int64(types.PtrSize)
	}

	return p
}
