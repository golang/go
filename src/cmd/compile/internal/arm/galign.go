// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm

import (
	"cmd/compile/internal/gc"
	"cmd/compile/internal/ssa"
	"cmd/internal/obj/arm"
)

func betypeinit() {
}

func Main() {
	gc.Thearch.LinkArch = &arm.Linkarm
	gc.Thearch.REGSP = arm.REGSP
	gc.Thearch.REGCTXT = arm.REGCTXT
	gc.Thearch.REGCALLX = arm.REG_R1
	gc.Thearch.REGCALLX2 = arm.REG_R2
	gc.Thearch.REGRETURN = arm.REG_R0
	gc.Thearch.REGMIN = arm.REG_R0
	gc.Thearch.REGMAX = arm.REGEXT
	gc.Thearch.FREGMIN = arm.REG_F0
	gc.Thearch.FREGMAX = arm.FREGEXT
	gc.Thearch.MAXWIDTH = (1 << 32) - 1
	gc.Thearch.ReservedRegs = resvd

	gc.Thearch.Betypeinit = betypeinit
	gc.Thearch.Defframe = defframe
	gc.Thearch.Gins = gins
	gc.Thearch.Proginfo = proginfo

	gc.Thearch.SSARegToReg = ssaRegToReg
	gc.Thearch.SSAMarkMoves = func(s *gc.SSAGenState, b *ssa.Block) {}
	gc.Thearch.SSAGenValue = ssaGenValue
	gc.Thearch.SSAGenBlock = ssaGenBlock

	gc.Main()
	gc.Exit(0)
}
