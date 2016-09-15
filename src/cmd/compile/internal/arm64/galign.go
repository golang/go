// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

import (
	"cmd/compile/internal/gc"
	"cmd/compile/internal/ssa"
	"cmd/internal/obj/arm64"
)

func betypeinit() {
}

func Main() {
	gc.Thearch.LinkArch = &arm64.Linkarm64
	gc.Thearch.REGSP = arm64.REGSP
	gc.Thearch.REGCTXT = arm64.REGCTXT
	gc.Thearch.REGCALLX = arm64.REGRT1
	gc.Thearch.REGCALLX2 = arm64.REGRT2
	gc.Thearch.REGRETURN = arm64.REG_R0
	gc.Thearch.REGMIN = arm64.REG_R0
	gc.Thearch.REGMAX = arm64.REG_R31
	gc.Thearch.REGZERO = arm64.REGZERO
	gc.Thearch.FREGMIN = arm64.REG_F0
	gc.Thearch.FREGMAX = arm64.REG_F31
	gc.Thearch.MAXWIDTH = 1 << 50
	gc.Thearch.ReservedRegs = resvd

	gc.Thearch.Betypeinit = betypeinit
	gc.Thearch.Defframe = defframe
	gc.Thearch.Proginfo = proginfo

	gc.Thearch.SSARegToReg = ssaRegToReg
	gc.Thearch.SSAMarkMoves = func(s *gc.SSAGenState, b *ssa.Block) {}
	gc.Thearch.SSAGenValue = ssaGenValue
	gc.Thearch.SSAGenBlock = ssaGenBlock

	gc.Main()
	gc.Exit(0)
}
