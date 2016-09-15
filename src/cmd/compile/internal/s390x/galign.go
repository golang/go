// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package s390x

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj/s390x"
)

func betypeinit() {
}

func Main() {
	gc.Thearch.LinkArch = &s390x.Links390x
	gc.Thearch.REGSP = s390x.REGSP
	gc.Thearch.REGCTXT = s390x.REGCTXT
	gc.Thearch.REGCALLX = s390x.REG_R3
	gc.Thearch.REGCALLX2 = s390x.REG_R4
	gc.Thearch.REGRETURN = s390x.REG_R3
	gc.Thearch.REGMIN = s390x.REG_R0
	gc.Thearch.REGMAX = s390x.REG_R15
	gc.Thearch.FREGMIN = s390x.REG_F0
	gc.Thearch.FREGMAX = s390x.REG_F15
	gc.Thearch.MAXWIDTH = 1 << 50
	gc.Thearch.ReservedRegs = resvd

	gc.Thearch.Betypeinit = betypeinit
	gc.Thearch.Defframe = defframe
	gc.Thearch.Proginfo = proginfo

	gc.Thearch.SSARegToReg = ssaRegToReg
	gc.Thearch.SSAMarkMoves = ssaMarkMoves
	gc.Thearch.SSAGenValue = ssaGenValue
	gc.Thearch.SSAGenBlock = ssaGenBlock

	gc.Main()
	gc.Exit(0)
}
