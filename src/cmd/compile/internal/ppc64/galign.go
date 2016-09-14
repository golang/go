// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ppc64

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/ppc64"
)

func betypeinit() {
	if gc.Ctxt.Flag_shared {
		gc.Thearch.ReservedRegs = append(gc.Thearch.ReservedRegs, ppc64.REG_R2)
		gc.Thearch.ReservedRegs = append(gc.Thearch.ReservedRegs, ppc64.REG_R12)
	}
}

func Main() {
	gc.Thearch.LinkArch = &ppc64.Linkppc64
	if obj.GOARCH == "ppc64le" {
		gc.Thearch.LinkArch = &ppc64.Linkppc64le
	}
	gc.Thearch.REGSP = ppc64.REGSP
	gc.Thearch.REGCTXT = ppc64.REGCTXT
	gc.Thearch.REGCALLX = ppc64.REG_R3
	gc.Thearch.REGCALLX2 = ppc64.REG_R4
	gc.Thearch.REGRETURN = ppc64.REG_R3
	gc.Thearch.REGMIN = ppc64.REG_R0
	gc.Thearch.REGMAX = ppc64.REG_R31
	gc.Thearch.FREGMIN = ppc64.REG_F0
	gc.Thearch.FREGMAX = ppc64.REG_F31
	gc.Thearch.MAXWIDTH = 1 << 50
	gc.Thearch.ReservedRegs = resvd

	gc.Thearch.Betypeinit = betypeinit
	gc.Thearch.Defframe = defframe
	gc.Thearch.Gins = gins
	gc.Thearch.Proginfo = proginfo

	gc.Thearch.SSARegToReg = ssaRegToReg
	gc.Thearch.SSAMarkMoves = ssaMarkMoves
	gc.Thearch.SSAGenValue = ssaGenValue
	gc.Thearch.SSAGenBlock = ssaGenBlock

	initvariants()
	initproginfo()

	gc.Main()
	gc.Exit(0)
}
