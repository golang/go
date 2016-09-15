// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package amd64

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
)

var (
	leaptr = x86.ALEAQ
)

func betypeinit() {
	if obj.GOARCH == "amd64p32" {
		leaptr = x86.ALEAL
	}
	if gc.Ctxt.Flag_dynlink || obj.GOOS == "nacl" {
		resvd = append(resvd, x86.REG_R15)
	}
	if gc.Ctxt.Framepointer_enabled || obj.GOOS == "nacl" {
		resvd = append(resvd, x86.REG_BP)
	}
	gc.Thearch.ReservedRegs = resvd
}

func Main() {
	gc.Thearch.LinkArch = &x86.Linkamd64
	if obj.GOARCH == "amd64p32" {
		gc.Thearch.LinkArch = &x86.Linkamd64p32
	}
	gc.Thearch.REGSP = x86.REGSP
	gc.Thearch.REGCTXT = x86.REGCTXT
	gc.Thearch.REGCALLX = x86.REG_BX
	gc.Thearch.REGCALLX2 = x86.REG_AX
	gc.Thearch.REGRETURN = x86.REG_AX
	gc.Thearch.REGMIN = x86.REG_AX
	gc.Thearch.REGMAX = x86.REG_R15
	gc.Thearch.FREGMIN = x86.REG_X0
	gc.Thearch.FREGMAX = x86.REG_X15
	gc.Thearch.MAXWIDTH = 1 << 50

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
