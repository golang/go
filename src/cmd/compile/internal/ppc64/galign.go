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
	if obj.Getgoarch() == "ppc64le" {
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
	gc.Thearch.Cgen_hmul = cgen_hmul
	gc.Thearch.Cgen_shift = cgen_shift
	gc.Thearch.Clearfat = clearfat
	gc.Thearch.Defframe = defframe
	gc.Thearch.Dodiv = dodiv
	gc.Thearch.Excise = excise
	gc.Thearch.Expandchecks = expandchecks
	gc.Thearch.Getg = getg
	gc.Thearch.Gins = gins
	gc.Thearch.Ginscmp = ginscmp
	gc.Thearch.Ginscon = ginscon
	gc.Thearch.Ginsnop = ginsnop
	gc.Thearch.Gmove = gmove
	gc.Thearch.Peep = peep
	gc.Thearch.Proginfo = proginfo
	gc.Thearch.Regtyp = regtyp
	gc.Thearch.Sameaddr = sameaddr
	gc.Thearch.Smallindir = smallindir
	gc.Thearch.Stackaddr = stackaddr
	gc.Thearch.Blockcopy = blockcopy
	gc.Thearch.Sudoaddable = sudoaddable
	gc.Thearch.Sudoclean = sudoclean
	gc.Thearch.Excludedregs = excludedregs
	gc.Thearch.RtoB = RtoB
	gc.Thearch.FtoB = RtoB
	gc.Thearch.BtoR = BtoR
	gc.Thearch.BtoF = BtoF
	gc.Thearch.Optoas = optoas
	gc.Thearch.Doregbits = doregbits
	gc.Thearch.Regnames = regnames

	initvariants()
	initproginfo()

	gc.Main()
	gc.Exit(0)
}
