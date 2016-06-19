// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mips64

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/mips"
)

func betypeinit() {
}

func Main() {
	gc.Thearch.LinkArch = &mips.Linkmips64
	if obj.Getgoarch() == "mips64le" {
		gc.Thearch.LinkArch = &mips.Linkmips64le
	}
	gc.Thearch.REGSP = mips.REGSP
	gc.Thearch.REGCTXT = mips.REGCTXT
	gc.Thearch.REGCALLX = mips.REG_R1
	gc.Thearch.REGCALLX2 = mips.REG_R2
	gc.Thearch.REGRETURN = mips.REGRET
	gc.Thearch.REGMIN = mips.REG_R0
	gc.Thearch.REGMAX = mips.REG_R31
	gc.Thearch.FREGMIN = mips.REG_F0
	gc.Thearch.FREGMAX = mips.REG_F31
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

	gc.Main()
	gc.Exit(0)
}
