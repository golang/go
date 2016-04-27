// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

import (
	"cmd/compile/internal/gc"
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
	gc.Thearch.Cgen_hmul = cgen_hmul
	gc.Thearch.AddSetCarry = AddSetCarry
	gc.Thearch.RightShiftWithCarry = RightShiftWithCarry
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
