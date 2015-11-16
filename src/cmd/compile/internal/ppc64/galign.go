// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ppc64

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/ppc64"
)

var thechar int = '9'

var thestring string = "ppc64"

var thelinkarch *obj.LinkArch

func linkarchinit() {
	thestring = obj.Getgoarch()
	gc.Thearch.Thestring = thestring
	if thestring == "ppc64le" {
		thelinkarch = &ppc64.Linkppc64le
	} else {
		thelinkarch = &ppc64.Linkppc64
	}
	gc.Thearch.Thelinkarch = thelinkarch
}

var MAXWIDTH int64 = 1 << 50

/*
 * go declares several platform-specific type aliases:
 * int, uint, and uintptr
 */
var typedefs = []gc.Typedef{
	{"int", gc.TINT, gc.TINT64},
	{"uint", gc.TUINT, gc.TUINT64},
	{"uintptr", gc.TUINTPTR, gc.TUINT64},
}

func betypeinit() {
	gc.Widthptr = 8
	gc.Widthint = 8
	gc.Widthreg = 8

	if gc.Ctxt.Flag_shared != 0 {
		gc.Thearch.ReservedRegs = append(gc.Thearch.ReservedRegs, ppc64.REG_R2)
		gc.Thearch.ReservedRegs = append(gc.Thearch.ReservedRegs, ppc64.REG_R12)
	}
}

func Main() {
	gc.Thearch.Thechar = thechar
	gc.Thearch.Thestring = thestring
	gc.Thearch.Thelinkarch = thelinkarch
	gc.Thearch.Typedefs = typedefs
	gc.Thearch.REGSP = ppc64.REGSP
	gc.Thearch.REGCTXT = ppc64.REGCTXT
	gc.Thearch.REGCALLX = ppc64.REG_R3
	gc.Thearch.REGCALLX2 = ppc64.REG_R4
	gc.Thearch.REGRETURN = ppc64.REG_R3
	gc.Thearch.REGMIN = ppc64.REG_R0
	gc.Thearch.REGMAX = ppc64.REG_R31
	gc.Thearch.FREGMIN = ppc64.REG_F0
	gc.Thearch.FREGMAX = ppc64.REG_F31
	gc.Thearch.MAXWIDTH = MAXWIDTH
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
	gc.Thearch.Linkarchinit = linkarchinit
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
