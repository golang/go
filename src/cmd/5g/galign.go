// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/arm"
)

var thechar int = '5'

var thestring string = "arm"

var thelinkarch *obj.LinkArch = &arm.Linkarm

func linkarchinit() {
}

var MAXWIDTH int64 = (1 << 32) - 1

/*
 * go declares several platform-specific type aliases:
 * int, uint, and uintptr
 */
var typedefs = []gc.Typedef{
	gc.Typedef{"int", gc.TINT, gc.TINT32},
	gc.Typedef{"uint", gc.TUINT, gc.TUINT32},
	gc.Typedef{"uintptr", gc.TUINTPTR, gc.TUINT32},
}

func betypeinit() {
	gc.Widthptr = 4
	gc.Widthint = 4
	gc.Widthreg = 4
}

func main() {
	gc.Thearch.Thechar = thechar
	gc.Thearch.Thestring = thestring
	gc.Thearch.Thelinkarch = thelinkarch
	gc.Thearch.Typedefs = typedefs
	gc.Thearch.REGSP = arm.REGSP
	gc.Thearch.REGCTXT = arm.REGCTXT
	gc.Thearch.REGCALLX = arm.REG_R1
	gc.Thearch.REGCALLX2 = arm.REG_R2
	gc.Thearch.REGRETURN = arm.REG_R0
	gc.Thearch.REGMIN = arm.REG_R0
	gc.Thearch.REGMAX = arm.REGEXT
	gc.Thearch.FREGMIN = arm.REG_F0
	gc.Thearch.FREGMAX = arm.FREGEXT
	gc.Thearch.MAXWIDTH = MAXWIDTH
	gc.Thearch.ReservedRegs = resvd

	gc.Thearch.Betypeinit = betypeinit
	gc.Thearch.Cgen64 = cgen64
	gc.Thearch.Cgen_hmul = cgen_hmul
	gc.Thearch.Cgen_shift = cgen_shift
	gc.Thearch.Clearfat = clearfat
	gc.Thearch.Cmp64 = cmp64
	gc.Thearch.Defframe = defframe
	gc.Thearch.Excise = excise
	gc.Thearch.Expandchecks = expandchecks
	gc.Thearch.Getg = getg
	gc.Thearch.Gins = gins
	gc.Thearch.Ginscon = ginscon
	gc.Thearch.Ginsnop = ginsnop
	gc.Thearch.Gmove = gmove
	gc.Thearch.Cgenindex = cgenindex
	gc.Thearch.Linkarchinit = linkarchinit
	gc.Thearch.Peep = peep
	gc.Thearch.Proginfo = proginfo
	gc.Thearch.Regtyp = regtyp
	gc.Thearch.Sameaddr = sameaddr
	gc.Thearch.Smallindir = smallindir
	gc.Thearch.Stackaddr = stackaddr
	gc.Thearch.Stackcopy = stackcopy
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
