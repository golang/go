// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
)

var thechar int = '8'

var thestring string = "386"

var thelinkarch *obj.LinkArch = &x86.Link386

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
	gc.Thearch.REGSP = x86.REGSP
	gc.Thearch.REGCTXT = x86.REGCTXT
	gc.Thearch.REGCALLX = x86.REG_BX
	gc.Thearch.REGCALLX2 = x86.REG_AX
	gc.Thearch.REGRETURN = x86.REG_AX
	gc.Thearch.REGMIN = x86.REG_AX
	gc.Thearch.REGMAX = x86.REG_DI
	switch v := obj.Getgo386(); v {
	case "387":
		gc.Thearch.FREGMIN = x86.REG_F0
		gc.Thearch.FREGMAX = x86.REG_F7
		gc.Thearch.Use387 = true
	case "sse2":
		gc.Thearch.FREGMIN = x86.REG_X0
		gc.Thearch.FREGMAX = x86.REG_X7
	default:
		gc.Fatal("unsupported setting GO386=%s", v)
	}
	gc.Thearch.MAXWIDTH = MAXWIDTH
	gc.Thearch.ReservedRegs = resvd

	gc.Thearch.Betypeinit = betypeinit
	gc.Thearch.Bgen_float = bgen_float
	gc.Thearch.Cgen64 = cgen64
	gc.Thearch.Cgen_bmul = cgen_bmul
	gc.Thearch.Cgen_float = cgen_float
	gc.Thearch.Cgen_hmul = cgen_hmul
	gc.Thearch.Cgen_shift = cgen_shift
	gc.Thearch.Clearfat = clearfat
	gc.Thearch.Cmp64 = cmp64
	gc.Thearch.Defframe = defframe
	gc.Thearch.Dodiv = cgen_div
	gc.Thearch.Excise = excise
	gc.Thearch.Expandchecks = expandchecks
	gc.Thearch.Getg = getg
	gc.Thearch.Gins = gins
	gc.Thearch.Ginscon = ginscon
	gc.Thearch.Ginsnop = ginsnop
	gc.Thearch.Gmove = gmove
	gc.Thearch.Igenindex = igenindex
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
	gc.Thearch.FtoB = FtoB
	gc.Thearch.BtoR = BtoR
	gc.Thearch.BtoF = BtoF
	gc.Thearch.Optoas = optoas
	gc.Thearch.Doregbits = doregbits
	gc.Thearch.Regnames = regnames

	gc.Main()
	gc.Exit(0)
}
