// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/obj"
	"cmd/internal/obj/i386"
)
import "cmd/internal/gc"

var thechar int = '8'

var thestring string = "386"

var thelinkarch *obj.LinkArch = &i386.Link386

func linkarchinit() {
}

var MAXWIDTH int64 = (1 << 32) - 1

/*
 * go declares several platform-specific type aliases:
 * int, uint, float, and uintptr
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
	gc.Thearch.REGSP = i386.REGSP
	gc.Thearch.REGCTXT = i386.REGCTXT
	gc.Thearch.MAXWIDTH = MAXWIDTH
	gc.Thearch.Anyregalloc = anyregalloc
	gc.Thearch.Betypeinit = betypeinit
	gc.Thearch.Bgen = bgen
	gc.Thearch.Cgen = cgen
	gc.Thearch.Cgen_call = cgen_call
	gc.Thearch.Cgen_callinter = cgen_callinter
	gc.Thearch.Cgen_ret = cgen_ret
	gc.Thearch.Clearfat = clearfat
	gc.Thearch.Defframe = defframe
	gc.Thearch.Excise = excise
	gc.Thearch.Expandchecks = expandchecks
	gc.Thearch.Gclean = gclean
	gc.Thearch.Ginit = ginit
	gc.Thearch.Gins = gins
	gc.Thearch.Ginscall = ginscall
	gc.Thearch.Igen = igen
	gc.Thearch.Linkarchinit = linkarchinit
	gc.Thearch.Peep = peep
	gc.Thearch.Proginfo = proginfo
	gc.Thearch.Regalloc = regalloc
	gc.Thearch.Regfree = regfree
	gc.Thearch.Regtyp = regtyp
	gc.Thearch.Sameaddr = sameaddr
	gc.Thearch.Smallindir = smallindir
	gc.Thearch.Stackaddr = stackaddr
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
