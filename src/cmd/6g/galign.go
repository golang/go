// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
)
import "cmd/internal/gc"

var thechar int = '6'

var thestring string = "amd64"

var thelinkarch *obj.LinkArch = &x86.Linkamd64

func linkarchinit() {
	if obj.Getgoarch() == "amd64p32" {
		thelinkarch = &x86.Linkamd64p32
		gc.Thearch.Thelinkarch = thelinkarch
		thestring = "amd64p32"
		gc.Thearch.Thestring = "amd64p32"
	}
}

var MAXWIDTH int64 = 1 << 50

var addptr int = x86.AADDQ

var movptr int = x86.AMOVQ

var leaptr int = x86.ALEAQ

var cmpptr int = x86.ACMPQ

/*
 * go declares several platform-specific type aliases:
 * int, uint, float, and uintptr
 */
var typedefs = []gc.Typedef{
	gc.Typedef{"int", gc.TINT, gc.TINT64},
	gc.Typedef{"uint", gc.TUINT, gc.TUINT64},
	gc.Typedef{"uintptr", gc.TUINTPTR, gc.TUINT64},
}

func betypeinit() {
	gc.Widthptr = 8
	gc.Widthint = 8
	gc.Widthreg = 8
	if obj.Getgoarch() == "amd64p32" {
		gc.Widthptr = 4
		gc.Widthint = 4
		addptr = x86.AADDL
		movptr = x86.AMOVL
		leaptr = x86.ALEAL
		cmpptr = x86.ACMPL
		typedefs[0].Sameas = gc.TINT32
		typedefs[1].Sameas = gc.TUINT32
		typedefs[2].Sameas = gc.TUINT32
	}

}

func main() {
	gc.Thearch.Thechar = thechar
	gc.Thearch.Thestring = thestring
	gc.Thearch.Thelinkarch = thelinkarch
	gc.Thearch.Typedefs = typedefs
	gc.Thearch.REGSP = x86.REGSP
	gc.Thearch.REGCTXT = x86.REGCTXT
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
