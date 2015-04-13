// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
)

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
 * int, uint, and uintptr
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

	if gc.Ctxt.Flag_dynlink {
		gc.Thearch.ReservedRegs = append(gc.Thearch.ReservedRegs, x86.REG_R15)
	}
}

func main() {
	if obj.Getgoos() == "nacl" {
		resvd = append(resvd, x86.REG_BP, x86.REG_R15)
	} else if obj.Framepointer_enabled != 0 {
		resvd = append(resvd, x86.REG_BP)
	}

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
	gc.Thearch.REGMAX = x86.REG_R15
	gc.Thearch.FREGMIN = x86.REG_X0
	gc.Thearch.FREGMAX = x86.REG_X15
	gc.Thearch.MAXWIDTH = MAXWIDTH
	gc.Thearch.ReservedRegs = resvd

	gc.Thearch.AddIndex = addindex
	gc.Thearch.Betypeinit = betypeinit
	gc.Thearch.Cgen_bmul = cgen_bmul
	gc.Thearch.Cgen_hmul = cgen_hmul
	gc.Thearch.Cgen_shift = cgen_shift
	gc.Thearch.Clearfat = clearfat
	gc.Thearch.Defframe = defframe
	gc.Thearch.Dodiv = dodiv
	gc.Thearch.Excise = excise
	gc.Thearch.Expandchecks = expandchecks
	gc.Thearch.Getg = getg
	gc.Thearch.Gins = gins
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
