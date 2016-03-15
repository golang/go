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
	thestring   = "amd64"
	thelinkarch = &x86.Linkamd64
	addptr      = x86.AADDQ
	movptr      = x86.AMOVQ
	leaptr      = x86.ALEAQ
	cmpptr      = x86.ACMPQ
)

func linkarchinit() {
	if obj.Getgoarch() == "amd64p32" {
		thelinkarch = &x86.Linkamd64p32
		gc.Thearch.Thelinkarch = thelinkarch
		thestring = "amd64p32"
		gc.Thearch.Thestring = "amd64p32"
	}
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
	}

	if gc.Ctxt.Flag_dynlink {
		gc.Thearch.ReservedRegs = append(gc.Thearch.ReservedRegs, x86.REG_R15)
	}
}

func Main() {
	if obj.Getgoos() == "nacl" {
		resvd = append(resvd, x86.REG_BP, x86.REG_R15)
	} else if obj.Framepointer_enabled != 0 {
		resvd = append(resvd, x86.REG_BP)
	}

	gc.Thearch.Thechar = '6'
	gc.Thearch.Thestring = thestring
	gc.Thearch.Thelinkarch = thelinkarch
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
	gc.Thearch.Ginsboolval = ginsboolval
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
	gc.Thearch.FtoB = FtoB
	gc.Thearch.BtoR = BtoR
	gc.Thearch.BtoF = BtoF
	gc.Thearch.Optoas = optoas
	gc.Thearch.Doregbits = doregbits
	gc.Thearch.Regnames = regnames

	gc.Thearch.SSARegToReg = ssaRegToReg
	gc.Thearch.SSAMarkMoves = ssaMarkMoves
	gc.Thearch.SSAGenValue = ssaGenValue
	gc.Thearch.SSAGenBlock = ssaGenBlock

	gc.Main()
	gc.Exit(0)
}
