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
}

func Main() {
	gc.Thearch.LinkArch = &ppc64.Linkppc64
	if obj.GOARCH == "ppc64le" {
		gc.Thearch.LinkArch = &ppc64.Linkppc64le
	}
	gc.Thearch.REGSP = ppc64.REGSP
	gc.Thearch.REGCTXT = ppc64.REGCTXT
	gc.Thearch.MAXWIDTH = 1 << 50

	gc.Thearch.Betypeinit = betypeinit
	gc.Thearch.Defframe = defframe
	gc.Thearch.Proginfo = proginfo

	gc.Thearch.SSARegToReg = ssaRegToReg
	gc.Thearch.SSAMarkMoves = ssaMarkMoves
	gc.Thearch.SSAGenValue = ssaGenValue
	gc.Thearch.SSAGenBlock = ssaGenBlock

	initvariants()
	initproginfo()

	gc.Main()
	gc.Exit(0)
}
