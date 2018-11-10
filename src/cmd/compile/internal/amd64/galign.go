// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package amd64

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
)

var leaptr = x86.ALEAQ

func Init() {
	gc.Thearch.LinkArch = &x86.Linkamd64
	if obj.GOARCH == "amd64p32" {
		gc.Thearch.LinkArch = &x86.Linkamd64p32
		leaptr = x86.ALEAL
	}
	gc.Thearch.REGSP = x86.REGSP
	gc.Thearch.MAXWIDTH = 1 << 50

	gc.Thearch.Defframe = defframe
	gc.Thearch.Proginfo = proginfo

	gc.Thearch.SSAMarkMoves = ssaMarkMoves
	gc.Thearch.SSAGenValue = ssaGenValue
	gc.Thearch.SSAGenBlock = ssaGenBlock
	gc.Thearch.ZeroAuto = zeroAuto
}
