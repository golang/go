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

func Init(arch *gc.Arch) {
	arch.LinkArch = &x86.Linkamd64
	if obj.GOARCH == "amd64p32" {
		arch.LinkArch = &x86.Linkamd64p32
		leaptr = x86.ALEAL
	}
	arch.REGSP = x86.REGSP
	arch.MAXWIDTH = 1 << 50

	arch.Defframe = defframe
	arch.Ginsnop = ginsnop

	arch.SSAMarkMoves = ssaMarkMoves
	arch.SSAGenValue = ssaGenValue
	arch.SSAGenBlock = ssaGenBlock

	x86.InstInit()
}
