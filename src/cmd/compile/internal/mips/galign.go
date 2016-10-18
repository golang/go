// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mips

import (
	"cmd/compile/internal/gc"
	"cmd/compile/internal/ssa"
	"cmd/internal/obj"
	"cmd/internal/obj/mips"
)

func Init() {
	gc.Thearch.LinkArch = &mips.Linkmips
	if obj.GOARCH == "mipsle" {
		gc.Thearch.LinkArch = &mips.Linkmipsle
	}
	gc.Thearch.REGSP = mips.REGSP
	gc.Thearch.MAXWIDTH = (1 << 31) - 1
	gc.Thearch.Defframe = defframe
	gc.Thearch.Proginfo = proginfo
	gc.Thearch.SSAMarkMoves = func(s *gc.SSAGenState, b *ssa.Block) {}
	gc.Thearch.SSAGenValue = ssaGenValue
	gc.Thearch.SSAGenBlock = ssaGenBlock
}
