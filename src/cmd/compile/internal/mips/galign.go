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

func Init(arch *gc.Arch) {
	arch.LinkArch = &mips.Linkmips
	if obj.GOARCH == "mipsle" {
		arch.LinkArch = &mips.Linkmipsle
	}
	arch.REGSP = mips.REGSP
	arch.MAXWIDTH = (1 << 31) - 1
	arch.Defframe = defframe
	arch.Ginsnop = ginsnop
	arch.Proginfo = proginfo
	arch.SSAMarkMoves = func(s *gc.SSAGenState, b *ssa.Block) {}
	arch.SSAGenValue = ssaGenValue
	arch.SSAGenBlock = ssaGenBlock
}
