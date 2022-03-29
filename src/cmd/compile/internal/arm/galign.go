// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm

import (
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssagen"
	"cmd/internal/obj/arm"
	"internal/buildcfg"
)

func Init(arch *ssagen.ArchInfo) {
	arch.LinkArch = &arm.Linkarm
	arch.REGSP = arm.REGSP
	arch.MAXWIDTH = (1 << 32) - 1
	arch.SoftFloat = buildcfg.GOARM == 5
	arch.ZeroRange = zerorange
	arch.Ginsnop = ginsnop

	arch.SSAMarkMoves = func(s *ssagen.State, b *ssa.Block) {}
	arch.SSAGenValue = ssaGenValue
	arch.SSAGenBlock = ssaGenBlock
}
