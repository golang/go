// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ppc64

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj/ppc64"
	"cmd/internal/objabi"
)

func Init(arch *gc.Arch) {
	arch.LinkArch = &ppc64.Linkppc64
	if objabi.GOARCH == "ppc64le" {
		arch.LinkArch = &ppc64.Linkppc64le
	}
	arch.REGSP = ppc64.REGSP
	arch.MAXWIDTH = 1 << 50

	arch.ZeroRange = zerorange
	arch.ZeroAuto = zeroAuto
	arch.Ginsnop = ginsnop2

	arch.SSAMarkMoves = ssaMarkMoves
	arch.SSAGenValue = ssaGenValue
	arch.SSAGenBlock = ssaGenBlock
}
