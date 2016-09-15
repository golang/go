// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package s390x

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj/s390x"
)

func betypeinit() {
}

func Main() {
	gc.Thearch.LinkArch = &s390x.Links390x
	gc.Thearch.REGSP = s390x.REGSP
	gc.Thearch.REGCTXT = s390x.REGCTXT
	gc.Thearch.MAXWIDTH = 1 << 50

	gc.Thearch.Betypeinit = betypeinit
	gc.Thearch.Defframe = defframe
	gc.Thearch.Proginfo = proginfo

	gc.Thearch.SSARegToReg = ssaRegToReg
	gc.Thearch.SSAMarkMoves = ssaMarkMoves
	gc.Thearch.SSAGenValue = ssaGenValue
	gc.Thearch.SSAGenBlock = ssaGenBlock

	gc.Main()
	gc.Exit(0)
}
