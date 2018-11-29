// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x86

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj/x86"
	"cmd/internal/objabi"
	"fmt"
	"os"
)

func Init(arch *gc.Arch) {
	arch.LinkArch = &x86.Link386
	arch.REGSP = x86.REGSP
	switch v := objabi.GO386; v {
	case "387":
		arch.Use387 = true
		arch.SSAGenValue = ssaGenValue387
		arch.SSAGenBlock = ssaGenBlock387
	case "sse2":
		arch.SSAGenValue = ssaGenValue
		arch.SSAGenBlock = ssaGenBlock
	default:
		fmt.Fprintf(os.Stderr, "unsupported setting GO386=%s\n", v)
		gc.Exit(1)
	}
	arch.MAXWIDTH = (1 << 32) - 1

	arch.ZeroRange = zerorange
	arch.ZeroAuto = zeroAuto
	arch.Ginsnop = ginsnop

	arch.SSAMarkMoves = ssaMarkMoves
}
