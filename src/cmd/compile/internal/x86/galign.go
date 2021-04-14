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
	arch.SSAGenValue = ssaGenValue
	arch.SSAGenBlock = ssaGenBlock
	arch.MAXWIDTH = (1 << 32) - 1
	switch v := objabi.GO386; v {
	case "sse2":
	case "softfloat":
		arch.SoftFloat = true
	case "387":
		fmt.Fprintf(os.Stderr, "unsupported setting GO386=387. Consider using GO386=softfloat instead.\n")
		gc.Exit(1)
	default:
		fmt.Fprintf(os.Stderr, "unsupported setting GO386=%s\n", v)
		gc.Exit(1)

	}

	arch.ZeroRange = zerorange
	arch.Ginsnop = ginsnop
	arch.Ginsnopdefer = ginsnop

	arch.SSAMarkMoves = ssaMarkMoves
}
