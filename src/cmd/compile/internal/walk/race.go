// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package walk

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"cmd/internal/sys"
)

func instrument(fn *ir.Func) {
	if fn.Pragma&ir.Norace != 0 || (fn.Linksym() != nil && fn.Linksym().ABIWrapper()) {
		return
	}

	if !base.Flag.Race || !base.Compiling(base.NoRacePkgs) {
		fn.SetInstrumentBody(true)
	}

	if base.Flag.Race {
		lno := base.Pos
		base.Pos = src.NoXPos
		if ssagen.Arch.LinkArch.Arch.Family != sys.AMD64 {
			fn.Enter.Prepend(mkcallstmt("racefuncenterfp"))
			fn.Exit.Append(mkcallstmt("racefuncexit"))
		} else {

			// nodpc is the PC of the caller as extracted by
			// getcallerpc. We use -widthptr(FP) for x86.
			// This only works for amd64. This will not
			// work on arm or others that might support
			// race in the future.

			nodpc := ir.NewNameAt(src.NoXPos, typecheck.Lookup(".fp"))
			nodpc.Class = ir.PPARAM
			nodpc.SetUsed(true)
			nodpc.SetType(types.Types[types.TUINTPTR])
			nodpc.SetFrameOffset(int64(-types.PtrSize))
			fn.Dcl = append(fn.Dcl, nodpc)
			fn.Enter.Prepend(mkcallstmt("racefuncenter", nodpc))
			fn.Exit.Append(mkcallstmt("racefuncexit"))
		}
		base.Pos = lno
	}
}
