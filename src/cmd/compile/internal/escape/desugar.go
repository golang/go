// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
)

// TODO(mdempsky): Desugaring doesn't belong during escape analysis,
// but for now it's the most convenient place for some rewrites.

// fixRecoverCall rewrites an ORECOVER call into ORECOVERFP,
// adding an explicit frame pointer argument.
// If call is not an ORECOVER call, it's left unmodified.
func fixRecoverCall(call *ir.CallExpr) {
	if call.Op() != ir.ORECOVER {
		return
	}

	pos := call.Pos()

	// FP is equal to caller's SP plus FixedFrameSize.
	var fp ir.Node = ir.NewCallExpr(pos, ir.OGETCALLERSP, nil, nil)
	if off := base.Ctxt.Arch.FixedFrameSize; off != 0 {
		fp = ir.NewBinaryExpr(fp.Pos(), ir.OADD, fp, ir.NewInt(off))
	}
	// TODO(mdempsky): Replace *int32 with unsafe.Pointer, without upsetting checkptr.
	fp = ir.NewConvExpr(pos, ir.OCONVNOP, types.NewPtr(types.Types[types.TINT32]), fp)

	call.SetOp(ir.ORECOVERFP)
	call.Args = []ir.Node{typecheck.Expr(fp)}
}
