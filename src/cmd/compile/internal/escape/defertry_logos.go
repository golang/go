// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
)

func pooledTryDeferClosure(n *ir.GoDeferStmt) *ir.ClosureExpr {
	if n.Op() != ir.ODEFER || n.DeferTry == nil || n.DeferAt != nil || base.Debug.NoDeferTryPool != 0 {
		return nil
	}
	call, ok := n.Call.(*ir.CallExpr)
	if !ok {
		return nil
	}
	clo, _ := call.Fun.(*ir.ClosureExpr)
	return clo
}

func capturedByPooledTryDefer(clo *ir.ClosureExpr, canonical *ir.Name) bool {
	return clo.DeferTryErr != nil && canonical != clo.DeferTryErr
}
