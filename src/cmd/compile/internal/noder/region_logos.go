// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
)

func logosKeepRegionBlock(stmts []ir.Node) bool {
	if !ir.LogosRegionsEnabled() {
		return false
	}
	for _, st := range stmts {
		if logosStmtOpens(st) {
			return true
		}
	}
	return false
}

func logosRegionBlock(stmts []ir.Node) ir.Node {
	return typecheck.Stmt(ir.NewBlockStmt(stmts[0].Pos(), stmts))
}

func logosStmtOpens(n ir.Node) bool {
	if n == nil {
		return false
	}
	if _, ok := n.(*ir.ClosureExpr); ok {
		return false
	}
	switch n.Op() {
	case ir.OBLOCK, ir.OFOR, ir.ORANGE, ir.OIF, ir.OSWITCH, ir.OSELECT:
		return false
	}
	if call, ok := n.(*ir.CallExpr); ok && ir.LogosRegionCallOp(call) == ir.LogosRegionOpen {
		return true
	}
	found := false
	ir.DoChildren(n, func(c ir.Node) bool {
		if logosStmtOpens(c) {
			found = true
			return true
		}
		return false
	})
	return found
}
