// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
)

// sinkReturnAddr delays allocating an addressable copy of a pointer parameter
// until the final return in a function with an if statement before that return.
// The copy is equivalent when the parameter's address cannot be observed
// anywhere else and no closure can modify the parameter after evaluation.
func sinkReturnAddr(fn *ir.Func) {
	if len(fn.Body) < 2 || fn.Body[len(fn.Body)-2].Op() != ir.OIF {
		return
	}
	ret, ok := fn.Body[len(fn.Body)-1].(*ir.ReturnStmt)
	if !ok || len(ret.Results) != 1 {
		return
	}
	value := ret.Results[0]
	for value.Op() == ir.OCONVNOP || value.Op() == ir.OCONVIFACE {
		value = value.(*ir.ConvExpr).X
	}
	addr, ok := value.(*ir.AddrExpr)
	if !ok || addr.Op() != ir.OADDR {
		return
	}
	param, ok := addr.X.(*ir.Name)
	if !ok || param.Class != ir.PPARAM || param.Curfn != fn || !param.Type().IsPtr() {
		return
	}

	if ir.Any(fn, func(n ir.Node) bool {
		switch n := n.(type) {
		case *ir.AddrExpr:
			return n != addr && n.Op() == ir.OADDR && ir.OuterValue(n.X) == param
		case *ir.ClosureExpr:
			for _, cv := range n.Func.ClosureVars {
				if cv.Canonical() == param {
					return true
				}
			}
		}
		return false
	}) {
		return
	}

	pos := addr.Pos()
	tmp := typecheck.TempAt(pos, fn, param.Type())
	tmp.SetEsc(ir.EscUnknown)
	tmp.SetAddrtaken(true)
	init := []ir.Node{
		ir.NewDecl(pos, ir.ODCL, tmp),
		ir.NewAssignStmt(pos, tmp, param),
	}
	typecheck.Stmts(fn, init)
	addr.PtrInit().Append(init...)
	addr.X = tmp
}
