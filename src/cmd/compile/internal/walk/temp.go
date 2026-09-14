// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package walk

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
)

// initStackTemp appends statements to init to initialize the given
// temporary variable to val, and then returns the expression &tmp.
func (w *walkState) initStackTemp(init *ir.Nodes, tmp *ir.Name, val ir.Node) *ir.AddrExpr {
	if val != nil && !types.Identical(tmp.Type(), val.Type()) {
		base.Fatalf("bad initial value for %L: %L", tmp, val)
	}
	w.appendWalkStmt(init, ir.NewAssignStmt(base.Pos, tmp, val))
	return typecheck.Expr(typecheck.NodAddr(tmp)).(*ir.AddrExpr)
}

// stackTempAddr returns the expression &tmp, where tmp is a newly
// allocated temporary variable of the given type. Statements to
// zero-initialize tmp are appended to init.
func (w *walkState) stackTempAddr(init *ir.Nodes, typ *types.Type) *ir.AddrExpr {
	n := typecheck.TempAt(base.Pos, w.curfunc, typ)
	n.SetNonMergeable(true)
	return w.initStackTemp(init, n, nil)
}

// stackBufAddr returns the expression &tmp, where tmp is a newly
// allocated temporary variable of type [len]elem. This variable is
// initialized, and elem must not contain pointers.
func (w *walkState) stackBufAddr(len int64, elem *types.Type) *ir.AddrExpr {
	if elem.HasPointers() {
		base.FatalfAt(base.Pos, "%v has pointers", elem)
	}
	tmp := typecheck.TempAt(base.Pos, w.curfunc, types.NewArray(elem, len))
	return typecheck.Expr(typecheck.NodAddr(tmp)).(*ir.AddrExpr)
}
