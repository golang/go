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
// temporary variable, and then returns the expression &tmp. If vardef
// is true, then the variable is initialized with OVARDEF, and the
// caller must ensure the variable is later assigned before use;
// otherwise, it's zero initialized.
//
// TODO(mdempsky): Change callers to provide tmp's initial value,
// rather than just vardef, to make this safer/easier to use.
func initStackTemp(init *ir.Nodes, tmp *ir.Name, vardef bool) *ir.AddrExpr {
	if vardef {
		init.Append(ir.NewUnaryExpr(base.Pos, ir.OVARDEF, tmp))
	} else {
		appendWalkStmt(init, ir.NewAssignStmt(base.Pos, tmp, nil))
	}
	return typecheck.Expr(typecheck.NodAddr(tmp)).(*ir.AddrExpr)
}

// stackTempAddr returns the expression &tmp, where tmp is a newly
// allocated temporary variable of the given type. Statements to
// zero-initialize tmp are appended to init.
func stackTempAddr(init *ir.Nodes, typ *types.Type) *ir.AddrExpr {
	return initStackTemp(init, typecheck.Temp(typ), false)
}

// stackBufAddr returns thte expression &tmp, where tmp is a newly
// allocated temporary variable of type [len]elem. This variable is
// initialized, and elem must not contain pointers.
func stackBufAddr(len int64, elem *types.Type) *ir.AddrExpr {
	if elem.HasPointers() {
		base.FatalfAt(base.Pos, "%v has pointers", elem)
	}
	tmp := typecheck.Temp(types.NewArray(elem, len))
	return typecheck.Expr(typecheck.NodAddr(tmp)).(*ir.AddrExpr)
}
