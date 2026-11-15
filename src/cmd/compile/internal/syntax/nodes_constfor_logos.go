// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

type ConstForStmt struct {
	Index *Name
	Field *Name
	Range Expr
	Body  *BlockStmt
	stmt
}

type FieldProjExpr struct {
	X      Expr
	Field  *Name
	Rbrack Pos
	expr
}
