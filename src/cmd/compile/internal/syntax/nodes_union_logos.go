// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

type SwitchExpr struct {
	Init   SimpleStmt
	Tag    Expr
	Arms   []*ArmClause
	Lbrace Pos
	Rbrace Pos
	expr
}

type ArmClause struct {
	Cases Expr
	Arrow Pos
	Body  Expr
	node
}

type AssertedStruct struct {
	Asserts []Expr
	Struct  *StructType
	Colon   Pos
	Lbrace  Pos
	expr
}
