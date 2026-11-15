// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

type ConstIfStmt struct {
	Cond Expr
	Then *BlockStmt

	Else Stmt
	stmt
}

type ConstSwitchCase struct {
	Type     Expr
	Tilde    bool
	TildePos Pos
	Body     []Stmt
	Case     Pos
	Colon    Pos
}

type ConstSwitchStmt struct {
	Targ   Expr
	Cases  []*ConstSwitchCase
	Lbrace Pos
	Rbrace Pos
	stmt
}
