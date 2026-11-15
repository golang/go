// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

type GroupStmt struct {
	Ctx   Expr
	Limit Expr
	Body  *BlockStmt
	stmt
}

type GroupChild struct {
	Body *BlockStmt
	stmt
}
