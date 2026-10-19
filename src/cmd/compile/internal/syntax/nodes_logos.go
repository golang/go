// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s syntax nodes for M1-S2 (`?` and `errdefer`).
// Upstream files reference them through hook lines only (G3).

// TryExpr is a loGo/s postfix `?` error-propagation expression (T-104.1).
//
// X is the operand. In valid code X is a *CallExpr whose last result has
// type error; the parser accepts any other operand shape as well and
// leaves placement and type checks to types2 (M1-S3), so a misplaced `?`
// keeps a precise position for its diagnostic.
//
// The position of a TryExpr is the position of `?`, mirroring the house
// style where SelectorExpr sits at `.` and CallExpr at `(`.
type TryExpr struct {
	X Expr // operand
	expr
}

// ErrdeferStmt is a loGo/s `errdefer <call>` cleanup statement (T-104.2).
//
// Call is the deferred cleanup. In valid code it is a *CallExpr, or a
// *TryExpr wrapping a *CallExpr for `errdefer f()?`; anything else is a
// syntax error reported by the parser ("errdefer requires a function
// call").
//
// The position of an ErrdeferStmt is the position of `errdefer`.
type ErrdeferStmt struct {
	Call Expr // cleanup call
	stmt
}
