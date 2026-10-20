// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s syntax node for M6-S6 lambdas (SPEC Sec
// 1.4, T-104.6). Upstream files reference it through hook lines only
// (G3); every rule below runs solely in LogosMode, so plain Go parses
// exactly as upstream.
//
// Lambdas come from the shared `(` rule (parser_tuple_logos.go) for the
// `(a, b) => ...` form and from logosLambdaOrExpr for the lone-name
// `x => ...` form. Parameter types always come from the target function
// type at check time, so Params carries names only.

// LambdaExpr is a loGo/s lambda `x => expr`, `x => { stmts }`, or
// `(a, b) => ...` (SPEC T-104.6).
//
// Params holds one or more parameter names. Body is either an Expr
// (expression form, possibly holding `?` per T-104.1) or a *BlockStmt
// (block form). Arrow is the position of `=>`.
//
// The position of a LambdaExpr is the position of its head: the lone
// name, or the `(` of the parameter list.
type LambdaExpr struct {
	Params []*Name
	Body   Node
	Arrow  Pos
	expr
}
