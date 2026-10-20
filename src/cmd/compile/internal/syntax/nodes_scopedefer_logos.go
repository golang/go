// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s syntax node for M6-S10 (`scope defer`).
// Upstream files reference it through hook lines only (G3); every rule
// below runs solely in LogosMode, so plain Go parses exactly as upstream.

// ScopeDeferStmt is a loGo/s `scope defer <call>` cleanup statement
// (T-104.7). It runs Call when control leaves the enclosing block.
//
// Call is the deferred cleanup. In valid code it is a *CallExpr; anything
// else is a syntax error reported by the parser ("scope defer requires a
// function call").
//
// The position of a ScopeDeferStmt is the position of `scope`.
type ScopeDeferStmt struct {
	Call Expr // cleanup call
	stmt
}
