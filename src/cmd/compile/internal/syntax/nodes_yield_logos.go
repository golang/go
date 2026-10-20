// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s syntax node for M6-S9 (`yield` generators).
// Upstream files reference it through hook lines only (G3); every rule
// below runs solely in LogosMode, so plain Go parses exactly as upstream.

// YieldStmt is a loGo/s `yield` generator step (T-104.8).
//
// Value is the yielded value: a single expression for `iter.Seq[T]`, or
// a two-element *ListExpr for `iter.Seq2[K, V]` (`yield k, v`). The
// parser accepts any expression list shape and leaves generator and
// arity checks to types2 (M6-S9), so a misplaced `yield` keeps a precise
// position for its diagnostic.
//
// The position of a YieldStmt is the position of `yield`.
type YieldStmt struct {
	Value Expr // yielded value or values
	stmt
}
