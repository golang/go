// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s syntax node for M6-S1 interpolated strings
// (SPEC Sec 1.4, T-104.4). Upstream files reference it through hook
// lines only (G3); the rule runs solely in LogosMode.

// InterpExpr is a loGo/s `$"..."` interpolated string expression. (The
// slice shortens this node to InterpLit; that name is taken by the
// LitKind the scanner reports for the raw `$"..."` token.)
//
// Parts holds the decoded literal text between operands (len(Parts) is
// always len(Exprs)+1); Exprs holds the `{...}` operand expressions in
// source order. `{{` and `}}` escapes are already folded into Parts, so
// lowering quotes Parts verbatim. Raw is the full source spelling of
// the literal (including `$"` and the closing quote); the printer emits
// it unchanged, so printing reparse to the same tree. Opens and Closes
// hold each operand's brace positions, so lowering can span the exact
// `{...}` source without trusting node positions past tokens that later
// passes rewrite.
//
// The position of an InterpExpr is the position of `$`.
type InterpExpr struct {
	Parts  []string
	Exprs  []Expr
	Opens  []Pos
	Closes []Pos
	Raw    string
	expr
}
