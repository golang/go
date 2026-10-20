// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s syntax node for M6-S4 spread (`..src`
// composite literal elements; SPEC T-104.5). Upstream files reference
// it through hook lines only (G3); the rule below runs solely in
// LogosMode, so plain Go parses exactly as upstream.

// SpreadExpr is a loGo/s spread element `..src` inside a struct
// composite literal (SPEC T-104.5).
//
// X is the source expression whose fields with identical names and
// identical types are copied into the literal. Explicit fields win over
// spread fields, and the source is evaluated once. The parser accepts
// any operand shape; placement (struct literals only) and type checks
// belong to types2 (spread_logos.go), so a misplaced spread keeps a
// precise position for its diagnostic.
//
// The position of a SpreadExpr is the position of the first `.`,
// mirroring the house style where SelectorExpr sits at `.`.
type SpreadExpr struct {
	X Expr // spread source
	expr
}
