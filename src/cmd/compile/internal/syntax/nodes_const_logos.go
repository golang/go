// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s syntax node for M5-S1 (const views
// `const T`; SPEC Sec 2.1, T-201). Upstream files reference it through
// hook lines only (G3); the rule below runs solely in LogosMode, so
// plain Go parses exactly as upstream.

// ConstType is a loGo/s const-view type `const T` (SPEC T-201).
//
// Elem is the viewed type T. The parser accepts `const` wherever a type
// starts in parameter, receiver, result, variable, and field types; the
// type checker qualifies it as a compile-time-only view that lowering
// erases. A doubled `const const T` nests here and flattens in types2,
// so printing stays idempotent without a special case.
//
// The position of a ConstType is the position of `const`.
type ConstType struct {
	Elem Expr // viewed type
	expr
}
