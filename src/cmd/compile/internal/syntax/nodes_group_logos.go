// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s syntax nodes for M8-S2 (`go group` and
// `go { }` children, SPEC Sec 2.6, T-206). Upstream files reference them
// through hook lines only (G3); every rule below runs solely in
// LogosMode, so plain Go parses exactly as upstream.

// GroupStmt is a loGo/s `go group (ctx)` or `go group (ctx, limit = N)`
// group statement (T-206). Ctx is the parent context expression; Limit
// is nil when no `limit =` element was written. Body holds the group
// children (`go { }` blocks and `go f(args)` calls) alongside the
// synchronous body statements.
//
// The position of a GroupStmt is the position of `go`.
type GroupStmt struct {
	Ctx   Expr // parent context
	Limit Expr // limit expression, or nil
	Body  *BlockStmt
	stmt
}

// GroupChild is a loGo/s `go { ... }` child block (T-206): its body runs
// as one group child of type `func() error`, with `?` returning to the
// group. The checker reports a child outside a group body with
// `[correctness::child_outside_group]`.
//
// The position of a GroupChild is the position of `go`.
type GroupChild struct {
	Body *BlockStmt
	stmt
}
