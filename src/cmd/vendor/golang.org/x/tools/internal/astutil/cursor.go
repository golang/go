// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package astutil

import (
	"go/ast"

	"golang.org/x/tools/go/ast/edge"
	"golang.org/x/tools/go/ast/inspector"
)

// UnparenCursor returns the cursor for an expression with any
// enclosing parentheses removed, similar to [ast.Unparen].
// It is often prudent to call this before switching on the
// type of cur.Node().
//
// See also [UnparenEnclosingCursor].
func UnparenCursor(cur inspector.Cursor) inspector.Cursor {
	for is[*ast.ParenExpr](cur) {
		cur, _ = cur.FirstChild()
	}
	return cur
}

// UnparenEnclosingCursor returns the first element of
// the [Cursor.Enclosing] sequence that is not itself enclosed
// in parens. It is often prudent to call this before switching on
// cur.ParentEdge().
//
// See also [UnparenCursor].
func UnparenEnclosingCursor(cur inspector.Cursor) inspector.Cursor {
	for cur.ParentEdgeKind() == edge.ParenExpr_X {
		cur = cur.Parent()
	}
	return cur
}
