// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cursorutil provides utility functions for working with [inspector.Cursor].
//
// It should create no additional dependencies beyond those of Cursor
// itself, so that functions can be promoted to the public API of
// Cursor in due course.
package cursorutil

import (
	"go/ast"

	"golang.org/x/tools/go/ast/edge"
	"golang.org/x/tools/go/ast/inspector"
	// no dependencies other than those of Cursor!
)

// EnclosingFile returns the Cursor for the enclosing File.
//
// TODO(adonovan): promote to Cursor.EnclosingFile method.
func EnclosingFile(cur inspector.Cursor) inspector.Cursor {
	// TODO(adonovan): optimize.
	for cur := range cur.Enclosing((*ast.File)(nil)) {
		return cur
	}
	panic("unreachable")
}

// FirstEnclosing returns the first value from [cursor.Enclosing] as
// both a designated type and a [inspector.Cursor] pointing to it.
//
// It returns the zero value if it is not found.
//
// A common usage is:
//
//	call, callCur := cursorutil.FirstEnclosing[*ast.CallExpr](cur)
//	if call == nil {
//		// Not Found
//	}
func FirstEnclosing[N ast.Node](cur inspector.Cursor) (N, inspector.Cursor) {
	var typ N
	for cur := range cur.Enclosing(typ) {
		return cur.Node().(N), cur
	}
	return typ, inspector.Cursor{}
}

// Unparen returns the cursor for an expression with any
// enclosing parentheses removed, similar to [ast.Unparen].
// It is often prudent to call this before switching on the
// type of cur.Node().
//
// See also [UnparenEnclosing].
//
// TODO(adonovan): promote to Cursor.Unparen method (go.dev/issue/78995).
func Unparen(cur inspector.Cursor) inspector.Cursor {
	for is[*ast.ParenExpr](cur) {
		cur, _ = cur.FirstChild()
	}
	return cur
}

// UnparenEnclosing returns the first element of
// the [Cursor.Enclosing] sequence that is not itself enclosed
// in parens. It is often prudent to call this before switching on
// cur.ParentEdge().
//
// See also [Unparen].
//
// TODO(adonovan): promote to Cursor.UnparenEnclosing method (go.dev/issue/78995).
func UnparenEnclosing(cur inspector.Cursor) inspector.Cursor {
	for cur.ParentEdgeKind() == edge.ParenExpr_X {
		cur = cur.Parent()
	}
	return cur
}

func is[T any](x any) bool {
	_, ok := x.(T)
	return ok
}
