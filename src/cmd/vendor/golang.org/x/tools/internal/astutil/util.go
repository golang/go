// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package astutil

import (
	"go/ast"
	"go/token"
)

// PreorderStack traverses the tree rooted at root,
// calling f before visiting each node.
//
// Each call to f provides the current node and traversal stack,
// consisting of the original value of stack appended with all nodes
// from root to n, excluding n itself. (This design allows calls
// to PreorderStack to be nested without double counting.)
//
// If f returns false, the traversal skips over that subtree. Unlike
// [ast.Inspect], no second call to f is made after visiting node n.
// In practice, the second call is nearly always used only to pop the
// stack, and it is surprisingly tricky to do this correctly; see
// https://go.dev/issue/73319.
//
// TODO(adonovan): replace with [ast.PreorderStack] when go1.25 is assured.
func PreorderStack(root ast.Node, stack []ast.Node, f func(n ast.Node, stack []ast.Node) bool) {
	before := len(stack)
	ast.Inspect(root, func(n ast.Node) bool {
		if n != nil {
			if !f(n, stack) {
				// Do not push, as there will be no corresponding pop.
				return false
			}
			stack = append(stack, n) // push
		} else {
			stack = stack[:len(stack)-1] // pop
		}
		return true
	})
	if len(stack) != before {
		panic("push/pop mismatch")
	}
}

// NodeContains reports whether the Pos/End range of node n encloses
// the given position pos.
//
// It is inclusive of both end points, to allow hovering (etc) when
// the cursor is immediately after a node.
//
// For unfortunate historical reasons, the Pos/End extent of an
// ast.File runs from the start of its package declaration---excluding
// copyright comments, build tags, and package documentation---to the
// end of its last declaration, excluding any trailing comments. So,
// as a special case, if n is an [ast.File], NodeContains uses
// n.FileStart <= pos && pos <= n.FileEnd to report whether the
// position lies anywhere within the file.
//
// Precondition: n must not be nil.
func NodeContains(n ast.Node, pos token.Pos) bool {
	var start, end token.Pos
	if file, ok := n.(*ast.File); ok {
		start, end = file.FileStart, file.FileEnd // entire file
	} else {
		start, end = n.Pos(), n.End()
	}
	return start <= pos && pos <= end
}
