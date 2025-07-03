// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package astutil

import (
	"fmt"
	"go/ast"
	"go/token"
	"strconv"
	"unicode/utf8"
)

// RangeInStringLiteral calculates the positional range within a string literal
// corresponding to the specified start and end byte offsets within the logical string.
func RangeInStringLiteral(lit *ast.BasicLit, start, end int) (token.Pos, token.Pos, error) {
	startPos, err := PosInStringLiteral(lit, start)
	if err != nil {
		return 0, 0, fmt.Errorf("start: %v", err)
	}
	endPos, err := PosInStringLiteral(lit, end)
	if err != nil {
		return 0, 0, fmt.Errorf("end: %v", err)
	}
	return startPos, endPos, nil
}

// PosInStringLiteral returns the position within a string literal
// corresponding to the specified byte offset within the logical
// string that it denotes.
func PosInStringLiteral(lit *ast.BasicLit, offset int) (token.Pos, error) {
	raw := lit.Value

	value, err := strconv.Unquote(raw)
	if err != nil {
		return 0, err
	}
	if !(0 <= offset && offset <= len(value)) {
		return 0, fmt.Errorf("invalid offset")
	}

	// remove quotes
	quote := raw[0] // '"' or '`'
	raw = raw[1 : len(raw)-1]

	var (
		i   = 0                // byte index within logical value
		pos = lit.ValuePos + 1 // position within literal
	)
	for raw != "" && i < offset {
		r, _, rest, _ := strconv.UnquoteChar(raw, quote) // can't fail
		sz := len(raw) - len(rest)                       // length of literal char in raw bytes
		pos += token.Pos(sz)
		raw = raw[sz:]
		i += utf8.RuneLen(r)
	}
	return pos, nil
}

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
