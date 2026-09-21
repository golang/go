// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package astutil

import (
	"fmt"
	"go/ast"
	"go/printer"
	"go/token"
	"strings"

	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/moreiters"
)

// NodeContains reports whether the Pos/End range of node n encloses
// the given range.
//
// It is inclusive of both end points, to allow hovering (etc) when
// the cursor is immediately after a node.
//
// Like [NodeRange], it treats the range of an [ast.File] as the
// file's complete extent.
//
// Precondition: n must not be nil.
func NodeContains(n ast.Node, rng Range) bool {
	return NodeRange(n).Contains(rng)
}

// NodeContainsPos reports whether the Pos/End range of node n encloses
// the given pos.
//
// Like [NodeRange], it treats the range of an [ast.File] as the
// file's complete extent.
func NodeContainsPos(n ast.Node, pos token.Pos) bool {
	return NodeRange(n).ContainsPos(pos)
}

// EnclosingFile returns the syntax tree for the file enclosing c.
//
// TODO(adonovan): promote this to a method of Cursor.
func EnclosingFile(c inspector.Cursor) *ast.File {
	c, _ = moreiters.First(c.Enclosing((*ast.File)(nil)))
	return c.Node().(*ast.File)
}

// DocComment returns the doc comment for a node, if any.
func DocComment(n ast.Node) *ast.CommentGroup {
	switch n := n.(type) {
	case *ast.FuncDecl:
		return n.Doc
	case *ast.GenDecl:
		return n.Doc
	case *ast.ValueSpec:
		return n.Doc
	case *ast.TypeSpec:
		return n.Doc
	case *ast.File:
		return n.Doc
	case *ast.ImportSpec:
		return n.Doc
	case *ast.Field:
		return n.Doc
	}
	return nil
}

// Format returns a string representation of the node n.
func Format(fset *token.FileSet, n ast.Node) string {
	var buf strings.Builder
	printer.Fprint(&buf, fset, n) // ignore errors
	return buf.String()
}

// -- Range --

// Range is a Pos interval.
// It implements [analysis.Range] and [ast.Node].
type Range struct{ Start, EndPos token.Pos }

// RangeOf constructs a Range.
//
// RangeOf exists to pacify the "unkeyed literal" (composites) vet
// check. It would be nice if there were a way for a type to add
// itself to the allowlist.
func RangeOf(start, end token.Pos) Range { return Range{start, end} }

// NodeRange returns the extent of node n as a Range.
//
// For unfortunate historical reasons, the Pos/End extent of an
// ast.File runs from the start of its package declaration---excluding
// copyright comments, build tags, and package documentation---to the
// end of its last declaration, excluding any trailing comments. So,
// as a special case, if n is an [ast.File], NodeContains uses
// n.FileStart <= pos && pos <= n.FileEnd to report whether the
// position lies anywhere within the file.
func NodeRange(n ast.Node) Range {
	if file, ok := n.(*ast.File); ok {
		return Range{file.FileStart, file.FileEnd} // entire file
	}
	return Range{n.Pos(), n.End()}
}

func (r Range) Pos() token.Pos { return r.Start }
func (r Range) End() token.Pos { return r.EndPos }

// ContainsPos reports whether the range (inclusive of both end points)
// includes the specified position.
func (r Range) ContainsPos(pos token.Pos) bool {
	return r.Contains(RangeOf(pos, pos))
}

// Contains reports whether the range (inclusive of both end points)
// includes the specified range.
func (r Range) Contains(rng Range) bool {
	return r.Start <= rng.Start && rng.EndPos <= r.EndPos
}

// IsValid reports whether the range is valid.
func (r Range) IsValid() bool { return r.Start.IsValid() && r.Start <= r.EndPos }

// --

// Select returns the syntax nodes identified by a user's text
// selection. It returns three nodes: the innermost node that wholly
// encloses the selection; and the first and last nodes that are
// wholly enclosed by the selection.
//
// For example, given this selection:
//
//	{ f(); g(); /* comment */ }
//	  ~~~~~~~~~~~
//
// Select returns the enclosing BlockStmt, the f() CallExpr, and the g() CallExpr.
//
// If the selection does not wholly enclose any nodes, Select returns an error
// and invalid start/end nodes, but it may return a valid enclosing node.
//
// Callers that require exactly one syntax tree (e.g. just f() or just
// g()) should check that the returned start and end nodes are
// identical.
//
// This function is intended to be called early in the handling of a
// user's request, since it is tolerant of sloppy selection including
// extraneous whitespace and comments. Use it in new code instead of
// PathEnclosingInterval. When the exact extent of a node is known,
// use [Cursor.FindByPos] instead.
//
// TODO(hxjiang): Consider refactoring the function signature. It is currently
// confusing that an error is returned even when a valid enclosing node is
// successfully found. Consider grouping all cursors into one struct.
func Select(curFile inspector.Cursor, start, end token.Pos) (_enclosing, _start, _end inspector.Cursor, _ error) {
	curEnclosing, ok := curFile.FindByPos(start, end)
	if !ok {
		return noCursor, noCursor, noCursor, fmt.Errorf("invalid selection")
	}

	// Find the first and last node wholly within the (start, end) range.
	// We'll narrow the effective selection to them, to exclude whitespace.
	// (This matches the functionality of PathEnclosingInterval.)
	var curStart, curEnd inspector.Cursor
	rng := RangeOf(start, end)
	for cur := range curEnclosing.Preorder() {
		if rng.Contains(NodeRange(cur.Node())) {
			// The start node has the least Pos.
			if !curStart.Valid() {
				curStart = cur
			}
			// The end node has the greatest End.
			// End positions do not change monotonically,
			// so we must compute the max.
			if !curEnd.Valid() ||
				cur.Node().End() > curEnd.Node().End() {
				curEnd = cur
			}
		}
	}
	if !curStart.Valid() {
		// The selection is valid (inside curEnclosing) but contains no
		// complete nodes. This happens for point selections (start == end),
		// or selections covering only only spaces, comments, and punctuation
		// tokens.
		// Return the enclosing node so the caller can still use the context.
		return curEnclosing, noCursor, noCursor, fmt.Errorf("invalid selection")
	}
	return curEnclosing, curStart, curEnd, nil
}

var noCursor inspector.Cursor

// MaybeParenthesize returns new, possibly wrapped in parens if needed
// to preserve operator precedence when it replaces old, whose parent
// is parentNode.
//
// (This would be more naturally written in terms of Cursor, but one of
// the callers--the inliner--does not have cursors handy.)
func MaybeParenthesize(parentNode ast.Node, old, new ast.Expr) ast.Expr {
	if needsParens(parentNode, old, new) {
		new = &ast.ParenExpr{X: new}
	}
	return new
}

func needsParens(parentNode ast.Node, old, new ast.Expr) bool {
	// An expression beneath a non-expression
	// has no precedence ambiguity.
	parent, ok := parentNode.(ast.Expr)
	if !ok {
		return false
	}

	precedence := func(n ast.Node) int {
		switch n := n.(type) {
		case *ast.UnaryExpr, *ast.StarExpr:
			return token.UnaryPrec
		case *ast.BinaryExpr:
			return n.Op.Precedence()
		}
		return -1
	}

	// Parens are not required if the new node
	// is not unary or binary.
	newprec := precedence(new)
	if newprec < 0 {
		return false
	}

	// Parens are required if parent and child are both
	// unary or binary and the parent has higher precedence.
	if precedence(parent) > newprec {
		return true
	}

	// Was the old node the operand of a postfix operator?
	//  f().sel
	//  f()[i:j]
	//  f()[i]
	//  f().(T)
	//  f()(x)
	switch parent := parent.(type) {
	case *ast.SelectorExpr:
		return parent.X == old
	case *ast.IndexExpr:
		return parent.X == old
	case *ast.SliceExpr:
		return parent.X == old
	case *ast.TypeAssertExpr:
		return parent.X == old
	case *ast.CallExpr:
		return parent.Fun == old
	}
	return false
}

func is[T any](n any) bool {
	_, ok := n.(T)
	return ok
}
