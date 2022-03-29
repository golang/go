// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
)

// Eval returns the type and, if constant, the value for the
// expression expr, evaluated at position pos of package pkg,
// which must have been derived from type-checking an AST with
// complete position information relative to the provided file
// set.
//
// The meaning of the parameters fset, pkg, and pos is the
// same as in CheckExpr. An error is returned if expr cannot
// be parsed successfully, or the resulting expr AST cannot be
// type-checked.
func Eval(fset *token.FileSet, pkg *Package, pos token.Pos, expr string) (_ TypeAndValue, err error) {
	// parse expressions
	node, err := parser.ParseExprFrom(fset, "eval", expr, 0)
	if err != nil {
		return TypeAndValue{}, err
	}

	info := &Info{
		Types: make(map[ast.Expr]TypeAndValue),
	}
	err = CheckExpr(fset, pkg, pos, node, info)
	return info.Types[node], err
}

// CheckExpr type checks the expression expr as if it had appeared at position
// pos of package pkg. Type information about the expression is recorded in
// info. The expression may be an identifier denoting an uninstantiated generic
// function or type.
//
// If pkg == nil, the Universe scope is used and the provided
// position pos is ignored. If pkg != nil, and pos is invalid,
// the package scope is used. Otherwise, pos must belong to the
// package.
//
// An error is returned if pos is not within the package or
// if the node cannot be type-checked.
//
// Note: Eval and CheckExpr should not be used instead of running Check
// to compute types and values, but in addition to Check, as these
// functions ignore the context in which an expression is used (e.g., an
// assignment). Thus, top-level untyped constants will return an
// untyped type rather then the respective context-specific type.
//
func CheckExpr(fset *token.FileSet, pkg *Package, pos token.Pos, expr ast.Expr, info *Info) (err error) {
	// determine scope
	var scope *Scope
	if pkg == nil {
		scope = Universe
		pos = token.NoPos
	} else if !pos.IsValid() {
		scope = pkg.scope
	} else {
		// The package scope extent (position information) may be
		// incorrect (files spread across a wide range of fset
		// positions) - ignore it and just consider its children
		// (file scopes).
		for _, fscope := range pkg.scope.children {
			if scope = fscope.Innermost(pos); scope != nil {
				break
			}
		}
		if scope == nil || debug {
			s := scope
			for s != nil && s != pkg.scope {
				s = s.parent
			}
			// s == nil || s == pkg.scope
			if s == nil {
				return fmt.Errorf("no position %s found in package %s", fset.Position(pos), pkg.name)
			}
		}
	}

	// initialize checker
	check := NewChecker(nil, fset, pkg, info)
	check.scope = scope
	check.pos = pos
	defer check.handleBailout(&err)

	// evaluate node
	var x operand
	check.rawExpr(&x, expr, nil, true) // allow generic expressions
	check.processDelayed(0)            // incl. all functions
	check.recordUntyped()

	return nil
}
