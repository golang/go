// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"fmt"
	"go/parser"
	"go/token"
)

// Eval returns the type and, if constant, the value for the
// expression expr, evaluated at position pos of package pkg,
// which must have been derived from type-checking an AST with
// complete position information relative to the provided file
// set.
//
// If the expression contains function literals, their bodies
// are ignored (i.e., the bodies are not type-checked).
//
// If pkg == nil, the Universe scope is used and the provided
// position pos is ignored. If pkg != nil, and pos is invalid,
// the package scope is used. Otherwise, pos must belong to the
// package.
//
// An error is returned if pos is not within the package or
// if the node cannot be evaluated.
//
// Note: Eval should not be used instead of running Check to compute
// types and values, but in addition to Check. Eval will re-evaluate
// its argument each time, and it also does not know about the context
// in which an expression is used (e.g., an assignment). Thus, top-
// level untyped constants will return an untyped type rather then the
// respective context-specific type.
//
func Eval(fset *token.FileSet, pkg *Package, pos token.Pos, expr string) (tv TypeAndValue, err error) {
	// determine scope
	var scope *Scope
	if pkg == nil {
		scope = Universe
		pos = token.NoPos
	} else if !pos.IsValid() {
		scope = pkg.scope
	} else {
		// The package scope extent (position information) may be
		// incorrect (files spread accross a wide range of fset
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
				return TypeAndValue{}, fmt.Errorf("no position %s found in package %s", fset.Position(pos), pkg.name)
			}
		}
	}

	// parse expressions
	node, err := parser.ParseExprFrom(fset, "eval", expr, 0)
	if err != nil {
		return TypeAndValue{}, err
	}

	// initialize checker
	check := NewChecker(nil, fset, pkg, nil)
	check.scope = scope
	check.pos = pos
	defer check.handleBailout(&err)

	// evaluate node
	var x operand
	check.rawExpr(&x, node, nil)
	return TypeAndValue{x.mode, x.typ, x.val}, err
}
