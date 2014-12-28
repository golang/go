// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements New, Eval and EvalNode.

package types

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
)

// New is a convenience function to create a new type from a given
// expression or type literal string evaluated in Universe scope.
// New(str) is shorthand for Eval(str, nil, nil), but only returns
// the type result, and panics in case of an error.
// Position info for objects in the result type is undefined.
//
func New(str string) Type {
	tv, err := Eval(str, nil, nil)
	if err != nil {
		panic(err)
	}
	return tv.Type
}

// Eval returns the type and, if constant, the value for the
// expression or type literal string str evaluated in scope.
// If the expression contains function literals, the function
// bodies are ignored (though they must be syntactically correct).
//
// If pkg == nil, the Universe scope is used and the provided
// scope is ignored. Otherwise, the scope must belong to the
// package (either the package scope, or nested within the
// package scope).
//
// An error is returned if the scope is incorrect, the string
// has syntax errors, or if it cannot be evaluated in the scope.
// Position info for objects in the result type is undefined.
//
// Note: Eval should not be used instead of running Check to compute
// types and values, but in addition to Check. Eval will re-evaluate
// its argument each time, and it also does not know about the context
// in which an expression is used (e.g., an assignment). Thus, top-
// level untyped constants will return an untyped type rather then the
// respective context-specific type.
//
func Eval(str string, pkg *Package, scope *Scope) (TypeAndValue, error) {
	node, err := parser.ParseExpr(str)
	if err != nil {
		return TypeAndValue{}, err
	}

	// Create a file set that looks structurally identical to the
	// one created by parser.ParseExpr for correct error positions.
	fset := token.NewFileSet()
	fset.AddFile("", len(str), fset.Base()).SetLinesForContent([]byte(str))

	return EvalNode(fset, node, pkg, scope)
}

// EvalNode is like Eval but instead of string it accepts
// an expression node and respective file set.
//
// An error is returned if the scope is incorrect
// if the node cannot be evaluated in the scope.
//
func EvalNode(fset *token.FileSet, node ast.Expr, pkg *Package, scope *Scope) (tv TypeAndValue, err error) {
	// verify package/scope relationship
	if pkg == nil {
		scope = Universe
	} else {
		s := scope
		for s != nil && s != pkg.scope {
			s = s.parent
		}
		// s == nil || s == pkg.scope
		if s == nil {
			return TypeAndValue{}, fmt.Errorf("scope does not belong to package %s", pkg.name)
		}
	}

	// initialize checker
	check := NewChecker(nil, fset, pkg, nil)
	check.scope = scope
	defer check.handleBailout(&err)

	// evaluate node
	var x operand
	check.rawExpr(&x, node, nil)
	return TypeAndValue{x.mode, x.typ, x.val}, nil
}
