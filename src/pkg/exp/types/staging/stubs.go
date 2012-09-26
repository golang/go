// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains unimplemented stubs so that the
// code in exp/types/staging compiles.

package types

import "go/ast"

// expr typechecks expression e and initializes x with the expression
// value or type. If an error occured, x.mode is set to invalid.
// A hint != nil is used as operand type for untyped shifted operands;
// iota >= 0 indicates that the expression is part of a constant declaration.
// cycleOk indicates whether it is ok for a type expression to refer to itself.
//
func (check *checker) exprOrType(x *operand, e ast.Expr, hint Type, iota int, cycleOk bool) {
	unimplemented()
}

// expr is like exprOrType but also checks that e represents a value (rather than a type).
func (check *checker) expr(x *operand, e ast.Expr, hint Type, iota int) {
	unimplemented()
}

// typ is like exprOrType but also checks that e represents a type (rather than a value).
// If an error occured, the result is Typ[Invalid].
//
func (check *checker) typ(e ast.Expr, cycleOk bool) Type {
	unimplemented()
	return nil
}

// assignNtoM typechecks a general assignment. If decl is set, the lhs operands
// must be identifiers. If their types are not set, they are deduced from the
// types of the corresponding rhs expressions. iota >= 0 indicates that the
// "assignment" is part of a constant declaration.
//
func (check *checker) assignNtoM(lhs, rhs []ast.Expr, decl bool, iota int) {
	unimplemented()
}

// assignment typechecks a single assignment of the form lhs := x. If decl is set,
// the lhs operand must be an identifier. If its type is not set, it is deduced
// from the type or value of x.
//
func (check *checker) assignment(lhs ast.Expr, x *operand, decl bool) {
	unimplemented()
}

// stmt typechecks statement s.
func (check *checker) stmt(s ast.Stmt) {
	unimplemented()
}
