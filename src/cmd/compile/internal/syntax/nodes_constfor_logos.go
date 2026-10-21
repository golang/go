// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s syntax nodes for M7-S2 (`const for` over
// concrete types) and M7-S3 (comptime generics) (SPEC Sec 2.5, T-205).
// Upstream files reference them through hook lines only (G3); every rule
// below runs solely in LogosMode, so plain Go parses exactly as upstream.
//
// A `const for` unrolls at lowering once per field of its struct type
// (declaration order); `val.[f]` projects the field named by the loop
// variable. Comptime generics reuse these nodes with a type parameter as
// the Fields argument; expansion (M7-S3) substitutes concrete types and
// unrolls per instantiation.

// ConstForStmt is a loGo/s `const for` statement (SPEC Sec 2.5, T-205):
//
//	const for [Index ","] Field := range comptime.Fields[T]() { ... }
//
// Index is nil when omitted (`const for f := range ...`); a blank `_`
// stays a *Name like any other index. Field is the loop field variable;
// Range is the `comptime.Fields[T]()` call (any other shape is a types2
// error, so positions stay precise); Body is the loop body.
//
// The position of a ConstForStmt is the position of `const`.
type ConstForStmt struct {
	Index *Name // loop index variable, or nil when omitted
	Field *Name // loop field variable
	Range Expr  // range operand (a comptime.Fields call in valid code)
	Body  *BlockStmt
	stmt
}

// FieldProjExpr is a loGo/s field projection `val.[f]` (SPEC Sec 2.5,
// T-205): the struct field of val named by the const-for field variable
// f. X is the struct value; Field is the loop variable name (resolved by
// types2 to the enclosing ConstForStmt); Rbrack is the position of `]`.
//
// The position of a FieldProjExpr is the position of `.`, mirroring the
// house style where SelectorExpr sits at `.`.
type FieldProjExpr struct {
	X      Expr  // struct value
	Field  *Name // const-for field variable
	Rbrack Pos   // position of `]`
	expr
}
