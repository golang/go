// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s syntax nodes of M2 (switch expressions and
// inline struct assertions). Upstream files reference them through hook
// lines only (G3).

// SwitchExpr is a loGo/s switch expression (SPEC Sec 2.3, T-203.1):
//
//	switch [Init ";"] [Tag] "{" Arm* "}"
//
// in expression position. Tag is nil (a value switch on true), an
// expression (a value switch), or a *TypeSwitchGuard (a type switch);
// the guard form follows the *SwitchStmt encoding. Every arm carries an
// expression result. Lowering turns the node into an immediately
// invoked function literal, so no SwitchExpr survives lowering.
//
// The position of a SwitchExpr is the position of `switch`. Lbrace and
// Rbrace are the positions of the braces that enclose the arms.
type SwitchExpr struct {
	Init   SimpleStmt
	Tag    Expr // incl. *TypeSwitchGuard
	Arms   []*ArmClause
	Lbrace Pos
	Rbrace Pos
	expr
}

// ArmClause is one arm of a SwitchExpr:
//
//	"case" ExprList "=>" Expr
//	"default" "=>" Expr
//
// Cases is nil for the default arm. Arrow is the position of `=>`. The
// position of an ArmClause is the position of `case` or `default`.
type ArmClause struct {
	Cases Expr // nil means default arm
	Arrow Pos
	Body  Expr
	node
}

// AssertedStruct is the right-hand side of a loGo/s struct declaration
// with inline interface assertions (SPEC Sec 2.3, T-203.3):
//
//	type S struct : I, J { ... }
//
// Struct holds the ordinary struct type; Asserts lists the interface
// types `*S` must implement, in source order. Colon and Lbrace are the
// positions of `:` and of the `{` that opens the field list. The node
// appears only as TypeDecl.Type of a non-alias declaration, and
// lowering replaces it with Struct.
//
// The position of an AssertedStruct is the position of `struct`.
type AssertedStruct struct {
	Asserts []Expr
	Struct  *StructType
	Colon   Pos
	Lbrace  Pos
	expr
}
