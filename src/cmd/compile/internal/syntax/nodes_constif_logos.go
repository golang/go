// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s syntax nodes for M7-S1 (`const if`) and
// M7-S4 (`const switch T`; SPEC Sec 2.5, T-205). Upstream files reference
// them through hook lines only (G3); every rule below runs solely in
// LogosMode, so plain Go parses exactly as upstream.

// ConstIfStmt is a loGo/s `const if` statement (SPEC Sec 2.5, T-205):
//
//	"const" "if" Cond Block ["else" (Block | ConstIf)]
//
// Cond must be a constant boolean (checked in types2, which reports
// [comptime::not_constant] otherwise); both branches type-check, and
// lowering keeps only the live branch's block. Else is nil, a *BlockStmt,
// or a *ConstIfStmt for `else const if` chains. A regular `else if`
// (with a non-const IfStmt) is outside the SPEC grammar and the parser
// rejects it.
//
// The position of a ConstIfStmt is the position of `const`.
type ConstIfStmt struct {
	Cond Expr
	Then *BlockStmt

	// Else is nil, a *BlockStmt, or a *ConstIfStmt.
	Else Stmt
	stmt
}

// ConstSwitchCase is one case of a loGo/s `const switch` statement. It
// is a plain struct, not a syntax node (Table N: the case list lives
// inline in ConstSwitchStmt; there is no ConstCaseClause node): Type is
// the case type as written (`int`, `~int`, or an interface type), Tilde
// reports whether a `~` preceded it, and Body holds the case statements.
// The walker and printer reach cases through the ConstSwitchStmt entries
// in walk_logos.go and printer_logos.go.
type ConstSwitchCase struct {
	Type     Expr
	Tilde    bool
	TildePos Pos
	Body     []Stmt
	Case     Pos
	Colon    Pos
}

// ConstSwitchStmt is a loGo/s `const switch` statement (SPEC Sec 2.5,
// T-205):
//
//	"const" "switch" Targ "{" { "case" Type ":" Stmts } "}"
//
// Targ is the switched type: the name of a type parameter for the
// run-time form, or a concrete type (which folds to the matching case
// at compile time). Each case names a single type; a comma list is
// rejected by the parser. There is no source-level `default` arm (the
// SPEC grammar lists ConstCase only): when no case matches, control
// falls through to the statements after the switch. Case types classify in types2:
// an exact type, a `~U` underlying match with a basic U, or an interface
// case.
//
// The position of a ConstSwitchStmt is the position of `const`. Rbrace
// is the position of the closing brace.
type ConstSwitchStmt struct {
	Targ   Expr
	Cases  []*ConstSwitchCase
	Lbrace Pos
	Rbrace Pos
	stmt
}
