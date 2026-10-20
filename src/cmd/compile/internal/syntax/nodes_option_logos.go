// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s syntax nodes for M3-S2 (option types `T?`,
// `??`, and `?.`; SPEC T-202, T-104.3). Upstream files reference them
// through hook lines only (G3); every rule below runs solely in LogosMode,
// so plain Go parses exactly as upstream.
//
// Table N (IMPLEMENTATION.md) defines no OptSelectorExpr lowering entity:
// lowering (a later slice) desugars these nodes into plain nodes inside
// generated func wrappers. The parse nodes below are named apart from
// that PLAN name; they exist so the parser, printer, walker, and types2
// can tell `?.` from `.` (SPEC [correctness::not_an_option]).

// OptionType is a loGo/s option type `T?` (SPEC T-202).
//
// Elem is the wrapped type T. The parser rejects the joined spelling
// `T??` with a syntax error, so direct nesting needs parentheses
// (`(T?)?`); the type checker still accepts a nested *Option value.
//
// The position of an OptionType is the position of `?`, mirroring TryExpr.
type OptionType struct {
	Elem Expr // wrapped type
	expr
}

// CoalesceExpr is a loGo/s `??` expression `x ?? y` (SPEC T-104.3).
//
// X must be an option; Y is the fallback evaluated only when X is none.
// `??` binds less tightly than `||` and groups to the right, so
// `a || b ?? c` parses as `(a || b) ?? c` and `a ?? b ?? c` as
// `a ?? (b ?? c)`.
//
// The position of a CoalesceExpr is the position of `??`.
type CoalesceExpr struct {
	X, Y Expr // left (option) and right (fallback) operands
	expr
}

// OptSelectExpr is a loGo/s optional selection `x?.Sel` (SPEC T-104.3).
//
// X must be an option; Sel is the selected field or method name. A call
// `x?.M(args)` parses as a *CallExpr whose Fun is an OptSelectExpr.
//
// The position of an OptSelectExpr is the position of `?.`, mirroring
// the house style where SelectorExpr sits at `.` and CallExpr at `(`.
type OptSelectExpr struct {
	X   Expr  // receiver (must be an option)
	Sel *Name // selected field or method
	expr
}
