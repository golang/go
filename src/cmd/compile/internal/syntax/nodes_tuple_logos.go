// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s syntax nodes for M6-S5 tuples (SPEC Sec
// 2.4, T-204.1, and Sec 5.2-5.3, T-104.6). Upstream files reference them
// through hook lines only (G3); the `(` rule below runs solely in
// LogosMode, so plain Go parses exactly as upstream.
//
// One `(` rule (parser_tuple_logos.go) serves tuple types, tuple
// literals, parenthesized expressions, and lambda parameter lists: it
// parses the element list once, then branches. `=>` after `)` forms a
// lambda (nodes_lambda_logos.go); a single element stays a parenthesized
// expression or type; two or more elements form a TupleType in type
// position or a TupleLit in expression position.

// TupleType is a loGo/s tuple type `(A, B)` or `(a A, b B)` (SPEC
// T-204.1) in a non-result position. Result lists keep the Go
// multi-value meaning and never produce this node.
//
// Fields holds two or more entries. An entry with a nil Name is an
// unnamed element; otherwise Name is the element name as written (the
// lowering spells it `T_<name>`). The parser accepts an optional `:`
// between a name and its type, so `(x: int, y: int)` and `(x int, y
// int)` mean the same.
//
// The position of a TupleType is the position of `(`.
type TupleType struct {
	Fields []*Field
	expr
}

// TupleLit is a loGo/s tuple literal `(a, b)` (SPEC Sec 5.2) in
// expression position: two or more values packed positionally. The
// target type (from an assignment, declaration, argument, or return)
// supplies any element names; the literal itself carries none.
//
// The position of a TupleLit is the position of `(`.
type TupleLit struct {
	Elems []Expr
	expr
}
