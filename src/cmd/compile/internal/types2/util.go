// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains various functionality that is
// different between go/types and types2. Factoring
// out this code allows more of the rest of the code
// to be shared.

package types2

import (
	"cmd/compile/internal/syntax"
	"go/constant"
	"go/token"
)

const isTypes2 = true

// cmpPos compares the positions p and q and returns a result r as follows:
//
// r <  0: p is before q
// r == 0: p and q are the same position (but may not be identical)
// r >  0: p is after q
//
// If p and q are in different files, p is before q if the filename
// of p sorts lexicographically before the filename of q.
func cmpPos(p, q syntax.Pos) int { return p.Cmp(q) }

// hasDots reports whether the last argument in the call is followed by ...
func hasDots(call *syntax.CallExpr) bool { return call.HasDots }

// dddErrPos returns the node (poser) for reporting an invalid ... use in a call.
func dddErrPos(call *syntax.CallExpr) *syntax.CallExpr {
	// TODO(gri) should use "..." instead of call position
	return call
}

// argErrPos returns the node (poser) for reportign an invalid argument count.
func argErrPos(call *syntax.CallExpr) *syntax.CallExpr { return call }

// ExprString returns a string representation of x.
func ExprString(x syntax.Node) string { return syntax.String(x) }

// startPos returns the start position of node n.
func startPos(n syntax.Node) syntax.Pos { return syntax.StartPos(n) }

// endPos returns the position of the first character immediately after node n.
func endPos(n syntax.Node) syntax.Pos { return syntax.EndPos(n) }

// makeFromLiteral returns the constant value for the given literal string and kind.
func makeFromLiteral(lit string, kind syntax.LitKind) constant.Value {
	return constant.MakeFromLiteral(lit, kind2tok[kind], 0)
}

var kind2tok = [...]token.Token{
	syntax.IntLit:    token.INT,
	syntax.FloatLit:  token.FLOAT,
	syntax.ImagLit:   token.IMAG,
	syntax.RuneLit:   token.CHAR,
	syntax.StringLit: token.STRING,
}
