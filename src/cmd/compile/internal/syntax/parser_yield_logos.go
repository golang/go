// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s parser logic for M6-S9 (`yield`
// generators). parser.go calls into it through hook lines only (G3);
// every rule below runs solely in LogosMode, so plain Go parses exactly
// as upstream.

// logosYield recognizes a `yield <expr>` statement and returns it. It
// returns nil when lhs cannot start one, leaving every ordinary Go use
// of the identifier (`yield := 1`, `yield = 2`, `yield(x)`, `yield.f()`,
// `yield[0]`, `yield++`, `yield <- v`, a bare reference, a label) on the
// upstream path. A `yield` followed by an operand-starting token can
// never continue an ordinary Go statement with `yield` as its first
// operand (`yield i` is not valid Go), so it starts a yield statement;
// anything else keeps its Go meaning, both inside and outside generator
// functions (types2 reports yield_outside_generator outside one).
func (p *parser) logosYield(lhs Expr) Stmt {
	if p.mode&LogosMode == 0 {
		return nil
	}
	name, ok := lhs.(*Name)
	if !ok || name.Value != "yield" {
		return nil
	}
	switch p.tok {
	case _Literal, _Func, _Struct, _Map, _Chan, _Interface, _Name:
		return p.yieldStmt(name.Pos())
	default:
		return nil
	}
}

// yieldStmt parses the yielded value after `yield` (pos). The value is
// the full expression list, so `yield k, v` keeps both values for
// `iter.Seq2`; a single value stays a plain expression.
func (p *parser) yieldStmt(pos Pos) Stmt {
	s := new(YieldStmt)
	s.pos = pos
	s.Value = p.exprList()
	return s
}
