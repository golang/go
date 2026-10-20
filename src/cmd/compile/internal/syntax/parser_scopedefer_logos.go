// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s parser logic for M6-S10 (`scope defer`).
// parser.go calls into it through hook lines only (G3); every rule below
// runs solely in LogosMode, so plain Go parses exactly as upstream.

// logosScopeDefer recognizes a `scope defer <call>` statement and returns
// it. It returns nil unless `scope` is followed by the `defer` keyword,
// leaving every ordinary Go use of the identifier (`scope := 1`,
// `scope = 2`, `scope(x)`, `scope.f()`, a bare reference, a label) on the
// upstream path.
func (p *parser) logosScopeDefer(lhs Expr) Stmt {
	if p.mode&LogosMode == 0 {
		return nil
	}
	name, ok := lhs.(*Name)
	if !ok || name.Value != "scope" {
		return nil
	}
	if p.tok != _Defer {
		return nil
	}
	return p.scopeDeferStmt(name.Pos())
}

// scopeDeferStmt parses the cleanup call after `scope defer` (pos). The
// call is a *CallExpr; anything else reports "scope defer requires a
// function call" once and keeps the parsed operand for recovery.
func (p *parser) scopeDeferStmt(pos Pos) Stmt {
	s := new(ScopeDeferStmt)
	s.pos = pos
	dpos := p.pos()
	p.next() // consume `defer`
	if p.tok == _Semi || p.tok == _Rbrace || p.tok == _EOF {
		p.syntaxErrorAt(dpos, "scope defer requires a function call")
		s.Call = p.badExpr()
		return s
	}
	x := p.pexpr(nil, false)
	if t, ok := Unparen(x).(*CallExpr); ok {
		s.Call = t
		return s
	}
	if _, bad := x.(*BadExpr); bad {
		s.Call = x
		return s
	}
	p.syntaxErrorAt(x.Pos(), "scope defer requires a function call")
	s.Call = x
	return s
}
