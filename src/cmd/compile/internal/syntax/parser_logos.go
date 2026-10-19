// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s parser logic for M1-S2 (`?` and `errdefer`).
// parser.go calls into it through hook lines only (G3); every rule below
// runs solely in LogosMode, so plain Go parses exactly as upstream.

// logosQuestion consumes a postfix `?` in LogosMode, wrapping x in a
// TryExpr positioned at `?`, and reports true. In plain mode it reports
// false without consuming anything; the scanner never produces _Question
// there, so the caller leaves the pexpr postfix loop exactly as upstream.
func (p *parser) logosQuestion(x Expr, pos Pos) (Expr, bool) {
	if p.mode&LogosMode == 0 {
		return x, false
	}
	t := new(TryExpr)
	t.pos = pos
	p.next()
	t.X = x
	return t, true
}

// logosCheckGoTry rejects `?` in go statements: the operand of a `go`
// statement starts a fresh goroutine, so there is no caller to propagate
// an error to. `defer f()?` stays legal; only _Go is rejected.
func (p *parser) logosCheckGoTry(s *CallStmt) {
	if s.Tok != _Go || p.mode&LogosMode == 0 {
		return
	}
	if t, ok := s.Call.(*TryExpr); ok {
		p.syntaxErrorAt(t.Pos(), "? not allowed in go statement")
	}
}

// logosErrdefer recognizes an `errdefer <call>` statement and returns it.
// It returns nil when lhs cannot start one, leaving every ordinary Go use
// of the identifier `errdefer` (`errdefer := 1`, `errdefer = 2`,
// `errdefer(x)`, `errdefer.f()`, a bare reference) on the upstream path.
func (p *parser) logosErrdefer(lhs Expr) Stmt {
	if p.mode&LogosMode == 0 {
		return nil
	}
	name, ok := lhs.(*Name)
	if !ok || name.Value != "errdefer" {
		return nil
	}
	// A token that continues an ordinary Go statement with `errdefer`
	// as its first operand keeps its Go meaning. Anything else starts
	// either an errdefer call (an operand follows on this line) or the
	// "requires a function call" error below (e.g. `errdefer 1`).
	switch p.tok {
	case _Define, _Assign, _AssignOp, _IncOp, _Arrow,
		_Lparen, _Dot, _Lbrack, _Lbrace,
		_Operator, _Star,
		_Comma, _Colon, _Semi, _Rparen, _Rbrack, _Rbrace, _EOF:
		return nil
	}
	return p.errdeferStmt(name.Pos())
}

// errdeferStmt parses the cleanup call after `errdefer` (pos). The call
// is a *CallExpr, or a *TryExpr wrapping one for `errdefer f()?`.
func (p *parser) errdeferStmt(pos Pos) Stmt {
	s := new(ErrdeferStmt)
	s.pos = pos
	x := p.pexpr(nil, false)
	if t, ok := Unparen(x).(*CallExpr); ok {
		s.Call = t
		return s
	}
	if t, ok := Unparen(x).(*TryExpr); ok {
		if _, ok := Unparen(t.X).(*CallExpr); ok {
			s.Call = t
			return s
		}
	}
	p.syntaxErrorAt(x.Pos(), "errdefer requires a function call")
	s.Call = x
	return s
}
