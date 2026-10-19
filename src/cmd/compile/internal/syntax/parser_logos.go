// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import "fmt"

// This file holds the loGo/s parser logic for M1-S2 (`?` and `errdefer`).
// parser.go calls into it through hook lines only (G3); every rule below
// runs solely in LogosMode, so plain Go parses exactly as upstream.

// logosQuestion consumes a postfix `?` in LogosMode, wrapping x in a
// TryExpr positioned at `?`. In plain mode it returns x unchanged: the
// scanner never produces _Question there, so the pexpr loop never takes
// its _Question case in plain mode and no `break loop` is needed.
func (p *parser) logosQuestion(x Expr, pos Pos) Expr {
	if p.mode&LogosMode == 0 {
		return x
	}
	t := new(TryExpr)
	t.pos = pos
	p.next()
	t.X = x
	return t
}

// logosCheckCallParen reports the upstream "must not be parenthesized"
// error when a go/defer call hides its parentheses inside a TryExpr:
// `defer (f())?` and `go (f())?`. A bare `defer (f())` is already
// rejected upstream; `go f()?` and `defer f()?` stay legal (a misplaced
// `?` is types2's misplaced_try, SPEC §1.4).
func (p *parser) logosCheckCallParen(s *CallStmt) {
	if p.mode&LogosMode == 0 {
		return
	}
	t, ok := s.Call.(*TryExpr)
	if !ok {
		return
	}
	for inner, ok := t.X.(*TryExpr); ok; inner, ok = t.X.(*TryExpr) {
		t = inner
	}
	if u := Unparen(t.X); u != t.X {
		p.errorAt(t.X.Pos(), fmt.Sprintf("expression in %s must not be parenthesized", s.Tok))
		t.X = u
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
	switch p.tok {
	case _Name, _Func:
		// Only an identifier or `func` may start an errdefer call (SPEC §5.1).
		return p.errdeferStmt(name.Pos())
	case _Define, _Assign, _AssignOp, _IncOp, _Arrow,
		_Lparen, _Dot, _Lbrack, _Lbrace,
		_Operator, _Star,
		_Comma, _Colon, _Semi, _Rparen, _Rbrack, _Rbrace, _EOF:
		// A token that continues an ordinary Go statement with `errdefer`
		// as its first operand keeps its Go meaning.
		return nil
	case _Literal, _Struct, _Map, _Chan, _Interface:
		// An operand that can never be an errdefer call (a literal or a
		// struct/map/chan/interface-led composite): report once and
		// consume it, so recovery stays in sync without follow-on errors.
		p.syntaxErrorAt(p.pos(), "errdefer requires a function call")
		s := new(ErrdeferStmt)
		s.pos = name.Pos()
		s.Call = p.pexpr(nil, false)
		return s
	default:
		// Anything else (notably keywords such as `if` or `go`) cannot
		// start an errdefer operand: keep the plain Go meaning, so the
		// statement reports the same single upstream diagnostic.
		return nil
	}
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
	if _, bad := x.(*BadExpr); bad {
		// pexpr already reported; do not add a second diagnostic.
		s.Call = x
		return s
	}
	p.syntaxErrorAt(x.Pos(), "errdefer requires a function call")
	s.Call = x
	return s
}
