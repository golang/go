// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import "fmt"

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
		return p.errdeferStmt(name.Pos())
	case _Define, _Assign, _AssignOp, _IncOp, _Arrow,
		_Lparen, _Dot, _Lbrack, _Lbrace,
		_Operator, _Star,
		_Comma, _Colon, _Semi, _Rparen, _Rbrack, _Rbrace, _EOF:
		return nil
	case _Literal, _Struct, _Map, _Chan, _Interface:
		p.syntaxErrorAt(p.pos(), "errdefer requires a function call")
		s := new(ErrdeferStmt)
		s.pos = name.Pos()
		s.Call = p.pexpr(nil, false)
		return s
	default:
		return nil
	}
}

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
		s.Call = x
		return s
	}
	p.syntaxErrorAt(x.Pos(), "errdefer requires a function call")
	s.Call = x
	return s
}
