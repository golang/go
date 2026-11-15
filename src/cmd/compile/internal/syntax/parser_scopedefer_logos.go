// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

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

func (p *parser) scopeDeferStmt(pos Pos) Stmt {
	s := new(ScopeDeferStmt)
	s.pos = pos
	dpos := p.pos()
	p.next()
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
