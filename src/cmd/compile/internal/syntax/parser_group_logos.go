// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

func (p *parser) logosGoStmt() Stmt {
	if p.mode&LogosMode == 0 || p.tok != _Go {
		return nil
	}
	goPos := p.pos()
	p.next()
	if p.tok == _Lbrace {
		s := new(GroupChild)
		s.pos = goPos
		s.Body = p.blockStmt("")
		return s
	}
	if p.tok == _Name && p.lit == "group" {
		gname := p.name()
		if p.tok == _Lparen {
			return p.groupStmt(goPos, gname)
		}
		return p.ordinaryGoStmt(goPos, gname)
	}
	return p.ordinaryGoStmt(goPos, nil)
}

func (p *parser) groupList(plains *[]Expr) (limit Expr) {
	if p.tok == _Name && p.lit == "limit" {
		lpos := p.pos()
		p.next()
		if p.tok == _Assign {
			p.next()
			return p.expr()
		}
		*plains = append(*plains, p.pexpr(NewName(lpos, "limit"), false))
		return nil
	}
	*plains = append(*plains, p.expr())
	return nil
}

func (p *parser) logosNamedArg(x Expr) Expr {
	key, ok := x.(*Name)
	if !ok || p.mode&LogosMode == 0 || p.tok != _Assign {
		return x
	}
	arg := new(KeyValueExpr)
	arg.pos = p.pos()
	p.next()
	arg.Key = key
	arg.Value = p.expr()
	return arg
}

func (p *parser) groupStmt(goPos Pos, gname *Name) Stmt {
	lparen := p.pos()
	p.next()
	var plains []Expr
	var limit Expr
	for p.tok != _Rparen && p.tok != _EOF {
		if e := p.groupList(&plains); e != nil {
			if limit != nil {
				p.syntaxErrorAt(e.Pos(), "go group accepts only one limit")
			} else {
				limit = e
			}
		}
		if p.tok != _Comma {
			break
		}
		p.next()
	}
	p.want(_Rparen)
	if p.tok == _Lbrace {
		s := new(GroupStmt)
		s.pos = goPos
		s.Ctx = p.groupContext(goPos, plains)
		s.Limit = limit
		s.Body = p.blockStmt("")
		return s
	}
	args := make([]Expr, 0, len(plains)+1)
	args = append(args, plains...)
	if limit != nil {
		p.syntaxErrorAt(limit.Pos(), "go group limit requires a block")
		args = append(args, limit)
	}
	call := new(CallExpr)
	call.pos = lparen
	call.Fun = gname
	call.ArgList = args
	return p.ordinaryGoStmt(goPos, p.pexpr(call, false))
}

func (p *parser) groupContext(goPos Pos, plains []Expr) Expr {
	if len(plains) == 1 {
		return plains[0]
	}
	if len(plains) == 0 {
		p.syntaxErrorAt(goPos, "go group requires a context")
	} else {
		p.syntaxErrorAt(goPos, "go group accepts one context and one limit")
	}
	return p.badExpr()
}

func (p *parser) ordinaryGoStmt(goPos Pos, lhs Expr) Stmt {
	var x Expr
	if lhs != nil {
		x = p.pexpr(lhs, false)
	} else {
		x = p.pexpr(nil, p.tok == _Lparen)
	}
	if t := Unparen(x); t != x {
		p.errorAt(x.Pos(), "expression in go must not be parenthesized")
		x = t
	}
	s := new(CallStmt)
	s.pos = goPos
	s.Tok = _Go
	s.Call = x
	p.logosCheckCallParen(s)
	return s
}
