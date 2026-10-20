// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s parser logic for M8-S2 (`go group` and
// `go { }` children, SPEC Sec 2.6, T-206). stmtOrNil calls into it
// through one hook line (G3); every rule below runs solely in LogosMode,
// so plain Go parses exactly as upstream.
//
// In LogosMode logosGoStmt takes over every `go` statement (defer keeps
// the upstream path): after `go`, `{` opens a GroupChild, the name
// `group` followed by `(` opens a group argument list that either forms
// a GroupStmt (when `{` follows `)`) or stays an ordinary go call, and
// anything else parses exactly as callStmt does.

// logosGoStmt parses a `go` statement in LogosMode, or reports nil so
// the caller keeps the upstream path. A nil result covers plain mode and
// `defer`; every `go` form below returns a statement once `go` is
// consumed.
func (p *parser) logosGoStmt() Stmt {
	if p.mode&LogosMode == 0 || p.tok != _Go {
		return nil
	}
	goPos := p.pos()
	p.next()
	if p.tok == _Lbrace {
		return p.groupChildStmt(goPos)
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

// groupChildStmt parses the block of a `go { ... }` child at goPos: the
// current token is `{`.
func (p *parser) groupChildStmt(goPos Pos) Stmt {
	s := new(GroupChild)
	s.pos = goPos
	s.Body = p.blockStmt("")
	return s
}

// groupList parses one `go group (...)` element: either the `limit =`
// value or one plain expression appended to plains. It reports whether
// the element was the limit value; callers keep the single-limit rule.
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

// groupStmt parses `go group (...)` at goPos: gname is the consumed
// `group` name and the current token is `(`. The parenthesized list
// holds one context expression and at most one `limit = <expr>`
// element; a `{` after `)` forms a GroupStmt, anything else an ordinary
// go call (where a `limit =` element is a syntax error).
func (p *parser) groupStmt(goPos Pos, gname *Name) Stmt {
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
	return p.groupCallFallback(goPos, gname, plains, limit)
}

// groupContext resolves the parenthesized plain expressions to the one
// group context, reporting the arity errors.
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

// groupCallFallback builds the ordinary go call for `go group (...)`
// without a block: `go group(x)` keeps its Go meaning as a call of a
// function named `group`, while a `limit =` element there is a syntax
// error.
func (p *parser) groupCallFallback(goPos Pos, gname *Name, plains []Expr, limit Expr) Stmt {
	args := make([]Expr, 0, len(plains)+1)
	args = append(args, plains...)
	if limit != nil {
		p.syntaxErrorAt(limit.Pos(), "go group limit requires a block")
		args = append(args, limit)
	}
	call := new(CallExpr)
	call.pos = gname.Pos()
	call.Fun = gname
	call.ArgList = args
	return p.ordinaryGoStmt(goPos, p.pexpr(call, false))
}

// ordinaryGoStmt finishes an ordinary `go` statement at goPos: lhs is
// the parsed call prefix (nil when the call starts at the current
// token), mirroring callStmt exactly, so `go group(x)` and every other
// non-group form keep their Go meaning.
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
