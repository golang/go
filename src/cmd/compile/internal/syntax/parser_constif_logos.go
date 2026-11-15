// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import "unicode/utf8"

type constPeek struct {
	s    *scanner
	r    int
	mark int
}

func logosPeekConstNext(p *parser) token {
	k := constPeek{s: &p.scanner}
	k.r = k.s.r - k.s.chw
	if !k.skipBlank() {
		return _EOF
	}
	return logosConstKeyword(k.word())
}

func logosConstKeyword(word string) token {
	switch word {
	case "if":
		return _If
	case "switch":
		return _Switch
	case "for":
		return _For
	case "func":
		return _Func
	}
	return _EOF
}

func (k *constPeek) avail() bool {
	for k.r >= k.s.e {
		if k.s.ioerr != nil {
			return false
		}
		shift := k.s.fillPeek()
		k.r -= shift
		k.mark -= shift
	}
	return true
}

func (k *constPeek) skipBlank() bool {
	for k.avail() {
		switch c := k.s.buf[k.r]; c {
		case ' ', '\t', '\n', '\r':
			k.r++
		case '/':
			if !k.skipComment() {
				return false
			}
		default:
			return true
		}
	}
	return false
}

func (k *constPeek) skipComment() bool {
	k.r++
	if !k.avail() {
		return false
	}
	switch k.s.buf[k.r] {
	case '/':
		return k.skipLine()
	case '*':
		return k.skipBlock()
	}
	return false
}

func (k *constPeek) skipLine() bool {
	for k.avail() {
		if k.s.buf[k.r] == '\n' {
			return true
		}
		k.r++
	}
	return false
}

func (k *constPeek) skipBlock() bool {
	k.r++
	for k.avail() {
		if k.s.buf[k.r] != '*' {
			k.r++
			continue
		}
		k.r++
		if !k.avail() {
			return false
		}
		if k.s.buf[k.r] == '/' {
			k.r++
			return true
		}
	}
	return false
}

func (k *constPeek) word() string {
	k.mark = k.r
	for k.avail() && isWordByte(k.s.buf[k.r]) {
		k.r++
	}
	if k.avail() && k.s.buf[k.r] >= utf8.RuneSelf {
		return ""
	}
	return string(k.s.buf[k.mark:k.r])
}

func isWordByte(c byte) bool {
	return c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9'
}

func (p *parser) logosConstStmt() Stmt {
	if p.mode&LogosMode == 0 || p.tok != _Const {
		return nil
	}
	switch logosPeekConstNext(p) {
	case _If:
		return p.constIfStmt()
	case _Switch:
		return p.constSwitchStmt()
	}
	return nil
}

func (p *parser) constIfStmt() Stmt {
	if trace {
		defer p.trace("constIfStmt")()
	}

	s := new(ConstIfStmt)
	s.pos = p.pos()
	p.next()
	p.next()
	outer := p.xnest
	p.xnest = -1
	s.Cond = p.expr()
	p.xnest = outer
	s.Then = p.blockStmt("const if clause")
	if p.got(_Else) {
		s.Else = p.constElse()
	}
	return s
}

func (p *parser) constElse() Stmt {
	switch {
	case p.tok == _Const && logosPeekConstNext(p) == _If:
		return p.constIfStmt()
	case p.tok == _Lbrace:
		return p.blockStmt("")
	}
	p.syntaxError("else must be followed by a statement block or const if")
	p.advance(_Rbrace)
	return nil
}

func (p *parser) constSwitchStmt() Stmt {
	if trace {
		defer p.trace("constSwitchStmt")()
	}

	s := new(ConstSwitchStmt)
	s.pos = p.pos()
	p.next()
	p.next()
	s.Targ = p.type_()
	s.Lbrace = p.pos()
	if !p.got(_Lbrace) {
		p.syntaxError("missing { after const switch type")
		p.advance(_Case, _Rbrace)
	}
	for p.tok != _EOF && p.tok != _Rbrace {
		s.Cases = append(s.Cases, p.constCase())
	}
	s.Rbrace = p.pos()
	p.want(_Rbrace)
	return s
}

func (p *parser) constCase() *ConstSwitchCase {
	if trace {
		defer p.trace("constCase")()
	}

	c := new(ConstSwitchCase)
	c.Case = p.pos()
	if p.tok != _Case {
		return p.constCaseRecover(c)
	}
	p.next()
	if p.tok == _Operator && p.op == Tilde {
		c.Tilde = true
		c.TildePos = p.pos()
		p.next()
	}
	c.Type = p.type_()
	if p.tok == _Comma {
		p.syntaxError("const switch case takes a single type, not a list")
		p.advance(_Colon, _Case, _Rbrace)
	}
	c.Colon = p.pos()
	p.want(_Colon)
	c.Body = p.stmtList()
	return c
}

func (p *parser) constCaseRecover(c *ConstSwitchCase) *ConstSwitchCase {
	p.syntaxError("expected case or }")
	p.advance(_Colon, _Case, _Rbrace)
	if p.tok != _Colon && p.tok != _Case && p.tok != _Rbrace && p.tok != _EOF {
		p.next()
	}
	c.Type = p.badExpr()
	c.Colon = p.pos()
	if p.got(_Colon) {
		c.Body = p.stmtList()
	}
	return c
}
