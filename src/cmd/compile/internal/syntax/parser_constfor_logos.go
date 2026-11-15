// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

func (p *parser) logosConstForStmt() Stmt {
	if p.mode&LogosMode == 0 || !p.peekConstWord() {
		return nil
	}
	pos := p.pos()
	p.next()
	p.want(_For)
	var index, field *Name
	first := p.name()
	if p.tok == _Comma {
		p.next()
		index = first
		field = p.name()
	} else {
		field = first
	}
	p.want(_Define)
	p.want(_Range)
	s := new(ConstForStmt)
	s.pos = pos
	s.Index = index
	s.Field = field
	outer := p.xnest
	p.xnest = -1
	s.Range = p.expr()
	p.xnest = outer
	s.Body = p.blockStmt("const for clause")
	return s
}

func (p *parser) peekConstWord() bool {
	q := p.r - p.chw
	get := func() (byte, bool) {
		for q >= p.e {
			if p.ioerr != nil {
				return 0, false
			}
			q -= p.fillPeek()
		}
		c := p.buf[q]
		q++
		return c, true
	}
	for {
		c, ok := get()
		if !ok {
			return false
		}
		if c == ' ' || c == '\t' || c == '\r' || c == '\n' {
			continue
		}
		if c == '/' {
			c2, ok := get()
			if !ok {
				return false
			}
			if c2 == '/' {
				for {
					c3, ok := get()
					if !ok || c3 == '\n' {
						break
					}
				}
				continue
			}
			if c2 == '*' {
				for {
					c3, ok := get()
					if !ok {
						return false
					}
					if c3 == '*' {
						c4, ok := get()
						if !ok {
							return false
						}
						if c4 == '/' {
							break
						}
						q--
					}
				}
				continue
			}
			return false
		}
		if c != 'f' {
			return false
		}
		var word [3]byte
		word[0] = c
		for i := 1; i < 3; i++ {
			c, ok := get()
			if !ok {
				return false
			}
			word[i] = c
		}
		if string(word[:]) != "for" {
			return false
		}
		c2, ok2 := get()
		if !ok2 {
			return true
		}
		return !isWordByte(c2)
	}
}

func (p *parser) logosFieldProj(x Expr, pos Pos) Expr {
	if p.mode&LogosMode == 0 || p.tok != _Lbrack {
		return nil
	}
	p.next()
	name := p.name()
	rbrack := p.pos()
	p.want(_Rbrack)
	t := new(FieldProjExpr)
	t.pos = pos
	t.X = x
	t.Field = name
	t.Rbrack = rbrack
	return t
}
