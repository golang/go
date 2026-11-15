// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

func (s *scanner) logosQuestionTail() bool {
	if s.mode&uint(LogosMode) == 0 {
		return false
	}
	switch s.ch {
	case '?':
		s.nextch()
		s.prec = precCoalesce
		s.tok = _Coalesce
		return true
	case '.':
		s.nextch()
		if !isDecimal(s.ch) {
			s.tok = _QuestionDot
			return true
		}
		s.rewind()
		s.nextch()
	}
	return false
}

func (p *parser) logosOptionType(typ Expr) Expr {
	if p.mode&LogosMode == 0 || typ == nil {
		return typ
	}
	for {
		switch p.tok {
		case _Question:
			t := new(OptionType)
			t.pos = p.pos()
			p.next()
			t.Elem = typ
			typ = t
		case _Coalesce:
			p.syntaxError("T?? is not allowed; use (T?)?")
			pos := p.pos()
			p.next()
			for range 2 {
				t := new(OptionType)
				t.pos = pos
				t.Elem = typ
				typ = t
			}
		default:
			return typ
		}
	}
}

func (p *parser) logosOptionResult(list []*Field) []*Field {
	if p.mode&LogosMode == 0 || p.tok != _Question && p.tok != _Coalesce {
		return list
	}
	if len(list) != 1 || list[0].Name != nil {
		p.syntaxError("? applies to one unnamed result type; write (T)? or (T?)?")
		p.next()
		return list
	}
	list[0].Type = p.logosOptionType(list[0].Type)
	return list
}

func (p *parser) logosTypeElem() Expr {
	if p.mode&LogosMode == 0 {
		return p.type_()
	}
	typ := p.typeOrNil()
	if typ == nil {
		typ = p.badExpr()
		p.syntaxError("expected type")
		p.advance(_Comma, _Colon, _Semi, _Rparen, _Rbrack, _Rbrace)
	}
	return typ
}

func (p *parser) logosCoalesce(x Expr, pos Pos) Expr {
	t := new(CoalesceExpr)
	t.pos = pos
	p.next()
	t.X = x
	t.Y = p.binaryExpr(nil, precCoalesce-1)
	return t
}

func (p *parser) logosOptSelect(x Expr, pos Pos) Expr {
	p.next()
	if p.tok != _Name {
		p.syntaxError("expected name after ?.")
		p.logosAdvanceSelector()
		return x
	}
	t := new(OptSelectExpr)
	t.pos = pos
	t.X = x
	t.Sel = p.name()
	return t
}
