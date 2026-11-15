// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

func (p *parser) logosOperand() Expr {
	if p.mode&LogosMode == 0 {
		return nil
	}
	switch p.tok {
	case _Switch:
		return p.switchExpr()
	}
	return nil
}

func (p *parser) switchExpr() *SwitchExpr {
	if trace {
		defer p.trace("switchExpr")()
	}

	e := new(SwitchExpr)
	e.pos = p.pos()

	e.Init, e.Tag, _ = p.header(_Switch)

	e.Lbrace = p.pos()
	if !p.got(_Lbrace) {
		p.syntaxError("missing { after switch clause")
		p.advance(_Case, _Default, _Rbrace)
	}
	p.xnest++
	for p.tok != _EOF && p.tok != _Rbrace {
		e.Arms = append(e.Arms, p.armClause())
		if !p.got(_Semi) && p.tok != _Rbrace {
			p.syntaxError("at end of switch arm")
			p.advance(_Semi, _Rbrace, _Case, _Default)
			p.got(_Semi)
		}
	}
	p.xnest--
	e.Rbrace = p.pos()
	p.want(_Rbrace)

	return e
}

func (p *parser) armClause() *ArmClause {
	if trace {
		defer p.trace("armClause")()
	}

	c := new(ArmClause)
	c.pos = p.pos()

	switch p.tok {
	case _Case:
		p.next()
		p.lambdaBan++
		c.Cases = p.exprList()
		p.lambdaBan--

	case _Default:
		p.next()

	default:
		p.syntaxError("expected case or default or }")
		p.advance(_FatArrow, _Case, _Default, _Rbrace)
	}

	c.Arrow = p.pos()
	p.want(_FatArrow)
	c.Body = p.expr()

	return c
}

func (p *parser) logosTypeDeclType(d *TypeDecl) Expr {
	if p.mode&LogosMode == 0 || d.Alias || p.tok != _Struct {
		return p.typeOrNil()
	}
	if trace {
		defer p.trace("logosTypeDeclType")()
	}

	typ := new(StructType)
	typ.pos = p.pos()
	p.next()

	var asserts []Expr
	var colon Pos
	if p.tok == _Colon {
		colon = p.pos()
		p.next()
		for {
			asserts = append(asserts, p.type_())
			if !p.got(_Comma) {
				break
			}
		}
	}

	lbrace := p.pos()
	p.want(_Lbrace)
	p.list("struct type", _Semi, _Rbrace, func() bool {
		p.fieldDecl(typ)
		return false
	})

	if len(asserts) == 0 {
		return typ
	}
	a := new(AssertedStruct)
	a.pos = typ.pos
	a.Asserts = asserts
	a.Struct = typ
	a.Colon = colon
	a.Lbrace = lbrace
	return a
}

func (p *parser) logosAdvanceSelector() {
	if p.mode&LogosMode != 0 {
		p.advance(_Semi, _Rparen, _Rbrace)
		return
	}
	p.advance(_Semi, _Rparen)
}
