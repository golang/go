// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s parser logic for M2: switch expressions with
// `=>` arms (M2-S5) and struct declarations with inline interface
// assertions (M2-S6). parser.go calls into it through hook lines only
// (G3); every rule below runs solely in LogosMode, so plain Go parses
// exactly as upstream. Closed unions need no grammar of their own: the
// SPEC spelling is an interface whose type set is a finite union, which
// the upstream interface grammar already parses.

// logosOperand parses the loGo/s operands that start with a keyword Go
// does not accept in operand position. It returns nil, consuming nothing,
// when the current token starts none of them, so the caller reports the
// upstream "expected expression" error.
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

// switchExpr parses a switch expression:
//
//	SwitchExpr = "switch" [ SimpleStmt ";" ] [ Tag ] "{" { Arm ";" } "}" .
//
// The header grammar is the one of a switch statement, so a type switch
// guard `[ v := ] x.(type)` reaches the node as a *TypeSwitchGuard tag.
// Arms are separated by semicolons, written or inserted at a line end;
// the semicolon before `}` is optional.
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
	// Composite literals are valid inside arms, even when the switch
	// expression itself sits in a statement header.
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

// armClause parses one switch expression arm:
//
//	Arm = "case" ExpressionList "=>" Expression | "default" "=>" Expression .
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

// logosTypeDeclType parses the right-hand side of the non-generic type
// declaration d. In LogosMode a struct type may list the interfaces the
// declared type's pointer must implement between `struct` and `{`:
//
//	StructDecl = "struct" [ ":" Type { "," Type } ] "{" { FieldDecl ";" } "}" .
//
// Without a list it returns the ordinary *StructType, so every other
// right-hand side (including aliases, which carry no assertions) takes
// the upstream path. A list on any other type stays a syntax error: the
// type ends before the `:`.
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

// logosAdvanceSelector recovers from a selector that lacks a name after
// the dot. In LogosMode a closing brace ends the skip, so the enclosing
// block still closes where the source closes it instead of swallowing the
// following declarations as part of the broken statement.
func (p *parser) logosAdvanceSelector() {
	if p.mode&LogosMode != 0 {
		p.advance(_Semi, _Rparen, _Rbrace)
		return
	}
	p.advance(_Semi, _Rparen)
}
