// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s parser logic for M6-S5 tuples and the
// shared `(` rule (SPEC Sec 2.4, T-204.1, and Sec 5.2-5.3, T-104.6).
// parser.go calls into it through hook lines only (G3); the rule below
// runs solely in LogosMode, so plain Go parses exactly as upstream.
//
// One `(` rule serves tuple types, tuple literals, parenthesized
// expressions, and lambda parameter lists: logosParen parses the element
// list once, then branches on what follows. A `)` followed by `=>`
// forms a *LambdaExpr (the body parses in parser_lambda_logos.go); a
// single element stays a parenthesized expression or type, exactly as
// upstream; two or more elements form a *TupleType in type position or
// a *TupleLit in expression position. A `:` between a name and its type
// also marks a named field, so `(x: int, y: int)` and `(x int, y int)`
// mean the same. No backtracking: each element parses once (#38).

// parenElem is one element of a `(` list: either a plain expression or
// type e, or a named tuple field (name with its type).
type parenElem struct {
	e    Expr  // value or type expression
	name *Name // non-nil for a named tuple field
	typ  Expr  // field type when name != nil
}

// logosParenOperand parses `(...)` in an expression position;
// logosParenType parses it in a type position. Both share logosParen.
func (p *parser) logosParenOperand(keepParens bool) (Expr, bool) {
	return p.logosParen(keepParens, false)
}

// logosParenType parses `(...)` in a type position.
func (p *parser) logosParenType() (Expr, bool) {
	return p.logosParen(false, true)
}

// logosParen parses `(...)` in LogosMode. keepParens mirrors the
// operand keep_parens flag (composite literals and go/defer calls keep
// parentheses); inType reports the typeOrNil position (a type context)
// as opposed to the operand position (an expression context). It
// reports true when it consumed the list, which is always the case in
// LogosMode once `(` is seen; plain Go never reaches it.
func (p *parser) logosParen(keepParens, inType bool) (Expr, bool) {
	if p.mode&LogosMode == 0 || p.tok != _Lparen {
		return nil, false
	}
	pos := p.pos()
	p.next()
	p.xnest++
	if p.tok == _Rparen {
		p.next()
		p.xnest--
		if p.tok == _FatArrow && p.lambdaBan == 0 {
			return p.logosParenLambda(pos, nil), true
		}
		p.syntaxError("expected expression")
		return p.badExpr(), true
	}
	var elems []parenElem
	for {
		if inType {
			elems = append(elems, p.logosParenTypeElem())
		} else {
			elems = append(elems, p.logosParenValueElem(keepParens))
		}
		if p.tok == _Comma {
			p.next()
			if p.tok == _Rparen {
				break
			}
			continue
		}
		break
	}
	p.xnest--
	p.want(_Rparen)

	// A `=>` after `)` forms a lambda over plain names (SPEC T-104.6),
	// except in switch-expression arm cases where `=>` delimits the arm.
	if p.tok == _FatArrow && p.lambdaBan == 0 {
		return p.logosParenLambda(pos, elems), true
	}

	// A single plain element stays parenthesized, exactly as upstream.
	if len(elems) == 1 && elems[0].name == nil {
		e := elems[0].e
		if !inType && p.tok == _Lbrace {
			// Mirror operand: parentheses stand around a composite
			// literal type when `{` follows.
			keepParens = true
		}
		if inType || !keepParens {
			return e, true
		}
		px := new(ParenExpr)
		px.pos = pos
		px.X = e
		return px, true
	}

	// A single named field is never a complete tuple (SPEC Sec 5.2
	// requires two or more elements).
	if len(elems) == 1 {
		if inType {
			p.syntaxError("tuple type needs at least two element types")
		} else {
			p.syntaxError("expected comma or ) in tuple literal")
		}
	}
	if inType {
		t := new(TupleType)
		t.pos = pos
		for _, el := range elems {
			f := new(Field)
			if el.name != nil {
				f.Name = el.name
				f.Type = el.typ
			} else {
				f.Type = el.e
			}
			t.Fields = append(t.Fields, f)
		}
		return t, true
	}
	t := new(TupleLit)
	t.pos = pos
	for _, el := range elems {
		if el.name != nil {
			t.Elems = append(t.Elems, el.typ)
		} else {
			t.Elems = append(t.Elems, el.e)
		}
	}
	return t, true
}

// logosParenLambda builds the lambda for `(elems) => body`. Every
// element must be a plain name; anything else is a syntax error at the
// arrow, and checking continues with the valid names.
func (p *parser) logosParenLambda(pos Pos, elems []parenElem) Expr {
	t := new(LambdaExpr)
	t.pos = pos
	t.Arrow = p.pos()
	var names []*Name
	for _, el := range elems {
		if el.name != nil {
			continue
		}
		if nm, ok := Unparen(el.e).(*Name); ok {
			names = append(names, nm)
			continue
		}
	}
	if len(names) != len(elems) {
		p.syntaxError("lambda parameters must be names")
	}
	t.Params = names
	p.next()
	t.Body = p.logosLambdaBody()
	return t
}

// logosParenValueElem parses one `(` element in an expression position.
// Operand-led heads parse exactly as in operand position and continue
// through pexpr and the binary operators, so calls, indexes, selections,
// composites, and `a + b` behave identically; anything else, including
// unary `*x` and `<-x`, parses as a full expression.
func (p *parser) logosParenValueElem(keepParens bool) parenElem {
	switch p.tok {
	case _Lparen:
		e, _ := p.logosParen(keepParens, false)
		return p.logosParenValueTail(e, keepParens)

	case _Func:
		// Mirror operand: a `{` after the signature makes a
		// function literal, otherwise it is a function type.
		pos := p.pos()
		p.next()
		_, ftyp := p.funcType("function type")
		if p.tok == _Lbrace {
			p.xnest++
			f := new(FuncLit)
			f.pos = pos
			f.Type = ftyp
			f.Body = p.funcBody()
			p.xnest--
			return p.logosParenValueTail(f, keepParens)
		}
		return p.logosParenValueTail(ftyp, keepParens)

	case _Const:
		// `const T` views (M5-S1) and `const T(x)` conversions
		// (M5-S5) start here, exactly as in operand position.
		return p.logosParenValueTail(p.logosConstOperand(), keepParens)

	case _Lbrack, _Chan, _Map, _Struct, _Interface:
		typ := p.typeOrNil()
		if typ == nil {
			typ = p.badExpr()
			p.syntaxError("expected type")
			p.advance(_Comma, _Rparen)
			return parenElem{e: typ}
		}
		return p.logosParenValueTail(typ, keepParens)
	}

	return p.logosParenValueField(p.expr())
}

// logosParenValueTail continues an operand-led element past its head:
// selectors, calls, and composites through pexpr, then binary operators,
// exactly the order expr uses for an expression that starts with head.
func (p *parser) logosParenValueTail(head Expr, keepParens bool) parenElem {
	return p.logosParenValueField(p.binaryExpr(p.pexpr(head, keepParens), 0))
}

// logosParenValueField splits a `name Type` element in an expression
// position. Two values in a row are never a valid expression, so no
// valid Go input parses differently; the enclosing list rejects the
// field with one error at `)`.
func (p *parser) logosParenValueField(e Expr) parenElem {
	if nm, ok := Unparen(e).(*Name); ok && p.tok == _Name {
		return parenElem{e: e, name: nm, typ: p.type_()}
	}
	return parenElem{e: e}
}

// logosParenTypeElem parses one `(` element in a type position: a type
// or a named field. Anything else is the upstream "expected type"
// error, so invalid Go reports exactly as in plain mode.
func (p *parser) logosParenTypeElem() parenElem {
	switch p.tok {
	case _Lparen:
		e, _ := p.logosParen(false, true)
		return parenElem{e: p.logosOptionType(e)}

	case _Name:
		e := p.qualifiedName(nil)
		if fe := p.logosParenTypeField(e); fe.name != nil {
			return fe
		}
		return parenElem{e: p.logosOptionType(e)}

	case _Star, _Arrow, _Lbrack, _Chan, _Map, _Struct, _Interface, _Func, _Const:
		typ := p.typeOrNil()
		if typ == nil {
			typ = p.badExpr()
			p.syntaxError("expected type")
			p.advance(_Comma, _Rparen)
		}
		return parenElem{e: p.logosOptionType(typ)}

	default:
		p.syntaxError("expected type")
		p.advance(_Comma, _Rparen)
		return parenElem{e: p.badExpr()}
	}
}

// logosParenTypeField splits a `name Type` (or `name: Type`) element
// into a named tuple field. Only a bare *Name starts a field.
func (p *parser) logosParenTypeField(e Expr) parenElem {
	nm, ok := Unparen(e).(*Name)
	if !ok {
		return parenElem{e: e}
	}
	if p.tok == _Colon {
		p.next()
		return parenElem{e: e, name: nm, typ: p.type_()}
	}
	switch p.tok {
	case _Name, _Star, _Lbrack, _Chan, _Map, _Struct, _Interface,
		_Func, _Const, _Lparen, _Arrow:
		return parenElem{e: e, name: nm, typ: p.type_()}
	}
	return parenElem{e: e}
}
