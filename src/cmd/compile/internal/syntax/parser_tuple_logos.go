// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

type parenElem struct {
	e    Expr
	name *Name
	typ  Expr
}

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

	if p.tok == _FatArrow && p.lambdaBan == 0 {
		return p.logosParenLambda(pos, elems), true
	}

	if len(elems) == 1 && elems[0].name == nil {
		e := elems[0].e
		if !inType && p.tok == _Lbrace {
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

func (p *parser) logosParenLambda(pos Pos, elems []parenElem) Expr {
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
		p.next()
		p.logosLambdaBody()
		return p.badExpr()
	}
	t := new(LambdaExpr)
	t.pos = pos
	t.Arrow = p.pos()
	t.Params = names
	p.next()
	t.Body = p.logosLambdaBody()
	return t
}

func (p *parser) logosParenValueElem(keepParens bool) parenElem {
	switch p.tok {
	case _Lparen:
		e, _ := p.logosParen(keepParens, false)
		return p.logosParenValueTail(e, keepParens)

	case _Func:
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

func (p *parser) logosParenValueTail(head Expr, keepParens bool) parenElem {
	return p.logosParenValueField(p.binaryExpr(p.pexpr(head, keepParens), 0))
}

func (p *parser) logosParenValueField(e Expr) parenElem {
	if nm, ok := Unparen(e).(*Name); ok && p.tok == _Name {
		return parenElem{e: e, name: nm, typ: p.type_()}
	}
	return parenElem{e: e}
}

func (p *parser) logosParenTypeElem() parenElem {
	switch p.tok {
	case _Lparen:
		e, _ := p.logosParen(false, true)
		return parenElem{e: p.logosOptionType(e)}

	case _Name:
		name := p.name()
		if p.tok == _Lbrack {
			return p.logosParenTypeBracket(name)
		}
		e := p.qualifiedName(name)
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

func (p *parser) logosParenTypeBracket(name *Name) parenElem {
	typ := p.arrayOrTArgs()
	inst, ok := typ.(*IndexExpr)
	if !ok {
		return parenElem{e: name, name: name, typ: typ}
	}
	inst.X = name
	return parenElem{e: p.logosOptionType(inst)}
}

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
