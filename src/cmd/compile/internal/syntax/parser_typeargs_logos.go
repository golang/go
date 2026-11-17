// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

func (p *parser) logosTypeArgExpr() Expr {
	if p.mode&LogosMode == 0 {
		return p.expr()
	}
	p.logosTypeArgDepth++
	x := p.expr()
	p.logosTypeArgDepth--
	return x
}

func (p *parser) logosTypeArgNameElem(keepParens bool) parenElem {
	name := p.name()
	field := parenElem{e: name}
	old := p.logosTypeArgField
	p.logosTypeArgField = &field
	e := p.pexpr(name, keepParens)
	p.logosTypeArgField = old
	if field.name != nil {
		return field
	}
	if logosTypeArgName(e) == name {
		switch p.tok {
		case _Colon:
			if e == name {
				p.next()
				return parenElem{e: e, name: name, typ: p.type_()}
			}
		case _Name, _Map, _Chan, _Const, _Func, _Struct, _Interface, _Arrow:
			return parenElem{e: e, name: name, typ: logosTypeArgArrays(e, p.type_())}
		}
	}
	return p.logosParenValueField(p.logosLambdaOrExpr(p.binaryExpr(e, 0)))
}

func (p *parser) logosTypeArgFieldPrefix(e Expr) bool {
	return p.logosTypeArgField != nil && logosTypeArgName(e) == p.logosTypeArgField.e
}

// '[' has been consumed. An empty bracket can only be a named slice field
// when it belongs to the exact initial name of this parenthesized element.
func (p *parser) logosTypeArgSlice(prefix Expr, pos Pos) bool {
	if p.tok != _Rbrack || !p.logosTypeArgFieldPrefix(prefix) {
		return false
	}
	p.next()
	p.logosTypeArgField.name = logosTypeArgName(prefix)
	p.logosTypeArgField.typ = logosTypeArgArrays(prefix, p.logosOptionType(p.sliceType(pos)))
	return true
}

// Ordinary arguments keep the existing expression grammar. Only a list
// containing an unambiguous named type field is retained as a tuple type;
// the checker still decides whether the enclosing brackets instantiate.
func (p *parser) logosTypeArgCall(fun Expr, pos Pos) ([]Expr, bool) {
	if !p.logosTypeArgFieldPrefix(fun) {
		return p.argList()
	}
	var elems []parenElem
	var hasDots, named bool
	p.xnest++
	p.list("argument list", _Comma, _Rparen, func() bool {
		el := p.logosParenValueElem(true)
		if el.name == nil {
			el.e = p.logosNamedArg(el.e)
		}
		elems = append(elems, el)
		named = named || el.name != nil
		hasDots = p.got(_DotDotDot)
		return hasDots
	})
	p.xnest--
	if named {
		t := new(TupleType)
		t.pos = pos
		for _, el := range elems {
			typ := el.e
			if el.name != nil {
				typ = el.typ
			}
			t.Fields = append(t.Fields, &Field{Name: el.name, Type: typ})
		}
		return []Expr{t}, hasDots
	}
	list := make([]Expr, len(elems))
	for i, el := range elems {
		list[i] = el.e
	}
	return list, hasDots
}
