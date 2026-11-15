// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

type yieldDim struct {
	pos   Pos
	len   Expr
	slice bool
}

func (p *parser) logosYield() (Expr, Stmt) {
	if p.mode&LogosMode == 0 || p.lit != "yield" {
		return p.exprList(), nil
	}
	name := p.name()
	if p.yieldOperandStart() {
		return nil, NewYieldStmt(name, p.exprList())
	}
	var head Expr = name
	if p.tok == _Lbrack {
		var typed bool
		if head, typed = p.yieldBracket(name); typed {
			return nil, NewYieldStmt(name, p.yieldTail(head))
		}
	}
	return p.yieldStatement(p.yieldTail(head))
}

func (p *parser) yieldTail(head Expr) Expr {
	return p.exprListTail(p.logosLambdaOrExpr(p.binaryExpr(p.pexpr(head, true), 0)))
}

func (p *parser) yieldOperandStart() bool {
	switch p.tok {
	case _Name, _Literal, _Func, _Struct, _Map, _Chan, _Interface, _Const, _Switch:
		return true
	case _Operator:
		return p.op == Not
	}
	return false
}

func (p *parser) yieldTypeStart() bool {
	switch p.tok {
	case _Name, _Star, _Func, _Struct, _Map, _Chan, _Interface:
		return true
	}
	return false
}

func (p *parser) yieldStatement(lhs Expr) (Expr, Stmt) {
	switch p.tok {
	case _AssignOp, _Assign, _Define, _IncOp, _Arrow, _Colon:
		return lhs, nil
	}
	if _, isCall := lhs.(*CallExpr); isCall {
		return lhs, nil
	}
	list := UnpackListExpr(lhs)
	value, word, ok := YieldOperand(list[0])
	if !ok {
		return lhs, nil
	}
	if len(list) > 1 {
		rest := *lhs.(*ListExpr)
		rest.ElemList = append([]Expr{value}, list[1:]...)
		value = &rest
	}
	return nil, NewYieldStmt(word, value)
}

func (p *parser) yieldBracket(name *Name) (head Expr, typed bool) {
	var dims []yieldDim
	instance := false
	for p.tok == _Lbrack {
		dim := yieldDim{pos: p.pos()}
		p.next()
		switch p.tok {
		case _Rbrack:
			dim.slice, typed = true, true
		case _DotDotDot:
			p.next()
			typed = true
		default:
			var comma bool
			dim.len, comma = p.typeList(false)
			instance = instance || comma
		}
		p.want(_Rbrack)
		dims = append(dims, dim)
	}
	if !typed && (instance || !p.yieldTypeStart()) {
		head = name
		for _, dim := range dims {
			index := new(IndexExpr)
			index.pos = dim.pos
			index.X = head
			index.Index = dim.len
			head = index
		}
		return head, false
	}
	head = p.logosTypeElem()
	for i := len(dims) - 1; i >= 0; i-- {
		head = yieldDimType(dims[i], head)
	}
	return head, true
}

func yieldDimType(dim yieldDim, elem Expr) Expr {
	if dim.slice {
		t := new(SliceType)
		t.pos = dim.pos
		t.Elem = elem
		return t
	}
	t := new(ArrayType)
	t.pos = dim.pos
	t.Len = dim.len
	t.Elem = elem
	return t
}
