// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

func YieldOperand(x Expr) (value Expr, word *Name, ok bool) {
	switch x := x.(type) {
	case *CallExpr:
		if name, isWord := x.Fun.(*Name); isWord && isYieldWord(name) {
			if len(x.ArgList) != 1 || x.HasDots {
				return nil, nil, false
			}
			paren := new(ParenExpr)
			paren.X = x.ArgList[0]
			paren.SetPos(x.Pos())
			return paren, name, true
		}
	case *Operation:
		if name, isWord := x.X.(*Name); isWord && x.Y != nil && isYieldWord(name) {
			if !yieldUnaryOp(x.Op) || isLambdaValue(x.Y) {
				return nil, nil, false
			}
			return yieldPrefix(x.Op, x.Y, x.Pos()), name, true
		}
	}
	left := yieldLeft(x)
	if left == nil {
		return nil, nil, false
	}
	value, word, ok = YieldOperand(left)
	if !ok {
		return nil, nil, false
	}
	return yieldWithLeft(x, value), word, true
}

func YieldSendOperand(s *SendStmt) (value Expr, word *Name, ok bool) {
	name, isWord := s.Chan.(*Name)
	if !isWord || !isYieldWord(name) || isLambdaValue(s.Value) {
		return nil, nil, false
	}
	return yieldPrefix(Recv, s.Value, s.Pos()), name, true
}

func YieldReserved(x Expr) (word *Name, ok bool) {
	for {
		left := yieldLeft(x)
		if left == nil {
			return nil, false
		}
		if name, isWord := left.(*Name); isWord && isYieldWord(name) {
			switch x.(type) {
			case *IndexExpr, *CallExpr:
				return name, true
			}
			return nil, false
		}
		x = left
	}
}

func NewYieldStmt(word *Name, value Expr) *YieldStmt {
	s := new(YieldStmt)
	s.SetPos(word.Pos())
	s.Value = value
	return s
}

func yieldLeft(x Expr) Expr {
	switch x := x.(type) {
	case *Operation:
		if x.Y != nil {
			return x.X
		}
	case *CallExpr:
		return x.Fun
	case *IndexExpr:
		return x.X
	case *SliceExpr:
		return x.X
	case *SelectorExpr:
		return x.X
	case *AssertExpr:
		return x.X
	case *CoalesceExpr:
		return x.X
	case *TryExpr:
		return x.X
	case *OptSelectExpr:
		return x.X
	}
	return nil
}

func yieldWithLeft(x, left Expr) Expr {
	switch x := x.(type) {
	case *Operation:
		c := *x
		c.X = left
		return &c
	case *CallExpr:
		c := *x
		c.Fun = left
		return &c
	case *IndexExpr:
		c := *x
		c.X = left
		return &c
	case *SliceExpr:
		c := *x
		c.X = left
		return &c
	case *SelectorExpr:
		c := *x
		c.X = left
		return &c
	case *AssertExpr:
		c := *x
		c.X = left
		return &c
	case *CoalesceExpr:
		c := *x
		c.X = left
		return &c
	case *TryExpr:
		c := *x
		c.X = left
		return &c
	case *OptSelectExpr:
		c := *x
		c.X = left
		return &c
	}
	return x
}

func yieldPrefix(op Operator, e Expr, anchor Pos) Expr {
	switch e := e.(type) {
	case *Operation:
		if e.Y != nil {
			c := *e
			c.X = yieldPrefix(op, e.X, anchor)
			return &c
		}
	case *CoalesceExpr:
		c := *e
		c.X = yieldPrefix(op, e.X, anchor)
		return &c
	}
	unary := &Operation{Op: op, X: e}
	unary.SetPos(anchor)
	return unary
}

func isYieldWord(name *Name) bool {
	return name.Value == "yield"
}

func isLambdaValue(e Expr) bool {
	_, ok := e.(*LambdaExpr)
	return ok
}

func yieldUnaryOp(op Operator) bool {
	switch op {
	case Add, Sub, Mul, And, Xor:
		return true
	}
	return false
}
