// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

func (p *parser) logosLambdaBody() Node {
	if p.tok == _Lbrace {
		return p.blockStmt("lambda")
	}
	return p.expr()
}

func (p *parser) logosLambdaOrExpr(x Expr) Expr {
	if p.mode&LogosMode == 0 || p.tok != _FatArrow || p.lambdaBan > 0 {
		return x
	}
	if nm, ok := Unparen(x).(*Name); ok {
		t := new(LambdaExpr)
		t.pos = StartPos(x)
		t.Arrow = p.pos()
		t.Params = []*Name{nm}
		p.next()
		t.Body = p.logosLambdaBody()
		return t
	}
	p.syntaxError("lambda parameters must be names")
	p.next()
	p.logosLambdaBody()
	return p.badExpr()
}
