// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s parser logic for M6-S6 lambdas (SPEC Sec
// 1.4, T-104.6). parser.go calls into it through hook lines only (G3);
// every rule below runs solely in LogosMode, so plain Go parses exactly
// as upstream. The scanner only produces _FatArrow in LogosMode, so the
// lone-name rule is unreachable in plain mode.

// logosLambdaBody parses a lambda body after `=>`: a block when `{`
// follows, otherwise a full expression (which may hold `?` per
// T-104.1).
func (p *parser) logosLambdaBody() Node {
	if p.tok == _Lbrace {
		return p.blockStmt("lambda")
	}
	return p.expr()
}

// logosLambdaOrExpr wraps a lone `x => ...` name into a *LambdaExpr.
// It runs at the end of expr(), so the head parses with full operator
// precedence and the body nests to the right. A non-name head is a
// syntax error at the arrow. In plain mode, or without `=>`, it returns
// x unchanged, so every plain-Go tree is identical.
func (p *parser) logosLambdaOrExpr(x Expr) Expr {
	if p.mode&LogosMode == 0 || p.tok != _FatArrow || p.lambdaBan > 0 {
		return x
	}
	t := new(LambdaExpr)
	t.pos = StartPos(x)
	t.Arrow = p.pos()
	if nm, ok := Unparen(x).(*Name); ok {
		t.Params = []*Name{nm}
	} else {
		p.syntaxError("lambda parameters must be names")
	}
	p.next()
	t.Body = p.logosLambdaBody()
	return t
}
