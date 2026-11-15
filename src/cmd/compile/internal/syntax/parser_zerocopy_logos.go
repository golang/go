// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

func (p *parser) logosConstOperand() Expr {
	if x := p.logosConstType(); x != nil {
		return x
	}
	x := p.badExpr()
	p.syntaxError("expected expression")
	p.advance(_Rparen, _Rbrack, _Rbrace)
	return x
}
