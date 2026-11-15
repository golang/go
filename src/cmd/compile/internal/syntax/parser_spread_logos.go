// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

func (p *parser) logosSpreadOrElem() Expr {
	if p.mode&LogosMode != 0 && p.tok == _Dot {
		pos := p.pos()
		p.next()
		if p.tok != _Dot {
			p.syntaxError("expected . for spread `..src` in struct literal")
			p.advance(_Comma, _Rbrace)
			return p.badExpr()
		}
		p.next()
		s := new(SpreadExpr)
		s.pos = pos
		s.X = p.bare_complitexpr()
		return s
	}
	return p.bare_complitexpr()
}
