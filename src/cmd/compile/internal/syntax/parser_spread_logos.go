// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s parser logic for M6-S4 spread (`..src`
// composite literal elements; SPEC T-104.5). parser.go calls into it
// through one hook line only (G3); the rule below runs solely in
// LogosMode, so plain Go parses exactly as upstream. The scanner splits
// `..` into two _Dot tokens (the `...` form stays _DotDotDot), so the
// parser cases for spread are unreachable in plain mode: no valid Go
// composite literal element starts with `.`.

// logosSpreadOrElem parses one composite literal element: a `..src`
// spread in LogosMode, otherwise the shared element rule. A spread
// followed by `:` becomes a keyed element whose key is not a name,
// which types2 rejects with its invalid-field diagnostic.
func (p *parser) logosSpreadOrElem() Expr {
	if se := p.logosSpreadElem(); se != nil {
		return se
	}
	return p.bare_complitexpr()
}

// logosSpreadElem parses a `..src` spread element at the head of a
// composite literal element. It returns nil, consuming nothing, unless
// in LogosMode the current token is `.`. A lone `.` is a syntax error
// at the offending token (no valid element starts with `.`), so the
// enclosing literal still closes where the source closes it. Two dots
// wrap the following element expression (parsed with the shared element
// rule, so nested spreads compose) in a SpreadExpr positioned at the
// first dot.
func (p *parser) logosSpreadElem() Expr {
	if p.mode&LogosMode == 0 || p.tok != _Dot {
		return nil
	}
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
