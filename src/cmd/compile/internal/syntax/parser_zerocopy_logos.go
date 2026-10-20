// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s parser logic for M5-S5 (zero-copy
// conversions; SPEC Sec 2.1, T-201). parser.go calls into it through a
// hook line only (G3); the rule below runs solely in LogosMode, so
// plain Go parses exactly as upstream. A `const T` type in operand
// position lets loGo/s spell const-view conversions such as
// `const []byte(s)`; pexpr already turns `Type(args)` into a CallExpr,
// so returning the ConstType here is enough for conversions. (A
// trailing `{` does not form a composite literal: pexpr only accepts
// the upstream complittype shapes there, so `const []byte{...}` stays
// a parse error and composites reach const views through the
// plain-literal-plus-assignment spelling.)

// logosConstOperand parses `const T` where an operand starts. The
// caller saw `const` in LogosMode; T itself parses through the M5-S1
// const-type rule, so `const []byte` views a byte slice. In plain mode
// it reproduces the upstream missing-expression error path.
func (p *parser) logosConstOperand() Expr {
	if x := p.logosConstType(); x != nil {
		return x
	}
	x := p.badExpr()
	p.syntaxError("expected expression")
	p.advance(_Rparen, _Rbrack, _Rbrace)
	return x
}
