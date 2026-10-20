// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// logosFieldDefault parses the `= const` default of a struct field in
// LogosMode (`F T = const` before the tag, T-104.5). In plain mode it
// returns typ unchanged, so `=` keeps its upstream syntax error. Only
// fieldDecl calls it, so embedded fields and parameter lists never
// carry defaults.
func (p *parser) logosFieldDefault(typ Expr) Expr {
	if p.mode&LogosMode == 0 || p.tok != _Assign {
		return typ
	}
	t := new(FieldDefault)
	t.pos = p.pos()
	p.next()
	t.Type = typ
	t.Default = p.expr()
	return t
}
