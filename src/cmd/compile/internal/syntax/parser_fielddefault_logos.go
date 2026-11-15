// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

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
