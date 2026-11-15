// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

func (p *parser) logosConstType() Expr {
	if p.mode&LogosMode == 0 {
		return nil
	}
	pos := p.pos()
	p.next()
	t := new(ConstType)
	t.pos = pos
	t.Elem = p.type_()
	return t
}
