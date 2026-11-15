// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

func (p *parser) logosConstDecl(list []Decl) []Decl {
	pos := p.pos()
	p.next()
	if p.mode&LogosMode != 0 && p.tok == _Func {
		funcPos := p.pos()
		pragma := p.takePragma()
		p.next()
		fn := p.funcDeclOrNil()
		fn.pos = funcPos
		if fn.Pragma == nil {
			fn.Pragma = pragma
		}
		d := &ConstFuncDecl{ConstPos: pos, Func: fn}
		d.pos = pos
		return append(list, d)
	}
	return p.appendGroup(list, p.constDecl)
}
