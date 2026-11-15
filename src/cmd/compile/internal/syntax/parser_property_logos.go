// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

func (p *parser) logosPropertyConst() *Field {
	if p.mode&LogosMode == 0 || p.tok != _Const {
		return nil
	}
	pos := p.pos()
	p.next()
	name := p.name()
	f := new(Field)
	f.pos = pos
	f.Name = name
	if p.tok == _Lparen {
		_, ft := p.funcType("interface method")
		f.Const = true
		f.Type = ft
		return f
	}
	t := new(PropertyType)
	t.pos = pos
	t.Const = true
	t.Type = p.type_()
	f.Type = t
	return f
}

func (p *parser) logosPropertyRest(name *Name, f *Field) *Field {
	if p.mode&LogosMode == 0 || !isExported(name.Value) {
		return nil
	}
	switch p.tok {
	case _Name, _Star, _Arrow, _Func, _Map, _Chan, _Struct, _Interface:
	default:
		return nil
	}
	t := new(PropertyType)
	t.pos = p.pos()
	t.Type = p.type_()
	f.Name = name
	f.Type = t
	return f
}
