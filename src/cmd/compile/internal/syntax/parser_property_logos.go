// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s parser logic for M4-S1 property interfaces
// (SPEC Sec 2.3, T-203.2; grammar Sec 5.2: PropertyElem ::=
// ["const"] Name Type). parser.go calls into it through hook lines only
// (G3); every rule below runs solely in LogosMode, so plain Go parses
// exactly as upstream.
//
// A property element is `const F T` (read-only) or `F T` (read-write)
// inside interface braces, mixed freely with methods and embedded
// elements. The element becomes a *Field whose Name is F and whose Type
// is a *PropertyType; types2 lowers it to Logos_GetF/Logos_SetF methods.

// logosPropertyConst parses a `const F T` property element. It returns
// nil, consuming nothing, unless in LogosMode the current token is
// `const`. Unlike the bare form, `const` never starts a valid plain-Go
// interface element, so no exportedness gate is needed here: no .go
// input parses differently.
func (p *parser) logosPropertyConst() *Field {
	if p.mode&LogosMode == 0 || p.tok != _Const {
		return nil
	}
	pos := p.pos()
	p.next()
	name := p.name()
	t := new(PropertyType)
	t.pos = pos
	t.Const = true
	t.Type = p.type_()
	f := new(Field)
	f.pos = pos
	f.Name = name
	f.Type = t
	return f
}

// logosPropertyRest parses the type half of a bare `F T` property
// element after methodDecl consumed F as name. It returns nil unless in
// LogosMode the token after F can start a property type but cannot
// continue a method or embedded element, leaving those shapes to the
// methodDecl switch. In particular `(` (method), `[` (generic method or
// instantiated embed), `|` (union), `.` (qualified embed), and every
// separator stay upstream, so LogosMode parses plain Go identically.
func (p *parser) logosPropertyRest(name *Name, f *Field) *Field {
	if p.mode&LogosMode == 0 || !isExported(name.Value) {
		return nil
	}
	// _Lparen continues a method, _Lbrack a generic method or an
	// instantiated embed, _Dot a qualified embed, _Operator("|") a
	// union, and separators end the element: all stay upstream. Note
	// _Lbrack is deliberately absent: `F [N]T` keeps the upstream
	// generic-method reading, so array and slice property types are
	// not expressible in S1.
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
