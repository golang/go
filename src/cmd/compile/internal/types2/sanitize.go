// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

func sanitizeInfo(info *Info) {
	var s sanitizer = make(map[Type]Type)

	// Note: Some map entries are not references.
	// If modified, they must be assigned back.

	for e, tv := range info.Types {
		tv.Type = s.typ(tv.Type)
		info.Types[e] = tv
	}

	for e, inf := range info.Inferred {
		for i, targ := range inf.Targs {
			inf.Targs[i] = s.typ(targ)
		}
		inf.Sig = s.typ(inf.Sig).(*Signature)
		info.Inferred[e] = inf
	}

	for _, obj := range info.Defs {
		if obj != nil {
			obj.setType(s.typ(obj.Type()))
		}
	}

	for _, obj := range info.Uses {
		if obj != nil {
			obj.setType(s.typ(obj.Type()))
		}
	}

	// TODO(gri) sanitize as needed
	// - info.Implicits
	// - info.Selections
	// - info.Scopes
	// - info.InitOrder
}

type sanitizer map[Type]Type

func (s sanitizer) typ(typ Type) Type {
	if t, found := s[typ]; found {
		return t
	}
	s[typ] = typ

	switch t := typ.(type) {
	case nil, *Basic, *bottom, *top:
		// nothing to do

	case *Array:
		t.elem = s.typ(t.elem)

	case *Slice:
		t.elem = s.typ(t.elem)

	case *Struct:
		s.varList(t.fields)

	case *Pointer:
		t.base = s.typ(t.base)

	case *Tuple:
		s.tuple(t)

	case *Signature:
		s.var_(t.recv)
		s.tuple(t.params)
		s.tuple(t.results)

	case *Sum:
		s.typeList(t.types)

	case *Interface:
		s.funcList(t.methods)
		s.typ(t.types)
		s.typeList(t.embeddeds)
		s.funcList(t.allMethods)
		s.typ(t.allTypes)

	case *Map:
		t.key = s.typ(t.key)
		t.elem = s.typ(t.elem)

	case *Chan:
		t.elem = s.typ(t.elem)

	case *Named:
		t.orig = s.typ(t.orig)
		t.underlying = s.typ(t.underlying)
		s.typeList(t.targs)
		s.funcList(t.methods)

	case *TypeParam:
		t.bound = s.typ(t.bound)

	case *instance:
		typ = t.expand()
		s[t] = typ

	default:
		unimplemented()
	}

	return typ
}

func (s sanitizer) var_(v *Var) {
	if v != nil {
		v.typ = s.typ(v.typ)
	}
}

func (s sanitizer) varList(list []*Var) {
	for _, v := range list {
		s.var_(v)
	}
}

func (s sanitizer) tuple(t *Tuple) {
	if t != nil {
		s.varList(t.vars)
	}
}

func (s sanitizer) func_(f *Func) {
	if f != nil {
		f.typ = s.typ(f.typ)
	}
}

func (s sanitizer) funcList(list []*Func) {
	for _, f := range list {
		s.func_(f)
	}
}

func (s sanitizer) typeList(list []Type) {
	for i, t := range list {
		list[i] = s.typ(t)
	}
}
