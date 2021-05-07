// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

// sanitizeInfo walks the types contained in info to ensure that all instances
// are expanded.
//
// This includes some objects that may be shared across concurrent
// type-checking passes (such as those in the universe scope), so we are
// careful here not to write types that are already sanitized. This avoids a
// data race as any shared types should already be sanitized.
func sanitizeInfo(info *Info) {
	var s sanitizer = make(map[Type]Type)

	// Note: Some map entries are not references.
	// If modified, they must be assigned back.

	for e, tv := range info.Types {
		if typ := s.typ(tv.Type); typ != tv.Type {
			tv.Type = typ
			info.Types[e] = tv
		}
	}

	inferred := getInferred(info)
	for e, inf := range inferred {
		changed := false
		for i, targ := range inf.Targs {
			if typ := s.typ(targ); typ != targ {
				inf.Targs[i] = typ
				changed = true
			}
		}
		if typ := s.typ(inf.Sig); typ != inf.Sig {
			inf.Sig = typ.(*Signature)
			changed = true
		}
		if changed {
			inferred[e] = inf
		}
	}

	for _, obj := range info.Defs {
		if obj != nil {
			if typ := s.typ(obj.Type()); typ != obj.Type() {
				obj.setType(typ)
			}
		}
	}

	for _, obj := range info.Uses {
		if obj != nil {
			if typ := s.typ(obj.Type()); typ != obj.Type() {
				obj.setType(typ)
			}
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
	if typ == nil {
		return nil
	}

	if t, found := s[typ]; found {
		return t
	}
	s[typ] = typ

	switch t := typ.(type) {
	case *Basic, *bottom, *top:
		// nothing to do

	case *Array:
		if elem := s.typ(t.elem); elem != t.elem {
			t.elem = elem
		}

	case *Slice:
		if elem := s.typ(t.elem); elem != t.elem {
			t.elem = elem
		}

	case *Struct:
		s.varList(t.fields)

	case *Pointer:
		if base := s.typ(t.base); base != t.base {
			t.base = base
		}

	case *Tuple:
		s.tuple(t)

	case *Signature:
		s.var_(t.recv)
		s.tuple(t.params)
		s.tuple(t.results)

	case *_Sum:
		s.typeList(t.types)

	case *Interface:
		s.funcList(t.methods)
		if types := s.typ(t.types); types != t.types {
			t.types = types
		}
		s.typeList(t.embeddeds)
		s.funcList(t.allMethods)
		if allTypes := s.typ(t.allTypes); allTypes != t.allTypes {
			t.allTypes = allTypes
		}

	case *Map:
		if key := s.typ(t.key); key != t.key {
			t.key = key
		}
		if elem := s.typ(t.elem); elem != t.elem {
			t.elem = elem
		}

	case *Chan:
		if elem := s.typ(t.elem); elem != t.elem {
			t.elem = elem
		}

	case *Named:
		if debug && t.check != nil {
			panic("internal error: Named.check != nil")
		}
		if orig := s.typ(t.orig); orig != t.orig {
			t.orig = orig
		}
		if under := s.typ(t.underlying); under != t.underlying {
			t.underlying = under
		}
		s.typeList(t.targs)
		s.funcList(t.methods)

	case *_TypeParam:
		if bound := s.typ(t.bound); bound != t.bound {
			t.bound = bound
		}

	case *instance:
		typ = t.expand()
		s[t] = typ

	default:
		panic("unimplemented")
	}

	return typ
}

func (s sanitizer) var_(v *Var) {
	if v != nil {
		if typ := s.typ(v.typ); typ != v.typ {
			v.typ = typ
		}
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
		if typ := s.typ(f.typ); typ != f.typ {
			f.typ = typ
		}
	}
}

func (s sanitizer) funcList(list []*Func) {
	for _, f := range list {
		s.func_(f)
	}
}

func (s sanitizer) typeList(list []Type) {
	for i, t := range list {
		if typ := s.typ(t); typ != t {
			list[i] = typ
		}
	}
}
