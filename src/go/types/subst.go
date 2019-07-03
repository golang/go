// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements instantiation of generic types
// through substitution of type parameters by actual
// types.

package types

func subst(typ Type, targs []Type) Type {
	if len(targs) == 0 {
		return typ
	}
	s := subster{targs, make(map[Type]Type)}
	return s.typ(typ)
}

type subster struct {
	targs []Type
	cache map[Type]Type
}

func (s *subster) typ(typ Type) (res Type) {
	// TODO(gri) this is not correct in the presence of cycles
	if t, hit := s.cache[typ]; hit {
		return t
	}
	defer func() {
		s.cache[typ] = res
	}()

	switch t := typ.(type) {
	case nil, *Basic: // TODO(gri) should nil be handled here?
		// nothing to do

	case *Array:
		elem := s.typ(t.elem)
		if elem != t.elem {
			return &Array{t.len, elem}
		}

	case *Slice:
		elem := s.typ(t.elem)
		if elem != t.elem {
			return &Slice{elem}
		}

	case *Struct:
		if fields, copied := s.varList(t.fields); copied {
			return &Struct{fields, t.tags}
		}

	case *Pointer:
		base := s.typ(t.base)
		if base != t.base {
			return &Pointer{base}
		}

	case *Tuple:
		return s.tuple(t)

	case *Signature:
		recv := s.var_(t.recv) // not strictly needed (receivers cannot be parametrized)
		params := s.tuple(t.params)
		results := s.tuple(t.results)
		if recv != t.recv || params != t.params || results != t.results {
			copy := *t
			copy.tparams = nil // TODO(gri) is this correct? (another indication that perhaps tparams belong to the function decl)
			copy.recv = recv
			copy.params = params
			copy.results = results
			return &copy
		}

	case *Interface:
		panic("subst not implemented for interfaces")

	case *Map:
		key := s.typ(t.key)
		elem := s.typ(t.elem)
		if key != t.key || elem != t.elem {
			return &Map{key, elem}
		}

	case *Chan:
		elem := s.typ(t.elem)
		if elem != t.elem {
			return &Chan{t.dir, elem}
		}

	case *Named:
		// TODO(gri) is this correct?
		// nothing to do

	case *TypeParam:
		if targ := s.targs[t.index]; targ != nil {
			return targ
		}

	case *Contract:
		panic("subst not implemented for contracts")

	default:
		panic("unimplemented")
	}

	return typ
}

func (s *subster) var_(v *Var) *Var {
	if v != nil {
		if typ := s.typ(v.typ); typ != v.typ {
			copy := *v
			copy.typ = typ
			return &copy
		}
	}
	return v
}

func (s *subster) tuple(t *Tuple) *Tuple {
	if t != nil {
		if vars, copied := s.varList(t.vars); copied {
			return &Tuple{vars}
		}
	}
	return t
}

func (s *subster) varList(in []*Var) (out []*Var, copied bool) {
	out = in
	for i, v := range in {
		if w := s.var_(v); w != v {
			if !copied {
				// first variable that got substituted => allocate new out slice
				// and copy all variables
				new := make([]*Var, len(in))
				copy(new, out)
				out = new
				copied = true
			}
			out[i] = w
		}
	}
	return
}
