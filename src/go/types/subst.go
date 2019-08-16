// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements instantiation of generic types
// through substitution of type parameters by actual
// types.

package types

import (
	"bytes"
)

// inst returns the instantiated type of tname.
func (check *Checker) inst(tname *TypeName, targs []Type) (res Type) {
	if check.conf.Trace {
		check.trace(tname.pos, "-- instantiating %s with %s", tname, typeListString(targs))
		check.indent++
		defer func() {
			check.indent--
			check.trace(tname.pos, "=> %s", res)
		}()
	}

	return check.subst(tname.typ, tname.tparams, targs)
}

// subst returns the type typ with its type parameters tparams replaced by
// the corresponding type arguments targs, recursively.
//
// TODO(gri) tparams is only passed so we can verify that the type parameters
// occuring in typ are the ones from typ's parameter list. We should be able
// to prove that this is always the case and then we don't need this extra
// argument anymore.
func (check *Checker) subst(typ Type, tparams []*TypeName, targs []Type) Type {
	// check.dump("%s: tparams %d, targs %d", typ, len(tparams), len(targs))
	assert(len(tparams) == len(targs))
	if len(tparams) == 0 {
		return typ
	}
	s := subster{check, make(map[Type]Type), tparams, targs}
	return s.typ(typ)
}

type subster struct {
	check   *Checker
	cache   map[Type]Type
	tparams []*TypeName
	targs   []Type
}

func (s *subster) typ(typ Type) (res Type) {
	// avoid repeating the same substitution for a given type
	// TODO(gri) is this correct in the presence of cycles?
	if typ, found := s.cache[typ]; found {
		return typ
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
		// TODO(gri) rethink the recv situation with respect to methods on parameterized types
		//recv := s.var_(t.recv) // not strictly needed (receivers cannot be parameterized) (?)
		recv := t.recv
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
		// if not all type parameters are known, create a parameterized type
		if IsParameterizedList(s.targs) {
			return &Parameterized{t.obj, s.targs}
		}

		// TODO(gri) revisit name creation (function local types, etc.) and factor out
		name := TypeString(t, nil) + "<" + typeListString(s.targs) + ">"
		if tname, found := s.check.typMap[name]; found {
			return tname.typ
		}

		// create a new named type and populate caches to avoid endless recursion
		// TODO(gri) should use actual instantiation position
		tname := NewTypeName(t.obj.pos, s.check.pkg, name, nil)
		s.check.typMap[name] = tname
		named := NewNamed(tname, nil, nil)
		s.cache[t] = named
		named.underlying = s.typ(t.underlying).Underlying()

		// instantiate custom methods as necessary
		for _, m := range t.methods {
			// methods may not have a fully set up signature yet
			s.check.objDecl(m, nil)
			sig := s.check.subst(m.typ, m.tparams, s.targs).(*Signature)
			m1 := NewFunc(m.pos, m.pkg, m.name, sig)
			// s.check.dump("%s: method %s => %s", name, m, m1)
			named.methods = append(named.methods, m1)
		}
		// TODO(gri) update the method receivers?
		return named

	case *Parameterized:
		// first, instantiate any arguments if necessary
		// TODO(gri) should this be done in check.inst
		// and thus for any caller of check.inst)?
		targs := make([]Type, len(t.targs))
		for i, a := range t.targs {
			targs[i] = s.typ(a) // TODO(gri) fix this
		}
		// then instantiate t
		return s.check.inst(t.tname, targs)

	case *TypeParam:
		// verify that the type parameter t is from the correct
		// parameterized type
		assert(s.tparams[t.index] == t.obj)
		// TODO(gri) targ may be nil in error messages from check.infer.
		// Eliminate that possibility and then we don't need this check.
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

func typeListString(targs []Type) string {
	var buf bytes.Buffer
	for i, arg := range targs {
		if i > 0 {
			buf.WriteString(", ")
		}
		WriteType(&buf, arg, nil)
	}
	return buf.String()
}
