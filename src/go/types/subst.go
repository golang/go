// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements instantiation of generic types
// through substitution of type parameters by actual
// types.

package types

import (
	"bytes"
	"go/token"
	"strings"
)

func (check *Checker) instantiate(pos token.Pos, typ Type, targs []Type) (res Type) {
	if check.conf.Trace {
		check.trace(pos, "-- instantiating %s with %s", typ, typeListString(targs))
		check.indent++
		defer func() {
			check.indent--
			var under Type
			if res != nil {
				under = res.Underlying()
			}
			check.trace(pos, "=> %s %s", res, under)
		}()
	}

	// TODO(gri) What is better here: work with TypeParams, or work with TypeNames?
	var tparams []*TypeName
	switch typ := typ.(type) {
	case *Named:
		tparams = typ.obj.tparams
	case *Signature:
		tparams = typ.tparams
	default:
		check.dump(">>> trying to instantiate %s", typ)
		unreachable() // only defined types and (defined) functions can be generic

	}

	// check bounds
	for i, tname := range tparams {
		tpar := tname.typ.(*TypeParam)
		if tpar.bound == nil {
			continue // no bound
		}

		// determine type parameter bound
		var iface *Interface
		switch bound := tpar.bound.Underlying().(type) {
		case *Interface:
			iface = bound
		case *Contract:
			iface = bound.ifaceAt(i)
		default:
			unreachable()
		}

		// use interface type of type parameter, if any
		// targ must implement iface
		targ := targs[i]
		if m, _ := check.missingMethod(targ, iface, true); m != nil {
			check.softErrorf(pos, "%s does not satisfy %s (missing method %s)", targ, tpar.bound, m)
			break
		}

		// targ's underlying type must also be one of the interface types listed, if any
		if len(iface.types) > 0 {
			utyp := targ.Underlying()

			// TODO(gri) Cannot handle a type argument that is itself parameterized for now
			switch utyp.(type) {
			case *Interface, *Contract:
				panic("unimplemented")
			}

			// TODO(gri) factor this out as sep. function?
			ok := false
			for _, t := range iface.types {
				// if we find one matching type, we're ok
				if Identical(utyp, t) {
					ok = true
					break
				}
			}

			if !ok {
				check.softErrorf(pos, "%s does not satisfy %s (%s not found in %s)", targ, tpar.bound, targ, iface)
				break
			}
		}
	}

	return check.subst(pos, typ, tparams, targs)
}

// subst returns the type typ with its type parameters tparams replaced by
// the corresponding type arguments targs, recursively.
func (check *Checker) subst(pos token.Pos, typ Type, tpars []*TypeName, targs []Type) Type {
	//assert(len(tpars) == len(targs)) // bounds may have only some type parameters
	if len(tpars) == 0 {
		return typ
	}
	subst := subster{pos, check, make(map[Type]Type), tpars, targs}
	return subst.typ(typ)
}

type subster struct {
	pos   token.Pos
	check *Checker
	cache map[Type]Type
	tpars []*TypeName
	targs []Type
}

func (subst *subster) typ(typ Type) Type {
	switch t := typ.(type) {
	case *Basic:
		// nothing to do

	case *Array:
		elem := subst.typ(t.elem)
		if elem != t.elem {
			return &Array{t.len, elem}
		}

	case *Slice:
		elem := subst.typ(t.elem)
		if elem != t.elem {
			return &Slice{elem}
		}

	case *Struct:
		if fields, copied := subst.varList(t.fields); copied {
			return &Struct{fields, t.tags}
		}

	case *Pointer:
		base := subst.typ(t.base)
		if base != t.base {
			return &Pointer{base}
		}

	case *Tuple:
		return subst.tuple(t)

	case *Signature:
		// TODO(gri) rethink the recv situation with respect to methods on parameterized types
		//recv := s.var_(t.recv) // not strictly needed (receivers cannot be parameterized) (?)
		recv := t.recv
		params := subst.tuple(t.params)
		results := subst.tuple(t.results)
		if recv != t.recv || params != t.params || results != t.results {
			copy := *t
			copy.tparams = nil // TODO(gri) is this correct? (another indication that perhaps tparams belong to the function decl)
			copy.recv = recv
			copy.params = params
			copy.results = results
			return &copy
		}

	case *Interface:
		// for now ignore embeddeds
		// TODO(gri) decide what to do
		assert(len(t.embeddeds) == 0)
		methods, mcopied := subst.funcList(t.methods)
		types, tcopied := subst.typeList(t.types)
		if mcopied || tcopied {
			iface := &Interface{methods: methods, types: types}
			iface.Complete()
			return iface
		}

	case *Map:
		key := subst.typ(t.key)
		elem := subst.typ(t.elem)
		if key != t.key || elem != t.elem {
			return &Map{key, elem}
		}

	case *Chan:
		elem := subst.typ(t.elem)
		if elem != t.elem {
			return &Chan{t.dir, elem}
		}

	case *Named:
		subst.check.indent++
		defer func() {
			subst.check.indent--
		}()
		dump := func(format string, args ...interface{}) {
			if subst.check.conf.Trace {
				subst.check.trace(subst.pos, format, args...)
			}
		}

		if len(t.obj.tparams) == 0 {
			dump(">>> %s is not parameterized", t)
			return t // type is not parameterized
		}

		var new_targs []Type

		if len(t.targs) > 0 {

			// already instantiated
			dump(">>> %s already instantiated", t)
			assert(len(t.targs) == len(t.obj.tparams))
			// For each (existing) type argument targ, determine if it needs
			// to be substituted; i.e., if it is or contains a type parameter
			// that has a type argument for it.
			for i, targ := range t.targs {
				dump(">>> %d targ = %s", i, targ)
				new_targ := subst.typ(targ)
				if new_targ != targ {
					dump(">>> substituted %d targ %s => %s", i, targ, new_targ)
					if new_targs == nil {
						new_targs = make([]Type, len(t.obj.tparams))
						copy(new_targs, t.targs)
					}
					new_targs[i] = new_targ
				}
			}

			if new_targs == nil {
				dump(">>> nothing to substitute in %s", t)
				return t // nothing to substitute
			}

		} else {

			// not yet instantiated
			dump(">>> first instantiation of %s", t)
			new_targs = subst.targs

		}

		// TODO(gri) revisit name creation (function local types, etc.) and factor out
		//           (also, stripArgNames call is an awful hack)
		name := stripArgNames(TypeString(t, nil)) + "<" + typeListString(new_targs) + ">"
		dump(">>> new type name: %s", name)
		if tname, found := subst.check.typMap[name]; found {
			dump(">>> instantiated %s found", tname)
			return tname.typ
		}

		// create a new named type and populate caches to avoid endless recursion
		tname := NewTypeName(subst.pos, subst.check.pkg, name, nil)
		tname.tparams = t.obj.tparams // new type is still parameterized
		subst.check.typMap[name] = tname
		named := NewNamed(tname, nil, nil)
		named.targs = new_targs
		subst.cache[t] = named
		dump(">>> subst %s(%s) with %s (new: %s)", t.underlying, subst.tpars, subst.targs, new_targs)
		named.underlying = subst.typ(t.underlying)

		named.methods = t.methods // for now
		// TODO(gri) how much work do we really need to do here?
		// instantiate custom methods as necessary
		// for _, m := range t.methods {
		// 	// methods may not have a fully set up signature yet
		// 	dump(">>> instantiate %s", m)
		// 	subst.check.objDecl(m, nil)
		// 	sig := subst.check.subst(m.pos, m.typ, subst.tpars /*m.tparams*/, subst.targs).(*Signature)
		// 	m1 := NewFunc(m.pos, m.pkg, m.name, sig)
		// 	dump(">>> %s: method %s => %s", name, m, m1)
		// 	named.methods = append(named.methods, m1)
		// }
		// TODO(gri) update the method receivers?
		return named

	case *TypeParam:
		assert(len(subst.tpars) == len(subst.targs)) // TODO(gri) don't need this?
		// TODO(gri) Can we do this with direct indexing somehow? Or use a map instead?
		for i, tpar := range subst.tpars {
			if tpar.typ == t {
				return subst.targs[i]
			}
		}

	default:
		panic("unimplemented")
	}

	return typ
}

func stripArgNames(s string) string {
	if i := strings.IndexByte(s, '<'); i > 0 {
		return s[:i]
	}
	return s
}

func (subst *subster) var_(v *Var) *Var {
	if v != nil {
		if typ := subst.typ(v.typ); typ != v.typ {
			copy := *v
			copy.typ = typ
			return &copy
		}
	}
	return v
}

func (subst *subster) tuple(t *Tuple) *Tuple {
	if t != nil {
		if vars, copied := subst.varList(t.vars); copied {
			return &Tuple{vars}
		}
	}
	return t
}

func (subst *subster) varList(in []*Var) (out []*Var, copied bool) {
	out = in
	for i, v := range in {
		if w := subst.var_(v); w != v {
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

func (subst *subster) func_(f *Func) *Func {
	assert(len(f.tparams) == 0)
	if f != nil {
		if typ := subst.typ(f.typ); typ != f.typ {
			copy := *f
			copy.typ = typ
			return &copy
		}
	}
	return f
}

func (subst *subster) funcList(in []*Func) (out []*Func, copied bool) {
	out = in
	for i, f := range in {
		if g := subst.func_(f); g != f {
			if !copied {
				// first function that got substituted => allocate new out slice
				// and copy all functions
				new := make([]*Func, len(in))
				copy(new, out)
				out = new
				copied = true
			}
			out[i] = g
		}
	}
	return
}

func (subst *subster) typeList(in []Type) (out []Type, copied bool) {
	out = in
	for i, t := range in {
		if u := subst.typ(t); u != t {
			if !copied {
				// first function that got substituted => allocate new out slice
				// and copy all functions
				new := make([]Type, len(in))
				copy(new, out)
				out = new
				copied = true
			}
			out[i] = u
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
