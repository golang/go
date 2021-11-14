// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements type parameter substitution.

package types

import "go/token"

type substMap map[*TypeParam]Type

// makeSubstMap creates a new substitution map mapping tpars[i] to targs[i].
// If targs[i] is nil, tpars[i] is not substituted.
func makeSubstMap(tpars []*TypeParam, targs []Type) substMap {
	assert(len(tpars) == len(targs))
	proj := make(substMap, len(tpars))
	for i, tpar := range tpars {
		proj[tpar] = targs[i]
	}
	return proj
}

func (m substMap) empty() bool {
	return len(m) == 0
}

func (m substMap) lookup(tpar *TypeParam) Type {
	if t := m[tpar]; t != nil {
		return t
	}
	return tpar
}

// subst returns the type typ with its type parameters tpars replaced by the
// corresponding type arguments targs, recursively. subst is pure in the sense
// that it doesn't modify the incoming type. If a substitution took place, the
// result type is different from the incoming type.
//
// If the given context is non-nil, it is used in lieu of check.Config.Context
func (check *Checker) subst(pos token.Pos, typ Type, smap substMap, ctxt *Context) Type {
	if smap.empty() {
		return typ
	}

	// common cases
	switch t := typ.(type) {
	case *Basic:
		return typ // nothing to do
	case *TypeParam:
		return smap.lookup(t)
	}

	// general case
	subst := subster{
		pos:   pos,
		smap:  smap,
		check: check,
		ctxt:  check.bestContext(ctxt),
	}
	return subst.typ(typ)
}

type subster struct {
	pos   token.Pos
	smap  substMap
	check *Checker // nil if called via Instantiate
	ctxt  *Context
}

func (subst *subster) typ(typ Type) Type {
	switch t := typ.(type) {
	case nil:
		// Call typOrNil if it's possible that typ is nil.
		panic("nil typ")

	case *Basic:
		// nothing to do

	case *Array:
		elem := subst.typOrNil(t.elem)
		if elem != t.elem {
			return &Array{len: t.len, elem: elem}
		}

	case *Slice:
		elem := subst.typOrNil(t.elem)
		if elem != t.elem {
			return &Slice{elem: elem}
		}

	case *Struct:
		if fields, copied := subst.varList(t.fields); copied {
			s := &Struct{fields: fields, tags: t.tags}
			s.markComplete()
			return s
		}

	case *Pointer:
		base := subst.typ(t.base)
		if base != t.base {
			return &Pointer{base: base}
		}

	case *Tuple:
		return subst.tuple(t)

	case *Signature:
		// TODO(gri) rethink the recv situation with respect to methods on parameterized types
		// recv := subst.var_(t.recv) // TODO(gri) this causes a stack overflow - explain
		recv := t.recv
		params := subst.tuple(t.params)
		results := subst.tuple(t.results)
		if recv != t.recv || params != t.params || results != t.results {
			return &Signature{
				rparams: t.rparams,
				// TODO(rFindley) why can't we nil out tparams here, rather than in instantiate?
				tparams: t.tparams,
				// instantiated signatures have a nil scope
				recv:     recv,
				params:   params,
				results:  results,
				variadic: t.variadic,
			}
		}

	case *Union:
		terms, copied := subst.termlist(t.terms)
		if copied {
			// term list substitution may introduce duplicate terms (unlikely but possible).
			// This is ok; lazy type set computation will determine the actual type set
			// in normal form.
			return &Union{terms, nil}
		}

	case *Interface:
		methods, mcopied := subst.funcList(t.methods)
		embeddeds, ecopied := subst.typeList(t.embeddeds)
		if mcopied || ecopied {
			iface := &Interface{methods: methods, embeddeds: embeddeds, implicit: t.implicit, complete: t.complete}
			return iface
		}

	case *Map:
		key := subst.typ(t.key)
		elem := subst.typ(t.elem)
		if key != t.key || elem != t.elem {
			return &Map{key: key, elem: elem}
		}

	case *Chan:
		elem := subst.typ(t.elem)
		if elem != t.elem {
			return &Chan{dir: t.dir, elem: elem}
		}

	case *Named:
		// dump is for debugging
		dump := func(string, ...interface{}) {}
		if subst.check != nil && trace {
			subst.check.indent++
			defer func() {
				subst.check.indent--
			}()
			dump = func(format string, args ...interface{}) {
				subst.check.trace(subst.pos, format, args...)
			}
		}

		// subst is called by expandNamed, so in this function we need to be
		// careful not to call any methods that would cause t to be expanded: doing
		// so would result in deadlock.
		//
		// So we call t.orig.TypeParams() rather than t.TypeParams() here and
		// below.
		if t.orig.TypeParams().Len() == 0 {
			dump(">>> %s is not parameterized", t)
			return t // type is not parameterized
		}

		var newTArgs []Type
		if t.targs.Len() != t.orig.TypeParams().Len() {
			return Typ[Invalid] // error reported elsewhere
		}

		// already instantiated
		dump(">>> %s already instantiated", t)
		// For each (existing) type argument targ, determine if it needs
		// to be substituted; i.e., if it is or contains a type parameter
		// that has a type argument for it.
		for i, targ := range t.targs.list() {
			dump(">>> %d targ = %s", i, targ)
			new_targ := subst.typ(targ)
			if new_targ != targ {
				dump(">>> substituted %d targ %s => %s", i, targ, new_targ)
				if newTArgs == nil {
					newTArgs = make([]Type, t.orig.TypeParams().Len())
					copy(newTArgs, t.targs.list())
				}
				newTArgs[i] = new_targ
			}
		}

		if newTArgs == nil {
			dump(">>> nothing to substitute in %s", t)
			return t // nothing to substitute
		}

		// before creating a new named type, check if we have this one already
		h := subst.ctxt.instanceHash(t.orig, newTArgs)
		dump(">>> new type hash: %s", h)
		if named := subst.ctxt.lookup(h, t.orig, newTArgs); named != nil {
			dump(">>> found %s", named)
			return named
		}

		// Create a new instance and populate the context to avoid endless
		// recursion. The position used here is irrelevant because validation only
		// occurs on t (we don't call validType on named), but we use subst.pos to
		// help with debugging.
		t.orig.resolve(subst.ctxt)
		return subst.check.instance(subst.pos, t.orig, newTArgs, subst.ctxt)

		// Note that if we were to expose substitution more generally (not just in
		// the context of a declaration), we'd have to substitute in
		// named.underlying as well.
		//
		// But this is unnecessary for now.

	case *TypeParam:
		return subst.smap.lookup(t)

	default:
		panic("unimplemented")
	}

	return typ
}

// typOrNil is like typ but if the argument is nil it is replaced with Typ[Invalid].
// A nil type may appear in pathological cases such as type T[P any] []func(_ T([]_))
// where an array/slice element is accessed before it is set up.
func (subst *subster) typOrNil(typ Type) Type {
	if typ == nil {
		return Typ[Invalid]
	}
	return subst.typ(typ)
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
			return &Tuple{vars: vars}
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

func (subst *subster) termlist(in []*Term) (out []*Term, copied bool) {
	out = in
	for i, t := range in {
		if u := subst.typ(t.typ); u != t.typ {
			if !copied {
				// first function that got substituted => allocate new out slice
				// and copy all functions
				new := make([]*Term, len(in))
				copy(new, out)
				out = new
				copied = true
			}
			out[i] = NewTerm(t.tilde, u)
		}
	}
	return
}
