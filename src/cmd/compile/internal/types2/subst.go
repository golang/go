// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements type parameter substitution.

package types2

import (
	"cmd/compile/internal/syntax"
)

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

// makeRenameMap is like makeSubstMap, but creates a map used to rename type
// parameters in from with the type parameters in to.
func makeRenameMap(from, to []*TypeParam) substMap {
	assert(len(from) == len(to))
	proj := make(substMap, len(from))
	for i, tpar := range from {
		proj[tpar] = to[i]
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
// corresponding type arguments targs, recursively. subst doesn't modify the
// incoming type. If a substitution took place, the result type is different
// from the incoming type.
//
// If expanding is non-nil, it is the instance type currently being expanded.
// One of expanding or ctxt must be non-nil.
func (check *Checker) subst(pos syntax.Pos, typ Type, smap substMap, expanding *Named, ctxt *Context) Type {
	assert(expanding != nil || ctxt != nil)

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
		pos:       pos,
		smap:      smap,
		check:     check,
		expanding: expanding,
		ctxt:      ctxt,
	}
	return subst.typ(typ)
}

type subster struct {
	pos       syntax.Pos
	smap      substMap
	check     *Checker // nil if called via Instantiate
	expanding *Named   // if non-nil, the instance that is being expanded
	ctxt      *Context
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
		// Preserve the receiver: it is handled during *Interface and *Named type
		// substitution.
		//
		// Naively doing the substitution here can lead to an infinite recursion in
		// the case where the receiver is an interface. For example, consider the
		// following declaration:
		//
		//  type T[A any] struct { f interface{ m() } }
		//
		// In this case, the type of f is an interface that is itself the receiver
		// type of all of its methods. Because we have no type name to break
		// cycles, substituting in the recv results in an infinite loop of
		// recv->interface->recv->interface->...
		recv := t.recv

		params := subst.tuple(t.params)
		results := subst.tuple(t.results)
		if params != t.params || results != t.results {
			return &Signature{
				rparams: t.rparams,
				// TODO(gri) why can't we nil out tparams here, rather than in instantiate?
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
			return &Union{terms}
		}

	case *Interface:
		methods, mcopied := subst.funcList(t.methods)
		embeddeds, ecopied := subst.typeList(t.embeddeds)
		if mcopied || ecopied {
			iface := subst.check.newInterface()
			iface.embeddeds = embeddeds
			iface.embedPos = t.embedPos
			iface.implicit = t.implicit
			assert(t.complete) // otherwise we are copying incomplete data
			iface.complete = t.complete
			// If we've changed the interface type, we may need to replace its
			// receiver if the receiver type is the original interface. Receivers of
			// *Named type are replaced during named type expansion.
			//
			// Notably, it's possible to reach here and not create a new *Interface,
			// even though the receiver type may be parameterized. For example:
			//
			//  type T[P any] interface{ m() }
			//
			// In this case the interface will not be substituted here, because its
			// method signatures do not depend on the type parameter P, but we still
			// need to create new interface methods to hold the instantiated
			// receiver. This is handled by Named.expandUnderlying.
			iface.methods, _ = replaceRecvType(methods, t, iface)

			// If check != nil, check.newInterface will have saved the interface for later completion.
			if subst.check == nil { // golang/go#61561: all newly created interfaces must be completed
				iface.typeSet()
			}
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
		if subst.check != nil && subst.check.conf.Trace {
			subst.check.indent++
			defer func() {
				subst.check.indent--
			}()
			dump = func(format string, args ...interface{}) {
				subst.check.trace(subst.pos, format, args...)
			}
		}

		// subst is called during expansion, so in this function we need to be
		// careful not to call any methods that would cause t to be expanded: doing
		// so would result in deadlock.
		//
		// So we call t.Origin().TypeParams() rather than t.TypeParams().
		orig := t.Origin()
		n := orig.TypeParams().Len()
		if n == 0 {
			dump(">>> %s is not parameterized", t)
			return t // type is not parameterized
		}

		var newTArgs []Type
		if t.TypeArgs().Len() != n {
			return Typ[Invalid] // error reported elsewhere
		}

		// already instantiated
		dump(">>> %s already instantiated", t)
		// For each (existing) type argument targ, determine if it needs
		// to be substituted; i.e., if it is or contains a type parameter
		// that has a type argument for it.
		for i, targ := range t.TypeArgs().list() {
			dump(">>> %d targ = %s", i, targ)
			new_targ := subst.typ(targ)
			if new_targ != targ {
				dump(">>> substituted %d targ %s => %s", i, targ, new_targ)
				if newTArgs == nil {
					newTArgs = make([]Type, n)
					copy(newTArgs, t.TypeArgs().list())
				}
				newTArgs[i] = new_targ
			}
		}

		if newTArgs == nil {
			dump(">>> nothing to substitute in %s", t)
			return t // nothing to substitute
		}

		// Create a new instance and populate the context to avoid endless
		// recursion. The position used here is irrelevant because validation only
		// occurs on t (we don't call validType on named), but we use subst.pos to
		// help with debugging.
		return subst.check.instance(subst.pos, orig, newTArgs, subst.expanding, subst.ctxt)

	case *TypeParam:
		return subst.smap.lookup(t)

	default:
		unreachable()
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
			return substVar(v, typ)
		}
	}
	return v
}

func substVar(v *Var, typ Type) *Var {
	copy := *v
	copy.typ = typ
	copy.origin = v.Origin()
	return &copy
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
			return substFunc(f, typ)
		}
	}
	return f
}

func substFunc(f *Func, typ Type) *Func {
	copy := *f
	copy.typ = typ
	copy.origin = f.Origin()
	return &copy
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

// replaceRecvType updates any function receivers that have type old to have
// type new. It does not modify the input slice; if modifications are required,
// the input slice and any affected signatures will be copied before mutating.
//
// The resulting out slice contains the updated functions, and copied reports
// if anything was modified.
func replaceRecvType(in []*Func, old, new Type) (out []*Func, copied bool) {
	out = in
	for i, method := range in {
		sig := method.Type().(*Signature)
		if sig.recv != nil && sig.recv.Type() == old {
			if !copied {
				// Allocate a new methods slice before mutating for the first time.
				// This is defensive, as we may share methods across instantiations of
				// a given interface type if they do not get substituted.
				out = make([]*Func, len(in))
				copy(out, in)
				copied = true
			}
			newsig := *sig
			newsig.recv = substVar(sig.recv, new)
			out[i] = substFunc(method, &newsig)
		}
	}
	return
}
