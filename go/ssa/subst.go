// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"go/types"

	"golang.org/x/tools/internal/typeparams"
)

// Type substituter for a fixed set of replacement types.
//
// A nil *subster is an valid, empty substitution map. It always acts as
// the identity function. This allows for treating parameterized and
// non-parameterized functions identically while compiling to ssa.
//
// Not concurrency-safe.
type subster struct {
	replacements map[*typeparams.TypeParam]types.Type // values should contain no type params
	cache        map[types.Type]types.Type            // cache of subst results
	ctxt         *typeparams.Context                  // cache for instantiation
	scope        *types.Scope                         // *types.Named declared within this scope can be substituted (optional)
	debug        bool                                 // perform extra debugging checks
	// TODO(taking): consider adding Pos
	// TODO(zpavlinovic): replacements can contain type params
	// when generating instances inside of a generic function body.
}

// Returns a subster that replaces tparams[i] with targs[i]. Uses ctxt as a cache.
// targs should not contain any types in tparams.
// scope is the (optional) lexical block of the generic function for which we are substituting.
func makeSubster(ctxt *typeparams.Context, scope *types.Scope, tparams *typeparams.TypeParamList, targs []types.Type, debug bool) *subster {
	assert(tparams.Len() == len(targs), "makeSubster argument count must match")

	subst := &subster{
		replacements: make(map[*typeparams.TypeParam]types.Type, tparams.Len()),
		cache:        make(map[types.Type]types.Type),
		ctxt:         ctxt,
		scope:        scope,
		debug:        debug,
	}
	for i := 0; i < tparams.Len(); i++ {
		subst.replacements[tparams.At(i)] = targs[i]
	}
	if subst.debug {
		subst.wellFormed()
	}
	return subst
}

// wellFormed asserts that subst was properly initialized.
func (subst *subster) wellFormed() {
	if subst == nil {
		return
	}
	// Check that all of the type params do not appear in the arguments.
	s := make(map[types.Type]bool, len(subst.replacements))
	for tparam := range subst.replacements {
		s[tparam] = true
	}
	for _, r := range subst.replacements {
		if reaches(r, s) {
			panic(subst)
		}
	}
}

// typ returns the type of t with the type parameter tparams[i] substituted
// for the type targs[i] where subst was created using tparams and targs.
func (subst *subster) typ(t types.Type) (res types.Type) {
	if subst == nil {
		return t // A nil subst is type preserving.
	}
	if r, ok := subst.cache[t]; ok {
		return r
	}
	defer func() {
		subst.cache[t] = res
	}()

	// fall through if result r will be identical to t, types.Identical(r, t).
	switch t := t.(type) {
	case *typeparams.TypeParam:
		r := subst.replacements[t]
		assert(r != nil, "type param without replacement encountered")
		return r

	case *types.Basic:
		return t

	case *types.Array:
		if r := subst.typ(t.Elem()); r != t.Elem() {
			return types.NewArray(r, t.Len())
		}
		return t

	case *types.Slice:
		if r := subst.typ(t.Elem()); r != t.Elem() {
			return types.NewSlice(r)
		}
		return t

	case *types.Pointer:
		if r := subst.typ(t.Elem()); r != t.Elem() {
			return types.NewPointer(r)
		}
		return t

	case *types.Tuple:
		return subst.tuple(t)

	case *types.Struct:
		return subst.struct_(t)

	case *types.Map:
		key := subst.typ(t.Key())
		elem := subst.typ(t.Elem())
		if key != t.Key() || elem != t.Elem() {
			return types.NewMap(key, elem)
		}
		return t

	case *types.Chan:
		if elem := subst.typ(t.Elem()); elem != t.Elem() {
			return types.NewChan(t.Dir(), elem)
		}
		return t

	case *types.Signature:
		return subst.signature(t)

	case *typeparams.Union:
		return subst.union(t)

	case *types.Interface:
		return subst.interface_(t)

	case *types.Named:
		return subst.named(t)

	default:
		panic("unreachable")
	}
}

// types returns the result of {subst.typ(ts[i])}.
func (subst *subster) types(ts []types.Type) []types.Type {
	res := make([]types.Type, len(ts))
	for i := range ts {
		res[i] = subst.typ(ts[i])
	}
	return res
}

func (subst *subster) tuple(t *types.Tuple) *types.Tuple {
	if t != nil {
		if vars := subst.varlist(t); vars != nil {
			return types.NewTuple(vars...)
		}
	}
	return t
}

type varlist interface {
	At(i int) *types.Var
	Len() int
}

// fieldlist is an adapter for structs for the varlist interface.
type fieldlist struct {
	str *types.Struct
}

func (fl fieldlist) At(i int) *types.Var { return fl.str.Field(i) }
func (fl fieldlist) Len() int            { return fl.str.NumFields() }

func (subst *subster) struct_(t *types.Struct) *types.Struct {
	if t != nil {
		if fields := subst.varlist(fieldlist{t}); fields != nil {
			tags := make([]string, t.NumFields())
			for i, n := 0, t.NumFields(); i < n; i++ {
				tags[i] = t.Tag(i)
			}
			return types.NewStruct(fields, tags)
		}
	}
	return t
}

// varlist reutrns subst(in[i]) or return nils if subst(v[i]) == v[i] for all i.
func (subst *subster) varlist(in varlist) []*types.Var {
	var out []*types.Var // nil => no updates
	for i, n := 0, in.Len(); i < n; i++ {
		v := in.At(i)
		w := subst.var_(v)
		if v != w && out == nil {
			out = make([]*types.Var, n)
			for j := 0; j < i; j++ {
				out[j] = in.At(j)
			}
		}
		if out != nil {
			out[i] = w
		}
	}
	return out
}

func (subst *subster) var_(v *types.Var) *types.Var {
	if v != nil {
		if typ := subst.typ(v.Type()); typ != v.Type() {
			if v.IsField() {
				return types.NewField(v.Pos(), v.Pkg(), v.Name(), typ, v.Embedded())
			}
			return types.NewVar(v.Pos(), v.Pkg(), v.Name(), typ)
		}
	}
	return v
}

func (subst *subster) union(u *typeparams.Union) *typeparams.Union {
	var out []*typeparams.Term // nil => no updates

	for i, n := 0, u.Len(); i < n; i++ {
		t := u.Term(i)
		r := subst.typ(t.Type())
		if r != t.Type() && out == nil {
			out = make([]*typeparams.Term, n)
			for j := 0; j < i; j++ {
				out[j] = u.Term(j)
			}
		}
		if out != nil {
			out[i] = typeparams.NewTerm(t.Tilde(), r)
		}
	}

	if out != nil {
		return typeparams.NewUnion(out)
	}
	return u
}

func (subst *subster) interface_(iface *types.Interface) *types.Interface {
	if iface == nil {
		return nil
	}

	// methods for the interface. Initially nil if there is no known change needed.
	// Signatures for the method where recv is nil. NewInterfaceType fills in the receivers.
	var methods []*types.Func
	initMethods := func(n int) { // copy first n explicit methods
		methods = make([]*types.Func, iface.NumExplicitMethods())
		for i := 0; i < n; i++ {
			f := iface.ExplicitMethod(i)
			norecv := changeRecv(f.Type().(*types.Signature), nil)
			methods[i] = types.NewFunc(f.Pos(), f.Pkg(), f.Name(), norecv)
		}
	}
	for i := 0; i < iface.NumExplicitMethods(); i++ {
		f := iface.ExplicitMethod(i)
		// On interfaces, we need to cycle break on anonymous interface types
		// being in a cycle with their signatures being in cycles with their receivers
		// that do not go through a Named.
		norecv := changeRecv(f.Type().(*types.Signature), nil)
		sig := subst.typ(norecv)
		if sig != norecv && methods == nil {
			initMethods(i)
		}
		if methods != nil {
			methods[i] = types.NewFunc(f.Pos(), f.Pkg(), f.Name(), sig.(*types.Signature))
		}
	}

	var embeds []types.Type
	initEmbeds := func(n int) { // copy first n embedded types
		embeds = make([]types.Type, iface.NumEmbeddeds())
		for i := 0; i < n; i++ {
			embeds[i] = iface.EmbeddedType(i)
		}
	}
	for i := 0; i < iface.NumEmbeddeds(); i++ {
		e := iface.EmbeddedType(i)
		r := subst.typ(e)
		if e != r && embeds == nil {
			initEmbeds(i)
		}
		if embeds != nil {
			embeds[i] = r
		}
	}

	if methods == nil && embeds == nil {
		return iface
	}
	if methods == nil {
		initMethods(iface.NumExplicitMethods())
	}
	if embeds == nil {
		initEmbeds(iface.NumEmbeddeds())
	}
	return types.NewInterfaceType(methods, embeds).Complete()
}

func (subst *subster) named(t *types.Named) types.Type {
	// A named type may be:
	// (1) ordinary named type (non-local scope, no type parameters, no type arguments),
	// (2) locally scoped type,
	// (3) generic (type parameters but no type arguments), or
	// (4) instantiated (type parameters and type arguments).
	tparams := typeparams.ForNamed(t)
	if tparams.Len() == 0 {
		if subst.scope != nil && !subst.scope.Contains(t.Obj().Pos()) {
			// Outside the current function scope?
			return t // case (1) ordinary
		}

		// case (2) locally scoped type.
		// Create a new named type to represent this instantiation.
		// We assume that local types of distinct instantiations of a
		// generic function are distinct, even if they don't refer to
		// type parameters, but the spec is unclear; see golang/go#58573.
		//
		// Subtle: We short circuit substitution and use a newly created type in
		// subst, i.e. cache[t]=n, to pre-emptively replace t with n in recursive
		// types during traversal. This both breaks infinite cycles and allows for
		// constructing types with the replacement applied in subst.typ(under).
		//
		// Example:
		// func foo[T any]() {
		//   type linkedlist struct {
		//     next *linkedlist
		//     val T
		//   }
		// }
		//
		// When the field `next *linkedlist` is visited during subst.typ(under),
		// we want the substituted type for the field `next` to be `*n`.
		n := types.NewNamed(t.Obj(), nil, nil)
		subst.cache[t] = n
		subst.cache[n] = n
		n.SetUnderlying(subst.typ(t.Underlying()))
		return n
	}
	targs := typeparams.NamedTypeArgs(t)

	// insts are arguments to instantiate using.
	insts := make([]types.Type, tparams.Len())

	// case (3) generic ==> targs.Len() == 0
	// Instantiating a generic with no type arguments should be unreachable.
	// Please report a bug if you encounter this.
	assert(targs.Len() != 0, "substition into a generic Named type is currently unsupported")

	// case (4) instantiated.
	// Substitute into the type arguments and instantiate the replacements/
	// Example:
	//    type N[A any] func() A
	//    func Foo[T](g N[T]) {}
	//  To instantiate Foo[string], one goes through {T->string}. To get the type of g
	//  one subsitutes T with string in {N with typeargs == {T} and typeparams == {A} }
	//  to get {N with TypeArgs == {string} and typeparams == {A} }.
	assert(targs.Len() == tparams.Len(), "typeargs.Len() must match typeparams.Len() if present")
	for i, n := 0, targs.Len(); i < n; i++ {
		inst := subst.typ(targs.At(i)) // TODO(generic): Check with rfindley for mutual recursion
		insts[i] = inst
	}
	r, err := typeparams.Instantiate(subst.ctxt, typeparams.NamedTypeOrigin(t), insts, false)
	assert(err == nil, "failed to Instantiate Named type")
	return r
}

func (subst *subster) signature(t *types.Signature) types.Type {
	tparams := typeparams.ForSignature(t)

	// We are choosing not to support tparams.Len() > 0 until a need has been observed in practice.
	//
	// There are some known usages for types.Types coming from types.{Eval,CheckExpr}.
	// To support tparams.Len() > 0, we just need to do the following [psuedocode]:
	//   targs := {subst.replacements[tparams[i]]]}; Instantiate(ctxt, t, targs, false)

	assert(tparams.Len() == 0, "Substituting types.Signatures with generic functions are currently unsupported.")

	// Either:
	// (1)non-generic function.
	//    no type params to substitute
	// (2)generic method and recv needs to be substituted.

	// Receivers can be either:
	// named
	// pointer to named
	// interface
	// nil
	// interface is the problematic case. We need to cycle break there!
	recv := subst.var_(t.Recv())
	params := subst.tuple(t.Params())
	results := subst.tuple(t.Results())
	if recv != t.Recv() || params != t.Params() || results != t.Results() {
		return typeparams.NewSignatureType(recv, nil, nil, params, results, t.Variadic())
	}
	return t
}

// reaches returns true if a type t reaches any type t' s.t. c[t'] == true.
// It updates c to cache results.
//
// reaches is currently only part of the wellFormed debug logic, and
// in practice c is initially only type parameters. It is not currently
// relied on in production.
func reaches(t types.Type, c map[types.Type]bool) (res bool) {
	if c, ok := c[t]; ok {
		return c
	}

	// c is populated with temporary false entries as types are visited.
	// This avoids repeat visits and break cycles.
	c[t] = false
	defer func() {
		c[t] = res
	}()

	switch t := t.(type) {
	case *typeparams.TypeParam, *types.Basic:
		return false
	case *types.Array:
		return reaches(t.Elem(), c)
	case *types.Slice:
		return reaches(t.Elem(), c)
	case *types.Pointer:
		return reaches(t.Elem(), c)
	case *types.Tuple:
		for i := 0; i < t.Len(); i++ {
			if reaches(t.At(i).Type(), c) {
				return true
			}
		}
	case *types.Struct:
		for i := 0; i < t.NumFields(); i++ {
			if reaches(t.Field(i).Type(), c) {
				return true
			}
		}
	case *types.Map:
		return reaches(t.Key(), c) || reaches(t.Elem(), c)
	case *types.Chan:
		return reaches(t.Elem(), c)
	case *types.Signature:
		if t.Recv() != nil && reaches(t.Recv().Type(), c) {
			return true
		}
		return reaches(t.Params(), c) || reaches(t.Results(), c)
	case *typeparams.Union:
		for i := 0; i < t.Len(); i++ {
			if reaches(t.Term(i).Type(), c) {
				return true
			}
		}
	case *types.Interface:
		for i := 0; i < t.NumEmbeddeds(); i++ {
			if reaches(t.Embedded(i), c) {
				return true
			}
		}
		for i := 0; i < t.NumExplicitMethods(); i++ {
			if reaches(t.ExplicitMethod(i).Type(), c) {
				return true
			}
		}
	case *types.Named:
		return reaches(t.Underlying(), c)
	default:
		panic("unreachable")
	}
	return false
}
