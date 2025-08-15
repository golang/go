// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
)

// An Alias represents an alias type.
//
// Alias types are created by alias declarations such as:
//
//	type A = int
//
// The type on the right-hand side of the declaration can be accessed
// using [Alias.Rhs]. This type may itself be an alias.
// Call [Unalias] to obtain the first non-alias type in a chain of
// alias type declarations.
//
// Like a defined ([Named]) type, an alias type has a name.
// Use the [Alias.Obj] method to access its [TypeName] object.
//
// Historically, Alias types were not materialized so that, in the example
// above, A's type was represented by a Basic (int), not an Alias
// whose [Alias.Rhs] is int. But Go 1.24 allows you to declare an
// alias type with type parameters or arguments:
//
//	type Set[K comparable] = map[K]bool
//	s := make(Set[String])
//
// and this requires that Alias types be materialized. Use the
// [Alias.TypeParams] and [Alias.TypeArgs] methods to access them.
//
// To ease the transition, the Alias type was introduced in go1.22,
// but the type-checker would not construct values of this type unless
// the GODEBUG=gotypesalias=1 environment variable was provided.
// Starting in go1.23, this variable is enabled by default.
// This setting also causes the predeclared type "any" to be
// represented as an Alias, not a bare [Interface].
type Alias struct {
	obj     *TypeName      // corresponding declared alias object
	orig    *Alias         // original, uninstantiated alias
	tparams *TypeParamList // type parameters, or nil
	targs   *TypeList      // type arguments, or nil
	fromRHS Type           // RHS of type alias declaration; may be an alias
	actual  Type           // actual (aliased) type; never an alias
}

// NewAlias creates a new Alias type with the given type name and rhs.
// If rhs is nil, the alias is incomplete.
func NewAlias(obj *TypeName, rhs Type) *Alias {
	alias := (*Checker)(nil).newAlias(obj, rhs)
	// Ensure that alias.actual is set (#65455).
	alias.cleanup()
	return alias
}

// Obj returns the type name for the declaration defining the alias type a.
// For instantiated types, this is same as the type name of the origin type.
func (a *Alias) Obj() *TypeName { return a.orig.obj }

func (a *Alias) String() string { return TypeString(a, nil) }

// Underlying returns the [underlying type] of the alias type a, which is the
// underlying type of the aliased type. Underlying types are never Named,
// TypeParam, or Alias types.
//
// [underlying type]: https://go.dev/ref/spec#Underlying_types.
func (a *Alias) Underlying() Type { return unalias(a).Underlying() }

// Origin returns the generic Alias type of which a is an instance.
// If a is not an instance of a generic alias, Origin returns a.
func (a *Alias) Origin() *Alias { return a.orig }

// TypeParams returns the type parameters of the alias type a, or nil.
// A generic Alias and its instances have the same type parameters.
func (a *Alias) TypeParams() *TypeParamList { return a.tparams }

// SetTypeParams sets the type parameters of the alias type a.
// The alias a must not have type arguments.
func (a *Alias) SetTypeParams(tparams []*TypeParam) {
	assert(a.targs == nil)
	a.tparams = bindTParams(tparams)
}

// TypeArgs returns the type arguments used to instantiate the Alias type.
// If a is not an instance of a generic alias, the result is nil.
func (a *Alias) TypeArgs() *TypeList { return a.targs }

// Rhs returns the type R on the right-hand side of an alias
// declaration "type A = R", which may be another alias.
func (a *Alias) Rhs() Type { return a.fromRHS }

// Unalias returns t if it is not an alias type;
// otherwise it follows t's alias chain until it
// reaches a non-alias type which is then returned.
// Consequently, the result is never an alias type.
// Returns nil if the alias is incomplete.
func Unalias(t Type) Type {
	if a0, _ := t.(*Alias); a0 != nil {
		return unalias(a0)
	}
	return t
}

func unalias(a0 *Alias) Type {
	if a0.actual != nil {
		return a0.actual
	}
	var t Type
	for a := a0; a != nil; a, _ = t.(*Alias) {
		t = a.fromRHS
	}

	// It's fine to memorize nil types since it's the zero value for actual.
	// It accomplishes nothing.
	a0.actual = t
	return t
}

// asNamed returns t as *Named if that is t's
// actual type. It returns nil otherwise.
func asNamed(t Type) *Named {
	n, _ := Unalias(t).(*Named)
	return n
}

// newAlias creates a new Alias type with the given type name and rhs.
// If rhs is nil, the alias is incomplete.
func (check *Checker) newAlias(obj *TypeName, rhs Type) *Alias {
	a := new(Alias)
	a.obj = obj
	a.orig = a
	a.fromRHS = rhs
	if obj.typ == nil {
		obj.typ = a
	}

	// Ensure that a.actual is set at the end of type checking.
	if check != nil {
		check.needsCleanup(a)
	}

	return a
}

// newAliasInstance creates a new alias instance for the given origin and type
// arguments, recording pos as the position of its synthetic object (for error
// reporting).
func (check *Checker) newAliasInstance(pos syntax.Pos, orig *Alias, targs []Type, expanding *Named, ctxt *Context) *Alias {
	assert(len(targs) > 0)
	obj := NewTypeName(pos, orig.obj.pkg, orig.obj.name, nil)
	rhs := check.subst(pos, orig.fromRHS, makeSubstMap(orig.TypeParams().list(), targs), expanding, ctxt)
	res := check.newAlias(obj, rhs)
	res.orig = orig
	res.tparams = orig.tparams
	res.targs = newTypeList(targs)
	return res
}

func (a *Alias) cleanup() {
	// Ensure a.actual is set before types are published,
	// so unalias is a pure "getter", not a "setter".
	unalias(a)
}
