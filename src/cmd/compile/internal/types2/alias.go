// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import "fmt"

// An Alias represents an alias type.
// Whether or not Alias types are created is controlled by the
// gotypesalias setting with the GODEBUG environment variable.
// For gotypesalias=1, alias declarations produce an Alias type.
// Otherwise, the alias information is only in the type name,
// which points directly to the actual (aliased) type.
type Alias struct {
	obj     *TypeName      // corresponding declared alias object
	tparams *TypeParamList // type parameters, or nil
	fromRHS Type           // RHS of type alias declaration; may be an alias
	actual  Type           // actual (aliased) type; never an alias
}

// NewAlias creates a new Alias type with the given type name and rhs.
// rhs must not be nil.
func NewAlias(obj *TypeName, rhs Type) *Alias {
	alias := (*Checker)(nil).newAlias(obj, rhs)
	// Ensure that alias.actual is set (#65455).
	unalias(alias)
	return alias
}

func (a *Alias) Obj() *TypeName   { return a.obj }
func (a *Alias) Underlying() Type { return unalias(a).Underlying() }
func (a *Alias) String() string   { return TypeString(a, nil) }

// TODO(adonovan): uncomment when proposal #66559 is accepted.
//
// // Rhs returns the type R on the right-hand side of an alias
// // declaration "type A = R", which may be another alias.
// func (a *Alias) Rhs() Type { return a.fromRHS }

// Unalias returns t if it is not an alias type;
// otherwise it follows t's alias chain until it
// reaches a non-alias type which is then returned.
// Consequently, the result is never an alias type.
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
	if t == nil {
		panic(fmt.Sprintf("non-terminated alias %s", a0.obj.name))
	}
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
// rhs must not be nil.
func (check *Checker) newAlias(obj *TypeName, rhs Type) *Alias {
	assert(rhs != nil)
	a := &Alias{obj, nil, rhs, nil}
	if obj.typ == nil {
		obj.typ = a
	}

	// Ensure that a.actual is set at the end of type checking.
	if check != nil {
		check.needsCleanup(a)
	}

	return a
}

func (a *Alias) cleanup() {
	Unalias(a)
}
