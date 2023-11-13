// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import "fmt"

// Names starting with a _ are intended to be exported eventually
// (go.dev/issue/63223).

// An _Alias represents an alias type.
type _Alias struct {
	obj     *TypeName // corresponding declared alias object
	fromRHS Type      // RHS of type alias declaration; may be an alias
	actual  Type      // actual (aliased) type; never an alias
}

// _NewAlias creates a new Alias type with the given type name and rhs.
// rhs must not be nil.
func _NewAlias(obj *TypeName, rhs Type) *_Alias {
	return (*Checker)(nil).newAlias(obj, rhs)
}

func (a *_Alias) Underlying() Type { return a.actual.Underlying() }
func (a *_Alias) String() string   { return TypeString(a, nil) }

// Type accessors

// _Unalias returns t if it is not an alias type;
// otherwise it follows t's alias chain until it
// reaches a non-alias type which is then returned.
// Consequently, the result is never an alias type.
func _Unalias(t Type) Type {
	if a0, _ := t.(*_Alias); a0 != nil {
		if a0.actual != nil {
			return a0.actual
		}
		for a := a0; ; {
			t = a.fromRHS
			a, _ = t.(*_Alias)
			if a == nil {
				break
			}
		}
		if t == nil {
			panic(fmt.Sprintf("non-terminated alias %s", a0.obj.name))
		}
		a0.actual = t
	}
	return t
}

// asNamed returns t as *Named if that is t's
// actual type. It returns nil otherwise.
func asNamed(t Type) *Named {
	n, _ := _Unalias(t).(*Named)
	return n
}

// newAlias creates a new Alias type with the given type name and rhs.
// rhs must not be nil.
func (check *Checker) newAlias(obj *TypeName, rhs Type) *_Alias {
	assert(rhs != nil)
	a := &_Alias{obj, rhs, nil}
	if obj.typ == nil {
		obj.typ = a
	}

	// Ensure that a.actual is set at the end of type checking.
	if check != nil {
		check.needsCleanup(a)
	}

	return a
}

func (a *_Alias) cleanup() {
	_Unalias(a)
}
