// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import "go/token"

// An instance represents an instantiated generic type syntactically
// (without expanding the instantiation). Type instances appear only
// during type-checking and are replaced by their fully instantiated
// (expanded) types before the end of type-checking.
type instance struct {
	check   *Checker    // for lazy instantiation
	pos     token.Pos   // position of type instantiation; for error reporting only
	base    *Named      // parameterized type to be instantiated
	targs   []Type      // type arguments
	poslist []token.Pos // position of each targ; for error reporting only
	value   Type        // base[targs...] after instantiation or Typ[Invalid]; nil if not yet set
}

// expand returns the instantiated (= expanded) type of t.
// The result is either an instantiated *Named type, or
// Typ[Invalid] if there was an error.
func (t *instance) expand() Type {
	v := t.value
	if v == nil {
		v = t.check.instantiate(t.pos, t.base, t.targs, t.poslist)
		if v == nil {
			v = Typ[Invalid]
		}
		t.value = v
	}
	// After instantiation we must have an invalid or a *Named type.
	if debug && v != Typ[Invalid] {
		_ = v.(*Named)
	}
	return v
}

// expand expands a type instance into its instantiated
// type and leaves all other types alone. expand does
// not recurse.
func expand(typ Type) Type {
	if t, _ := typ.(*instance); t != nil {
		return t.expand()
	}
	return typ
}

// expandf is set to expand.
// Call expandf when calling expand causes compile-time cycle error.
var expandf func(Type) Type

func init() { expandf = expand }

func (t *instance) Underlying() Type { return t }
func (t *instance) String() string   { return TypeString(t, nil) }
