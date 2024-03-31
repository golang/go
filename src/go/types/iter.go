// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import "iter"

// This file defines go1.23 iterator methods for a variety of data
// types. They are not mirrored to cmd/compile/internal/types2, as
// there is no point doing so until the bootstrap compiler it at least
// go1.23; therefore go1.23-style range statements should not be used
// in code common to types and types2, though clients of go/types are
// free to use them.

// Methods returns a go1.23 iterator over all the methods of an
// interface, ordered by Id.
//
// Example: for m := range t.Methods() { ... }
func (t *Interface) Methods() iter.Seq[*Func] {
	return func(yield func(m *Func) bool) {
		for i := range t.NumMethods() {
			if !yield(t.Method(i)) {
				break
			}
		}
	}
}

// ExplicitMethods returns a go1.23 iterator over the explicit methods of
// an interface, ordered by Id.
//
// Example: for m := range t.ExplicitMethods() { ... }
func (t *Interface) ExplicitMethods() iter.Seq[*Func] {
	return func(yield func(m *Func) bool) {
		for i := range t.NumExplicitMethods() {
			if !yield(t.ExplicitMethod(i)) {
				break
			}
		}
	}
}

// EmbeddedTypes returns a go1.23 iterator over the types embedded within an interface.
//
// Example: for e := range t.EmbeddedTypes() { ... }
func (t *Interface) EmbeddedTypes() iter.Seq[Type] {
	return func(yield func(e Type) bool) {
		for i := range t.NumEmbeddeds() {
			if !yield(t.EmbeddedType(i)) {
				break
			}
		}
	}
}

// Methods returns a go1.23 iterator over the declared methods of a named type.
//
// Example: for m := range t.Methods() { ... }
func (t *Named) Methods() iter.Seq[*Func] {
	return func(yield func(m *Func) bool) {
		for i := range t.NumMethods() {
			if !yield(t.Method(i)) {
				break
			}
		}
	}
}

// Children returns a go1.23 iterator over the child scopes nested within scope s.
//
// Example: for child := range scope.Children() { ... }
func (s *Scope) Children() iter.Seq[*Scope] {
	return func(yield func(child *Scope) bool) {
		for i := range s.NumChildren() {
			if !yield(s.Child(i)) {
				break
			}
		}
	}
}

// Fields returns a go1.23 iterator over the fields of a struct type.
//
// Example: for field := range s.Fields() { ... }
func (s *Struct) Fields() iter.Seq[*Var] {
	return func(yield func(field *Var) bool) {
		for i := range s.NumFields() {
			if !yield(s.Field(i)) {
				break
			}
		}
	}
}

// Variables returns a go1.23 iterator over the variables of a tuple type.
//
// Example: for v := range tuple.Variables() { ... }
func (t *Tuple) Variables() iter.Seq[*Var] {
	return func(yield func(v *Var) bool) {
		for i := range t.Len() {
			if !yield(t.At(i)) {
				break
			}
		}
	}
}

// MethodSet returns a go1.23 iterator over the methods of a method set.
//
// Example: for method := range s.Methods() { ... }
func (s *MethodSet) Methods() iter.Seq[*Selection] {
	return func(yield func(method *Selection) bool) {
		for i := range s.Len() {
			if !yield(s.At(i)) {
				break
			}
		}
	}
}

// Terms returns a go1.23 iterator over the terms of a union.
//
// Example: for term := range union.Terms() { ... }
func (u *Union) Terms() iter.Seq[*Term] {
	return func(yield func(term *Term) bool) {
		for i := range u.Len() {
			if !yield(u.Term(i)) {
				break
			}
		}
	}
}

// TypeParams returns a go1.23 iterator over a list of type parameters.
//
// Example: for tparam := range l.TypeParams() { ... }
func (l *TypeParamList) TypeParams() iter.Seq[*TypeParam] {
	return func(yield func(tparam *TypeParam) bool) {
		for i := range l.Len() {
			if !yield(l.At(i)) {
				break
			}
		}
	}
}

// Types returns a go1.23 iterator over the elements of a list of types.
//
// Example: for t := range l.Types() { ... }
func (l *TypeList) Types() iter.Seq[Type] {
	return func(yield func(t Type) bool) {
		for i := range l.Len() {
			if !yield(l.At(i)) {
				break
			}
		}
	}
}
