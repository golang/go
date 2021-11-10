// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package generics contains the new syntax supporting generic programming in
// Go.
package generics

// Variables with an instantiated type should be shown.
var X Type[int]

// Parameterized types should be shown.
type Type[P any] struct {
	Field P
}

// Constructors for parameterized types should be shown.
func Constructor[lowerCase any]() Type[lowerCase] {
	return Type[lowerCase]{}
}

// MethodA uses a different name for its receiver type parameter.
func (t Type[A]) MethodA(p A) {}

// MethodB has a blank receiver type parameter.
func (t Type[_]) MethodB() {}

// MethodC has a lower-case receiver type parameter.
func (t Type[c]) MethodC() {}

// Constraint is a constraint interface with two type parameters.
type Constraint[P, Q interface{ string | ~int | Type[int] }] interface {
	~int | ~byte | Type[string]
	M() P
}

// int16 shadows the predeclared type int16.
type int16 int

// NewEmbeddings demonstrates how we filter the new embedded elements.
type NewEmbeddings interface {
	string // should not be filtered
	int16
	struct{ f int }
	~struct{ f int }
	*struct{ f int }
	struct{ f int } | ~struct{ f int }
}

// Func has an instantiated constraint.
func Func[T Constraint[string, Type[int]]]() {}

// AnotherFunc has an implicit constraint interface.
//
// Neither type parameters nor regular parameters should be filtered.
func AnotherFunc[T ~struct{ f int }](_ struct{ f int }) {}

// AFuncType demonstrates filtering of parameters and type parameters. Here we
// don't filter type parameters (to be consistent with function declarations),
// but DO filter the RHS.
type AFuncType[T ~struct{ f int }] func(_ struct{ f int })

// See issue #49477: type parameters should not be interpreted as named types
// for the purpose of determining whether a function is a factory function.

// Slice is not a factory function.
func Slice[T any]() []T {
	return nil
}

// Single is not a factory function.
func Single[T any]() *T {
	return nil
}
