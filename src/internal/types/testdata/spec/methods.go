// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// Non-interface methods may declare type parameters.

type T struct{}

func (T) m[P any](x P) P { return x }

func _() {
	// A generic method must be instantiated before it is called.
	var x T
	var _ int = x.m[int](1) // explicit instantiation
	var _ int = x.m(2)      // instantiation via type inference
	var _ int = x /* ERROR "cannot use x.m(3.14) (value of type float64)" */ .m(3.14)

	// Receivers of generic method calls may be complex expressions:
	// Instantiation must work not just on simple operands.
	var a [10]T
	_ = a[0].m[int]  // explicit instantiation
	_ = a[1].m(2.72) // instantiation via type inference

	var m map[string][]struct{ T }
	_ = m["foo"][0].T.m[float32]
	_ = m["foo"][0].T.m(2.72)

	_ = m["foo"][0].m[float32] // method promotion with explicit instantiation
	_ = m["foo"][0].m(2.72)    // method promotion with instantiation via type inference

	// A generic method expression may be assigned to a function after instantiation.
	var _ func(T, int) int = T.m[int] // explicit instantiation
	var _ func(T, int) int = T.m      // instantiation via type inference

	// A generic method value may be assigned to a function after instantiation.
	var _ func(int) int = x.m[int] // explicit instantiation
	var _ func(int) int = x.m      // instantiation via type inference
}

// Generic methods may be added to generic types.
type G[F any] struct {
	f F
}

// The constraint for the method parameter may refer to the receiver type parameter.
func (g G[F]) m[H interface{ convert(F) H }]() (r H) {
	return r.convert(g.f)
}

// But the usual restrictions for type terms still apply.
func (G[F]) m2[P F /* ERROR "cannot use a type parameter as constraint" */ ]() {}
func (G[F]) m3[P *F]() {} // this is ok

// Generic methods don't satisfy interfaces.
type I[P any] interface {
	m(P) P
}

var _ I[int] = T /* ERROR "(wrong type for method m)\n\t\thave m[P any](P) P\n\t\twant m(int) int" */ {}

// A method declaring type parameters is generic even if it doesn't use the type parameters in its signature.
type U struct {}

func (U) m[_ any](x int) int { return x }

var _ I[int] = U /* ERROR "wrong type for method m)\n\t\thave m[_ any](int) int\n\t\twant m(int) int" */ {}

type J interface {
	m()
}

type V struct {}

func (V) m[_ any]() {}

var _ J = V /* ERROR "wrong type for method m)\n\t\thave m[_ any]()\n\t\twant m()" */ {}

// In particular, interface inference must not unify a generic method's
// own type parameter into an inference variable.
func need[X any](I[X]) {}

func _() {
	need(T /* ERROR "type T of T{} does not match I[X] (cannot infer X)" */ {})
}

// Test case from parser smoke test.

type List[E any] []E

func (l List[E]) Map[F any](m func(E) F) (r List[F]) {
	for _, x := range l {
		r = append(r, m(x))
	}
	return
}

func _() {
	l := List[string]{"foo", "foobar", "42"}
	r := l.Map(func(s string) int { return len(s)})
	_ = r
}

func _[E, F any](l List[E]) List[F] {
	var f func(List[E], func(E) F) List[F] = List[E].Map  // method expression & type inference
	return f(l, func(E) F { var f F; return f })
}

func _[E, F any](l List[E]) List[F] {
	var f func(func(E) F) List[F] = l.Map  // method value & type inference
	return f(func(E) F { var f F; return f })
}
