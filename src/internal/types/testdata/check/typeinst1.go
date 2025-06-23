// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type List[E any] []E
var _ List[List[List[int]]]
var _ List[List[List[int]]] = []List[List[int]]{}

type (
	T1[P1 any] struct {
		f1 T2[P1, float32]
	}

	T2[P2, P3 any] struct {
		f2 P2
		f3 P3
	}
)

func _() {
	var x1 T1[int]
	var x2 T2[int, float32]

	x1.f1.f2 = 0
	x1.f1 = x2
}

type T3[P any] T1[T2[P, P]]

func _() {
	var x1 T3[int]
	var x2 T2[int, int]
	x1.f1.f2 = x2
}

func f[P any] (x P) List[P] {
	return List[P]{x}
}

var (
	_ []int = f(0)
	_ []float32 = f[float32](10)
	_ List[complex128] = f(1i)
	_ []List[int] = f(List[int]{})
        _ List[List[int]] = []List[int]{}
        _ = []List[int]{}
)

// Parameterized types with methods

func (l List[E]) Head() (_ E, _ bool) {
	if len(l) > 0 {
		return l[0], true
	}
	return
}

// A test case for instantiating types with other types (extracted from map.go2)

type Pair[K any] struct {
	key K
}

type Receiver[T any] struct {
	values T
}

type Iterator[K any] struct {
	r Receiver[Pair[K]]
}

func Values [T any] (r Receiver[T]) T {
        return r.values
}

func (it Iterator[K]) Next() K {
        return Values[Pair[K]](it.r).key
}

// A more complex test case testing type bounds (extracted from linalg.go2 and reduced to essence)

type NumericAbs[T any] interface {
	Abs() T
}

func AbsDifference[T NumericAbs[T]](x T) { panic(0) }

// For now, a lone type parameter is not permitted as RHS in a type declaration (issue #45639).
// type OrderedAbs[T any] T
// 
// func (a OrderedAbs[T]) Abs() OrderedAbs[T]
// 
// func OrderedAbsDifference[T any](x T) {
// 	AbsDifference(OrderedAbs[T](x))
// }

// same code, reduced to essence

func g[P interface{ m() P }](x P) { panic(0) }

// For now, a lone type parameter is not permitted as RHS in a type declaration (issue #45639).
// type T4[P any] P
// 
// func (_ T4[P]) m() T4[P]
// 
// func _[Q any](x Q) {
// 	g(T4[Q](x))
// }

// Another test case that caused  problems in the past

type T5[_ interface { a() }, _ interface{}] struct{}

type A[P any] struct{ x P }

func (_ A[P]) a() {}

var _ T5[A[int], int]

// Invoking methods with parameterized receiver types uses
// type inference to determine the actual type arguments matching
// the receiver type parameters from the actual receiver argument.
// Go does implicit address-taking and dereferenciation depending
// on the actual receiver and the method's receiver type. To make
// type inference work, the type-checker matches "pointer-ness"
// of the actual receiver and the method's receiver type.
// The following code tests this mechanism.

type R1[A any] struct{}
func (_ R1[A]) vm()
func (_ *R1[A]) pm()

func _[T any](r R1[T], p *R1[T]) {
	r.vm()
	r.pm()
	p.vm()
	p.pm()
}

type R2[A, B any] struct{}
func (_ R2[A, B]) vm()
func (_ *R2[A, B]) pm()

func _[T any](r R2[T, int], p *R2[string, T]) {
	r.vm()
	r.pm()
	p.vm()
	p.pm()
}

// It is ok to have multiple embedded unions.
type _ interface {
	m0()
	~int | ~string | ~bool
	~float32 | ~float64
	m1()
	m2()
	~complex64 | ~complex128
	~rune
}

// Type sets may contain each type at most once.
type _ interface {
	~int|~ /* ERROR "overlapping terms ~int" */ int
	~int|int /* ERROR "overlapping terms int" */
	int|int /* ERROR "overlapping terms int" */
}

type _ interface {
	~struct{f int} | ~struct{g int} | ~ /* ERROR "overlapping terms" */ struct{f int}
}

// Interface term lists can contain any type, incl. *Named types.
// Verify that we use the underlying type(s) of the type(s) in the
// term list when determining if an operation is permitted.

type MyInt int
func add1[T interface{MyInt}](x T) T {
	return x + 1
}

type MyString string
func double[T interface{MyInt|MyString}](x T) T {
	return x + x
}

// Embedding of interfaces with term lists leads to interfaces
// with term lists that are the intersection of the embedded
// term lists.

type E0 interface {
	~int | ~bool | ~string
}

type E1 interface {
	~int | ~float64 | ~string
}

type E2 interface {
	~float64
}

type I0 interface {
	E0
}

func f0[T I0]() {}
var _ = f0[int]
var _ = f0[bool]
var _ = f0[string]
var _ = f0[float64 /* ERROR "does not satisfy I0" */ ]

type I01 interface {
	E0
	E1
}

func f01[T I01]() {}
var _ = f01[int]
var _ = f01[bool /* ERROR "does not satisfy I0" */ ]
var _ = f01[string]
var _ = f01[float64 /* ERROR "does not satisfy I0" */ ]

type I012 interface {
	E0
	E1
	E2
}

func f012[T I012]() {}
var _ = f012[int /* ERRORx `cannot satisfy I012.*empty type set` */ ]
var _ = f012[bool /* ERRORx `cannot satisfy I012.*empty type set` */ ]
var _ = f012[string /* ERRORx `cannot satisfy I012.*empty type set` */ ]
var _ = f012[float64 /* ERRORx `cannot satisfy I012.*empty type set` */ ]

type I12 interface {
	E1
	E2
}

func f12[T I12]() {}
var _ = f12[int /* ERROR "does not satisfy I12" */ ]
var _ = f12[bool /* ERROR "does not satisfy I12" */ ]
var _ = f12[string /* ERROR "does not satisfy I12" */ ]
var _ = f12[float64]

type I0_ interface {
	E0
	~int
}

func f0_[T I0_]() {}
var _ = f0_[int]
var _ = f0_[bool /* ERROR "does not satisfy I0_" */ ]
var _ = f0_[string /* ERROR "does not satisfy I0_" */ ]
var _ = f0_[float64 /* ERROR "does not satisfy I0_" */ ]

// Using a function instance as a type is an error.
var _ f0 // ERROR "not a type"
var _ f0 /* ERROR "not a type" */ [int]

// Empty type sets can only be satisfied by empty type sets.
type none interface {
	// force an empty type set
        int
        string
}

func ff[T none]() {}
func gg[T any]() {}
func hh[T ~int]() {}

func _[T none]() {
	_ = ff[int /* ERROR "cannot satisfy none (empty type set)" */ ]
	_ = ff[T]  // pathological but ok because T's type set is empty, too
	_ = gg[int]
	_ = gg[T]
	_ = hh[int]
	_ = hh[T]
}
