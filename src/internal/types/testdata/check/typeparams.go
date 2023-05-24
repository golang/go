// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// import "io" // for type assertion tests

var _ any // ok to use any anywhere
func _[_ any, _ interface{any}](any) {
        var _ any
}

func identity[T any](x T) T { return x }

func _[_ any](x int) int { panic(0) }
func _[T any](T /* ERROR "redeclared" */ T)() {}
func _[T, T /* ERROR "redeclared" */ any]() {}

// Constraints (incl. any) may be parenthesized.
func _[_ (any)]() {}
func _[_ (interface{})]() {}

func reverse[T any](list []T) []T {
        rlist := make([]T, len(list))
        i := len(list)
        for _, x := range list {
                i--
                rlist[i] = x
        }
        return rlist
}

var _ = reverse /* ERROR "cannot use generic function reverse" */
var _ = reverse[int, float32 /* ERROR "got 2 type arguments" */ ] ([]int{1, 2, 3})
var _ = reverse[int]([ /* ERROR "cannot use" */ ]float32{1, 2, 3})
var f = reverse[chan int]
var _ = f(0 /* ERRORx `cannot use 0 .* as \[\]chan int` */ )

func swap[A, B any](a A, b B) (B, A) { return b, a }

var _ = swap /* ERROR "multiple-value" */ [int, float32](1, 2)
var f32, i = swap[int, float32](swap[float32, int](1, 2))
var _ float32 = f32
var _ int = i

func swapswap[A, B any](a A, b B) (A, B) {
        return swap[B, A](b, a)
}

type F[A, B any] func(A, B) (B, A)

func min[T interface{ ~int }](x, y T) T {
        if x < y {
                return x
        }
        return y
}

func _[T interface{~int | ~float32}](x, y T) bool { return x < y }
func _[T any](x, y T) bool { return x /* ERROR "type parameter T is not comparable" */ < y }
func _[T interface{~int | ~float32 | ~bool}](x, y T) bool { return x /* ERROR "type parameter T is not comparable" */ < y }

func _[T C1[T]](x, y T) bool { return x /* ERROR "type parameter T is not comparable" */ < y }
func _[T C2[T]](x, y T) bool { return x < y }

type C1[T any] interface{}
type C2[T any] interface{ ~int | ~float32 }

func new[T any]() *T {
        var x T
        return &x
}

var _ = new /* ERROR "cannot use generic function new" */
var _ *int = new[int]()

func _[T any](map[T /* ERROR "invalid map key type T (missing comparable constraint)" */]int) {} // w/o constraint we don't know if T is comparable

func f1[T1 any](struct{T1 /* ERRORx `cannot be a .* type parameter` */ }) int { panic(0) }
var _ = f1[int](struct{T1}{})
type T1 = int

func f2[t1 any](struct{t1 /* ERRORx `cannot be a .* type parameter` */ ; x float32}) int { panic(0) }
var _ = f2[t1](struct{t1; x float32}{})
type t1 = int


func f3[A, B, C any](A, struct{x B}, func(A, struct{x B}, *C)) int { panic(0) }

var _ = f3[int, rune, bool](1, struct{x rune}{}, nil)

// indexing

func _[T any] (x T, i int) { _ = x /* ERROR "cannot index" */ [i] }
func _[T interface{ ~int }] (x T, i int) { _ = x /* ERROR "cannot index" */ [i] }
func _[T interface{ ~string }] (x T, i int) { _ = x[i] }
func _[T interface{ ~[]int }] (x T, i int) { _ = x[i] }
func _[T interface{ ~[10]int | ~*[20]int | ~map[int]int }] (x T, i int) { _ = x /* ERROR "cannot index" */ [i] } // map and non-map types
func _[T interface{ ~string | ~[]byte }] (x T, i int) { _ = x[i] }
func _[T interface{ ~[]int | ~[1]rune }] (x T, i int) { _ = x /* ERROR "cannot index" */ [i] }
func _[T interface{ ~string | ~[]rune }] (x T, i int) { _ = x /* ERROR "cannot index" */ [i] }

// indexing with various combinations of map types in type sets (see issue #42616)
func _[T interface{ ~[]E | ~map[int]E }, E any](x T, i int) { _ = x /* ERROR "cannot index" */ [i] } // map and non-map types
func _[T interface{ ~[]E }, E any](x T, i int) { _ = &x[i] }
func _[T interface{ ~map[int]E }, E any](x T, i int) { _, _ = x[i] } // comma-ok permitted
func _[T interface{ ~map[int]E }, E any](x T, i int) { _ = &x /* ERROR "cannot take address" */ [i] }
func _[T interface{ ~map[int]E | ~map[uint]E }, E any](x T, i int) { _ = x /* ERROR "cannot index" */ [i] } // different map element types
func _[T interface{ ~[]E | ~map[string]E }, E any](x T, i int) { _ = x /* ERROR "cannot index" */ [i] } // map and non-map types

// indexing with various combinations of array and other types in type sets
func _[T interface{ [10]int }](x T, i int) { _ = x[i]; _ = x[9]; _ = x[10 /* ERROR "out of bounds" */ ] }
func _[T interface{ [10]byte | string }](x T, i int) { _ = x[i]; _ = x[9]; _ = x[10 /* ERROR "out of bounds" */ ] }
func _[T interface{ [10]int | *[20]int | []int }](x T, i int) { _ = x[i]; _ = x[9]; _ = x[10 /* ERROR "out of bounds" */ ] }

// indexing with strings and non-variable arrays (assignment not permitted)
func _[T string](x T) { _ = x[0]; x /* ERROR "cannot assign" */ [0] = 0 }
func _[T []byte | string](x T) { x /* ERROR "cannot assign" */ [0] = 0 }
func _[T [10]byte]() { f := func() (x T) { return }; f /* ERROR "cannot assign" */ ()[0] = 0 }
func _[T [10]byte]() { f := func() (x *T) { return }; f /* ERROR "cannot index" */ ()[0] = 0 }
func _[T [10]byte]() { f := func() (x *T) { return }; (*f())[0] = 0 }
func _[T *[10]byte]() { f := func() (x T) { return }; f()[0] = 0 }

// slicing

func _[T interface{ ~[10]E }, E any] (x T, i, j, k int) { var _ []E = x[i:j] }
func _[T interface{ ~[10]E }, E any] (x T, i, j, k int) { var _ []E = x[i:j:k] }
func _[T interface{ ~[]byte }] (x T, i, j, k int) { var _ T = x[i:j] }
func _[T interface{ ~[]byte }] (x T, i, j, k int) { var _ T = x[i:j:k] }
func _[T interface{ ~string }] (x T, i, j, k int) { var _ T = x[i:j] }
func _[T interface{ ~string }] (x T, i, j, k int) { var _ T = x[i:j:k /* ERROR "3-index slice of string" */ ] }

type myByte1 []byte
type myByte2 []byte
func _[T interface{ []byte | myByte1 | myByte2 }] (x T, i, j, k int) { var _ T = x[i:j:k] }
func _[T interface{ []byte | myByte1 | []int }] (x T, i, j, k int) { var _ T = x /* ERROR "no core type" */ [i:j:k] }

func _[T interface{ []byte | myByte1 | myByte2 | string }] (x T, i, j, k int) { var _ T = x[i:j] }
func _[T interface{ []byte | myByte1 | myByte2 | string }] (x T, i, j, k int) { var _ T = x[i:j:k /* ERROR "3-index slice of string" */ ] }
func _[T interface{ []byte | myByte1 | []int | string }] (x T, i, j, k int) { var _ T = x /* ERROR "no core type" */ [i:j] }

// len/cap built-ins

func _[T any](x T) { _ = len(x /* ERROR "invalid argument" */ ) }
func _[T interface{ ~int }](x T) { _ = len(x /* ERROR "invalid argument" */ ) }
func _[T interface{ ~string | ~[]byte | ~int }](x T) { _ = len(x /* ERROR "invalid argument" */ ) }
func _[T interface{ ~string }](x T) { _ = len(x) }
func _[T interface{ ~[10]int }](x T) { _ = len(x) }
func _[T interface{ ~[]byte }](x T) { _ = len(x) }
func _[T interface{ ~map[int]int }](x T) { _ = len(x) }
func _[T interface{ ~chan int }](x T) { _ = len(x) }
func _[T interface{ ~string | ~[]byte | ~chan int }](x T) { _ = len(x) }

func _[T any](x T) { _ = cap(x /* ERROR "invalid argument" */ ) }
func _[T interface{ ~int }](x T) { _ = cap(x /* ERROR "invalid argument" */ ) }
func _[T interface{ ~string | ~[]byte | ~int }](x T) { _ = cap(x /* ERROR "invalid argument" */ ) }
func _[T interface{ ~string }](x T) { _ = cap(x /* ERROR "invalid argument" */ ) }
func _[T interface{ ~[10]int }](x T) { _ = cap(x) }
func _[T interface{ ~[]byte }](x T) { _ = cap(x) }
func _[T interface{ ~map[int]int }](x T) { _ = cap(x /* ERROR "invalid argument" */ ) }
func _[T interface{ ~chan int }](x T) { _ = cap(x) }
func _[T interface{ ~[]byte | ~chan int }](x T) { _ = cap(x) }

// range iteration

func _[T interface{}](x T) {
        for range x /* ERROR "cannot range" */ {}
}

type myString string

func _[
        B1 interface{ string },
        B2 interface{ string | myString },

        C1 interface{ chan int },
        C2 interface{ chan int | <-chan int },
        C3 interface{ chan<- int },

        S1 interface{ []int },
        S2 interface{ []int | [10]int },

        A1 interface{ [10]int },
        A2 interface{ [10]int | []int },

        P1 interface{ *[10]int },
        P2 interface{ *[10]int | *[]int },

        M1 interface{ map[string]int },
        M2 interface{ map[string]int | map[string]string },
]() {
        var b0 string
        for range b0 {}
        for _ = range b0 {}
        for _, _ = range b0 {}

        var b1 B1
        for range b1 {}
        for _ = range b1 {}
        for _, _ = range b1 {}

        var b2 B2
        for range b2 {}

        var c0 chan int
        for range c0 {}
        for _ = range c0 {}
        for _, _ /* ERROR "permits only one iteration variable" */ = range c0 {}

        var c1 C1
        for range c1 {}
        for _ = range c1 {}
        for _, _ /* ERROR "permits only one iteration variable" */ = range c1 {}

        var c2 C2
        for range c2 {}

        var c3 C3
        for range c3 /* ERROR "receive from send-only channel" */ {}

        var s0 []int
        for range s0 {}
        for _ = range s0 {}
        for _, _ = range s0 {}

        var s1 S1
        for range s1 {}
        for _ = range s1 {}
        for _, _ = range s1 {}

        var s2 S2
        for range s2 /* ERRORx `cannot range over s2.*no core type` */ {}

        var a0 []int
        for range a0 {}
        for _ = range a0 {}
        for _, _ = range a0 {}

        var a1 A1
        for range a1 {}
        for _ = range a1 {}
        for _, _ = range a1 {}

        var a2 A2
        for range a2 /* ERRORx `cannot range over a2.*no core type` */ {}

        var p0 *[10]int
        for range p0 {}
        for _ = range p0 {}
        for _, _ = range p0 {}

        var p1 P1
        for range p1 {}
        for _ = range p1 {}
        for _, _ = range p1 {}

        var p2 P2
        for range p2 /* ERRORx `cannot range over p2.*no core type` */ {}

        var m0 map[string]int
        for range m0 {}
        for _ = range m0 {}
        for _, _ = range m0 {}

        var m1 M1
        for range m1 {}
        for _ = range m1 {}
        for _, _ = range m1 {}

        var m2 M2
        for range m2 /* ERRORx `cannot range over m2.*no core type` */ {}
}

// type inference checks

var _ = new /* ERROR "cannot infer T" */ ()

func f4[A, B, C any](A, B) C { panic(0) }

var _ = f4 /* ERROR "cannot infer C" */ (1, 2)
var _ = f4[int, float32, complex128](1, 2)

func f5[A, B, C any](A, []*B, struct{f []C}) int { panic(0) }

var _ = f5[int, float32, complex128](0, nil, struct{f []complex128}{})
var _ = f5 /* ERROR "cannot infer" */ (0, nil, struct{f []complex128}{})
var _ = f5(0, []*float32{new[float32]()}, struct{f []complex128}{})

func f6[A any](A, []A) int { panic(0) }

var _ = f6(0, nil)

func f6nil[A any](A) int { panic(0) }

var _ = f6nil /* ERROR "cannot infer" */ (nil)

// type inference with variadic functions

func f7[T any](...T) T { panic(0) }

var _ int = f7 /* ERROR "cannot infer T" */ ()
var _ int = f7(1)
var _ int = f7(1, 2)
var _ int = f7([]int{}...)
var _ int = f7 /* ERROR "cannot use" */ ([]float64{}...)
var _ float64 = f7([]float64{}...)
var _ = f7[float64](1, 2.3)
var _ = f7(float64(1), 2.3)
var _ = f7(1, 2.3)
var _ = f7(1.2, 3)

func f8[A, B any](A, B, ...B) int { panic(0) }

var _ = f8(1) /* ERROR "not enough arguments" */
var _ = f8(1, 2.3)
var _ = f8(1, 2.3, 3.4, 4.5)
var _ = f8(1, 2.3, 3.4, 4)
var _ = f8[int, float64](1, 2.3, 3.4, 4)

var _ = f8[int, float64](0, 0, nil...) // test case for #18268

// init functions cannot have type parameters

func init() {}
func init[_ /* ERROR "func init must have no type parameters" */ any]() {}
func init[P /* ERROR "func init must have no type parameters" */ any]() {}

type T struct {}

func (T) m1() {}
func (T) m2[ /* ERROR "method must have no type parameters" */ _ any]() {}
func (T) m3[ /* ERROR "method must have no type parameters" */ P any]() {}

// type inference across parameterized types

type S1[P any] struct { f P }

func f9[P any](x S1[P]) {}

func _() {
        f9[int](S1[int]{42})
	f9(S1[int]{42})
}

type S2[A, B, C any] struct{}

func f10[X, Y, Z any](a S2[X, int, Z], b S2[X, Y, bool]) {}

func _[P any]() {
        f10[int, float32, string](S2[int, int, string]{}, S2[int, float32, bool]{})
        f10(S2[int, int, string]{}, S2[int, float32, bool]{})
        f10(S2[P, int, P]{}, S2[P, float32, bool]{})
}

// corner case for type inference
// (was bug: after instantiating f11, the type-checker didn't mark f11 as non-generic)

func f11[T any]() {}

func _() {
	f11[int]()
}

// the previous example was extracted from

// For now, a lone type parameter is not permitted as RHS in a type declaration (issue #45639).
// func f12[T interface{m() T}]() {}
// 
// type A[T any] T
// 
// func (a A[T]) m() A[T]
// 
// func _[T any]() {
// 	f12[A[T]]()
// }

// method expressions

func (_ S1[P]) m()

func _() {
	m := S1[int].m
	m(struct { f int }{42})
}

func _[T any] (x T) {
        m := S1[T].m
        m(S1[T]{x})
}

type I1[A any] interface {
        m1(A)
}

var _ I1[int] = r1[int]{}

type r1[T any] struct{}

func (_ r1[T]) m1(T)

type I2[A, B any] interface {
        m1(A)
        m2(A) B
}

var _ I2[int, float32] = R2[int, float32]{}

type R2[P, Q any] struct{}

func (_ R2[X, Y]) m1(X)
func (_ R2[X, Y]) m2(X) Y

// type assertions and type switches over generic types
// NOTE: These are currently disabled because it's unclear what the correct
// approach is, and one can always work around by assigning the variable to
// an interface first.

// // ReadByte1 corresponds to the ReadByte example in the draft design.
// func ReadByte1[T io.Reader](r T) (byte, error) {
// 	if br, ok := r.(io.ByteReader); ok {
// 		return br.ReadByte()
// 	}
// 	var b [1]byte
// 	_, err := r.Read(b[:])
// 	return b[0], err
// }
//
// // ReadBytes2 is like ReadByte1 but uses a type switch instead.
// func ReadByte2[T io.Reader](r T) (byte, error) {
//         switch br := r.(type) {
//         case io.ByteReader:
//                 return br.ReadByte()
//         }
// 	var b [1]byte
// 	_, err := r.Read(b[:])
// 	return b[0], err
// }
//
// // type assertions and type switches over generic types are strict
// type I3 interface {
//         m(int)
// }
//
// type I4 interface {
//         m() int // different signature from I3.m
// }
//
// func _[T I3](x I3, p T) {
//         // type assertions and type switches over interfaces are not strict
//         _ = x.(I4)
//         switch x.(type) {
//         case I4:
//         }
//
//         // type assertions and type switches over generic types are strict
//         _ = p /* ERROR "cannot have dynamic type I4" */.(I4)
//         switch p.(type) {
//         case I4 /* ERROR "cannot have dynamic type I4" */ :
//         }
// }

// type assertions and type switches over generic types lead to errors for now

func _[T any](x T) {
	_ = x /* ERROR "cannot use type assertion" */ .(int)
	switch x /* ERROR "cannot use type switch" */ .(type) {
	}

	// work-around
	var t interface{} = x
	_ = t.(int)
	switch t.(type) {
	}
}

func _[T interface{~int}](x T) {
	_ = x /* ERROR "cannot use type assertion" */ .(int)
	switch x /* ERROR "cannot use type switch" */ .(type) {
	}

	// work-around
	var t interface{} = x
	_ = t.(int)
	switch t.(type) {
	}
}

// error messages related to type bounds mention those bounds
type C[P any] interface{}

func _[P C[P]] (x P) {
	x.m /* ERROR "x.m undefined" */ ()
}

type I interface {}

func _[P I] (x P) {
	x.m /* ERROR "type P has no field or method m" */ ()
}

func _[P interface{}] (x P) {
	x.m /* ERROR "type P has no field or method m" */ ()
}

func _[P any] (x P) {
	x.m /* ERROR "type P has no field or method m" */ ()
}
