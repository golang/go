// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains regression tests for bugs found.

package p

import "io"
import "context"

func eql[T comparable](x, y T) bool {
	return x == y
}

func _[X comparable, Y interface{comparable; m()}]() {
	var x X
	var y Y
	eql(x, y /* ERROR does not match */ ) // interfaces of different types
	eql(x, x)
	eql(y, y)
	eql(y, nil /* ERROR cannot use nil as Y value in argument to eql */ )
	eql[io /* ERROR does not implement comparable */ .Reader](nil, nil)
}

// If we have a receiver of pointer to type parameter type (below: *T)
// we don't have any methods, like for interfaces.
type C[T any] interface {
    m()
}

// using type bound C
func _[T C[T]](x *T) {
	x.m /* ERROR x\.m undefined */ ()
}

// using an interface literal as bound
func _[T interface{ m() }](x *T) {
	x.m /* ERROR x\.m undefined */ ()
}

func f2[_ interface{ m1(); m2() }]() {}

type T struct{}
func (T) m1()
func (*T) m2()

func _() {
	f2[T /* ERROR m2 has pointer receiver */ ]()
	f2[*T]()
}

// When a type parameter is used as an argument to instantiate a parameterized
// type, the type argument's type set must be a subset of the instantiated type
// parameter's type set.
type T1[P interface{~uint}] struct{}

func _[P any]() {
    _ = T1[P /* ERROR P does not implement interface{~uint} */ ]{}
}

// This is the original (simplified) program causing the same issue.
type Unsigned interface {
	~uint
}

type T2[U Unsigned] struct {
    s U
}

func (u T2[U]) Add1() U {
    return u.s + 1
}

func NewT2[U any]() T2[U /* ERROR U does not implement Unsigned */ ] {
    return T2[U /* ERROR U does not implement Unsigned */ ]{}
}

func _() {
    u := NewT2[string]()
    _ = u.Add1()
}

// When we encounter an instantiated type such as Elem[T] we must
// not "expand" the instantiation when the type to be instantiated
// (Elem in this case) is not yet fully set up.
type Elem[T any] struct {
	next *Elem[T]
	list *List[T]
}

type List[T any] struct {
	root Elem[T]
}

func (l *List[T]) Init() {
	l.root.next = &l.root
}

// This is the original program causing the same issue.
type Element2[TElem any] struct {
	next, prev *Element2[TElem]
	list *List2[TElem]
	Value TElem
}

type List2[TElem any] struct {
	root Element2[TElem]
	len  int
}

func (l *List2[TElem]) Init() *List2[TElem] {
	l.root.next = &l.root
	l.root.prev = &l.root
	l.len = 0
	return l
}

// Self-recursive instantiations must work correctly.
type A[P any] struct { _ *A[P] }

type AB[P any] struct { _ *BA[P] }
type BA[P any] struct { _ *AB[P] }

// And a variation that also caused a problem with an
// unresolved underlying type.
type Element3[TElem any] struct {
	next, prev *Element3[TElem]
	list *List3[TElem]
	Value TElem
}

func (e *Element3[TElem]) Next() *Element3[TElem] {
	if p := e.next; e.list != nil && p != &e.list.root {
		return p
	}
	return nil
}

type List3[TElem any] struct {
	root Element3[TElem]
	len  int
}

// Infinite generic type declarations must lead to an error.
type inf1[T any] struct{ _ inf1 /* ERROR illegal cycle */ [T] }
type inf2[T any] struct{ inf2 /* ERROR illegal cycle */ [T] }

// The implementation of conversions T(x) between integers and floating-point
// numbers checks that both T and x have either integer or floating-point
// type. When the type of T or x is a type parameter, the respective simple
// predicate disjunction in the implementation was wrong because if a type set
// contains both an integer and a floating-point type, the type parameter is
// neither an integer or a floating-point number.
func convert[T1, T2 interface{~int | ~uint | ~float32}](v T1) T2 {
	return T2(v)
}

func _() {
	convert[int, uint](5)
}

// When testing binary operators, for +, the operand types must either be
// both numeric, or both strings. The implementation had the same problem
// with this check as the conversion issue above (issue #39623).

func issue39623[T interface{~int | ~string}](x, y T) T {
	return x + y
}

// Simplified, from https://go2goplay.golang.org/p/efS6x6s-9NI:
func Sum[T interface{~int | ~string}](s []T) (sum T) {
	for _, v := range s {
		sum += v
	}
	return
}

// Assignability of an unnamed pointer type to a type parameter that
// has a matching underlying type.
func _[T interface{}, PT interface{~*T}] (x T) PT {
    return &x
}

// Indexing of type parameters containing type parameters in their constraint terms:
func at[T interface{ ~[]E }, E interface{}](x T, i int) E {
        return x[i]
}

// Conversion of a local type to a type parameter.
func _[T interface{~int}](x T) {
	type myint int
	var _ int = int(x)
	var _ T = 42
	var _ T = T(myint(42))
}

// Indexing a type parameter with an array type bound checks length.
// (Example by mdempsky@.)
func _[T interface { ~[10]int }](x T) {
	_ = x[9] // ok
	_ = x[20 /* ERROR out of bounds */ ]
}

// Pointer indirection of a type parameter.
func _[T interface{ ~*int }](p T) int {
	return *p
}

// Channel sends and receives on type parameters.
func _[T interface{ ~chan int }](ch T) int {
	ch <- 0
	return <- ch
}

// Calling of a generic variable.
func _[T interface{ ~func() }](f T) {
	f()
	go f()
}

type F1 func()
type F2 func()
func _[T interface{ func()|F1|F2 }](f T) {
	f()
	go f()
}

// We must compare against the (possibly underlying) types of term list
// elements when checking if a constraint is satisfied by a type.
// The underlying type of each term must be computed after the
// interface has been instantiated as its constraint may contain
// a type parameter that was substituted with a defined type.
// Test case from an (originally) failing example.

type sliceOf[E any] interface{ ~[]E }

func append[T interface{}, S sliceOf[T], T2 interface{}](s S, t ...T2) S { panic(0) }

var f           func()
var cancelSlice []context.CancelFunc
var _ = append[context.CancelFunc, []context.CancelFunc, context.CancelFunc](cancelSlice, f)

// A generic function must be instantiated with a type, not a value.

func g[T any](T) T { panic(0) }

var _ = g[int]
var _ = g[nil /* ERROR is not a type */ ]
var _ = g(0)
