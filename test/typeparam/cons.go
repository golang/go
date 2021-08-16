// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// lice

package main

import "fmt"

// Overriding the predeclare "any", so it can be used as a type constraint or a type
// argument
type any interface{}

type Function[a, b any] interface {
	Apply(x a) b
}

type incr struct{ n int }

func (this incr) Apply(x int) int {
	return x + this.n
}

type pos struct{}

func (this pos) Apply(x int) bool {
	return x > 0
}

type compose[a, b, c any] struct {
	f Function[a, b]
	g Function[b, c]
}

func (this compose[a, b, c]) Apply(x a) c {
	return this.g.Apply(this.f.Apply(x))
}

type _Eq[a any] interface {
	Equal(a) bool
}

type Int int

func (this Int) Equal(that int) bool {
	return int(this) == that
}

type List[a any] interface {
	Match(casenil Function[Nil[a], any], casecons Function[Cons[a], any]) any
}

type Nil[a any] struct {
}

func (xs Nil[a]) Match(casenil Function[Nil[a], any], casecons Function[Cons[a], any]) any {
	return casenil.Apply(xs)
}

type Cons[a any] struct {
	Head a
	Tail List[a]
}

func (xs Cons[a]) Match(casenil Function[Nil[a], any], casecons Function[Cons[a], any]) any {
	return casecons.Apply(xs)
}

type mapNil[a, b any] struct {
}

func (m mapNil[a, b]) Apply(_ Nil[a]) any {
	return Nil[b]{}
}

type mapCons[a, b any] struct {
	f Function[a, b]
}

func (m mapCons[a, b]) Apply(xs Cons[a]) any {
	return Cons[b]{m.f.Apply(xs.Head), Map[a, b](m.f, xs.Tail)}
}

func Map[a, b any](f Function[a, b], xs List[a]) List[b] {
	return xs.Match(mapNil[a, b]{}, mapCons[a, b]{f}).(List[b])
}

func main() {
	var xs List[int] = Cons[int]{3, Cons[int]{6, Nil[int]{}}}
	var ys List[int] = Map[int, int](incr{-5}, xs)
	var xz List[bool] = Map[int, bool](pos{}, ys)
	cs1 := xz.(Cons[bool])
	cs2 := cs1.Tail.(Cons[bool])
	_, ok := cs2.Tail.(Nil[bool])
	if cs1.Head != false || cs2.Head != true || !ok {
		panic(fmt.Sprintf("got %v, %v, %v, expected false, true, true",
			cs1.Head, cs2.Head, ok))
	}
}
