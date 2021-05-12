// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// lice

package main

import "fmt"

// Overriding the predeclare "any", so it can be used as a type constraint or a type
// argument
type any interface{}

type _Function[a, b any] interface {
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
	f _Function[a, b]
	g _Function[b, c]
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

type _List[a any] interface {
	Match(casenil _Function[_Nil[a], any], casecons _Function[_Cons[a], any]) any
}

type _Nil[a any] struct{
}

func (xs _Nil[a]) Match(casenil _Function[_Nil[a], any], casecons _Function[_Cons[a], any]) any {
	return casenil.Apply(xs)
}

type _Cons[a any] struct {
	Head a
	Tail _List[a]
}

func (xs _Cons[a]) Match(casenil _Function[_Nil[a], any], casecons _Function[_Cons[a], any]) any {
	return casecons.Apply(xs)
}

type mapNil[a, b any] struct{
}

func (m mapNil[a, b]) Apply(_ _Nil[a]) any {
	return _Nil[b]{}
}

type mapCons[a, b any] struct {
	f _Function[a, b]
}

func (m mapCons[a, b]) Apply(xs _Cons[a]) any {
	return _Cons[b]{m.f.Apply(xs.Head), _Map[a, b](m.f, xs.Tail)}
}

func _Map[a, b any](f _Function[a, b], xs _List[a]) _List[b] {
	return xs.Match(mapNil[a, b]{}, mapCons[a, b]{f}).(_List[b])
}

func main() {
	var xs _List[int] = _Cons[int]{3, _Cons[int]{6, _Nil[int]{}}}
	var ys _List[int] = _Map[int, int](incr{-5}, xs)
	var xz _List[bool] = _Map[int, bool](pos{}, ys)
	cs1 := xz.(_Cons[bool])
	cs2 := cs1.Tail.(_Cons[bool])
	_, ok := cs2.Tail.(_Nil[bool])
	if cs1.Head != false || cs2.Head != true || !ok {
		panic(fmt.Sprintf("got %v, %v, %v, expected false, true, true",
			cs1.Head, cs2.Head, ok))
	}
}
