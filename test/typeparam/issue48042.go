// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"reflect"
)

type G[T any] interface {
	g() func()(*T)
}
type Foo[T any] struct {

}
// OCALL
func (l *Foo[T]) f1() (*T) {
	return g[T]()()
}
// OCALLFUNC
func (l *Foo[T]) f2() (*T) {
	var f = g[T]
	return f()()
}
// OCALLMETH
func (l *Foo[T]) f3() (*T) {
	return l.g()()
}
// OCALLINTER
func (l *Foo[T]) f4() (*T) {
	var g G[T] = l
	return g.g()()
}
// ODYNAMICDOTTYPE
func (l *Foo[T]) f5() (*T) {
	var x interface{}
	x = g[T]
	return x.(func()func()(*T))()()
}
func (l *Foo[T]) g() func() (*T) {
	return func() (*T) {
		t := new(T)
		reflect.ValueOf(t).Elem().SetInt(100)
		return t
	}
}
func g[T any]() func() (*T) {
	return func() (*T) {
		t := new(T)
		reflect.ValueOf(t).Elem().SetInt(100)
		return t
	}
}

func main() {
	foo := Foo[int]{}
	// Make sure the function conversion is correct
	if n := *(foo.f1()) ; n != 100{
		panic(fmt.Sprintf("%v",n))
	}
	if n := *(foo.f2()) ; n != 100{
		panic(fmt.Sprintf("%v",n))
	}
	if n := *(foo.f3()) ; n != 100{
		panic(fmt.Sprintf("%v",n))
	}
	if n := *(foo.f4()) ; n != 100{
		panic(fmt.Sprintf("%v",n))
	}
	if n := *(foo.f5()) ; n != 100{
		panic(fmt.Sprintf("%v",n))
	}
}