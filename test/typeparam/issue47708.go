// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type FooType[T any] interface {
	Foo(BarType[T]) string
}
type BarType[T any] interface {
	Bar(FooType[T]) string
}

// For now, a lone type parameter is not permitted as RHS in a type declaration (issue #45639).
// type Baz[T any] T
// func (l Baz[T]) Foo(v BarType[T]) string {
// 	return v.Bar(l)
// }
// type Bob[T any] T
// func (l Bob[T]) Bar(v FooType[T]) string {
// 	if v,ok := v.(Baz[T]);ok{
// 		return fmt.Sprintf("%v%v",v,l)
// 	}
// 	return ""
// }

func main() {
	// For now, a lone type parameter is not permitted as RHS in a type declaration (issue #45639).
	// var baz Baz[int] = 123
	// var bob Bob[int] = 456
	//
	// if got, want := baz.Foo(bob), "123456"; got != want {
	// 	panic(fmt.Sprintf("got %s want %s", got, want))
	// }
}
