//go:build ignore
// +build ignore

package main

import (
	"fmt"
	"os"
)

type S[T any] struct{ t T }

var theSint S[int]
var theSbool S[bool]

func (s *S[T]) String() string {
	print(s) // @pointsto command-line-arguments.theSbool | command-line-arguments.theSint
	return ""
}

func Type[T any]() {
	var x *T
	print(x) // @types *int | *bool
}

func Caller[T any]() {
	var s *S[T]
	_ = s.String()
}

var a int
var b bool

type t[T any] struct {
	a *map[string]chan *T
}

func fn[T any](a *T) {
	m := make(map[string]chan *T)
	m[""] = make(chan *T, 1)
	m[""] <- a
	x := []t[T]{t[T]{a: &m}}
	print(x) // @pointstoquery <-(*x[i].a)[key] command-line-arguments.a | command-line-arguments.b
}

func main() {
	// os.Args is considered intrinsically allocated,
	// but may also be set explicitly (e.g. on Windows), hence '...'.
	print(os.Args) // @pointsto <command-line args> | ...
	fmt.Println("Hello!", &theSint)
	fmt.Println("World!", &theSbool)

	Type[int]()      // call
	f := Type[bool]  // call through a variable
	_ = Type[string] // not called so will not appear in Type's print.
	f()

	Caller[int]()
	Caller[bool]()

	fn(&a)
	fn(&b)
}

// @calls (*fmt.pp).handleMethods -> (*command-line-arguments.S[int]).String[int]
// @calls (*fmt.pp).handleMethods -> (*command-line-arguments.S[bool]).String[bool]
// @calls command-line-arguments.Caller[int] -> (*command-line-arguments.S[int]).String[int]
// @calls command-line-arguments.Caller[bool] -> (*command-line-arguments.S[bool]).String[bool]
