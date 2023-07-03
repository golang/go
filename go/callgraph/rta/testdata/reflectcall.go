//go:build ignore
// +build ignore

// Test of a reflective call to an address-taken function.
//
// Dynamically, this program executes both print statements.
// RTA should report the hello methods as reachable,
// even though there are no dynamic calls of type func(U)
// and the type T is not live.

package main

import "reflect"

type T int
type U int // to ensure the hello methods' signatures are unique

func (T) hello(U) { println("hello") }

type T2 int

func (T2) Hello(U, U) { println("T2.Hello") }

func main() {
	u := reflect.ValueOf(U(0))

	// reflective call to bound method closure T.hello
	reflect.ValueOf(T(0).hello).Call([]reflect.Value{u})

	// reflective call to exported method "Hello" of rtype T2.
	reflect.ValueOf(T2(0)).Method(0).Call([]reflect.Value{u, u})
}

// WANT:
//
//  edge (reflect.Value).Call --synthetic call--> (T).hello$bound
//  edge (T).hello$bound --static method call--> (T).hello
//  edge main --static function call--> reflect.ValueOf
//  edge main --static method call--> (reflect.Value).Call
//  edge (*T2).Hello --static method call--> (T2).Hello
//
//  reachable (T).hello
//  reachable (T).hello$bound
//  reachable (T2).Hello
//
// !rtype T
//  rtype T2
//  rtype U
