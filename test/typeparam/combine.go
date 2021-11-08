// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

type Gen[A any] func() (A, bool)

func Combine[T1, T2, T any](g1 Gen[T1], g2 Gen[T2], join func(T1, T2) T) Gen[T] {
	return func() (T, bool) {
		var t T
		t1, ok := g1()
		if !ok {
			return t, false
		}
		t2, ok := g2()
		if !ok {
			return t, false
		}
		return join(t1, t2), true
	}
}

type Pair[A, B any] struct {
	A A
	B B
}

func _NewPair[A, B any](a A, b B) Pair[A, B] {
	return Pair[A, B]{a, b}
}

func Combine2[A, B any](ga Gen[A], gb Gen[B]) Gen[Pair[A, B]] {
	return Combine(ga, gb, _NewPair[A, B])
}

func main() {
	var g1 Gen[int] = func() (int, bool) { return 3, true }
	var g2 Gen[string] = func() (string, bool) { return "x", false }
	var g3 Gen[string] = func() (string, bool) { return "y", true }

	gc := Combine(g1, g2, _NewPair[int, string])
	if got, ok := gc(); ok {
		panic(fmt.Sprintf("got %v, %v, wanted -/false", got, ok))
	}
	gc2 := Combine2(g1, g2)
	if got, ok := gc2(); ok {
		panic(fmt.Sprintf("got %v, %v, wanted -/false", got, ok))
	}

	gc3 := Combine(g1, g3, _NewPair[int, string])
	if got, ok := gc3(); !ok || got.A != 3 || got.B != "y" {
		panic(fmt.Sprintf("got %v, %v, wanted {3, y}, true", got, ok))
	}
	gc4 := Combine2(g1, g3)
	if got, ok := gc4(); !ok || got.A != 3 || got.B != "y" {
		panic(fmt.Sprintf("got %v, %v, wanted {3, y}, true", got, ok))
	}
}
