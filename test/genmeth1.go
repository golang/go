// run -goexperiment genericmethods

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that generic methods order type arguments correctly.

package main

import (
	"fmt"
	"strings"
)

type S[A, B any] struct {
	a A
	b B
}

func (s S[A, B]) m() string {
	return typeStr(s.a, s.b)
}

func (s S[A, B]) n[C, D any]() string {
	var c C
	var d D
	return typeStr(s.a, s.b, c, d)
}

func typeStr(args ...any) string {
	s := ""
	for i, arg := range args {
		if i > 0 {
			s += "->"
		}
		s += strings.TrimPrefix(fmt.Sprintf("%T", arg), "main.") // trim for brevity
	}
	return s
}

func main() {
	type T1 int8
	type T2 int16
	type T3 int32
	type T4 int64

	// method calls
	// static dictionary on type
	check(S[T1, T2]{}.m(), "T1->T2")
	check(S[T2, T1]{}.m(), "T2->T1")
	// static dictionary on method
	check(S[T1, T2]{}.n[T3, T4](), "T1->T2->T3->T4")
	check(S[T4, T1]{}.n[T2, T3](), "T4->T1->T2->T3")
	check(S[T3, T4]{}.n[T1, T2](), "T3->T4->T1->T2")
	check(S[T2, T3]{}.n[T4, T1](), "T2->T3->T4->T1")
	// dynamic dictionary on type
	check(mCal[T1, T2](), "T1->T2")
	check(mCal[T2, T1](), "T2->T1")
	// dynamic dictionary on method
	check(nCal[T1, T2, T3, T4](), "T1->T2->T3->T4")
	check(nCal[T4, T1, T2, T3](), "T4->T1->T2->T3")
	check(nCal[T3, T4, T1, T2](), "T3->T4->T1->T2")
	check(nCal[T2, T3, T4, T1](), "T2->T3->T4->T1")

	// method values
	// static dictionary on type
	mv1 := S[T1, T2]{}.m
	check(mv1(), "T1->T2")
	mv2 := S[T2, T1]{}.m
	check(mv2(), "T2->T1")
	// static dictionary on method
	mv3 := S[T1, T2]{}.n[T3, T4]
	check(mv3(), "T1->T2->T3->T4")
	mv4 := S[T4, T1]{}.n[T2, T3]
	check(mv4(), "T4->T1->T2->T3")
	mv5 := S[T3, T4]{}.n[T1, T2]
	check(mv5(), "T3->T4->T1->T2")
	mv6 := S[T2, T3]{}.n[T4, T1]
	check(mv6(), "T2->T3->T4->T1")
	// dynamic dictionary on type
	check(mVal[T1, T2]()(), "T1->T2")
	check(mVal[T2, T1]()(), "T2->T1")
	// dynamic dictionary on method
	check(nVal[T1, T2, T3, T4]()(), "T1->T2->T3->T4")
	check(nVal[T4, T1, T2, T3]()(), "T4->T1->T2->T3")
	check(nVal[T3, T4, T1, T2]()(), "T3->T4->T1->T2")
	check(nVal[T2, T3, T4, T1]()(), "T2->T3->T4->T1")

	// method expressions
	// static dictionary on type
	me1 := S[T1, T2].m
	check(me1(S[T1, T2]{}), "T1->T2")
	me2 := S[T2, T1].m
	check(me2(S[T2, T1]{}), "T2->T1")
	// static dictionary on method
	me3 := S[T1, T2].n[T3, T4]
	check(me3(S[T1, T2]{}), "T1->T2->T3->T4")
	me4 := S[T4, T1].n[T2, T3]
	check(me4(S[T4, T1]{}), "T4->T1->T2->T3")
	me5 := S[T3, T4].n[T1, T2]
	check(me5(S[T3, T4]{}), "T3->T4->T1->T2")
	me6 := S[T2, T3].n[T4, T1]
	check(me6(S[T2, T3]{}), "T2->T3->T4->T1")
	// dynamic dictionary on type
	check(mExp[T1, T2]()(S[T1, T2]{}), "T1->T2")
	check(mExp[T2, T1]()(S[T2, T1]{}), "T2->T1")
	// dynamic dictionary on method
	check(nExp[T1, T2, T3, T4]()(S[T1, T2]{}), "T1->T2->T3->T4")
	check(nExp[T4, T1, T2, T3]()(S[T4, T1]{}), "T4->T1->T2->T3")
	check(nExp[T3, T4, T1, T2]()(S[T3, T4]{}), "T3->T4->T1->T2")
	check(nExp[T2, T3, T4, T1]()(S[T2, T3]{}), "T2->T3->T4->T1")
}

func check(got, want string) {
	if got != want {
		panic(fmt.Sprintf("got %s, want %s", got, want))
	}
}

// piping type arguments via type parameters for dynamic dictionaries
func mCal[A, B any]() string {
	return S[A, B]{}.m()
}

func mVal[A, B any]() func() string {
	return S[A, B]{}.m
}

func mExp[A, B any]() func(S[A, B]) string {
	return S[A, B].m
}

func nCal[A, B, C, D any]() string {
	return S[A, B]{}.n[C, D]()
}

func nVal[A, B, C, D any]() func() string {
	return S[A, B]{}.n[C, D]
}

func nExp[A, B, C, D any]() func(S[A, B]) string {
	return S[A, B].n[C, D]
}
