// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strconv"
)

// Various implementations of fromStrings().

type Setter[B any] interface {
	Set(string)
	*B
}

// Takes two type parameters where PT = *T
func fromStrings1[T any, PT Setter[T]](s []string) []T {
	result := make([]T, len(s))
	for i, v := range s {
		// The type of &result[i] is *T which is in the type list
		// of Setter, so we can convert it to PT.
		p := PT(&result[i])
		// PT has a Set method.
		p.Set(v)
	}
	return result
}

func fromStrings1a[T any, PT Setter[T]](s []string) []PT {
	result := make([]PT, len(s))
	for i, v := range s {
		// The type new(T) is *T which is in the type list
		// of Setter, so we can convert it to PT.
		result[i] = PT(new(T))
		p := result[i]
		// PT has a Set method.
		p.Set(v)
	}
	return result
}

// Takes one type parameter and a set function
func fromStrings2[T any](s []string, set func(*T, string)) []T {
	results := make([]T, len(s))
	for i, v := range s {
		set(&results[i], v)
	}
	return results
}

type Setter2 interface {
	Set(string)
}

// Takes only one type parameter, but causes a panic (see below)
func fromStrings3[T Setter2](s []string) []T {
	results := make([]T, len(s))
	for i, v := range s {
		// Panics if T is a pointer type because receiver is T(nil).
		results[i].Set(v)
	}
	return results
}

// Two concrete types with the appropriate Set method.

type SettableInt int

func (p *SettableInt) Set(s string) {
	i, err := strconv.Atoi(s)
	if err != nil {
		panic(err)
	}
	*p = SettableInt(i)
}

type SettableString struct {
	s string
}

func (x *SettableString) Set(s string) {
	x.s = s
}

func main() {
	s := fromStrings1[SettableInt, *SettableInt]([]string{"1"})
	if len(s) != 1 || s[0] != 1 {
		panic(fmt.Sprintf("got %v, want %v", s, []int{1}))
	}

	s2 := fromStrings1a[SettableInt, *SettableInt]([]string{"1"})
	if len(s2) != 1 || *s2[0] != 1 {
		x := 1
		panic(fmt.Sprintf("got %v, want %v", s2, []*int{&x}))
	}

	// Test out constraint type inference, which should determine that the second
	// type param is *SettableString.
	ps := fromStrings1[SettableString]([]string{"x", "y"})
	if len(ps) != 2 || ps[0] != (SettableString{"x"}) || ps[1] != (SettableString{"y"}) {
		panic(s)
	}

	s = fromStrings2([]string{"1"}, func(p *SettableInt, s string) { p.Set(s) })
	if len(s) != 1 || s[0] != 1 {
		panic(fmt.Sprintf("got %v, want %v", s, []int{1}))
	}

	defer func() {
		if recover() == nil {
			panic("did not panic as expected")
		}
	}()
	// This should type check but should panic at run time,
	// because it will make a slice of *SettableInt and then call
	// Set on a nil value.
	fromStrings3[*SettableInt]([]string{"1"})
}
