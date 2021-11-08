// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test illustrates how a type bound method (String below) can be implemented
// either by a concrete type (myint below) or a instantiated generic type
// (StringInt[myint] below).

package main

import (
	"fmt"
	"reflect"
	"strconv"
)

type myint int

//go:noinline
func (m myint) String() string {
	return strconv.Itoa(int(m))
}

type Stringer interface {
	String() string
}

func stringify[T Stringer](s []T) (ret []string) {
	for _, v := range s {
		// Test normal bounds method call on type param
		x1 := v.String()

		// Test converting type param to its bound interface first
		v1 := Stringer(v)
		x2 := v1.String()

		// Test method expression with type param type
		f1 := T.String
		x3 := f1(v)

		// Test creating and calling closure equivalent to the method expression
		f2 := func(v1 T) string {
			return Stringer(v1).String()
		}
		x4 := f2(v)

		if x1 != x2 || x2 != x3 || x3 != x4 {
			panic(fmt.Sprintf("Mismatched values %v, %v, %v, %v\n", x1, x2, x3, x4))
		}

		ret = append(ret, v.String())
	}
	return ret
}

type Ints interface {
	~int32 | ~int
}

// For now, a lone type parameter is not permitted as RHS in a type declaration (issue #45639).
// type StringInt[T Ints] T
//
// //go:noinline
// func (m StringInt[T]) String() string {
// 	return strconv.Itoa(int(m))
// }

type StringStruct[T Ints] struct {
	f T
}

func (m StringStruct[T]) String() string {
	return strconv.Itoa(int(m.f))
}

func main() {
	x := []myint{myint(1), myint(2), myint(3)}

	// stringify on a normal type, whose bound method is associated with the base type.
	got := stringify(x)
	want := []string{"1", "2", "3"}
	if !reflect.DeepEqual(got, want) {
		panic(fmt.Sprintf("got %s, want %s", got, want))
	}

	// For now, a lone type parameter is not permitted as RHS in a type declaration (issue #45639).
	// x2 := []StringInt[myint]{StringInt[myint](5), StringInt[myint](7), StringInt[myint](6)}
	//
	// // stringify on an instantiated type, whose bound method is associated with
	// // the generic type StringInt[T], which maps directly to T.
	// got2 := stringify(x2)
	// want2 := []string{"5", "7", "6"}
	// if !reflect.DeepEqual(got2, want2) {
	// 	panic(fmt.Sprintf("got %s, want %s", got2, want2))
	// }

	// stringify on an instantiated type, whose bound method is associated with
	// the generic type StringStruct[T], which maps to a struct containing T.
	x3 := []StringStruct[myint]{StringStruct[myint]{f: 11}, StringStruct[myint]{f: 10}, StringStruct[myint]{f: 9}}

	got3 := stringify(x3)
	want3 := []string{"11", "10", "9"}
	if !reflect.DeepEqual(got3, want3) {
		panic(fmt.Sprintf("got %s, want %s", got3, want3))
	}
}
