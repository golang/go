// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test method calls on type parameters

package main

import (
	"fmt"
	"reflect"
	"strconv"
)

// Simple constraint
type Stringer interface {
	String() string
}

func stringify[T Stringer](s []T) (ret []string) {
	for _, v := range s {
		ret = append(ret, v.String())
	}
	return ret
}

type myint int

func (i myint) String() string {
	return strconv.Itoa(int(i))
}

// Constraint with an embedded interface, but still only requires String()
type Stringer2 interface {
	CanBeStringer2() int
	SubStringer2
}

type SubStringer2 interface {
	CanBeSubStringer2() int
	String() string
}

func stringify2[T Stringer2](s []T) (ret []string) {
	for _, v := range s {
		ret = append(ret, v.String())
	}
	return ret
}

func (myint) CanBeStringer2() int {
	return 0
}

func (myint) CanBeSubStringer2() int {
	return 0
}

// Test use of method values that are not called
func stringify3[T Stringer](s []T) (ret []string) {
	for _, v := range s {
		f := v.String
		ret = append(ret, f())
	}
	return ret
}

func main() {
	x := []myint{myint(1), myint(2), myint(3)}

	got := stringify(x)
	want := []string{"1", "2", "3"}
	if !reflect.DeepEqual(got, want) {
		panic(fmt.Sprintf("got %s, want %s", got, want))
	}

	got = stringify2(x)
	if !reflect.DeepEqual(got, want) {
		panic(fmt.Sprintf("got %s, want %s", got, want))
	}

	got = stringify3(x)
	if !reflect.DeepEqual(got, want) {
		panic(fmt.Sprintf("got %s, want %s", got, want))
	}
}
