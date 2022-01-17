// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

type Temp[T any] struct {
}

var temp, temp1 any
var ch any

func (it Temp[T]) HasNext() bool {
	var ok bool
	temp1 = <-ch.(chan T)
	// test conversion of T to interface{} during an OAS2RECV
	temp, ok = <-ch.(chan T)
	return ok
}

type MyInt int

func (i MyInt) String() string {
	return "a"
}

type Stringer interface {
	String() string
}

type Temp2[T Stringer] struct {
}

var temp2 Stringer

func (it Temp2[T]) HasNext() string {
	var x map[int]T

	var ok bool
	// test conversion of T to Stringer during an OAS2MAPR
	temp2, ok = x[43]
	_ = ok
	return temp2.String()
}

func main() {
	ch1 := make(chan int, 2)
	ch1 <- 5
	ch1 <- 6
	ch = ch1
	iter := Temp[int]{}
	iter.HasNext()

	iter2 := Temp2[MyInt]{}
	if got, want := iter2.HasNext(), "a"; got != want {
		panic(fmt.Sprintf("got %v, want %v", got, want))
	}

}
