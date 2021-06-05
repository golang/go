// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that we can convert type parameters to both empty
// and nonempty interfaces, and named and nonnamed versions
// thereof.

package main

import "fmt"

type E interface{}

func f[T any](x T) interface{} {
	var i interface{} = x
	return i
}
func g[T any](x T) E {
	var i E = x
	return i
}

type C interface {
	foo() int
}

type myInt int

func (x myInt) foo() int {
	return int(x+1)
}

func h[T C](x T) interface{foo() int} {
	var i interface{foo()int} = x
	return i
}
func i[T C](x T) C {
	var i C = x
	return i
}

func main() {
	if got, want := f[int](7), 7; got != want {
		panic(fmt.Sprintf("got %d want %d", got, want))
	}
	if got, want := g[int](7), 7; got != want {
		panic(fmt.Sprintf("got %d want %d", got, want))
	}
	if got, want := h[myInt](7).foo(), 8; got != want {
		panic(fmt.Sprintf("got %d want %d", got, want))
	}
	if got, want := i[myInt](7).foo(), 8; got != want {
		panic(fmt.Sprintf("got %d want %d", got, want))
	}
}
