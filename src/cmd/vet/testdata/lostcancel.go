// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testdata

import (
	"context"
	"log"
	"os"
	"testing"
)

// Check the three functions and assignment forms (var, :=, =) we look for.
// (Do these early: line numbers are fragile.)
func _() {
	var ctx, cancel = context.WithCancel() // ERROR "the cancel function is not used on all paths \(possible context leak\)"
} // ERROR "this return statement may be reached without using the cancel var defined on line 17"

func _() {
	ctx, cancel2 := context.WithDeadline() // ERROR "the cancel2 function is not used..."
} // ERROR "may be reached without using the cancel2 var defined on line 21"

func _() {
	var ctx context.Context
	var cancel3 func()
	ctx, cancel3 = context.WithTimeout() // ERROR "function is not used..."
} // ERROR "this return statement may be reached without using the cancel3 var defined on line 27"

func _() {
	ctx, _ := context.WithCancel()  // ERROR "the cancel function returned by context.WithCancel should be called, not discarded, to avoid a context leak"
	ctx, _ = context.WithTimeout()  // ERROR "the cancel function returned by context.WithTimeout should be called, not discarded, to avoid a context leak"
	ctx, _ = context.WithDeadline() // ERROR "the cancel function returned by context.WithDeadline should be called, not discarded, to avoid a context leak"
}

func _() {
	ctx, cancel := context.WithCancel()
	defer cancel() // ok
}

func _() {
	ctx, cancel := context.WithCancel() // ERROR "not used on all paths"
	if condition {
		cancel()
	}
	return // ERROR "this return statement may be reached without using the cancel var"
}

func _() {
	ctx, cancel := context.WithCancel()
	if condition {
		cancel()
	} else {
		// ok: infinite loop
		for {
			print(0)
		}
	}
}

func _() {
	ctx, cancel := context.WithCancel() // ERROR "not used on all paths"
	if condition {
		cancel()
	} else {
		for i := 0; i < 10; i++ {
			print(0)
		}
	}
} // ERROR "this return statement may be reached without using the cancel var"

func _() {
	ctx, cancel := context.WithCancel()
	// ok: used on all paths
	switch someInt {
	case 0:
		new(testing.T).FailNow()
	case 1:
		log.Fatal()
	case 2:
		cancel()
	case 3:
		print("hi")
		fallthrough
	default:
		os.Exit(1)
	}
}

func _() {
	ctx, cancel := context.WithCancel() // ERROR "not used on all paths"
	switch someInt {
	case 0:
		new(testing.T).FailNow()
	case 1:
		log.Fatal()
	case 2:
		cancel()
	case 3:
		print("hi") // falls through to implicit return
	default:
		os.Exit(1)
	}
} // ERROR "this return statement may be reached without using the cancel var"

func _(ch chan int) int {
	ctx, cancel := context.WithCancel() // ERROR "not used on all paths"
	select {
	case <-ch:
		new(testing.T).FailNow()
	case y <- ch:
		print("hi") // falls through to implicit return
	case ch <- 1:
		cancel()
	default:
		os.Exit(1)
	}
} // ERROR "this return statement may be reached without using the cancel var"

func _(ch chan int) int {
	ctx, cancel := context.WithCancel()
	// A blocking select must execute one of its cases.
	select {
	case <-ch:
		panic()
	}
}

func _() {
	go func() {
		ctx, cancel := context.WithCancel() // ERROR "not used on all paths"
		print(ctx)
	}() // ERROR "may be reached without using the cancel var"
}

var condition bool
var someInt int

// Regression test for Go issue 16143.
func _() {
	var x struct{ f func() }
	x.f()
}

// Regression test for Go issue 16230.
func _() (ctx context.Context, cancel func()) {
	ctx, cancel = context.WithCancel()
	return // a naked return counts as a load of the named result values
}

// Same as above, but for literal function.
var _ = func() (ctx context.Context, cancel func()) {
	ctx, cancel = context.WithCancel()
	return
}
