// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import (
	"context"
	"log"
	"os"
	"testing"
	"time"
)

var bg = context.Background()

// Check the three functions and assignment forms (var, :=, =) we look for.
// (Do these early: line numbers are fragile.)
func _() {
	var _, cancel = context.WithCancel(bg) // want `the cancel function is not used on all paths \(possible context leak\)`
	if false {
		_ = cancel
	}
} // want "this return statement may be reached without using the cancel var defined on line 20"

func _() {
	_, cancel2 := context.WithDeadline(bg, time.Time{}) // want "the cancel2 function is not used..."
	if false {
		_ = cancel2
	}
} // want "may be reached without using the cancel2 var defined on line 27"

func _() {
	var cancel3 func()
	_, cancel3 = context.WithTimeout(bg, 0) // want "function is not used..."
	if false {
		_ = cancel3
	}
} // want "this return statement may be reached without using the cancel3 var defined on line 35"

func _() {
	ctx, _ := context.WithCancel(bg)               // want "the cancel function returned by context.WithCancel should be called, not discarded, to avoid a context leak"
	ctx, _ = context.WithTimeout(bg, 0)            // want "the cancel function returned by context.WithTimeout should be called, not discarded, to avoid a context leak"
	ctx, _ = context.WithDeadline(bg, time.Time{}) // want "the cancel function returned by context.WithDeadline should be called, not discarded, to avoid a context leak"
	_ = ctx
}

func _() {
	_, cancel := context.WithCancel(bg)
	defer cancel() // ok
}

func _() {
	_, cancel := context.WithCancel(bg) // want "not used on all paths"
	if condition {
		cancel()
	}
	return // want "this return statement may be reached without using the cancel var"
}

func _() {
	_, cancel := context.WithCancel(bg)
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
	_, cancel := context.WithCancel(bg) // want "not used on all paths"
	if condition {
		cancel()
	} else {
		for i := 0; i < 10; i++ {
			print(0)
		}
	}
} // want "this return statement may be reached without using the cancel var"

func _() {
	_, cancel := context.WithCancel(bg)
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
	_, cancel := context.WithCancel(bg) // want "not used on all paths"
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
} // want "this return statement may be reached without using the cancel var"

func _(ch chan int) {
	_, cancel := context.WithCancel(bg) // want "not used on all paths"
	select {
	case <-ch:
		new(testing.T).FailNow()
	case ch <- 2:
		print("hi") // falls through to implicit return
	case ch <- 1:
		cancel()
	default:
		os.Exit(1)
	}
} // want "this return statement may be reached without using the cancel var"

func _(ch chan int) {
	_, cancel := context.WithCancel(bg)
	// A blocking select must execute one of its cases.
	select {
	case <-ch:
		panic(0)
	}
	if false {
		_ = cancel
	}
}

func _() {
	go func() {
		ctx, cancel := context.WithCancel(bg) // want "not used on all paths"
		if false {
			_ = cancel
		}
		print(ctx)
	}() // want "may be reached without using the cancel var"
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
	ctx, cancel = context.WithCancel(bg)
	return // a naked return counts as a load of the named result values
}

// Same as above, but for literal function.
var _ = func() (ctx context.Context, cancel func()) {
	ctx, cancel = context.WithCancel(bg)
	return
}
