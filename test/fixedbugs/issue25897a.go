// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure the runtime can scan args of an unstarted goroutine
// which starts with a reflect-generated function.

package main

import (
	"reflect"
	"runtime"
)

const N = 100

func main() {
	runtime.GOMAXPROCS(1)
	// Run GC in a loop. This makes it more likely GC will catch
	// an unstarted goroutine then if we were to GC after kicking
	// everything off.
	go func() {
		for {
			runtime.GC()
		}
	}()
	c := make(chan bool, N)
	for i := 0; i < N; i++ {
		// Test both with an argument and without because this
		// affects whether the compiler needs to generate a
		// wrapper closure for the "go" statement.
		f := reflect.MakeFunc(reflect.TypeOf(((func(*int))(nil))),
			func(args []reflect.Value) []reflect.Value {
				c <- true
				return nil
			}).Interface().(func(*int))
		go f(nil)

		g := reflect.MakeFunc(reflect.TypeOf(((func())(nil))),
			func(args []reflect.Value) []reflect.Value {
				c <- true
				return nil
			}).Interface().(func())
		go g()
	}
	for i := 0; i < N*2; i++ {
		<-c
	}
}
