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

type T struct {
}

func (t *T) Foo(c chan bool) {
	c <- true
}

func main() {
	t := &T{}
	runtime.GOMAXPROCS(1)
	c := make(chan bool, N)
	for i := 0; i < N; i++ {
		f := reflect.ValueOf(t).MethodByName("Foo").Interface().(func(chan bool))
		go f(c)
	}
	runtime.GC()
	for i := 0; i < N; i++ {
		<-c
	}
}
