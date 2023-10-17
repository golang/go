// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Perform tracebackdefers with a deferred reflection method.

package main

import "reflect"

type T struct{}

func (T) M() {
}

func F(args []reflect.Value) (results []reflect.Value) {
	return nil
}

func main() {
	done := make(chan bool)
	go func() {
		// Test reflect.makeFuncStub.
		t := reflect.TypeOf((func())(nil))
		f := reflect.MakeFunc(t, F).Interface().(func())
		defer f()
		growstack(10000)
		done <- true
	}()
	<-done
	go func() {
		// Test reflect.methodValueCall.
		f := reflect.ValueOf(T{}).Method(0).Interface().(func())
		defer f()
		growstack(10000)
		done <- true
	}()
	<-done
}

func growstack(x int) {
	if x == 0 {
		return
	}
	growstack(x - 1)
}
