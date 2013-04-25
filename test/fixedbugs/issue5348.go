// run

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 5348: finalizers keep data live for a surprising amount of time

package main

import (
	"runtime"
)

type T struct {
	S *string
}

func newString(s string) *string {
	return &s
}

var c = make(chan int)

func foo() {
	t := &T{S: newString("foo")}
	runtime.SetFinalizer(t, func(p *T) { c <- 0 })
	runtime.SetFinalizer(t.S, func(p *string) { c <- 0 })
}

func main() {
	foo()
	runtime.GC()
	<-c
	runtime.GC()
	<-c
}
