// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure return values aren't scanned until they
// are initialized, when calling functions and methods
// via reflect.

package main

import (
	"io"
	"reflect"
	"runtime"
	"unsafe"
)

var badPtr uintptr

var sink []byte

func init() {
	// Allocate large enough to use largeAlloc.
	b := make([]byte, 1<<16-1)
	sink = b // force heap allocation
	//  Any space between the object and the end of page is invalid to point to.
	badPtr = uintptr(unsafe.Pointer(&b[len(b)-1])) + 1
}

func f(d func(error) error) error {
	// Initialize callee args section with a bad pointer.
	g(badPtr, badPtr, badPtr, badPtr)

	// Then call a function which returns a pointer.
	// That return slot starts out holding a bad pointer.
	return d(io.EOF)
}

//go:noinline
func g(x, y, z, w uintptr) {
}

type T struct {
}

func (t *T) Foo(e error) error {
	runtime.GC()
	return e
}

func main() {
	// Functions
	d := reflect.MakeFunc(reflect.TypeOf(func(e error) error { return e }),
		func(args []reflect.Value) []reflect.Value {
			runtime.GC()
			return args
		}).Interface().(func(error) error)
	f(d)

	// Methods
	x := reflect.ValueOf(&T{}).Method(0).Interface().(func(error) error)
	f(x)
}
