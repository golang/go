// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure return values aren't scanned until they
// are initialized, when calling functions and methods
// via reflect.

package main

import (
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

func f(d func() *byte) *byte {
	// Initialize callee args section with a bad pointer.
	g(badPtr)

	// Then call a function which returns a pointer.
	// That return slot starts out holding a bad pointer.
	return d()
}

//go:noinline
func g(x uintptr) {
}

type T struct {
}

func (t *T) Foo() *byte {
	runtime.GC()
	return nil
}

func main() {
	// Functions
	d := reflect.MakeFunc(reflect.TypeOf(func() *byte { return nil }),
		func(args []reflect.Value) []reflect.Value {
			runtime.GC()
			return []reflect.Value{reflect.ValueOf((*byte)(nil))}
		}).Interface().(func() *byte)
	f(d)

	// Methods
	e := reflect.ValueOf(&T{}).Method(0).Interface().(func() *byte)
	f(e)
}
