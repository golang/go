// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 30041: copying results of a reflect-generated
// call on stack should not have write barrier.

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

type ft func() *int

var fn ft

func rf([]reflect.Value) []reflect.Value {
	a := reflect.ValueOf((*int)(nil))
	return []reflect.Value{a}
}

const N = 1000

func main() {
	fn = reflect.MakeFunc(reflect.TypeOf(fn), rf).Interface().(ft)

	// Keep running GC so the write barrier is on.
	go func() {
		for i := 0; i < N; i++ {
			runtime.GC()
		}
	}()

	var x [10]uintptr
	for i := range x {
		x[i] = badPtr
	}
	for i := 0; i < N; i++ {
		runtime.Gosched()
		use(x) // prepare bad pointers on stack
		fn()
	}
}

//go:noinline
func use([10]uintptr) {}
