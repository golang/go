// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test makes sure unsafe-uintptr arguments are handled correctly.

package main

import (
	"runtime"
	"unsafe"
)

var done = make(chan bool, 1)

func setup() unsafe.Pointer {
	s := "ok"
	runtime.SetFinalizer(&s, func(p *string) { *p = "FAIL" })
	return unsafe.Pointer(&s)
}

//go:noinline
//go:uintptrescapes
func test(s string, p uintptr) int {
	runtime.GC()
	if *(*string)(unsafe.Pointer(p)) != "ok" {
		panic(s + " return unexpected result")
	}
	done <- true
	return 0
}

//go:noinline
func f() int {
	return test("return", uintptr(setup()))
}

func main() {
	test("normal", uintptr(setup()))
	<-done

	go test("go", uintptr(setup()))
	<-done

	func() {
		defer test("defer", uintptr(setup()))
	}()
	<-done

	f()
	<-done
}
