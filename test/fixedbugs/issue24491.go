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
func test(s string, p uintptr) {
	runtime.GC()
	if *(*string)(unsafe.Pointer(p)) != "ok" {
		panic(s + " return unexpected result")
	}
	done <- true
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
}
