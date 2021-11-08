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
func test(s string, p, q uintptr, rest ...uintptr) int {
	runtime.GC()
	runtime.GC()

	if *(*string)(unsafe.Pointer(p)) != "ok" {
		panic(s + ": p failed")
	}
	if *(*string)(unsafe.Pointer(q)) != "ok" {
		panic(s + ": q failed")
	}
	for _, r := range rest {
		if *(*string)(unsafe.Pointer(r)) != "ok" {
			panic(s + ": r[i] failed")
		}
	}

	done <- true
	return 0
}

//go:noinline
func f() int {
	return test("return", uintptr(setup()), uintptr(setup()), uintptr(setup()), uintptr(setup()))
}

type S struct{}

//go:noinline
//go:uintptrescapes
func (S) test(s string, p, q uintptr, rest ...uintptr) int {
	return test(s, p, q, rest...)
}

func main() {
	test("normal", uintptr(setup()), uintptr(setup()), uintptr(setup()), uintptr(setup()))
	<-done

	go test("go", uintptr(setup()), uintptr(setup()), uintptr(setup()), uintptr(setup()))
	<-done

	func() {
		defer test("defer", uintptr(setup()), uintptr(setup()), uintptr(setup()), uintptr(setup()))
	}()
	<-done

	func() {
		for {
			defer test("defer in for loop", uintptr(setup()), uintptr(setup()), uintptr(setup()), uintptr(setup()))
			break
		}
	}()

	<-done
	func() {
		s := &S{}
		defer s.test("method call", uintptr(setup()), uintptr(setup()), uintptr(setup()), uintptr(setup()))
	}()
	<-done

	func() {
		s := &S{}
		for {
			defer s.test("defer method loop", uintptr(setup()), uintptr(setup()), uintptr(setup()), uintptr(setup()))
			break
		}
	}()
	<-done

	f()
	<-done
}
