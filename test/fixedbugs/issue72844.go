// run

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:noinline
func nilPtrFunc() *[4]int {
	return nil
}

var nilPtrVar *[4]int

func testLen1() {
	_ = len(*nilPtrFunc())
}

func testLen2() {
	_ = len(nilPtrFunc())
}

func testLen3() {
	_ = len(*nilPtrVar)
}

func testLen4() {
	_ = len(nilPtrVar)
}

func testRange1() {
	for range *nilPtrFunc() {
	}
}
func testRange2() {
	for range nilPtrFunc() {
	}
}
func testRange3() {
	for range *nilPtrVar {
	}
}
func testRange4() {
	for range nilPtrVar {
	}
}

func main() {
	//shouldPanic(testLen1)
	shouldNotPanic(testLen2)
	shouldNotPanic(testLen3)
	shouldNotPanic(testLen4)
	//shouldPanic(testRange1)
	shouldNotPanic(testRange2)
	shouldNotPanic(testRange3)
	shouldNotPanic(testRange4)
}

func shouldPanic(f func()) {
	defer func() {
		if e := recover(); e == nil {
			panic("should have panicked")
		}
	}()
	f()
}
func shouldNotPanic(f func()) {
	f()
}
