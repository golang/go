// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests type assertion expressions and statements

package main

import (
	"fmt"
	"runtime"
)

type (
	S struct{}
	T struct{}

	I interface {
		F()
	}
)

var (
	s *S
	t *T
)

func (s *S) F() {}
func (t *T) F() {}

func e2t_ssa(e interface{}) *T {
	return e.(*T)
}

func i2t_ssa(i I) *T {
	return i.(*T)
}

func testAssertE2TOk() {
	if got := e2t_ssa(t); got != t {
		fmt.Printf("e2t_ssa(t)=%v want %v", got, t)
		failed = true
	}
}

func testAssertE2TPanic() {
	var got *T
	defer func() {
		if got != nil {
			fmt.Printf("e2t_ssa(s)=%v want nil", got)
			failed = true
		}
		e := recover()
		err, ok := e.(*runtime.TypeAssertionError)
		if !ok {
			fmt.Printf("e2t_ssa(s) panic type %T", e)
			failed = true
		}
		want := "interface conversion: interface {} is *main.S, not *main.T"
		if err.Error() != want {
			fmt.Printf("e2t_ssa(s) wrong error, want '%s', got '%s'\n", want, err.Error())
			failed = true
		}
	}()
	got = e2t_ssa(s)
	fmt.Printf("e2t_ssa(s) should panic")
	failed = true
}

func testAssertI2TOk() {
	if got := i2t_ssa(t); got != t {
		fmt.Printf("i2t_ssa(t)=%v want %v", got, t)
		failed = true
	}
}

func testAssertI2TPanic() {
	var got *T
	defer func() {
		if got != nil {
			fmt.Printf("i2t_ssa(s)=%v want nil", got)
			failed = true
		}
		e := recover()
		err, ok := e.(*runtime.TypeAssertionError)
		if !ok {
			fmt.Printf("i2t_ssa(s) panic type %T", e)
			failed = true
		}
		want := "interface conversion: main.I is *main.S, not *main.T"
		if err.Error() != want {
			fmt.Printf("i2t_ssa(s) wrong error, want '%s', got '%s'\n", want, err.Error())
			failed = true
		}
	}()
	got = i2t_ssa(s)
	fmt.Printf("i2t_ssa(s) should panic")
	failed = true
}

func e2t2_ssa(e interface{}) (*T, bool) {
	t, ok := e.(*T)
	return t, ok
}

func i2t2_ssa(i I) (*T, bool) {
	t, ok := i.(*T)
	return t, ok
}

func testAssertE2T2() {
	if got, ok := e2t2_ssa(t); !ok || got != t {
		fmt.Printf("e2t2_ssa(t)=(%v, %v) want (%v, %v)", got, ok, t, true)
		failed = true
	}
	if got, ok := e2t2_ssa(s); ok || got != nil {
		fmt.Printf("e2t2_ssa(s)=(%v, %v) want (%v, %v)", got, ok, nil, false)
		failed = true
	}
}

func testAssertI2T2() {
	if got, ok := i2t2_ssa(t); !ok || got != t {
		fmt.Printf("i2t2_ssa(t)=(%v, %v) want (%v, %v)", got, ok, t, true)
		failed = true
	}
	if got, ok := i2t2_ssa(s); ok || got != nil {
		fmt.Printf("i2t2_ssa(s)=(%v, %v) want (%v, %v)", got, ok, nil, false)
		failed = true
	}
}

var failed = false

func main() {
	testAssertE2TOk()
	testAssertE2TPanic()
	testAssertI2TOk()
	testAssertI2TPanic()
	testAssertE2T2()
	testAssertI2T2()
	if failed {
		panic("failed")
	}
}
