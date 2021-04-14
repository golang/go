// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests type assertion expressions and statements

package main

import (
	"runtime"
	"testing"
)

type (
	S struct{}
	U struct{}

	I interface {
		F()
	}
)

var (
	s *S
	u *U
)

func (s *S) F() {}
func (u *U) F() {}

func e2t_ssa(e interface{}) *U {
	return e.(*U)
}

func i2t_ssa(i I) *U {
	return i.(*U)
}

func testAssertE2TOk(t *testing.T) {
	if got := e2t_ssa(u); got != u {
		t.Errorf("e2t_ssa(u)=%v want %v", got, u)
	}
}

func testAssertE2TPanic(t *testing.T) {
	var got *U
	defer func() {
		if got != nil {
			t.Errorf("e2t_ssa(s)=%v want nil", got)
		}
		e := recover()
		err, ok := e.(*runtime.TypeAssertionError)
		if !ok {
			t.Errorf("e2t_ssa(s) panic type %T", e)
		}
		want := "interface conversion: interface {} is *main.S, not *main.U"
		if err.Error() != want {
			t.Errorf("e2t_ssa(s) wrong error, want '%s', got '%s'", want, err.Error())
		}
	}()
	got = e2t_ssa(s)
	t.Errorf("e2t_ssa(s) should panic")

}

func testAssertI2TOk(t *testing.T) {
	if got := i2t_ssa(u); got != u {
		t.Errorf("i2t_ssa(u)=%v want %v", got, u)
	}
}

func testAssertI2TPanic(t *testing.T) {
	var got *U
	defer func() {
		if got != nil {
			t.Errorf("i2t_ssa(s)=%v want nil", got)
		}
		e := recover()
		err, ok := e.(*runtime.TypeAssertionError)
		if !ok {
			t.Errorf("i2t_ssa(s) panic type %T", e)
		}
		want := "interface conversion: main.I is *main.S, not *main.U"
		if err.Error() != want {
			t.Errorf("i2t_ssa(s) wrong error, want '%s', got '%s'", want, err.Error())
		}
	}()
	got = i2t_ssa(s)
	t.Errorf("i2t_ssa(s) should panic")
}

func e2t2_ssa(e interface{}) (*U, bool) {
	u, ok := e.(*U)
	return u, ok
}

func i2t2_ssa(i I) (*U, bool) {
	u, ok := i.(*U)
	return u, ok
}

func testAssertE2T2(t *testing.T) {
	if got, ok := e2t2_ssa(u); !ok || got != u {
		t.Errorf("e2t2_ssa(u)=(%v, %v) want (%v, %v)", got, ok, u, true)
	}
	if got, ok := e2t2_ssa(s); ok || got != nil {
		t.Errorf("e2t2_ssa(s)=(%v, %v) want (%v, %v)", got, ok, nil, false)
	}
}

func testAssertI2T2(t *testing.T) {
	if got, ok := i2t2_ssa(u); !ok || got != u {
		t.Errorf("i2t2_ssa(u)=(%v, %v) want (%v, %v)", got, ok, u, true)
	}
	if got, ok := i2t2_ssa(s); ok || got != nil {
		t.Errorf("i2t2_ssa(s)=(%v, %v) want (%v, %v)", got, ok, nil, false)
	}
}

// TestTypeAssertion tests type assertions.
func TestTypeAssertion(t *testing.T) {
	testAssertE2TOk(t)
	testAssertE2TPanic(t)
	testAssertI2TOk(t)
	testAssertI2TPanic(t)
	testAssertE2T2(t)
	testAssertI2T2(t)
}
