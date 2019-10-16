// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"reflect"
	"runtime"
	"testing"
)

// Make sure open-coded defer exit code is not lost, even when there is an
// unconditional panic (hence no return from the function)
func TestUnconditionalPanic(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected unconditional panic")
		}
	}()
	panic("panic should be recovered")
}

var glob int = 3

// Test an open-coded defer and non-open-coded defer - make sure both defers run
// and call recover()
func TestOpenAndNonOpenDefers(t *testing.T) {
	for {
		// Non-open defer because in a loop
		defer func(n int) {
			if recover() == nil {
				t.Fatal("expected testNonOpen panic")
			}
		}(3)
		if glob > 2 {
			break
		}
	}
	testOpen(t, 47)
	panic("testNonOpenDefer")
}

//go:noinline
func testOpen(t *testing.T, arg int) {
	defer func(n int) {
		if recover() == nil {
			t.Fatal("expected testOpen panic")
		}
	}(4)
	if arg > 2 {
		panic("testOpenDefer")
	}
}

// Test a non-open-coded defer and an open-coded defer - make sure both defers run
// and call recover()
func TestNonOpenAndOpenDefers(t *testing.T) {
	testOpen(t, 47)
	for {
		// Non-open defer because in a loop
		defer func(n int) {
			if recover() == nil {
				t.Fatal("expected testNonOpen panic")
			}
		}(3)
		if glob > 2 {
			break
		}
	}
	panic("testNonOpenDefer")
}

var list []int

// Make sure that conditional open-coded defers are activated correctly and run in
// the correct order.
func TestConditionalDefers(t *testing.T) {
	list = make([]int, 0, 10)

	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
		want := []int{4, 2, 1}
		if !reflect.DeepEqual(want, list) {
			t.Fatal(fmt.Sprintf("wanted %v, got %v", want, list))
		}

	}()
	testConditionalDefers(8)
}

func testConditionalDefers(n int) {
	doappend := func(i int) {
		list = append(list, i)
	}

	defer doappend(1)
	if n > 5 {
		defer doappend(2)
		if n > 8 {
			defer doappend(3)
		} else {
			defer doappend(4)
		}
	}
	panic("test")
}

// Test that there is no compile-time or run-time error if an open-coded defer
// call is removed by constant propagation and dead-code elimination.
func TestDisappearingDefer(t *testing.T) {
	switch runtime.GOOS {
	case "invalidOS":
		defer func() {
			t.Fatal("Defer shouldn't run")
		}()
	}
}

// This tests an extra recursive panic behavior that is only specified in the
// code.  Suppose a first panic P1 happens and starts processing defer calls.  If
// a second panic P2 happens while processing defer call D in frame F, then defer
// call processing is restarted (with some potentially new defer calls created by
// D or its callees).  If the defer processing reaches the started defer call D
// again in the defer stack, then the original panic P1 is aborted and cannot
// continue panic processing or be recovered.  If the panic P2 does a recover at
// some point, it will naturally the original panic P1 from the stack, since the
// original panic had to be in frame F or a descendant of F.
func TestAbortedPanic(t *testing.T) {
	defer func() {
		// The first panic should have been "aborted", so there is
		// no other panic to recover
		r := recover()
		if r != nil {
			t.Fatal(fmt.Sprintf("wanted nil recover, got %v", r))
		}
	}()
	defer func() {
		r := recover()
		if r != "panic2" {
			t.Fatal(fmt.Sprintf("wanted %v, got %v", "panic2", r))
		}
	}()
	defer func() {
		panic("panic2")
	}()
	panic("panic1")
}

// This tests that recover() does not succeed unless it is called directly from a
// defer function that is directly called by the panic.  Here, we first call it
// from a defer function that is created by the defer function called directly by
// the panic.  In
func TestRecoverMatching(t *testing.T) {
	defer func() {
		r := recover()
		if r != "panic1" {
			t.Fatal(fmt.Sprintf("wanted %v, got %v", "panic1", r))
		}
	}()
	defer func() {
		defer func() {
			// Shouldn't succeed, even though it is called directly
			// from a defer function, since this defer function was
			// not directly called by the panic.
			r := recover()
			if r != nil {
				t.Fatal(fmt.Sprintf("wanted nil recover, got %v", r))
			}
		}()
	}()
	panic("panic1")
}
