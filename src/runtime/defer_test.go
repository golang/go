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
		if recover() != "testUnconditional" {
			t.Fatal("expected unconditional panic")
		}
	}()
	panic("testUnconditional")
}

var glob int = 3

// Test an open-coded defer and non-open-coded defer - make sure both defers run
// and call recover()
func TestOpenAndNonOpenDefers(t *testing.T) {
	for {
		// Non-open defer because in a loop
		defer func(n int) {
			if recover() != "testNonOpenDefer" {
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
		if recover() != "testOpenDefer" {
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
			if recover() != "testNonOpenDefer" {
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
		if recover() != "testConditional" {
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
	panic("testConditional")
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
// code. Suppose a first panic P1 happens and starts processing defer calls. If a
// second panic P2 happens while processing defer call D in frame F, then defer
// call processing is restarted (with some potentially new defer calls created by
// D or its callees). If the defer processing reaches the started defer call D
// again in the defer stack, then the original panic P1 is aborted and cannot
// continue panic processing or be recovered. If the panic P2 does a recover at
// some point, it will naturally remove the original panic P1 from the stack
// (since the original panic had to be in frame F or a descendant of F).
func TestAbortedPanic(t *testing.T) {
	defer func() {
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

type nonSSAable [128]byte

type bigStruct struct {
	x, y, z, w, p, q int64
}

type containsBigStruct struct {
	element bigStruct
}

func mknonSSAable() nonSSAable {
	globint1++
	return nonSSAable{0, 0, 0, 0, 5}
}

var globint1, globint2, globint3 int

//go:noinline
func sideeffect(n int64) int64 {
	globint2++
	return n
}

func sideeffect2(in containsBigStruct) containsBigStruct {
	globint3++
	return in
}

// Test that nonSSAable arguments to defer are handled correctly and only evaluated once.
func TestNonSSAableArgs(t *testing.T) {
	globint1 = 0
	globint2 = 0
	globint3 = 0
	var save1 byte
	var save2 int64
	var save3 int64
	var save4 int64

	defer func() {
		if globint1 != 1 {
			t.Fatal(fmt.Sprintf("globint1:  wanted: 1, got %v", globint1))
		}
		if save1 != 5 {
			t.Fatal(fmt.Sprintf("save1:  wanted: 5, got %v", save1))
		}
		if globint2 != 1 {
			t.Fatal(fmt.Sprintf("globint2:  wanted: 1, got %v", globint2))
		}
		if save2 != 2 {
			t.Fatal(fmt.Sprintf("save2:  wanted: 2, got %v", save2))
		}
		if save3 != 4 {
			t.Fatal(fmt.Sprintf("save3:  wanted: 4, got %v", save3))
		}
		if globint3 != 1 {
			t.Fatal(fmt.Sprintf("globint3:  wanted: 1, got %v", globint3))
		}
		if save4 != 4 {
			t.Fatal(fmt.Sprintf("save1:  wanted: 4, got %v", save4))
		}
	}()

	// Test function returning a non-SSAable arg
	defer func(n nonSSAable) {
		save1 = n[4]
	}(mknonSSAable())
	// Test composite literal that is not SSAable
	defer func(b bigStruct) {
		save2 = b.y
	}(bigStruct{1, 2, 3, 4, 5, sideeffect(6)})

	// Test struct field reference that is non-SSAable
	foo := containsBigStruct{}
	foo.element.z = 4
	defer func(element bigStruct) {
		save3 = element.z
	}(foo.element)
	defer func(element bigStruct) {
		save4 = element.z
	}(sideeffect2(foo).element)
}

//go:noinline
func doPanic() {
	panic("Test panic")
}

func TestDeferForFuncWithNoExit(t *testing.T) {
	cond := 1
	defer func() {
		if cond != 2 {
			t.Fatal(fmt.Sprintf("cond: wanted 2, got %v", cond))
		}
		if recover() != "Test panic" {
			t.Fatal("Didn't find expected panic")
		}
	}()
	x := 0
	// Force a stack copy, to make sure that the &cond pointer passed to defer
	// function is properly updated.
	growStackIter(&x, 1000)
	cond = 2
	doPanic()

	// This function has no exit/return, since it ends with an infinite loop
	for {
	}
}

// Test case approximating issue #37664, where a recursive function (interpreter)
// may do repeated recovers/re-panics until it reaches the frame where the panic
// can actually be handled. The recurseFnPanicRec() function is testing that there
// are no stale defer structs on the defer chain after the interpreter() sequence,
// by writing a bunch of 0xffffffffs into several recursive stack frames, and then
// doing a single panic-recover which would invoke any such stale defer structs.
func TestDeferWithRepeatedRepanics(t *testing.T) {
	interpreter(0, 6, 2)
	recurseFnPanicRec(0, 10)
	interpreter(0, 5, 1)
	recurseFnPanicRec(0, 10)
	interpreter(0, 6, 3)
	recurseFnPanicRec(0, 10)
}

func interpreter(level int, maxlevel int, rec int) {
	defer func() {
		e := recover()
		if e == nil {
			return
		}
		if level != e.(int) {
			//fmt.Fprintln(os.Stderr, "re-panicing, level", level)
			panic(e)
		}
		//fmt.Fprintln(os.Stderr, "Recovered, level", level)
	}()
	if level+1 < maxlevel {
		interpreter(level+1, maxlevel, rec)
	} else {
		//fmt.Fprintln(os.Stderr, "Initiating panic")
		panic(rec)
	}
}

func recurseFnPanicRec(level int, maxlevel int) {
	defer func() {
		recover()
	}()
	recurseFn(level, maxlevel)
}

var saveInt uint32

func recurseFn(level int, maxlevel int) {
	a := [40]uint32{0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff}
	if level+1 < maxlevel {
		// Make sure a array is referenced, so it is not optimized away
		saveInt = a[4]
		recurseFn(level+1, maxlevel)
	} else {
		panic("recurseFn panic")
	}
}

// Try to reproduce issue #37688, where a pointer to an open-coded defer struct is
// mistakenly held, and that struct keeps a pointer to a stack-allocated defer
// struct, and that stack-allocated struct gets overwritten or the stack gets
// moved, so a memory error happens on GC.
func TestIssue37688(t *testing.T) {
	for j := 0; j < 10; j++ {
		g2()
		g3()
	}
}

type foo struct {
}

//go:noinline
func (f *foo) method1() {
}

//go:noinline
func (f *foo) method2() {
}

func g2() {
	var a foo
	ap := &a
	// The loop forces this defer to be heap-allocated and the remaining two
	// to be stack-allocated.
	for i := 0; i < 1; i++ {
		defer ap.method1()
	}
	defer ap.method2()
	defer ap.method1()
	ff1(ap, 1, 2, 3, 4, 5, 6, 7, 8, 9)
	// Try to get the stack to be moved by growing it too large, so
	// existing stack-allocated defer becomes invalid.
	rec1(2000)
}

func g3() {
	// Mix up the stack layout by adding in an extra function frame
	g2()
}

var globstruct struct {
	a, b, c, d, e, f, g, h, i int
}

func ff1(ap *foo, a, b, c, d, e, f, g, h, i int) {
	defer ap.method1()

	// Make a defer that has a very large set of args, hence big size for the
	// defer record for the open-coded frame (which means it won't use the
	// defer pool)
	defer func(ap *foo, a, b, c, d, e, f, g, h, i int) {
		if v := recover(); v != nil {
		}
		globstruct.a = a
		globstruct.b = b
		globstruct.c = c
		globstruct.d = d
		globstruct.e = e
		globstruct.f = f
		globstruct.g = g
		globstruct.h = h
	}(ap, a, b, c, d, e, f, g, h, i)
	panic("ff1 panic")
}

func rec1(max int) {
	if max > 0 {
		rec1(max - 1)
	}
}

func TestIssue43921(t *testing.T) {
	defer func() {
		expect(t, 1, recover())
	}()
	func() {
		// Prevent open-coded defers
		for {
			defer func() {}()
			break
		}

		defer func() {
			defer func() {
				expect(t, 4, recover())
			}()
			panic(4)
		}()
		panic(1)

	}()
}

func expect(t *testing.T, n int, err interface{}) {
	if n != err {
		t.Fatalf("have %v, want %v", err, n)
	}
}

func TestIssue43920(t *testing.T) {
	var steps int

	defer func() {
		expect(t, 1, recover())
	}()
	defer func() {
		defer func() {
			defer func() {
				expect(t, 5, recover())
			}()
			defer panic(5)
			func() {
				panic(4)
			}()
		}()
		defer func() {
			expect(t, 3, recover())
		}()
		defer panic(3)
	}()
	func() {
		defer step(t, &steps, 1)
		panic(1)
	}()
}

func step(t *testing.T, steps *int, want int) {
	*steps++
	if *steps != want {
		t.Fatalf("have %v, want %v", *steps, want)
	}
}

func TestIssue43941(t *testing.T) {
	var steps int = 7
	defer func() {
		step(t, &steps, 14)
		expect(t, 4, recover())
	}()
	func() {
		func() {
			defer func() {
				defer func() {
					expect(t, 3, recover())
				}()
				defer panic(3)
				panic(2)
			}()
			defer func() {
				expect(t, 1, recover())
			}()
			defer panic(1)
		}()
		defer func() {}()
		defer func() {}()
		defer step(t, &steps, 10)
		defer step(t, &steps, 9)
		step(t, &steps, 8)
	}()
	func() {
		defer step(t, &steps, 13)
		defer step(t, &steps, 12)
		func() {
			defer step(t, &steps, 11)
			panic(4)
		}()

		// Code below isn't executed,
		// but removing it breaks the test case.
		defer func() {}()
		defer panic(-1)
		defer step(t, &steps, -1)
		defer step(t, &steps, -1)
		defer func() {}()
	}()
}
