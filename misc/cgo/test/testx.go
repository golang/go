// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test cases for cgo.
// Both the import "C" prologue and the main file are sorted by issue number.
// This file contains //export directives on Go functions
// and so it must NOT contain C definitions (only declarations).
// See test.go for C definitions.

package cgotest

import (
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"
)

/*
// threads
extern void doAdd(int, int);

// issue 1328
extern void BackIntoGo(void);
void IntoC(void);

// issue 1560
// mysleep returns the absolute start time in ms.
long long mysleep(int seconds);

// twoSleep returns the absolute start time of the first sleep
// in ms.
long long twoSleep(int);

// issue 3775
void lockOSThreadCallback(void);
inline static void lockOSThreadC(void)
{
        lockOSThreadCallback();
}
int usleep(unsigned usec);

// issue 4054 part 2 - part 1 in test.go
typedef enum {
	A = 0,
	B,
	C,
	D,
	E,
	F,
	G,
	H,
	II,
	J,
} issue4054b;

// issue 5548

extern int issue5548_in_c(void);

// issue 6833

extern unsigned long long issue6833Func(unsigned int, unsigned long long);

// issue 6907

extern int CheckIssue6907C(_GoString_);

// issue 7665

extern void f7665(void);

// issue 7978
// Stack tracing didn't work during cgo code after calling a Go
// callback.  Make sure GC works and the stack trace is correct.

#include <stdint.h>

void issue7978cb(void);

#if defined(__APPLE__) && defined(__arm__)
// on Darwin/ARM, libSystem doesn't provide implementation of the __sync_fetch_and_add
// primitive, and although gcc supports it, it doesn't inline its definition.
// Clang could inline its definition, so we require clang on Darwin/ARM.
#if defined(__clang__)
#define HAS_SYNC_FETCH_AND_ADD 1
#else
#define HAS_SYNC_FETCH_AND_ADD 0
#endif
#else
#define HAS_SYNC_FETCH_AND_ADD 1
#endif

// use ugly atomic variable sync since that doesn't require calling back into
// Go code or OS dependencies
static void issue7978c(uint32_t *sync) {
#if HAS_SYNC_FETCH_AND_ADD
	while(__sync_fetch_and_add(sync, 0) != 0)
		;
	__sync_fetch_and_add(sync, 1);
	while(__sync_fetch_and_add(sync, 0) != 2)
		;
	issue7978cb();
	__sync_fetch_and_add(sync, 1);
	while(__sync_fetch_and_add(sync, 0) != 6)
		;
#endif
}

// issue 8331 part 2 - part 1 in test.go
// A typedef of an unnamed struct is the same struct when
// #include'd twice.  No runtime test; just make sure it compiles.
#include "issue8331.h"

// issue 20910
void callMulti(void);

// issue 28772 part 2 - part 1 in issuex.go
#define issue28772Constant2 2

*/
import "C"

// exports

//export ReturnIntLong
func ReturnIntLong() (int, C.long) {
	return 1, 2
}

//export gc
func gc() {
	runtime.GC()
}

// threads

var sum struct {
	sync.Mutex
	i int
}

//export Add
func Add(x int) {
	defer func() {
		recover()
	}()
	sum.Lock()
	sum.i += x
	sum.Unlock()
	var p *int
	*p = 2
}

func testCthread(t *testing.T) {
	if runtime.GOOS == "darwin" && (runtime.GOARCH == "arm" || runtime.GOARCH == "arm64") {
		t.Skip("the iOS exec wrapper is unable to properly handle the panic from Add")
	}
	sum.i = 0
	C.doAdd(10, 6)

	want := 10 * (10 - 1) / 2 * 6
	if sum.i != want {
		t.Fatalf("sum=%d, want %d", sum.i, want)
	}
}

// issue 1328

//export BackIntoGo
func BackIntoGo() {
	x := 1

	for i := 0; i < 10000; i++ {
		xvariadic(x)
		if x != 1 {
			panic("x is not 1?")
		}
	}
}

func xvariadic(x ...interface{}) {
}

func test1328(t *testing.T) {
	C.IntoC()
}

// issue 1560

var sleepDone = make(chan int64)

// parallelSleep returns the absolute difference between the start time
// of the two sleeps.
func parallelSleep(n int) int64 {
	t := int64(C.twoSleep(C.int(n))) - <-sleepDone
	if t < 0 {
		return -t
	}
	return t
}

//export BackgroundSleep
func BackgroundSleep(n int32) {
	go func() {
		sleepDone <- int64(C.mysleep(C.int(n)))
	}()
}

func testParallelSleep(t *testing.T) {
	sleepSec := 1
	dt := time.Duration(parallelSleep(sleepSec)) * time.Millisecond
	t.Logf("difference in start time for two sleep(%d) is %v", sleepSec, dt)
	// bug used to run sleeps in serial, producing a 2*sleepSec-second delay.
	// we detect if the start times of those sleeps are > 0.5*sleepSec-second.
	if dt >= time.Duration(sleepSec)*time.Second/2 {
		t.Fatalf("parallel %d-second sleeps slept for %f seconds", sleepSec, dt.Seconds())
	}
}

// issue 2462

//export exportbyte
func exportbyte() byte {
	return 0
}

//export exportbool
func exportbool() bool {
	return false
}

//export exportrune
func exportrune() rune {
	return 0
}

//export exporterror
func exporterror() error {
	return nil
}

//export exportint
func exportint() int {
	return 0
}

//export exportuint
func exportuint() uint {
	return 0
}

//export exportuintptr
func exportuintptr() uintptr {
	return (uintptr)(0)
}

//export exportint8
func exportint8() int8 {
	return 0
}

//export exportuint8
func exportuint8() uint8 {
	return 0
}

//export exportint16
func exportint16() int16 {
	return 0
}

//export exportuint16
func exportuint16() uint16 {
	return 0
}

//export exportint32
func exportint32() int32 {
	return 0
}

//export exportuint32
func exportuint32() uint32 {
	return 0
}

//export exportint64
func exportint64() int64 {
	return 0
}

//export exportuint64
func exportuint64() uint64 {
	return 0
}

//export exportfloat32
func exportfloat32() float32 {
	return 0
}

//export exportfloat64
func exportfloat64() float64 {
	return 0
}

//export exportcomplex64
func exportcomplex64() complex64 {
	return 0
}

//export exportcomplex128
func exportcomplex128() complex128 {
	return 0
}

// issue 3741

//export exportSliceIn
func exportSliceIn(s []byte) bool {
	return len(s) == cap(s)
}

//export exportSliceOut
func exportSliceOut() []byte {
	return []byte{1}
}

//export exportSliceInOut
func exportSliceInOut(s []byte) []byte {
	return s
}

// issue 3775

func init() {
	if runtime.GOOS == "android" {
		return
	}
	// Same as test3775 but run during init so that
	// there are two levels of internal runtime lock
	// (1 for init, 1 for cgo).
	// This would have been broken by CL 11663043.
	C.lockOSThreadC()
}

func test3775(t *testing.T) {
	if runtime.GOOS == "android" {
		return
	}
	// Used to panic because of the UnlockOSThread below.
	C.lockOSThreadC()
}

//export lockOSThreadCallback
func lockOSThreadCallback() {
	runtime.LockOSThread()
	runtime.UnlockOSThread()
	go C.usleep(10000)
	runtime.Gosched()
}

// issue 4054 part 2 - part 1 in test.go

var issue4054b = []int{C.A, C.B, C.C, C.D, C.E, C.F, C.G, C.H, C.II, C.J}

//export issue5548FromC
func issue5548FromC(s string, i int) int {
	if len(s) == 4 && s == "test" && i == 42 {
		return 12345
	}
	println("got", len(s), i)
	return 9876
}

func test5548(t *testing.T) {
	if x := C.issue5548_in_c(); x != 12345 {
		t.Errorf("issue5548_in_c = %d, want %d", x, 12345)
	}
}

// issue 6833

//export GoIssue6833Func
func GoIssue6833Func(aui uint, aui64 uint64) uint64 {
	return aui64 + uint64(aui)
}

func test6833(t *testing.T) {
	ui := 7
	ull := uint64(0x4000300020001000)
	v := uint64(C.issue6833Func(C.uint(ui), C.ulonglong(ull)))
	exp := uint64(ui) + ull
	if v != exp {
		t.Errorf("issue6833Func() returns %x, expected %x", v, exp)
	}
}

// issue 6907

const CString = "C string"

//export CheckIssue6907Go
func CheckIssue6907Go(s string) C.int {
	if s == CString {
		return 1
	}
	return 0
}

func test6907Go(t *testing.T) {
	if got := C.CheckIssue6907C(CString); got != 1 {
		t.Errorf("C.CheckIssue6907C() == %d, want %d", got, 1)
	}
}

// issue 7665

//export f7665
func f7665() {}

var bad7665 unsafe.Pointer = C.f7665
var good7665 uintptr = uintptr(C.f7665)

func test7665(t *testing.T) {
	if bad7665 == nil || uintptr(bad7665) != good7665 {
		t.Errorf("ptrs = %p, %#x, want same non-nil pointer", bad7665, good7665)
	}
}

// issue 7978

var issue7978sync uint32

func issue7978check(t *testing.T, wantFunc string, badFunc string, depth int) {
	runtime.GC()
	buf := make([]byte, 65536)
	trace := string(buf[:runtime.Stack(buf, true)])
	for _, goroutine := range strings.Split(trace, "\n\n") {
		if strings.Contains(goroutine, "test.issue7978go") {
			trace := strings.Split(goroutine, "\n")
			// look for the expected function in the stack
			for i := 0; i < depth; i++ {
				if badFunc != "" && strings.Contains(trace[1+2*i], badFunc) {
					t.Errorf("bad stack: found %s in the stack:\n%s", badFunc, goroutine)
					return
				}
				if strings.Contains(trace[1+2*i], wantFunc) {
					return
				}
			}
			t.Errorf("bad stack: didn't find %s in the stack:\n%s", wantFunc, goroutine)
			return
		}
	}
	t.Errorf("bad stack: goroutine not found. Full stack dump:\n%s", trace)
}

func issue7978wait(store uint32, wait uint32) {
	if store != 0 {
		atomic.StoreUint32(&issue7978sync, store)
	}
	for atomic.LoadUint32(&issue7978sync) != wait {
		runtime.Gosched()
	}
}

//export issue7978cb
func issue7978cb() {
	// Force a stack growth from the callback to put extra
	// pressure on the runtime. See issue #17785.
	growStack(64)
	issue7978wait(3, 4)
}

func growStack(n int) int {
	var buf [128]int
	if n == 0 {
		return 0
	}
	return buf[growStack(n-1)]
}

func issue7978go() {
	C.issue7978c((*C.uint32_t)(&issue7978sync))
	issue7978wait(7, 8)
}

func test7978(t *testing.T) {
	if runtime.Compiler == "gccgo" {
		t.Skip("gccgo can not do stack traces of C code")
	}
	if C.HAS_SYNC_FETCH_AND_ADD == 0 {
		t.Skip("clang required for __sync_fetch_and_add support on darwin/arm")
	}
	debug.SetTraceback("2")
	issue7978sync = 0
	go issue7978go()
	// test in c code, before callback
	issue7978wait(0, 1)
	issue7978check(t, "_Cfunc_issue7978c(", "", 1)
	// test in go code, during callback
	issue7978wait(2, 3)
	issue7978check(t, "test.issue7978cb(", "test.issue7978go", 3)
	// test in c code, after callback
	issue7978wait(4, 5)
	issue7978check(t, "_Cfunc_issue7978c(", "_cgoexpwrap", 1)
	// test in go code, after return from cgo
	issue7978wait(6, 7)
	issue7978check(t, "test.issue7978go(", "", 3)
	atomic.StoreUint32(&issue7978sync, 8)
}

// issue 8331 part 2

var issue8331Var C.issue8331

// issue 20910

//export multi
func multi() (*C.char, C.int) {
	return C.CString("multi"), 0
}

func test20910(t *testing.T) {
	C.callMulti()
}

// issue 28772 part 2

const issue28772Constant2 = C.issue28772Constant2

//export issue29878exported
func issue29878exported(arg int8) uint64 {
	return uint64(arg)
}
