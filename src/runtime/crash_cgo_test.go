// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo

package runtime_test

import (
	"os/exec"
	"runtime"
	"strings"
	"testing"
)

func TestCgoCrashHandler(t *testing.T) {
	testCrashHandler(t, true)
}

func TestCgoSignalDeadlock(t *testing.T) {
	if testing.Short() && runtime.GOOS == "windows" {
		t.Skip("Skipping in short mode") // takes up to 64 seconds
	}
	got := executeTest(t, cgoSignalDeadlockSource, nil)
	want := "OK\n"
	if got != want {
		t.Fatalf("expected %q, but got %q", want, got)
	}
}

func TestCgoTraceback(t *testing.T) {
	got := executeTest(t, cgoTracebackSource, nil)
	want := "OK\n"
	if got != want {
		t.Fatalf("expected %q, but got %q", want, got)
	}
}

func TestCgoExternalThreadPanic(t *testing.T) {
	if runtime.GOOS == "plan9" {
		t.Skipf("no pthreads on %s", runtime.GOOS)
	}
	csrc := cgoExternalThreadPanicC
	if runtime.GOOS == "windows" {
		csrc = cgoExternalThreadPanicC_windows
	}
	got := executeTest(t, cgoExternalThreadPanicSource, nil, "main.c", csrc)
	want := "panic: BOOM"
	if !strings.Contains(got, want) {
		t.Fatalf("want failure containing %q. output:\n%s\n", want, got)
	}
}

func TestCgoExternalThreadSIGPROF(t *testing.T) {
	// issue 9456.
	switch runtime.GOOS {
	case "plan9", "windows":
		t.Skipf("no pthreads on %s", runtime.GOOS)
	case "darwin":
		// static constructor needs external linking, but we don't support
		// external linking on OS X 10.6.
		out, err := exec.Command("uname", "-r").Output()
		if err != nil {
			t.Fatalf("uname -r failed: %v", err)
		}
		// OS X 10.6 == Darwin 10.x
		if strings.HasPrefix(string(out), "10.") {
			t.Skipf("no external linking on OS X 10.6")
		}
	}
	got := executeTest(t, cgoExternalThreadSIGPROFSource, nil)
	want := "OK\n"
	if got != want {
		t.Fatalf("expected %q, but got %q", want, got)
	}
}

const cgoSignalDeadlockSource = `
package main

import "C"

import (
	"fmt"
	"runtime"
	"time"
)

func main() {
	runtime.GOMAXPROCS(100)
	ping := make(chan bool)
	go func() {
		for i := 0; ; i++ {
			runtime.Gosched()
			select {
			case done := <-ping:
				if done {
					ping <- true
					return
				}
				ping <- true
			default:
			}
			func() {
				defer func() {
					recover()
				}()
				var s *string
				*s = ""
			}()
		}
	}()
	time.Sleep(time.Millisecond)
	for i := 0; i < 64; i++ {
		go func() {
			runtime.LockOSThread()
			select {}
		}()
		go func() {
			runtime.LockOSThread()
			select {}
		}()
		time.Sleep(time.Millisecond)
		ping <- false
		select {
		case <-ping:
		case <-time.After(time.Second):
			fmt.Printf("HANG\n")
			return
		}
	}
	ping <- true
	select {
	case <-ping:
	case <-time.After(time.Second):
		fmt.Printf("HANG\n")
		return
	}
	fmt.Printf("OK\n")
}
`

const cgoTracebackSource = `
package main

/* void foo(void) {} */
import "C"

import (
	"fmt"
	"runtime"
)

func main() {
	C.foo()
	buf := make([]byte, 1)
	runtime.Stack(buf, true)
	fmt.Printf("OK\n")
}
`

const cgoExternalThreadPanicSource = `
package main

// void start(void);
import "C"

func main() {
	C.start()
	select {}
}

//export gopanic
func gopanic() {
	panic("BOOM")
}
`

const cgoExternalThreadPanicC = `
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

void gopanic(void);

static void*
die(void* x)
{
	gopanic();
	return 0;
}

void
start(void)
{
	pthread_t t;
	if(pthread_create(&t, 0, die, 0) != 0)
		printf("pthread_create failed\n");
}
`

const cgoExternalThreadPanicC_windows = `
#include <stdlib.h>
#include <stdio.h>

void gopanic(void);

static void*
die(void* x)
{
	gopanic();
	return 0;
}

void
start(void)
{
	if(_beginthreadex(0, 0, die, 0, 0, 0) != 0)
		printf("_beginthreadex failed\n");
}
`

const cgoExternalThreadSIGPROFSource = `
package main

/*
#include <stdint.h>
#include <signal.h>
#include <pthread.h>

volatile int32_t spinlock;

static void *thread1(void *p) {
	(void)p;
	while (spinlock == 0)
		;
	pthread_kill(pthread_self(), SIGPROF);
	spinlock = 0;
	return NULL;
}
__attribute__((constructor)) void issue9456() {
	pthread_t tid;
	pthread_create(&tid, 0, thread1, NULL);
}
*/
import "C"

import (
	"runtime"
	"sync/atomic"
	"unsafe"
)

func main() {
	// This test intends to test that sending SIGPROF to foreign threads
	// before we make any cgo call will not abort the whole process, so
	// we cannot make any cgo call here. See http://golang.org/issue/9456.
	atomic.StoreInt32((*int32)(unsafe.Pointer(&C.spinlock)), 1)
	for atomic.LoadInt32((*int32)(unsafe.Pointer(&C.spinlock))) == 1 {
		runtime.Gosched()
	}
	println("OK")
}
`
