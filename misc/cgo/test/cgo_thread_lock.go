// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux,freebsd,openbsd

package cgotest

/*
#include <unistd.h>
#include <sys/syscall.h>
void Gosched(void);
static int Ctid(void) { Gosched(); return syscall(SYS_gettid); }
*/
import "C"

import (
	"runtime"
	"syscall"
	"testing"
	"time"
)

//export Gosched
func Gosched() {
	runtime.Gosched()
}

func init() {
	testThreadLockFunc = testThreadLock
}

func testThreadLock(t *testing.T) {
	stop := make(chan int)
	go func() {
		// We need the G continue running,
		// so the M has a chance to run this G.
		for {
			select {
			case <-stop:
				return
			case <-time.After(time.Millisecond * 100):
			}
		}
	}()
	defer close(stop)

	for i := 0; i < 1000; i++ {
		if C.int(syscall.Gettid()) != C.Ctid() {
			t.Fatalf("cgo has not locked OS thread")
		}
	}
}
