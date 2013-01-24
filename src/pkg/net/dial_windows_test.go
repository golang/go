// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"sync"
	"syscall"
	"testing"
	"time"
	"unsafe"
)

var handleCounter struct {
	once sync.Once
	proc *syscall.Proc
}

func numHandles(t *testing.T) int {

	handleCounter.once.Do(func() {
		d, err := syscall.LoadDLL("kernel32.dll")
		if err != nil {
			t.Fatalf("LoadDLL: %v\n", err)
		}
		handleCounter.proc, err = d.FindProc("GetProcessHandleCount")
		if err != nil {
			t.Fatalf("FindProc: %v\n", err)
		}
	})

	cp, err := syscall.GetCurrentProcess()
	if err != nil {
		t.Fatalf("GetCurrentProcess: %v\n", err)
	}
	var n uint32
	r, _, err := handleCounter.proc.Call(uintptr(cp), uintptr(unsafe.Pointer(&n)))
	if r == 0 {
		t.Fatalf("GetProcessHandleCount: %v\n", error(err))
	}
	return int(n)
}

func testDialTimeoutHandleLeak(t *testing.T) (before, after int) {
	before = numHandles(t)
	// See comment in TestDialTimeout about why we use this address.
	c, err := DialTimeout("tcp", "127.0.71.111:49151", 200*time.Millisecond)
	after = numHandles(t)
	if err == nil {
		c.Close()
		t.Fatalf("unexpected: connected to %s", c.RemoteAddr())
	}
	terr, ok := err.(timeout)
	if !ok {
		t.Fatalf("got error %q; want error with timeout interface", err)
	}
	if !terr.Timeout() {
		t.Fatalf("got error %q; not a timeout", err)
	}
	return
}

func TestDialTimeoutHandleLeak(t *testing.T) {
	if !canUseConnectEx("tcp") {
		t.Skip("skipping test; no ConnectEx found.")
	}
	testDialTimeoutHandleLeak(t) // ignore first call results
	before, after := testDialTimeoutHandleLeak(t)
	if before != after {
		t.Fatalf("handle count is different before=%d and after=%d", before, after)
	}
}
