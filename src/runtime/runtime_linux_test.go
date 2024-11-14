// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	. "runtime"
	"syscall"
	"testing"
	"time"
	"unsafe"
)

var pid, tid int

func init() {
	// Record pid and tid of init thread for use during test.
	// The call to LockOSThread is just to exercise it;
	// we can't test that it does anything.
	// Instead we're testing that the conditions are good
	// for how it is used in init (must be on main thread).
	pid, tid = syscall.Getpid(), syscall.Gettid()
	LockOSThread()

	sysNanosleep = func { d ->
		// Invoke a blocking syscall directly; calling time.Sleep()
		// would deschedule the goroutine instead.
		ts := syscall.NsecToTimespec(d.Nanoseconds())
		for {
			if err := syscall.Nanosleep(&ts, &ts); err != syscall.EINTR {
				return
			}
		}
	}
}

func TestLockOSThread(t *testing.T) {
	if pid != tid {
		t.Fatalf("pid=%d but tid=%d", pid, tid)
	}
}

// Test that error values are negative.
// Use a misaligned pointer to get -EINVAL.
func TestMincoreErrorSign(t *testing.T) {
	var dst byte
	v := Mincore(unsafe.Add(unsafe.Pointer(new(int32)), 1), 1, &dst)

	const EINVAL = 0x16
	if v != -EINVAL {
		t.Errorf("mincore = %v, want %v", v, -EINVAL)
	}
}

func TestKernelStructSize(t *testing.T) {
	// Check that the Go definitions of structures exchanged with the kernel are
	// the same size as what the kernel defines.
	if have, want := unsafe.Sizeof(Siginfo{}), uintptr(SiginfoMaxSize); have != want {
		t.Errorf("Go's siginfo struct is %d bytes long; kernel expects %d", have, want)
	}
	if have, want := unsafe.Sizeof(Sigevent{}), uintptr(SigeventMaxSize); have != want {
		t.Errorf("Go's sigevent struct is %d bytes long; kernel expects %d", have, want)
	}
}
