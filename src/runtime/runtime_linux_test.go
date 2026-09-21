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

	sysNanosleep = func(d time.Duration) {
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

func TestParseRelease(t *testing.T) {
	tests := []struct {
		in                  string
		major, minor, patch int
		ok                  bool
	}{
		{"6.1.0", 6, 1, 0, true},
		{"5.15.0-91-generic", 5, 15, 0, true},
		{"4.19.0+", 4, 19, 0, true},
		{"6.6.0-rc1", 6, 6, 0, true},
		// Synology embedded Linux appends a platform identifier
		// after an underscore.
		{"3.4.35_hi3535", 3, 4, 35, true},
		{"2.6.32_synology", 2, 6, 32, true},
		{"3.10", 3, 10, 0, true},
		// A single component is not enough; major+minor required.
		{"3", 0, 0, 0, false},
		{"3-rc1", 0, 0, 0, false},
		{"", 0, 0, 0, false},
		{"bogus", 0, 0, 0, false},
	}
	for _, tt := range tests {
		major, minor, patch, ok := ParseRelease(tt.in)
		if major != tt.major || minor != tt.minor || patch != tt.patch || ok != tt.ok {
			t.Errorf("ParseRelease(%q) = (%d, %d, %d, %v); want (%d, %d, %d, %v)",
				tt.in, major, minor, patch, ok, tt.major, tt.minor, tt.patch, tt.ok)
		}
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
