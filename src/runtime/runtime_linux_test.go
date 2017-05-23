// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	. "runtime"
	"syscall"
	"testing"
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
}

func TestLockOSThread(t *testing.T) {
	if pid != tid {
		t.Fatalf("pid=%d but tid=%d", pid, tid)
	}
}

// Test that error values are negative. Use address 1 (a misaligned
// pointer) to get -EINVAL.
func TestMincoreErrorSign(t *testing.T) {
	var dst byte
	v := Mincore(unsafe.Pointer(uintptr(1)), 1, &dst)

	const EINVAL = 0x16
	if v != -EINVAL {
		t.Errorf("mincore = %v, want %v", v, -EINVAL)
	}
}
