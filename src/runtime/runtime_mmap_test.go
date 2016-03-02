// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd solaris

package runtime_test

import (
	"runtime"
	"runtime/internal/sys"
	"testing"
)

// Test that the error value returned by mmap is positive, as that is
// what the code in mem_bsd.go, mem_darwin.go, and mem_linux.go expects.
// See the uses of ENOMEM in sysMap in those files.
func TestMmapErrorSign(t *testing.T) {
	p := runtime.Mmap(nil, ^uintptr(0)&^(sys.PhysPageSize-1), 0, runtime.MAP_ANON|runtime.MAP_PRIVATE, -1, 0)

	// The runtime.mmap function is nosplit, but t.Errorf is not.
	// Reset the pointer so that we don't get an "invalid stack
	// pointer" error from t.Errorf if we call it.
	v := uintptr(p)
	p = nil

	if v != runtime.ENOMEM {
		t.Errorf("mmap = %v, want %v", v, runtime.ENOMEM)
	}
}
