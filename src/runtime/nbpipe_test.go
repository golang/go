// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package runtime_test

import (
	"runtime"
	"syscall"
	"testing"
	"unsafe"
)

func TestNonblockingPipe(t *testing.T) {
	// NonblockingPipe is the test name for nonblockingPipe.
	r, w, errno := runtime.NonblockingPipe()
	if errno != 0 {
		t.Fatal(syscall.Errno(errno))
	}
	defer runtime.Close(w)

	checkIsPipe(t, r, w)
	checkNonblocking(t, r, "reader")
	checkCloseonexec(t, r, "reader")
	checkNonblocking(t, w, "writer")
	checkCloseonexec(t, w, "writer")

	// Test that fcntl returns an error as expected.
	if runtime.Close(r) != 0 {
		t.Fatalf("Close(%d) failed", r)
	}
	val, errno := runtime.Fcntl(r, syscall.F_GETFD, 0)
	if val != -1 {
		t.Errorf("Fcntl succeeded unexpectedly")
	} else if syscall.Errno(errno) != syscall.EBADF {
		t.Errorf("Fcntl failed with error %v, expected %v", syscall.Errno(errno), syscall.EBADF)
	}
}

func checkIsPipe(t *testing.T, r, w int32) {
	bw := byte(42)
	if n := runtime.Write(uintptr(w), unsafe.Pointer(&bw), 1); n != 1 {
		t.Fatalf("Write(w, &b, 1) == %d, expected 1", n)
	}
	var br byte
	if n := runtime.Read(r, unsafe.Pointer(&br), 1); n != 1 {
		t.Fatalf("Read(r, &b, 1) == %d, expected 1", n)
	}
	if br != bw {
		t.Errorf("pipe read %d, expected %d", br, bw)
	}
}

func checkNonblocking(t *testing.T, fd int32, name string) {
	t.Helper()
	flags, errno := runtime.Fcntl(fd, syscall.F_GETFL, 0)
	if flags == -1 {
		t.Errorf("fcntl(%s, F_GETFL) failed: %v", name, syscall.Errno(errno))
	} else if flags&syscall.O_NONBLOCK == 0 {
		t.Errorf("O_NONBLOCK not set in %s flags %#x", name, flags)
	}
}

func checkCloseonexec(t *testing.T, fd int32, name string) {
	t.Helper()
	flags, errno := runtime.Fcntl(fd, syscall.F_GETFD, 0)
	if flags == -1 {
		t.Errorf("fcntl(%s, F_GETFD) failed: %v", name, syscall.Errno(errno))
	} else if flags&syscall.FD_CLOEXEC == 0 {
		t.Errorf("FD_CLOEXEC not set in %s flags %#x", name, flags)
	}
}
