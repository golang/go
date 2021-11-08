// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package runtime_test

import (
	"runtime"
	"syscall"
	"testing"
	"unsafe"
)

func TestNonblockingPipe(t *testing.T) {
	t.Parallel()

	// NonblockingPipe is the test name for nonblockingPipe.
	r, w, errno := runtime.NonblockingPipe()
	if errno != 0 {
		t.Fatal(syscall.Errno(errno))
	}
	defer func() {
		runtime.Close(r)
		runtime.Close(w)
	}()

	checkIsPipe(t, r, w)
	checkNonblocking(t, r, "reader")
	checkCloseonexec(t, r, "reader")
	checkNonblocking(t, w, "writer")
	checkCloseonexec(t, w, "writer")
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
	flags, errno := fcntl(uintptr(fd), syscall.F_GETFL, 0)
	if errno != 0 {
		t.Errorf("fcntl(%s, F_GETFL) failed: %v", name, syscall.Errno(errno))
	} else if flags&syscall.O_NONBLOCK == 0 {
		t.Errorf("O_NONBLOCK not set in %s flags %#x", name, flags)
	}
}

func checkCloseonexec(t *testing.T, fd int32, name string) {
	t.Helper()
	flags, errno := fcntl(uintptr(fd), syscall.F_GETFD, 0)
	if errno != 0 {
		t.Errorf("fcntl(%s, F_GETFD) failed: %v", name, syscall.Errno(errno))
	} else if flags&syscall.FD_CLOEXEC == 0 {
		t.Errorf("FD_CLOEXEC not set in %s flags %#x", name, flags)
	}
}

func TestSetNonblock(t *testing.T) {
	t.Parallel()

	r, w, errno := runtime.Pipe()
	if errno != 0 {
		t.Fatal(syscall.Errno(errno))
	}
	defer func() {
		runtime.Close(r)
		runtime.Close(w)
	}()

	checkIsPipe(t, r, w)

	runtime.SetNonblock(r)
	runtime.SetNonblock(w)
	checkNonblocking(t, r, "reader")
	checkNonblocking(t, w, "writer")

	runtime.Closeonexec(r)
	runtime.Closeonexec(w)
	checkCloseonexec(t, r, "reader")
	checkCloseonexec(t, w, "writer")
}
