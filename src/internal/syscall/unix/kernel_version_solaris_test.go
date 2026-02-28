// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build solaris

package unix_test

import (
	"internal/syscall/unix"
	"runtime"
	"syscall"
	"testing"
)

func TestSupportSockNonblockCloexec(t *testing.T) {
	// Test that SupportSockNonblockCloexec returns true if socket succeeds with SOCK_NONBLOCK and SOCK_CLOEXEC.
	s, err := syscall.Socket(syscall.AF_INET, syscall.SOCK_STREAM|syscall.SOCK_NONBLOCK|syscall.SOCK_CLOEXEC, 0)
	if err == nil {
		syscall.Close(s)
	}
	wantSock := err != syscall.EPROTONOSUPPORT && err != syscall.EINVAL
	gotSock := unix.SupportSockNonblockCloexec()
	if wantSock != gotSock {
		t.Fatalf("SupportSockNonblockCloexec, got %t; want %t", gotSock, wantSock)
	}

	// Test that SupportAccept4 returns true if accept4 is available.
	for {
		_, _, err = syscall.Accept4(0, syscall.SOCK_NONBLOCK|syscall.SOCK_CLOEXEC)
		if err != syscall.EINTR {
			break
		}
	}
	wantAccept4 := err != syscall.ENOSYS
	gotAccept4 := unix.SupportAccept4()
	if wantAccept4 != gotAccept4 {
		t.Fatalf("SupportAccept4, got %t; want %t", gotAccept4, wantAccept4)
	}

	// Test that the version returned by KernelVersion matches expectations.
	major, minor := unix.KernelVersion()
	t.Logf("Kernel version: %d.%d", major, minor)
	if runtime.GOOS == "illumos" {
		if gotSock && gotAccept4 && (major < 5 || (major == 5 && minor < 11)) {
			t.Fatalf("SupportSockNonblockCloexec and SupportAccept4 are true, but kernel version is older than 5.11, SunOS version: %d.%d", major, minor)
		}
		if !gotSock && !gotAccept4 && (major > 5 || (major == 5 && minor >= 11)) {
			t.Errorf("SupportSockNonblockCloexec and SupportAccept4 are false, but kernel version is 5.11 or newer, SunOS version: %d.%d", major, minor)
		}
	} else { // Solaris
		if gotSock && gotAccept4 && (major < 11 || (major == 11 && minor < 4)) {
			t.Fatalf("SupportSockNonblockCloexec and SupportAccept4 are true, but kernel version is older than 11.4, Solaris version: %d.%d", major, minor)
		}
		if !gotSock && !gotAccept4 && (major > 11 || (major == 11 && minor >= 4)) {
			t.Errorf("SupportSockNonblockCloexec and SupportAccept4 are false, but kernel version is 11.4 or newer, Solaris version: %d.%d", major, minor)
		}
	}
}
