// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (386 || amd64 || arm)

package linux_test

import (
	"internal/runtime/syscall/linux"
	"internal/testenv"
	"os"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"unsafe"
)

func TestSyscallErrnoBoundary(t *testing.T) {
	const childEnv = "GO_TEST_SYSCALL_ERRNO_BOUNDARY"
	if os.Getenv(childEnv) != "1" {
		cmd := testenv.Command(t, testenv.Executable(t), "-test.run=^TestSyscallErrnoBoundary$", "-test.v")
		cmd.Env = append(cmd.Environ(), childEnv+"=1")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("child test failed: %v\n%s", err, out)
		}
		if strings.Contains(string(out), "--- SKIP: TestSyscallErrnoBoundary") {
			t.Skipf("child test skipped: %s", out)
		}
		return
	}

	// A seccomp filter lets the kernel return the otherwise hard-to-produce
	// boundary value -4095. Run it in a child because the filter cannot be
	// removed from the current thread.
	runtime.LockOSThread()
	const (
		prSetNoNewPrivs   = 38
		seccompModeFilter = 2
		seccompRetErrno   = 0x00050000
		seccompRetAllow   = 0x7fff0000
		bpfLoadSyscall    = 0x20 // BPF_LD | BPF_W | BPF_ABS
		bpfJumpEqual      = 0x15 // BPF_JMP | BPF_JEQ | BPF_K
		bpfReturn         = 0x06 // BPF_RET | BPF_K
		maxErrno          = 4095
	)
	filter := [...]syscall.SockFilter{
		{Code: bpfLoadSyscall},
		// Leave getpid available for the runtime's signal delivery.
		{Code: bpfJumpEqual, K: syscall.SYS_GETPPID, Jf: 1},
		{Code: bpfReturn, K: seccompRetErrno | maxErrno},
		{Code: bpfReturn, K: seccompRetAllow},
	}
	prog := syscall.SockFprog{Len: uint16(len(filter)), Filter: &filter[0]}
	if _, _, errno := syscall.RawSyscall6(syscall.SYS_PRCTL, prSetNoNewPrivs, 1, 0, 0, 0, 0); errno != 0 {
		t.Skipf("PR_SET_NO_NEW_PRIVS unavailable: %v", errno)
	}
	_, _, errno := syscall.RawSyscall6(syscall.SYS_PRCTL, syscall.PR_SET_SECCOMP, seccompModeFilter, uintptr(unsafe.Pointer(&prog)), 0, 0, 0)
	runtime.KeepAlive(filter)
	runtime.KeepAlive(prog)
	if errno != 0 {
		t.Skipf("seccomp filter unavailable: %v", errno)
	}

	// Keep all calls on this goroutine so the thread's filter applies.
	for _, tc := range []struct {
		name string
		call func() (uintptr, uintptr, uintptr)
	}{
		{"internal Syscall6", func() (uintptr, uintptr, uintptr) {
			return linux.Syscall6(syscall.SYS_GETPPID, 0, 0, 0, 0, 0, 0)
		}},
		{"RawSyscall", func() (uintptr, uintptr, uintptr) {
			r1, r2, e := syscall.RawSyscall(syscall.SYS_GETPPID, 0, 0, 0)
			return r1, r2, uintptr(e)
		}},
		{"RawSyscall6", func() (uintptr, uintptr, uintptr) {
			r1, r2, e := syscall.RawSyscall6(syscall.SYS_GETPPID, 0, 0, 0, 0, 0, 0)
			return r1, r2, uintptr(e)
		}},
		{"Syscall", func() (uintptr, uintptr, uintptr) {
			r1, r2, e := syscall.Syscall(syscall.SYS_GETPPID, 0, 0, 0)
			return r1, r2, uintptr(e)
		}},
		{"Syscall6", func() (uintptr, uintptr, uintptr) {
			r1, r2, e := syscall.Syscall6(syscall.SYS_GETPPID, 0, 0, 0, 0, 0, 0)
			return r1, r2, uintptr(e)
		}},
	} {
		r1, r2, errno := tc.call()
		if r1 != ^uintptr(0) || r2 != 0 || errno != maxErrno {
			t.Errorf("%s: got (%#x, %#x, %d), want (%#x, 0, %d)", tc.name, r1, r2, errno, ^uintptr(0), maxErrno)
		}
	}
}
