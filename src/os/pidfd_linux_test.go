// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"errors"
	"internal/syscall/unix"
	"internal/testenv"
	"os"
	"os/exec"
	"syscall"
	"testing"
)

func TestFindProcessViaPidfd(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	if err := os.CheckPidfdOnce(); err != nil {
		// Non-pidfd code paths tested in exec_unix_test.go.
		t.Skipf("skipping: pidfd not available: %v", err)
	}

	p, err := os.StartProcess(testenv.GoToolPath(t), []string{"go"}, &os.ProcAttr{})
	if err != nil {
		t.Fatalf("starting test process: %v", err)
	}
	p.Wait()

	// Use pid of a non-existing process.
	proc, err := os.FindProcess(p.Pid)
	// FindProcess should never return errors on Unix.
	if err != nil {
		t.Fatalf("FindProcess: got error %v, want <nil>", err)
	}
	// FindProcess should never return nil Process.
	if proc == nil {
		t.Fatal("FindProcess: got nil, want non-nil")
	}
	if proc.Status() != os.StatusDone {
		t.Fatalf("got process status: %v, want %d", proc.Status(), os.StatusDone)
	}

	// Check that all Process' public methods work as expected with
	// "done" Process.
	if err := proc.Kill(); err != os.ErrProcessDone {
		t.Errorf("Kill: got %v, want %v", err, os.ErrProcessDone)
	}
	if err := proc.Signal(os.Kill); err != os.ErrProcessDone {
		t.Errorf("Signal: got %v, want %v", err, os.ErrProcessDone)
	}
	if _, err := proc.Wait(); !errors.Is(err, syscall.ECHILD) {
		t.Errorf("Wait: got %v, want %v", err, os.ErrProcessDone)
	}
	// Release never returns errors on Unix.
	if err := proc.Release(); err != nil {
		t.Fatalf("Release: got %v, want <nil>", err)
	}
}

func TestStartProcessWithPidfd(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	if err := os.CheckPidfdOnce(); err != nil {
		// Non-pidfd code paths tested in exec_unix_test.go.
		t.Skipf("skipping: pidfd not available: %v", err)
	}

	var pidfd int
	p, err := os.StartProcess(testenv.GoToolPath(t), []string{"go"}, &os.ProcAttr{
		Sys: &syscall.SysProcAttr{
			PidFD: &pidfd,
		},
	})
	if err != nil {
		t.Fatalf("starting test process: %v", err)
	}
	defer syscall.Close(pidfd)

	if _, err := p.Wait(); err != nil {
		t.Fatalf("Wait: got %v, want <nil>", err)
	}

	// Check the pidfd is still valid
	err = unix.PidFDSendSignal(uintptr(pidfd), syscall.Signal(0))
	if !errors.Is(err, syscall.ESRCH) {
		t.Errorf("SendSignal: got %v, want %v", err, syscall.ESRCH)
	}
}

// Issue #69284
func TestPidfdLeak(t *testing.T) {
	exe := testenv.Executable(t)

	// Find the next 10 descriptors.
	// We need to get more than one descriptor in practice;
	// the pidfd winds up not being the next descriptor.
	const count = 10
	want := make([]int, count)
	for i := range count {
		var err error
		want[i], err = syscall.Open(exe, syscall.O_RDONLY, 0)
		if err != nil {
			t.Fatal(err)
		}
	}

	// Close the descriptors.
	for _, d := range want {
		syscall.Close(d)
	}

	// Start a process 10 times.
	for range 10 {
		// For testing purposes this has to be an absolute path.
		// Otherwise we will fail finding the executable
		// and won't start a process at all.
		cmd := exec.Command("/noSuchExecutable")
		cmd.Run()
	}

	// Open the next 10 descriptors again.
	got := make([]int, count)
	for i := range count {
		var err error
		got[i], err = syscall.Open(exe, syscall.O_RDONLY, 0)
		if err != nil {
			t.Fatal(err)
		}
	}

	// Close the descriptors
	for _, d := range got {
		syscall.Close(d)
	}

	t.Logf("got %v", got)
	t.Logf("want %v", want)

	// Allow some slack for runtime epoll descriptors and the like.
	if got[count-1] > want[count-1]+5 {
		t.Errorf("got descriptor %d, want %d", got[count-1], want[count-1])
	}
}
