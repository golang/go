// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package os_test

import (
	"errors"
	"internal/race"
	"internal/testenv"
	"math"
	. "os"
	"runtime"
	"syscall"
	"testing"
	"time"
)

func TestErrProcessDone(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	p, err := StartProcess(testenv.GoToolPath(t), []string{"go"}, &ProcAttr{})
	if err != nil {
		t.Fatalf("starting test process: %v", err)
	}
	p.Wait()
	if got := p.Signal(Kill); got != ErrProcessDone {
		t.Errorf("got %v want %v", got, ErrProcessDone)
	}
}

// Lookup of a process that does not exist at time of lookup.
func TestProcessAlreadyDone(t *testing.T) {
	// Theoretically MaxInt32 is a valid PID, but the chance of it actually
	// being used is extremely unlikely.
	pid := math.MaxInt32
	if runtime.GOOS == "solaris" || runtime.GOOS == "illumos" {
		// Solaris/Illumos have a lower limit, above which wait returns
		// EINVAL (see waitid in usr/src/uts/common/os/exit.c in
		// illumos). This is configurable via sysconf(_SC_MAXPID), but
		// we'll just take the default.
		pid = 30000 - 1
	}

	p, err := FindProcess(pid)
	if err != nil {
		t.Fatalf("FindProcess(math.MaxInt32) got err %v, want nil", err)
	}

	if ps, err := p.Wait(); !errors.Is(err, syscall.ECHILD) {
		t.Errorf("Wait() got err %v (ps %+v), want %v", err, ps, syscall.ECHILD)
	}

	if err := p.Release(); err != nil {
		t.Errorf("Release() got err %v, want nil", err)
	}
}

func TestUNIXProcessAlive(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	p, err := StartProcess(testenv.GoToolPath(t), []string{"sleep", "1"}, &ProcAttr{})
	if err != nil {
		t.Skipf("starting test process: %v", err)
	}
	defer p.Kill()

	proc, err := FindProcess(p.Pid)
	if err != nil {
		t.Errorf("OS reported error for running process: %v", err)
	}
	err = proc.Signal(syscall.Signal(0))
	if err != nil {
		t.Errorf("OS reported error for running process: %v", err)
	}
}

func TestProcessBadPID(t *testing.T) {
	p, err := FindProcess(-1)
	if err != nil {
		t.Fatalf("unexpected FindProcess error: %v", err)
	}
	err = p.Signal(syscall.Signal(0))
	if err == nil {
		t.Error("p.Signal succeeded unexpectedly")
	}
}

// Test that a Signal that races with Release never passes the PID that
// Release stores, -1, to kill: that signals every process that the
// caller has permission to signal. See go.dev/issue/83341.
func TestSignalReleaseRace(t *testing.T) {
	if race.Enabled {
		// Release writes the Pid field without synchronization.
		// See the comment in Release.
		t.Skip("skipping: Signal and Release race on Process.Pid")
	}
	t.Parallel()

	// Signaling a nonexistent process fails, so Signal can only
	// succeed if it signals some other PID. Signal 0 has no effect
	// on the processes that kill(-1, 0) reaches.
	// As in TestProcessAlreadyDone, assume that MaxInt32 is not in use.
	//
	// The Process must use the PID rather than a process handle,
	// which FindProcess cannot be relied on to give us.
	p := &Process{Pid: math.MaxInt32}

	unlock := p.LockSignal()
	errc := make(chan error, 1)
	go func() {
		errc <- p.Signal(syscall.Signal(0))
	}()

	// Give Signal time to check the PID and block.
	// If it has not got that far the test passes without
	// testing anything, but it cannot fail spuriously.
	time.Sleep(10 * time.Millisecond)

	// This is the first thing that Release does, and the only
	// thing that it has done when a racing Signal goes wrong.
	// Calling Release here would also mark the Process as released,
	// which Signal checks after it has acquired the lock.
	p.Pid = -1
	unlock()

	if err := <-errc; err == nil {
		t.Errorf("Signal(0) of nonexistent process succeeded after Pid was set to -1")
	}
}
