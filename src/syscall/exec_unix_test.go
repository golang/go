// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris

package syscall_test

import (
	"internal/testenv"
	"io"
	"math/rand"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"syscall"
	"testing"
	"time"
	"unsafe"
)

type command struct {
	pipe io.WriteCloser
	proc *exec.Cmd
	test *testing.T
}

func (c *command) Info() (pid, pgrp int) {
	pid = c.proc.Process.Pid

	pgrp, err := syscall.Getpgid(pid)
	if err != nil {
		c.test.Fatal(err)
	}

	return
}

func (c *command) Start() {
	if err := c.proc.Start(); err != nil {
		c.test.Fatal(err)
	}
}

func (c *command) Stop() {
	c.pipe.Close()
	if err := c.proc.Wait(); err != nil {
		c.test.Fatal(err)
	}
}

func create(t *testing.T) *command {
	testenv.MustHaveExec(t)

	proc := exec.Command("cat")
	stdin, err := proc.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}

	return &command{stdin, proc, t}
}

func parent() (pid, pgrp int) {
	return syscall.Getpid(), syscall.Getpgrp()
}

func TestZeroSysProcAttr(t *testing.T) {
	ppid, ppgrp := parent()

	cmd := create(t)

	cmd.Start()
	defer cmd.Stop()

	cpid, cpgrp := cmd.Info()

	if cpid == ppid {
		t.Fatalf("Parent and child have the same process ID")
	}

	if cpgrp != ppgrp {
		t.Fatalf("Child is not in parent's process group")
	}
}

func TestSetpgid(t *testing.T) {
	ppid, ppgrp := parent()

	cmd := create(t)

	cmd.proc.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Start()
	defer cmd.Stop()

	cpid, cpgrp := cmd.Info()

	if cpid == ppid {
		t.Fatalf("Parent and child have the same process ID")
	}

	if cpgrp == ppgrp {
		t.Fatalf("Parent and child are in the same process group")
	}

	if cpid != cpgrp {
		t.Fatalf("Child's process group is not the child's process ID")
	}
}

func TestPgid(t *testing.T) {
	ppid, ppgrp := parent()

	cmd1 := create(t)

	cmd1.proc.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd1.Start()
	defer cmd1.Stop()

	cpid1, cpgrp1 := cmd1.Info()

	if cpid1 == ppid {
		t.Fatalf("Parent and child 1 have the same process ID")
	}

	if cpgrp1 == ppgrp {
		t.Fatalf("Parent and child 1 are in the same process group")
	}

	if cpid1 != cpgrp1 {
		t.Fatalf("Child 1's process group is not its process ID")
	}

	cmd2 := create(t)

	cmd2.proc.SysProcAttr = &syscall.SysProcAttr{
		Setpgid: true,
		Pgid:    cpgrp1,
	}
	cmd2.Start()
	defer cmd2.Stop()

	cpid2, cpgrp2 := cmd2.Info()

	if cpid2 == ppid {
		t.Fatalf("Parent and child 2 have the same process ID")
	}

	if cpgrp2 == ppgrp {
		t.Fatalf("Parent and child 2 are in the same process group")
	}

	if cpid2 == cpgrp2 {
		t.Fatalf("Child 2's process group is its process ID")
	}

	if cpid1 == cpid2 {
		t.Fatalf("Child 1 and 2 have the same process ID")
	}

	if cpgrp1 != cpgrp2 {
		t.Fatalf("Child 1 and 2 are not in the same process group")
	}
}

func TestForeground(t *testing.T) {
	signal.Ignore(syscall.SIGTTIN, syscall.SIGTTOU)

	tty, err := os.OpenFile("/dev/tty", os.O_RDWR, 0)
	if err != nil {
		t.Skipf("Can't test Foreground. Couldn't open /dev/tty: %s", err)
	}

	// This should really be pid_t, however _C_int (aka int32) is generally
	// equivalent.
	fpgrp := int32(0)

	errno := syscall.Ioctl(tty.Fd(), syscall.TIOCGPGRP, uintptr(unsafe.Pointer(&fpgrp)))
	if errno != 0 {
		t.Fatalf("TIOCGPGRP failed with error code: %s", errno)
	}

	if fpgrp == 0 {
		t.Fatalf("Foreground process group is zero")
	}

	ppid, ppgrp := parent()

	cmd := create(t)

	cmd.proc.SysProcAttr = &syscall.SysProcAttr{
		Ctty:       int(tty.Fd()),
		Foreground: true,
	}
	cmd.Start()

	cpid, cpgrp := cmd.Info()

	if cpid == ppid {
		t.Fatalf("Parent and child have the same process ID")
	}

	if cpgrp == ppgrp {
		t.Fatalf("Parent and child are in the same process group")
	}

	if cpid != cpgrp {
		t.Fatalf("Child's process group is not the child's process ID")
	}

	cmd.Stop()

	errno = syscall.Ioctl(tty.Fd(), syscall.TIOCSPGRP, uintptr(unsafe.Pointer(&fpgrp)))
	if errno != 0 {
		t.Fatalf("TIOCSPGRP failed with error code: %s", errno)
	}

	signal.Reset()
}

// Test a couple of cases that SysProcAttr can't handle. Issue 29458.
func TestInvalidExec(t *testing.T) {
	t.Parallel()
	t.Run("SetCtty-Foreground", func(t *testing.T) {
		t.Parallel()
		cmd := create(t)
		cmd.proc.SysProcAttr = &syscall.SysProcAttr{
			Setctty:    true,
			Foreground: true,
			Ctty:       0,
		}
		if err := cmd.proc.Start(); err == nil {
			t.Error("expected error setting both SetCtty and Foreground")
		}
	})
	t.Run("invalid-Ctty", func(t *testing.T) {
		t.Parallel()
		cmd := create(t)
		cmd.proc.SysProcAttr = &syscall.SysProcAttr{
			Setctty: true,
			Ctty:    3,
		}
		if err := cmd.proc.Start(); err == nil {
			t.Error("expected error with invalid Ctty value")
		}
	})
}

// TestExec is for issue #41702.
func TestExec(t *testing.T) {
	testenv.MustHaveExec(t)
	cmd := exec.Command(os.Args[0], "-test.run=TestExecHelper")
	cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=2")
	o, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("%s\n%v", o, err)
	}
}

// TestExecHelper is used by TestExec. It does nothing by itself.
// In testing on macOS 10.14, this used to fail with
// "signal: illegal instruction" more than half the time.
func TestExecHelper(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "2" {
		return
	}

	// We don't have to worry about restoring these values.
	// We are in a child process that only runs this test,
	// and we are going to call syscall.Exec anyhow.
	runtime.GOMAXPROCS(50)
	os.Setenv("GO_WANT_HELPER_PROCESS", "3")

	stop := time.Now().Add(time.Second)
	for i := 0; i < 100; i++ {
		go func(i int) {
			r := rand.New(rand.NewSource(int64(i)))
			for time.Now().Before(stop) {
				r.Uint64()
			}
		}(i)
	}

	time.Sleep(10 * time.Millisecond)

	argv := []string{os.Args[0], "-test.run=TestExecHelper"}
	syscall.Exec(os.Args[0], argv, os.Environ())

	t.Error("syscall.Exec returned")
}
