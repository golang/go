// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris

package syscall_test

import (
	"internal/testenv"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"syscall"
	"testing"
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

	fpgrp := 0

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
