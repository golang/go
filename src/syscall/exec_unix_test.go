// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package syscall_test

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"io"
	"math/rand"
	"os"
	"os/exec"
	"os/signal"
	"strconv"
	"syscall"
	"testing"
	"time"
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
	defer signal.Reset()

	tty, err := os.OpenFile("/dev/tty", os.O_RDWR, 0)
	if err != nil {
		t.Skipf("Can't test Foreground. Couldn't open /dev/tty: %s", err)
	}
	defer tty.Close()

	ttyFD := int(tty.Fd())

	fpgrp, err := syscall.Tcgetpgrp(ttyFD)
	if err != nil {
		t.Fatalf("Tcgetpgrp failed: %v", err)
	}
	if fpgrp == 0 {
		t.Fatalf("Foreground process group is zero")
	}

	ppid, ppgrp := parent()

	cmd := create(t)

	cmd.proc.SysProcAttr = &syscall.SysProcAttr{
		Ctty:       ttyFD,
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

	// This call fails on darwin/arm64. The failure doesn't matter, though.
	// This is just best effort.
	syscall.Tcsetpgrp(ttyFD, fpgrp)
}

func TestForegroundSignal(t *testing.T) {
	tty, err := os.OpenFile("/dev/tty", os.O_RDWR, 0)
	if err != nil {
		t.Skipf("couldn't open /dev/tty: %s", err)
	}
	defer tty.Close()

	ttyFD := int(tty.Fd())

	fpgrp, err := syscall.Tcgetpgrp(ttyFD)
	if err != nil {
		t.Fatalf("Tcgetpgrp failed: %v", err)
	}
	if fpgrp == 0 {
		t.Fatalf("Foreground process group is zero")
	}

	defer func() {
		signal.Ignore(syscall.SIGTTIN, syscall.SIGTTOU)
		syscall.Tcsetpgrp(ttyFD, fpgrp)
		signal.Reset()
	}()

	ch1 := make(chan os.Signal, 1)
	ch2 := make(chan bool)

	signal.Notify(ch1, syscall.SIGTTIN, syscall.SIGTTOU)
	defer signal.Stop(ch1)

	cmd := create(t)

	go func() {
		cmd.proc.SysProcAttr = &syscall.SysProcAttr{
			Ctty:       ttyFD,
			Foreground: true,
		}
		cmd.Start()
		cmd.Stop()
		close(ch2)
	}()

	timer := time.NewTimer(30 * time.Second)
	defer timer.Stop()
	for {
		select {
		case sig := <-ch1:
			t.Errorf("unexpected signal %v", sig)
		case <-ch2:
			// Success.
			return
		case <-timer.C:
			t.Fatal("timed out waiting for child process")
		}
	}
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
	cmd := exec.Command(os.Args[0], "-test.run=^TestExecHelper$")
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

	argv := []string{os.Args[0], "-test.run=^TestExecHelper$"}
	syscall.Exec(os.Args[0], argv, os.Environ())

	t.Error("syscall.Exec returned")
}

// Test that rlimit values are restored by exec.
func TestRlimitRestored(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "" {
		fmt.Println(syscall.OrigRlimitNofile().Cur)
		os.Exit(0)
	}

	orig := syscall.OrigRlimitNofile()
	if orig == nil {
		t.Skip("skipping test because rlimit not adjusted at startup")
	}

	exe := testenv.Executable(t)
	cmd := testenv.Command(t, exe, "-test.run=^TestRlimitRestored$")
	cmd = testenv.CleanCmdEnv(cmd)
	cmd.Env = append(cmd.Env, "GO_WANT_HELPER_PROCESS=1")

	out, err := cmd.CombinedOutput()
	if len(out) > 0 {
		t.Logf("%s", out)
	}
	if err != nil {
		t.Fatalf("subprocess failed: %v", err)
	}
	s := string(bytes.TrimSpace(out))
	v, err := strconv.ParseUint(s, 10, 64)
	if err != nil {
		t.Fatalf("could not parse %q as number: %v", s, v)
	}

	if v != uint64(orig.Cur) {
		t.Errorf("exec rlimit = %d, want %d", v, orig)
	}
}

func TestForkExecNilArgv(t *testing.T) {
	defer func() {
		if p := recover(); p != nil {
			t.Fatal("forkExec panicked")
		}
	}()

	// We don't really care what the result of forkExec is, just that it doesn't
	// panic, so we choose something we know won't actually spawn a process (probably).
	syscall.ForkExec("/dev/null", nil, nil)
}
