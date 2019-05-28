// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris

package exec_test

import (
	"os/user"
	"runtime"
	"strconv"
	"syscall"
	"testing"
	"time"
)

func TestCredentialNoSetGroups(t *testing.T) {
	if runtime.GOOS == "android" {
		t.Skip("unsupported on Android")
	}

	u, err := user.Current()
	if err != nil {
		t.Fatalf("error getting current user: %v", err)
	}

	uid, err := strconv.Atoi(u.Uid)
	if err != nil {
		t.Fatalf("error converting Uid=%s to integer: %v", u.Uid, err)
	}

	gid, err := strconv.Atoi(u.Gid)
	if err != nil {
		t.Fatalf("error converting Gid=%s to integer: %v", u.Gid, err)
	}

	// If NoSetGroups is true, setgroups isn't called and cmd.Run should succeed
	cmd := helperCommand(t, "echo", "foo")
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Credential: &syscall.Credential{
			Uid:         uint32(uid),
			Gid:         uint32(gid),
			NoSetGroups: true,
		},
	}

	if err = cmd.Run(); err != nil {
		t.Errorf("Failed to run command: %v", err)
	}
}

// For issue #19314: make sure that SIGSTOP does not cause the process
// to appear done.
func TestWaitid(t *testing.T) {
	t.Parallel()

	cmd := helperCommand(t, "sleep")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}

	// The sleeps here are unnecessary in the sense that the test
	// should still pass, but they are useful to make it more
	// likely that we are testing the expected state of the child.
	time.Sleep(100 * time.Millisecond)

	if err := cmd.Process.Signal(syscall.SIGSTOP); err != nil {
		cmd.Process.Kill()
		t.Fatal(err)
	}

	ch := make(chan error)
	go func() {
		ch <- cmd.Wait()
	}()

	time.Sleep(100 * time.Millisecond)

	if err := cmd.Process.Signal(syscall.SIGCONT); err != nil {
		t.Error(err)
		syscall.Kill(cmd.Process.Pid, syscall.SIGCONT)
	}

	cmd.Process.Kill()

	<-ch
}
