// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build freebsd || linux

package syscall_test

import (
	"bufio"
	"fmt"
	"internal/testenv"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"os/user"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
)

// TestDeathSignalSetuid verifies that a command run with a different UID still
// receives PDeathsig; it is a regression test for https://go.dev/issue/9686.
func TestDeathSignalSetuid(t *testing.T) {
	if testing.Short() {
		t.Skipf("skipping test that copies its binary into temp dir")
	}

	// Copy the test binary to a location that another user can read/execute
	// after we drop privileges.
	//
	// TODO(bcmills): Why do we believe that another users will be able to
	// execute a binary in this directory? (It could be mounted noexec.)
	tempDir := t.TempDir()
	os.Chmod(tempDir, 0755)

	tmpBinary := filepath.Join(tempDir, filepath.Base(os.Args[0]))

	src, err := os.Open(os.Args[0])
	if err != nil {
		t.Fatalf("cannot open binary %q, %v", os.Args[0], err)
	}
	defer src.Close()

	dst, err := os.OpenFile(tmpBinary, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0755)
	if err != nil {
		t.Fatalf("cannot create temporary binary %q, %v", tmpBinary, err)
	}
	if _, err := io.Copy(dst, src); err != nil {
		t.Fatalf("failed to copy test binary to %q, %v", tmpBinary, err)
	}
	err = dst.Close()
	if err != nil {
		t.Fatalf("failed to close test binary %q, %v", tmpBinary, err)
	}

	cmd := testenv.Command(t, tmpBinary)
	cmd.Env = append(cmd.Environ(), "GO_DEATHSIG_PARENT=1")
	chldStdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatalf("failed to create new stdin pipe: %v", err)
	}
	chldStdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("failed to create new stdout pipe: %v", err)
	}
	stderr := new(strings.Builder)
	cmd.Stderr = stderr

	err = cmd.Start()
	defer func() {
		chldStdin.Close()
		cmd.Wait()
		if stderr.Len() > 0 {
			t.Logf("stderr:\n%s", stderr)
		}
	}()
	if err != nil {
		t.Fatalf("failed to start first child process: %v", err)
	}

	chldPipe := bufio.NewReader(chldStdout)

	if got, err := chldPipe.ReadString('\n'); got == "start\n" {
		syscall.Kill(cmd.Process.Pid, syscall.SIGTERM)

		want := "ok\n"
		if got, err = chldPipe.ReadString('\n'); got != want {
			t.Fatalf("expected %q, received %q, %v", want, got, err)
		}
	} else if got == "skip\n" {
		t.Skipf("skipping: parent could not run child program as selected user")
	} else {
		t.Fatalf("did not receive start from child, received %q, %v", got, err)
	}
}

func deathSignalParent() {
	var (
		u   *user.User
		err error
	)
	if os.Getuid() == 0 {
		tryUsers := []string{"nobody"}
		if testenv.Builder() != "" {
			tryUsers = append(tryUsers, "gopher")
		}
		for _, name := range tryUsers {
			u, err = user.Lookup(name)
			if err == nil {
				break
			}
			fmt.Fprintf(os.Stderr, "Lookup(%q): %v\n", name, err)
		}
	}
	if u == nil {
		// If we couldn't find an unprivileged user to run as, try running as
		// the current user. (Empirically this still causes the call to Start to
		// fail with a permission error if running as a non-root user on Linux.)
		u, err = user.Current()
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}

	uid, err := strconv.ParseUint(u.Uid, 10, 32)
	if err != nil {
		fmt.Fprintf(os.Stderr, "invalid UID: %v\n", err)
		os.Exit(1)
	}
	gid, err := strconv.ParseUint(u.Gid, 10, 32)
	if err != nil {
		fmt.Fprintf(os.Stderr, "invalid GID: %v\n", err)
		os.Exit(1)
	}

	cmd := exec.Command(os.Args[0])
	cmd.Env = append(os.Environ(),
		"GO_DEATHSIG_PARENT=",
		"GO_DEATHSIG_CHILD=1",
	)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	attrs := syscall.SysProcAttr{
		Pdeathsig:  syscall.SIGUSR1,
		Credential: &syscall.Credential{Uid: uint32(uid), Gid: uint32(gid)},
	}
	cmd.SysProcAttr = &attrs

	fmt.Fprintf(os.Stderr, "starting process as user %q\n", u.Username)
	if err := cmd.Start(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		if testenv.SyscallIsNotSupported(err) {
			fmt.Println("skip")
			os.Exit(0)
		}
		os.Exit(1)
	}
	cmd.Wait()
	os.Exit(0)
}

func deathSignalChild() {
	c := make(chan os.Signal, 1)
	signal.Notify(c, syscall.SIGUSR1)
	go func() {
		<-c
		fmt.Println("ok")
		os.Exit(0)
	}()
	fmt.Println("start")

	buf := make([]byte, 32)
	os.Stdin.Read(buf)

	// We expected to be signaled before stdin closed
	fmt.Println("not ok")
	os.Exit(1)
}
