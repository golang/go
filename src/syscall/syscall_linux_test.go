// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall_test

import (
	"bufio"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

func TestMain(m *testing.M) {
	if os.Getenv("GO_DEATHSIG_PARENT") == "1" {
		deathSignalParent()
	} else if os.Getenv("GO_DEATHSIG_CHILD") == "1" {
		deathSignalChild()
	}

	os.Exit(m.Run())
}

func TestLinuxDeathSignal(t *testing.T) {
	if os.Getuid() != 0 {
		t.Skip("skipping root only test")
	}

	// Copy the test binary to a location that a non-root user can read/execute
	// after we drop privileges
	tempDir, err := ioutil.TempDir("", "TestDeathSignal")
	if err != nil {
		t.Fatalf("cannot create temporary directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
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

	cmd := exec.Command(tmpBinary)
	cmd.Env = []string{"GO_DEATHSIG_PARENT=1"}
	chldStdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal("failed to create new stdin pipe: %v", err)
	}
	chldStdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal("failed to create new stdout pipe: %v", err)
	}
	cmd.Stderr = os.Stderr

	err = cmd.Start()
	defer cmd.Wait()
	if err != nil {
		t.Fatalf("failed to start first child process: %v", err)
	}

	chldPipe := bufio.NewReader(chldStdout)

	if got, err := chldPipe.ReadString('\n'); got == "start\n" {
		syscall.Kill(cmd.Process.Pid, syscall.SIGTERM)

		go func() {
			time.Sleep(5 * time.Second)
			chldStdin.Close()
		}()

		want := "ok\n"
		if got, err = chldPipe.ReadString('\n'); got != want {
			t.Fatalf("expected %q, received %q, %v", want, got, err)
		}
	} else {
		t.Fatalf("did not receive start from child, received %q, %v", got, err)
	}
}

func deathSignalParent() {
	cmd := exec.Command(os.Args[0])
	cmd.Env = []string{"GO_DEATHSIG_CHILD=1"}
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	attrs := syscall.SysProcAttr{
		Pdeathsig: syscall.SIGUSR1,
		// UID/GID 99 is the user/group "nobody" on RHEL/Fedora and is
		// unused on Ubuntu
		Credential: &syscall.Credential{Uid: 99, Gid: 99},
	}
	cmd.SysProcAttr = &attrs

	err := cmd.Start()
	if err != nil {
		fmt.Fprintf(os.Stderr, "death signal parent error: %v\n")
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
