// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"bufio"
	"bytes"
	"errors"
	"internal/testenv"
	"io"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"testing"
)

// TestSignalPid1 verifies that a Go program running as PID 1 with no
// SIGTERM handler provides a sane exit code upon receiving SIGTERM.
//
// The test is Linux-specific because it uses CLONE_NEWPID to run as PID 1.
func TestSignalPid1(t *testing.T) {
	t.Parallel()

	exe, err := buildTestProg(t, "testprog")
	if err != nil {
		t.Fatal(err)
	}

	cmd := testenv.Command(t, exe, "SignalPid1")
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Cloneflags: syscall.CLONE_NEWPID | syscall.CLONE_NEWUSER,
		UidMappings: []syscall.SysProcIDMap{
			{ContainerID: 0, HostID: os.Getuid(), Size: 1},
		},
		GidMappings: []syscall.SysProcIDMap{
			{ContainerID: 0, HostID: os.Getgid(), Size: 1},
		},
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		t.Skipf("cannot create PID namespace (may require unprivileged user namespaces): %v", err)
	}

	waited := false
	defer func() {
		if !waited {
			cmd.Process.Kill()
			cmd.Wait()
		}
	}()

	// Wait for child to signal readiness.
	r := bufio.NewReader(stdout)
	line, err := r.ReadString('\n')
	if err != nil {
		t.Fatalf("reading from child: %v", err)
	}
	if strings.TrimRight(line, "\n") != "ready" {
		t.Fatalf("unexpected output from child: %q", line)
	}
	go io.Copy(io.Discard, r) // Drain any further output.

	const (
		signal      = syscall.SIGTERM
		expExitCode = int(128 + signal)
	)
	// Send signal from outside the child PID namespace.
	if err := cmd.Process.Signal(signal); err != nil {
		t.Fatalf("sending signal %d (%q): %v", signal, signal, err)
	}

	err = cmd.Wait()
	waited = true
	t.Logf("child: %v", err)
	if s := stderr.String(); s != "" {
		t.Fatalf("child stderr: %s", s)
	}
	if exitErr, ok := errors.AsType[*exec.ExitError](err); ok {
		if status, ok := exitErr.Sys().(syscall.WaitStatus); ok {
			if ec := status.ExitStatus(); ec == expExitCode {
				return // PASS.
			}
		}
	}

	t.Errorf("Want child exited with %d, got: %+v", expExitCode, err)
}
