// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package main_test

import (
	"bufio"
	"context"
	"internal/testenv"
	"io"
	"os"
	"os/exec"
	"slices"
	"strings"
	"syscall"
	"testing"
)

func TestGoBuildUmask(t *testing.T) {
	// Do not use tg.parallel; avoid other tests seeing umask manipulation.
	mask := syscall.Umask(0077) // prohibit low bits
	defer syscall.Umask(mask)

	tg := testgo(t)
	defer tg.cleanup()
	tg.tempFile("x.go", `package main; func main() {}`)

	// We have set a umask, but if the parent directory happens to have a default
	// ACL, the umask may be ignored. To prevent spurious failures from an ACL,
	// we compare the file created by "go build" against a file written explicitly
	// by os.WriteFile.
	//
	// (See https://go.dev/issue/62724, https://go.dev/issue/17909.)
	control := tg.path("control")
	tg.creatingTemp(control)
	if err := os.WriteFile(control, []byte("#!/bin/sh\nexit 0"), 0777); err != nil {
		t.Fatal(err)
	}
	cfi, err := os.Stat(control)
	if err != nil {
		t.Fatal(err)
	}

	exe := tg.path("x")
	tg.creatingTemp(exe)
	tg.run("build", "-o", exe, tg.path("x.go"))
	fi, err := os.Stat(exe)
	if err != nil {
		t.Fatal(err)
	}
	got, want := fi.Mode(), cfi.Mode()
	if got == want {
		t.Logf("wrote x with mode %v", got)
	} else {
		t.Fatalf("wrote x with mode %v, wanted no 0077 bits (%v)", got, want)
	}
}

// TestTestInterrupt verifies the fix for issue #60203.
//
// If the whole process group for a 'go test' invocation receives
// SIGINT (as would be sent by pressing ^C on a console),
// it should return quickly, not deadlock.
func TestTestInterrupt(t *testing.T) {
	if testing.Short() {
		t.Skipf("skipping in short mode: test executes many subprocesses")
	}
	// Don't run this test in parallel, for the same reason.

	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOROOT", testGOROOT)

	ctx, cancel := context.WithCancel(context.Background())
	cmd := testenv.CommandContext(t, ctx, tg.goTool(), "test", "std", "-short", "-count=1")
	cmd.Dir = tg.execDir

	// Override $TMPDIR when running the tests: since we're terminating the tests
	// with a signal they might fail to clean up some temp files, and we don't
	// want that to cause an "unexpected files" failure at the end of the run.
	cmd.Env = append(slices.Clip(tg.env), tempEnvName()+"="+t.TempDir())

	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setpgid: true,
	}
	cmd.Cancel = func() error {
		pgid := cmd.Process.Pid
		return syscall.Kill(-pgid, syscall.SIGINT)
	}

	pipe, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("running %v", cmd)
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}

	stdout := new(strings.Builder)
	r := bufio.NewReader(pipe)
	line, err := r.ReadString('\n')
	if err != nil {
		t.Fatal(err)
	}
	stdout.WriteString(line)

	// The output line for some test was written, so we know things are in progress.
	//
	// Cancel the rest of the run by sending SIGINT to the process group:
	// it should finish up and exit with a nonzero status,
	// not have to be killed with SIGKILL.
	cancel()

	io.Copy(stdout, r)
	if stdout.Len() > 0 {
		t.Logf("stdout:\n%s", stdout)
	}
	err = cmd.Wait()

	ee, _ := err.(*exec.ExitError)
	if ee == nil {
		t.Fatalf("unexpectedly finished with nonzero status")
	}
	if len(ee.Stderr) > 0 {
		t.Logf("stderr:\n%s", ee.Stderr)
	}
	if !ee.Exited() {
		t.Fatalf("'go test' did not exit after interrupt: %v", err)
	}

	t.Logf("interrupted tests without deadlocking")
}
