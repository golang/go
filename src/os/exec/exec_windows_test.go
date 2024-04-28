// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package exec_test

import (
	"fmt"
	"internal/testenv"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
)

var (
	quitSignal os.Signal = nil
	pipeSignal os.Signal = syscall.SIGPIPE
)

func init() {
	registerHelperCommand("pipehandle", cmdPipeHandle)
}

func cmdPipeHandle(args ...string) {
	handle, _ := strconv.ParseUint(args[0], 16, 64)
	pipe := os.NewFile(uintptr(handle), "")
	_, err := fmt.Fprint(pipe, args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "writing to pipe failed: %v\n", err)
		os.Exit(1)
	}
	pipe.Close()
}

func TestPipePassing(t *testing.T) {
	t.Parallel()

	r, w, err := os.Pipe()
	if err != nil {
		t.Error(err)
	}
	const marker = "arrakis, dune, desert planet"
	childProc := helperCommand(t, "pipehandle", strconv.FormatUint(uint64(w.Fd()), 16), marker)
	childProc.SysProcAttr = &syscall.SysProcAttr{AdditionalInheritedHandles: []syscall.Handle{syscall.Handle(w.Fd())}}
	err = childProc.Start()
	if err != nil {
		t.Error(err)
	}
	w.Close()
	response, err := io.ReadAll(r)
	if err != nil {
		t.Error(err)
	}
	r.Close()
	if string(response) != marker {
		t.Errorf("got %q; want %q", string(response), marker)
	}
	err = childProc.Wait()
	if err != nil {
		t.Error(err)
	}
}

func TestNoInheritHandles(t *testing.T) {
	t.Parallel()

	cmd := testenv.Command(t, "cmd", "/c exit 88")
	cmd.SysProcAttr = &syscall.SysProcAttr{NoInheritHandles: true}
	err := cmd.Run()
	exitError, ok := err.(*exec.ExitError)
	if !ok {
		t.Fatalf("got error %v; want ExitError", err)
	}
	if exitError.ExitCode() != 88 {
		t.Fatalf("got exit code %d; want 88", exitError.ExitCode())
	}
}

// start a child process without the user code explicitly starting
// with a copy of the parent's SYSTEMROOT.
// (See issue 25210.)
func TestChildCriticalEnv(t *testing.T) {
	t.Parallel()
	cmd := helperCommand(t, "echoenv", "SYSTEMROOT")

	// Explicitly remove SYSTEMROOT from the command's environment.
	var env []string
	for _, kv := range cmd.Environ() {
		k, _, ok := strings.Cut(kv, "=")
		if !ok || !strings.EqualFold(k, "SYSTEMROOT") {
			env = append(env, kv)
		}
	}
	cmd.Env = env

	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(string(out)) == "" {
		t.Error("no SYSTEMROOT found")
	}
}

func TestIssue66586(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	path := filepath.Join(testenv.GOROOT(t), "bin", "go")
	cmd := exec.Cmd{Path: path, Args: []string{path, "version"}}
	err := cmd.Run()
	if err != nil {
		t.Fatal(err)
	}
	if path != cmd.Path {
		t.Fatalf("unexpected path: %s", cmd.Path)
	}
}
