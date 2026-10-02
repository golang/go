// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package exec_test

import (
	"bufio"
	"fmt"
	"internal/testenv"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
)

var (
	quitSignal os.Signal = nil
	pipeSignal os.Signal = syscall.SIGPIPE
)

func init() {
	registerHelperCommand("pipehandle", cmdPipeHandle)
	registerHelperCommand("crtpipehandle", cmdCRTPipeHandle)
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

func cmdCRTPipeHandle(args ...string) {
	get_osfhandle := syscall.NewLazyDLL("msvcrt.dll").NewProc("_get_osfhandle")

	h3, _, _ := get_osfhandle.Call(3)
	if h3 == uintptr(syscall.InvalidHandle) {
		fmt.Fprintf(os.Stderr, "_get_osfhandle: pipe 3 is invalid\n")
		os.Exit(1)
	}
	pipe3 := os.NewFile(h3, "in")
	defer pipe3.Close()

	h4, _, _ := get_osfhandle.Call(4)
	if h4 == uintptr(syscall.InvalidHandle) {
		fmt.Fprintf(os.Stderr, "_get_osfhandle: pipe 4 is invalid\n")
		os.Exit(1)
	}
	pipe4 := os.NewFile(h4, "out")
	defer pipe4.Close()

	br := bufio.NewReader(pipe3)
	line, _, err := br.ReadLine()
	if err != nil {
		fmt.Fprintf(os.Stderr, "reading pipe failed: %v\n", err)
		os.Exit(1)
	}
	if string(line) == "ping" {
		_, err := fmt.Fprintf(pipe4, "%s\n", args[0])
		if err != nil {
			fmt.Fprintf(os.Stderr, "writing to pipe failed: %v\n", err)
			os.Exit(1)
		}
	} else {
		fmt.Fprintf(os.Stderr, "unexpected content from pipe: %q\n", line)
		os.Exit(1)
	}
}

func TestCRTPipePassing(t *testing.T) {
	crt := syscall.NewLazyDLL("msvcrt.dll")
	if err := crt.Load(); err != nil {
		t.Skipf("can't run test due to missing msvcrt.dll: %v", err)
	}

	r3, w3, err := os.Pipe()
	if err != nil {
		t.Errorf("failed to create pipe 3: %v", err)
	}
	defer func() {
		r3.Close()
		w3.Close()
	}()

	r4, w4, err := os.Pipe()
	if err != nil {
		t.Errorf("failed to create pipe 4: %v", err)
	}
	defer func() {
		r4.Close()
		w4.Close()
	}()

	const marker = "pong"
	childProc := helperCommand(t, "crtpipehandle", marker)
	childProc.SysProcAttr = &syscall.SysProcAttr{
		AdditionalInheritedHandles: []syscall.Handle{
			syscall.Handle(r3.Fd()),
			syscall.Handle(w4.Fd()),
		},
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		output, err := childProc.CombinedOutput()
		if err != nil {
			t.Errorf("child proc exited: %v. output:\n%s", err, output)
			r3.Close()
			w4.Close()
		}
		wg.Done()
	}()

	_, err = fmt.Fprint(w3, "ping\n")
	if err != nil {
		t.Errorf("writing pipe failed: %v", err)
	}

	br := bufio.NewReader(r4)
	response, _, err := br.ReadLine()
	if err != nil {
		t.Error(err)
	}
	if string(response) != marker {
		t.Errorf("got %q; want %q", string(response), marker)
	}
	wg.Wait()
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
