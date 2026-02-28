// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"bufio"
	"bytes"
	"fmt"
	"internal/testenv"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
)

func TestVectoredHandlerExceptionInNonGoThread(t *testing.T) {
	if *flagQuick {
		t.Skip("-quick")
	}
	if strings.HasPrefix(testenv.Builder(), "windows-amd64-2012") {
		testenv.SkipFlaky(t, 49681)
	}
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)
	testenv.MustHaveExecPath(t, "gcc")
	testprog.Lock()
	defer testprog.Unlock()
	dir := t.TempDir()

	// build c program
	dll := filepath.Join(dir, "veh.dll")
	cmd := exec.Command("gcc", "-shared", "-o", dll, "testdata/testwinlibthrow/veh.c")
	out, err := testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("failed to build c exe: %s\n%s", err, out)
	}

	// build go exe
	exe := filepath.Join(dir, "test.exe")
	cmd = exec.Command(testenv.GoToolPath(t), "build", "-o", exe, "testdata/testwinlibthrow/main.go")
	out, err = testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("failed to build go library: %s\n%s", err, out)
	}

	// run test program in same thread
	cmd = exec.Command(exe)
	out, err = testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err == nil {
		t.Fatal("error expected")
	}
	if _, ok := err.(*exec.ExitError); ok && len(out) > 0 {
		if !bytes.Contains(out, []byte("Exception 0x2a")) {
			t.Fatalf("unexpected failure while running executable: %s\n%s", err, out)
		}
	} else {
		t.Fatalf("unexpected error while running executable: %s\n%s", err, out)
	}
	// run test program in a new thread
	cmd = exec.Command(exe, "thread")
	out, err = testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err == nil {
		t.Fatal("error expected")
	}
	if err, ok := err.(*exec.ExitError); ok {
		if err.ExitCode() != 42 {
			t.Fatalf("unexpected failure while running executable: %s\n%s", err, out)
		}
	} else {
		t.Fatalf("unexpected error while running executable: %s\n%s", err, out)
	}
}

func TestVectoredHandlerDontCrashOnLibrary(t *testing.T) {
	if *flagQuick {
		t.Skip("-quick")
	}

	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)
	testenv.MustHaveExecPath(t, "gcc")
	testprog.Lock()
	defer testprog.Unlock()
	dir := t.TempDir()

	// build go dll
	dll := filepath.Join(dir, "testwinlib.dll")
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", dll, "-buildmode", "c-shared", "testdata/testwinlib/main.go")
	out, err := testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("failed to build go library: %s\n%s", err, out)
	}

	// build c program
	exe := filepath.Join(dir, "test.exe")
	cmd = exec.Command("gcc", "-L"+dir, "-I"+dir, "-ltestwinlib", "-o", exe, "testdata/testwinlib/main.c")
	out, err = testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("failed to build c exe: %s\n%s", err, out)
	}

	// run test program
	cmd = exec.Command(exe)
	out, err = testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("failure while running executable: %s\n%s", err, out)
	}
	var expectedOutput string
	if runtime.GOARCH == "arm64" {
		// TODO: remove when windows/arm64 support SEH stack unwinding.
		expectedOutput = "exceptionCount: 1\ncontinueCount: 1\nunhandledCount: 0\n"
	} else {
		expectedOutput = "exceptionCount: 1\ncontinueCount: 1\nunhandledCount: 1\n"
	}
	// cleaning output
	cleanedOut := strings.ReplaceAll(string(out), "\r\n", "\n")
	if cleanedOut != expectedOutput {
		t.Errorf("expected output %q, got %q", expectedOutput, cleanedOut)
	}
}

func sendCtrlBreak(pid int) error {
	kernel32, err := syscall.LoadDLL("kernel32.dll")
	if err != nil {
		return fmt.Errorf("LoadDLL: %v\n", err)
	}
	generateEvent, err := kernel32.FindProc("GenerateConsoleCtrlEvent")
	if err != nil {
		return fmt.Errorf("FindProc: %v\n", err)
	}
	result, _, err := generateEvent.Call(syscall.CTRL_BREAK_EVENT, uintptr(pid))
	if result == 0 {
		return fmt.Errorf("GenerateConsoleCtrlEvent: %v\n", err)
	}
	return nil
}

// TestCtrlHandler tests that Go can gracefully handle closing the console window.
// See https://golang.org/issues/41884.
func TestCtrlHandler(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	// build go program
	exe := filepath.Join(t.TempDir(), "test.exe")
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", exe, "testdata/testwinsignal/main.go")
	out, err := testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("failed to build go exe: %v\n%s", err, out)
	}

	// run test program
	cmd = exec.Command(exe)
	var stdout strings.Builder
	var stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	inPipe, err := cmd.StdinPipe()
	if err != nil {
		t.Fatalf("Failed to create stdin pipe: %v", err)
	}
	// keep inPipe alive until the end of the test
	defer inPipe.Close()

	// in a new command window
	const _CREATE_NEW_CONSOLE = 0x00000010
	cmd.SysProcAttr = &syscall.SysProcAttr{
		CreationFlags: _CREATE_NEW_CONSOLE,
		HideWindow:    true,
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	defer func() {
		cmd.Process.Kill()
		cmd.Wait()
	}()

	// check child exited gracefully, did not timeout
	if err := cmd.Wait(); err != nil {
		t.Fatalf("Program exited with error: %v\n%s", err, &stderr)
	}

	// check child received, handled SIGTERM
	if expected, got := syscall.SIGTERM.String(), strings.TrimSpace(stdout.String()); expected != got {
		t.Fatalf("Expected '%s' got: %s", expected, got)
	}
}

// TestLibraryCtrlHandler tests that Go DLL allows calling program to handle console control events.
// See https://golang.org/issues/35965.
func TestLibraryCtrlHandler(t *testing.T) {
	if *flagQuick {
		t.Skip("-quick")
	}
	if runtime.GOARCH != "amd64" {
		t.Skip("this test can only run on windows/amd64")
	}
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)
	testenv.MustHaveExecPath(t, "gcc")
	testprog.Lock()
	defer testprog.Unlock()
	dir := t.TempDir()

	// build go dll
	dll := filepath.Join(dir, "dummy.dll")
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", dll, "-buildmode", "c-shared", "testdata/testwinlibsignal/dummy.go")
	out, err := testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("failed to build go library: %s\n%s", err, out)
	}

	// build c program
	exe := filepath.Join(dir, "test.exe")
	cmd = exec.Command("gcc", "-o", exe, "testdata/testwinlibsignal/main.c")
	out, err = testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("failed to build c exe: %s\n%s", err, out)
	}

	// run test program
	cmd = exec.Command(exe)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	outPipe, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("Failed to create stdout pipe: %v", err)
	}
	outReader := bufio.NewReader(outPipe)

	cmd.SysProcAttr = &syscall.SysProcAttr{
		CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP,
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}

	errCh := make(chan error, 1)
	go func() {
		if line, err := outReader.ReadString('\n'); err != nil {
			errCh <- fmt.Errorf("could not read stdout: %v", err)
		} else if strings.TrimSpace(line) != "ready" {
			errCh <- fmt.Errorf("unexpected message: %v", line)
		} else {
			errCh <- sendCtrlBreak(cmd.Process.Pid)
		}
	}()

	if err := <-errCh; err != nil {
		t.Fatal(err)
	}
	if err := cmd.Wait(); err != nil {
		t.Fatalf("Program exited with error: %v\n%s", err, &stderr)
	}
}

func TestIssue59213(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("skipping windows only test")
	}
	if *flagQuick {
		t.Skip("-quick")
	}
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)

	goEnv := func(arg string) string {
		cmd := testenv.Command(t, testenv.GoToolPath(t), "env", arg)
		cmd.Stderr = new(bytes.Buffer)

		line, err := cmd.Output()
		if err != nil {
			t.Fatalf("%v: %v\n%s", cmd, err, cmd.Stderr)
		}
		out := string(bytes.TrimSpace(line))
		t.Logf("%v: %q", cmd, out)
		return out
	}

	cc := goEnv("CC")
	cgoCflags := goEnv("CGO_CFLAGS")

	t.Parallel()

	tmpdir := t.TempDir()
	dllfile := filepath.Join(tmpdir, "test.dll")
	exefile := filepath.Join(tmpdir, "gotest.exe")

	// build go dll
	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-o", dllfile, "-buildmode", "c-shared", "testdata/testwintls/main.go")
	out, err := testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("failed to build go library: %s\n%s", err, out)
	}

	// build c program
	cmd = testenv.Command(t, cc, "-o", exefile, "testdata/testwintls/main.c")
	testenv.CleanCmdEnv(cmd)
	cmd.Env = append(cmd.Env, "CGO_CFLAGS="+cgoCflags)
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("failed to build c exe: %s\n%s", err, out)
	}

	// run test program
	cmd = testenv.Command(t, exefile, dllfile, "GoFunc")
	out, err = testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("failed: %s\n%s", err, out)
	}
}
