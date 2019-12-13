// +build windows

package runtime_test

import (
	"bufio"
	"bytes"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
)

func TestVectoredHandlerDontCrashOnLibrary(t *testing.T) {
	if *flagQuick {
		t.Skip("-quick")
	}
	if runtime.GOARCH != "amd64" {
		t.Skip("this test can only run on windows/amd64")
	}
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveExecPath(t, "gcc")
	testprog.Lock()
	defer testprog.Unlock()
	dir, err := ioutil.TempDir("", "go-build")
	if err != nil {
		t.Fatalf("failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(dir)

	// build go dll
	dll := filepath.Join(dir, "testwinlib.dll")
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", dll, "--buildmode", "c-shared", "testdata/testwinlib/main.go")
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
	expectedOutput := "exceptionCount: 1\ncontinueCount: 1\n"
	// cleaning output
	cleanedOut := strings.ReplaceAll(string(out), "\r\n", "\n")
	if cleanedOut != expectedOutput {
		t.Errorf("expected output %q, got %q", expectedOutput, cleanedOut)
	}
}

func sendCtrlBreak(t *testing.T, pid int) {
	kernel32, err := syscall.LoadDLL("kernel32.dll")
	if err != nil {
		t.Fatalf("LoadDLL: %v\n", err)
	}
	generateEvent, err := kernel32.FindProc("GenerateConsoleCtrlEvent")
	if err != nil {
		t.Fatalf("FindProc: %v\n", err)
	}
	result, _, err := generateEvent.Call(syscall.CTRL_BREAK_EVENT, uintptr(pid))
	if result == 0 {
		t.Fatalf("GenerateConsoleCtrlEvent: %v\n", err)
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
	testenv.MustHaveExecPath(t, "gcc")
	testprog.Lock()
	defer testprog.Unlock()
	dir, err := ioutil.TempDir("", "go-build")
	if err != nil {
		t.Fatalf("failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(dir)

	// build go dll
	dll := filepath.Join(dir, "dummy.dll")
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", dll, "--buildmode", "c-shared", "testdata/testwinlibsignal/dummy.go")
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

	sentCtrl := make(chan bool)
	go func() {
		defer close(sentCtrl)
		if line, err := outReader.ReadString('\n'); err != nil {
			t.Fatalf("Could not read stdout: %v", err)
		} else if strings.TrimSpace(line) != "ready" {
			t.Fatalf("Unexpected message: %v", line)
		}
		sendCtrlBreak(t, cmd.Process.Pid)
	}()

	<-sentCtrl
	if err := cmd.Wait(); err != nil {
		t.Fatalf("Program exited with error: %v\n%s", err, &stderr)
	}
}
