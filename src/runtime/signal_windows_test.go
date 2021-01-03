// +build windows

package runtime_test

import (
	"bufio"
	"bytes"
	"fmt"
	"internal/testenv"
	"io/ioutil"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"time"
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
	dir, err := os.MkdirTemp("", "go-build")
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
	if *flagQuick {
		t.Skip("-quick")
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

	// build go program
	exe := filepath.Join(dir, "test.exe")
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", exe, "testdata/testwinsignal/main.go")
	out, err := testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("failed to build go exe: %v\n%s", err, out)
	}

	// udp socket for synchronization
	conn, err := net.ListenPacket("udp", "[::1]:0")
	if err != nil {
		t.Fatalf("ListenPacket failed: %v", err)
	}
	defer conn.Close()
	conn.SetDeadline(time.Now().Add(5 * time.Second))

	// run test program
	cmd = exec.Command("cmd.exe", "/c", "start", exe, conn.LocalAddr().String())
	if err := cmd.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}

	// read pid
	var data [512]byte
	n, _, err := conn.ReadFrom(data[:])
	if err != nil {
		t.Fatalf("ReadFrom failed: %v", err)
	}

	// gracefully kill pid
	err = exec.Command("taskkill.exe", "/pid", string(data[:n])).Run()
	if err != nil {
		t.Fatalf("failed to kill: %v", err)
	}

	// check received signal
	n, _, err = conn.ReadFrom(data[:])
	if err != nil {
		t.Fatalf("ReadFrom failed: %v", err)
	}
	expected := syscall.SIGTERM.String()
	if n != len(expected) && string(data[:len(expected)]) != expected {
		t.Fatalf("Expected '%s' got: %s", expected, data[:n])
	}

	if err := cmd.Wait(); err != nil {
		t.Fatalf("Program exited with error: %v", err)
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
	dir, err := os.MkdirTemp("", "go-build")
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
