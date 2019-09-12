// +build windows

package runtime_test

import (
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
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
