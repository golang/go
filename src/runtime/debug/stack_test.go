// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package debug_test

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	. "runtime/debug"
	"strings"
	"testing"
)

func TestMain(m *testing.M) {
	switch os.Getenv("GO_RUNTIME_DEBUG_TEST_ENTRYPOINT") {
	case "dumpgoroot":
		fmt.Println(runtime.GOROOT())
		os.Exit(0)

	case "setcrashoutput":
		f, err := os.Create(os.Getenv("CRASHOUTPUT"))
		if err != nil {
			log.Fatal(err)
		}
		if err := SetCrashOutput(f); err != nil {
			log.Fatal(err) // e.g. EMFILE
		}
		println("hello")
		panic("oops")
	}

	// default: run the tests.
	os.Exit(m.Run())
}

type T int

func (t *T) ptrmethod() []byte {
	return Stack()
}
func (t T) method() []byte {
	return t.ptrmethod()
}

/*
The traceback should look something like this, modulo line numbers and hex constants.
Don't worry much about the base levels, but check the ones in our own package.

	goroutine 10 [running]:
	runtime/debug.Stack(0x0, 0x0, 0x0)
		/Users/r/go/src/runtime/debug/stack.go:28 +0x80
	runtime/debug.(*T).ptrmethod(0xc82005ee70, 0x0, 0x0, 0x0)
		/Users/r/go/src/runtime/debug/stack_test.go:15 +0x29
	runtime/debug.T.method(0x0, 0x0, 0x0, 0x0)
		/Users/r/go/src/runtime/debug/stack_test.go:18 +0x32
	runtime/debug.TestStack(0xc8201ce000)
		/Users/r/go/src/runtime/debug/stack_test.go:37 +0x38
	testing.tRunner(0xc8201ce000, 0x664b58)
		/Users/r/go/src/testing/testing.go:456 +0x98
	created by testing.RunTests
		/Users/r/go/src/testing/testing.go:561 +0x86d
*/
func TestStack(t *testing.T) {
	b := T(0).method()
	lines := strings.Split(string(b), "\n")
	if len(lines) < 6 {
		t.Fatal("too few lines")
	}

	// If built with -trimpath, file locations should start with package paths.
	// Otherwise, file locations should start with a GOROOT/src prefix
	// (for whatever value of GOROOT is baked into the binary, not the one
	// that may be set in the environment).
	fileGoroot := ""
	if envGoroot := os.Getenv("GOROOT"); envGoroot != "" {
		// Since GOROOT is set explicitly in the environment, we can't be certain
		// that it is the same GOROOT value baked into the binary, and we can't
		// change the value in-process because runtime.GOROOT uses the value from
		// initial (not current) environment. Spawn a subprocess to determine the
		// real baked-in GOROOT.
		t.Logf("found GOROOT %q from environment; checking embedded GOROOT value", envGoroot)
		testenv.MustHaveExec(t)
		exe, err := os.Executable()
		if err != nil {
			t.Fatal(err)
		}
		cmd := exec.Command(exe)
		cmd.Env = append(os.Environ(), "GOROOT=", "GO_RUNTIME_DEBUG_TEST_ENTRYPOINT=dumpgoroot")
		out, err := cmd.Output()
		if err != nil {
			t.Fatal(err)
		}
		fileGoroot = string(bytes.TrimSpace(out))
	} else {
		// Since GOROOT is not set in the environment, its value (if any) must come
		// from the path embedded in the binary.
		fileGoroot = runtime.GOROOT()
	}
	filePrefix := ""
	if fileGoroot != "" {
		fileGoroot = filepath.ToSlash(filepath.Clean(fileGoroot))
		if fileGoroot == "/" {
			filePrefix = "/src/"
		} else {
			filePrefix = fileGoroot + "/src/"
		}
	}

	n := 0
	frame := func(file, code string) {
		t.Helper()

		line := lines[n]
		if !strings.Contains(line, code) {
			t.Errorf("expected %q in %q", code, line)
		}
		n++

		line = lines[n]

		wantPrefix := "\t" + filePrefix + file
		if !strings.HasPrefix(line, wantPrefix) {
			t.Errorf("in line %q, expected prefix %q", line, wantPrefix)
		}
		n++
	}
	n++

	frame("runtime/debug/stack.go", "runtime/debug.Stack")
	frame("runtime/debug/stack_test.go", "runtime/debug_test.(*T).ptrmethod")
	frame("runtime/debug/stack_test.go", "runtime/debug_test.T.method")
	frame("runtime/debug/stack_test.go", "runtime/debug_test.TestStack")
	frame("testing/testing.go", "")
}

func TestSetCrashOutput(t *testing.T) {
	testenv.MustHaveExec(t)
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}

	crashOutput := filepath.Join(t.TempDir(), "crash.out")

	cmd := exec.Command(exe)
	cmd.Stderr = new(strings.Builder)
	cmd.Env = append(os.Environ(), "GO_RUNTIME_DEBUG_TEST_ENTRYPOINT=setcrashoutput", "CRASHOUTPUT="+crashOutput)
	err = cmd.Run()
	stderr := fmt.Sprint(cmd.Stderr)
	if err == nil {
		t.Fatalf("child process succeeded unexpectedly (stderr: %s)", stderr)
	}
	t.Logf("child process finished with error %v and stderr <<%s>>", err, stderr)

	// Read the file the child process should have written.
	// It should contain a crash report such as this:
	//
	// panic: oops
	//
	// goroutine 1 [running]:
	// runtime/debug_test.TestMain(0x1400007e0a0)
	// 	GOROOT/src/runtime/debug/stack_test.go:33 +0x18c
	// main.main()
	// 	_testmain.go:71 +0x170
	data, err := os.ReadFile(crashOutput)
	if err != nil {
		t.Fatalf("child process failed to write crash report: %v", err)
	}
	crash := string(data)
	t.Logf("crash = <<%s>>", crash)
	t.Logf("stderr = <<%s>>", stderr)

	// Check that the crash file and the stderr both contain the panic and stack trace.
	for _, want := range []string{
		"panic: oops",
		"goroutine 1",
		"debug_test.TestMain",
	} {
		if !strings.Contains(crash, want) {
			t.Errorf("crash output does not contain %q", want)
		}
		if !strings.Contains(stderr, want) {
			t.Errorf("stderr output does not contain %q", want)
		}
	}

	// Check that stderr, but not crash, contains the output of println().
	printlnOnly := "hello"
	if strings.Contains(crash, printlnOnly) {
		t.Errorf("crash output contains %q, but should not", printlnOnly)
	}
	if !strings.Contains(stderr, printlnOnly) {
		t.Errorf("stderr output does not contain %q, but should", printlnOnly)
	}
}
