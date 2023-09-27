// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package debug_test

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	. "runtime/debug"
	"strings"
	"testing"
)

func TestMain(m *testing.M) {
	if os.Getenv("GO_RUNTIME_DEBUG_TEST_DUMP_GOROOT") != "" {
		fmt.Println(runtime.GOROOT())
		os.Exit(0)
	}
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
		cmd.Env = append(os.Environ(), "GOROOT=", "GO_RUNTIME_DEBUG_TEST_DUMP_GOROOT=1")
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
		filePrefix = filepath.ToSlash(fileGoroot) + "/src/"
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
