// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package runtime_test

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"syscall"
	"testing"
)

func canGenerateCore(t *testing.T) bool {
	// Ensure there is enough RLIMIT_CORE available to generate a full core.
	var lim syscall.Rlimit
	err := syscall.Getrlimit(syscall.RLIMIT_CORE, &lim)
	if err != nil {
		t.Fatalf("error getting rlimit: %v", err)
	}
	// Minimum RLIMIT_CORE max to allow. This is a conservative estimate.
	// Most systems allow infinity.
	const minRlimitCore = 100 << 20 // 100 MB
	if lim.Max < minRlimitCore {
		t.Skipf("RLIMIT_CORE max too low: %#+v", lim)
	}

	// Make sure core pattern will send core to the current directory.
	b, err := os.ReadFile("/proc/sys/kernel/core_pattern")
	if err != nil {
		t.Fatalf("error reading core_pattern: %v", err)
	}
	if string(b) != "core\n" {
		t.Skipf("Unexpected core pattern %q", string(b))
	}

	coreUsesPID := false
	b, err = os.ReadFile("/proc/sys/kernel/core_uses_pid")
	if err == nil {
		switch string(bytes.TrimSpace(b)) {
		case "0":
		case "1":
			coreUsesPID = true
		default:
			t.Skipf("unexpected core_uses_pid value %q", string(b))
		}
	}
	return coreUsesPID
}

const coreSignalSource = `
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime/debug"
	"syscall"
)

var pipeFD = flag.Int("pipe-fd", -1, "FD of write end of control pipe")

func enableCore() {
	debug.SetTraceback("crash")

	var lim syscall.Rlimit
	err := syscall.Getrlimit(syscall.RLIMIT_CORE, &lim)
	if err != nil {
		panic(fmt.Sprintf("error getting rlimit: %v", err))
	}
	lim.Cur = lim.Max
	fmt.Fprintf(os.Stderr, "Setting RLIMIT_CORE = %+#v\n", lim)
	err = syscall.Setrlimit(syscall.RLIMIT_CORE, &lim)
	if err != nil {
		panic(fmt.Sprintf("error setting rlimit: %v", err))
	}
}

func main() {
	flag.Parse()

	enableCore()

	// Ready to go. Notify parent.
	if err := syscall.Close(*pipeFD); err != nil {
		panic(fmt.Sprintf("error closing control pipe fd %d: %v", *pipeFD, err))
	}

	for {}
}
`

// TestGdbCoreSignalBacktrace tests that gdb can unwind the stack correctly
// through a signal handler in a core file
func TestGdbCoreSignalBacktrace(t *testing.T) {
	if runtime.GOOS != "linux" {
		// N.B. This test isn't fundamentally Linux-only, but it needs
		// to know how to enable/find core files on each OS.
		t.Skip("Test only supported on Linux")
	}
	if runtime.GOARCH != "386" && runtime.GOARCH != "amd64" {
		// TODO(go.dev/issue/25218): Other architectures use sigreturn
		// via VDSO, which we somehow don't handle correctly.
		t.Skip("Backtrace through signal handler only works on 386 and amd64")
	}

	checkGdbEnvironment(t)
	t.Parallel()
	checkGdbVersion(t)

	coreUsesPID := canGenerateCore(t)

	// Build the source code.
	dir := t.TempDir()
	src := filepath.Join(dir, "main.go")
	err := os.WriteFile(src, []byte(coreSignalSource), 0644)
	if err != nil {
		t.Fatalf("failed to create file: %v", err)
	}
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", "a.exe", "main.go")
	cmd.Dir = dir
	out, err := testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("building source %v\n%s", err, out)
	}

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("error creating control pipe: %v", err)
	}
	defer r.Close()

	// Start the test binary.
	cmd = testenv.Command(t, "./a.exe", "-pipe-fd=3")
	cmd.Dir = dir
	cmd.ExtraFiles = []*os.File{w}
	var output bytes.Buffer
	cmd.Stdout = &output // for test logging
	cmd.Stderr = &output

	if err := cmd.Start(); err != nil {
		t.Fatalf("error starting test binary: %v", err)
	}
	w.Close()

	pid := cmd.Process.Pid

	// Wait for child to be ready.
	var buf [1]byte
	if _, err := r.Read(buf[:]); err != io.EOF {
		t.Fatalf("control pipe read get err %v want io.EOF", err)
	}

	// 💥
	if err := cmd.Process.Signal(os.Signal(syscall.SIGABRT)); err != nil {
		t.Fatalf("erroring signaling child: %v", err)
	}

	err = cmd.Wait()
	t.Logf("child output:\n%s", output.String())
	if err == nil {
		t.Fatalf("Wait succeeded, want SIGABRT")
	}
	ee, ok := err.(*exec.ExitError)
	if !ok {
		t.Fatalf("Wait err got %T %v, want exec.ExitError", ee, ee)
	}
	ws, ok := ee.Sys().(syscall.WaitStatus)
	if !ok {
		t.Fatalf("Sys got %T %v, want syscall.WaitStatus", ee.Sys(), ee.Sys())
	}
	if ws.Signal() != syscall.SIGABRT {
		t.Fatalf("Signal got %d want SIGABRT", ws.Signal())
	}
	if !ws.CoreDump() {
		t.Fatalf("CoreDump got %v want true", ws.CoreDump())
	}

	coreFile := "core"
	if coreUsesPID {
		coreFile += fmt.Sprintf(".%d", pid)
	}

	// Execute gdb commands.
	args := []string{"-nx", "-batch",
		"-iex", "add-auto-load-safe-path " + filepath.Join(testenv.GOROOT(t), "src", "runtime"),
		"-ex", "backtrace",
		filepath.Join(dir, "a.exe"),
		filepath.Join(dir, coreFile),
	}
	cmd = testenv.Command(t, "gdb", args...)

	got, err := cmd.CombinedOutput()
	t.Logf("gdb output:\n%s", got)
	if err != nil {
		t.Fatalf("gdb exited with error: %v", err)
	}

	// We don't know which thread the fatal signal will land on, but we can still check for basics:
	//
	// 1. A frame in the signal handler: runtime.sigtramp
	// 2. GDB detection of the signal handler: <signal handler called>
	// 3. A frame before the signal handler: this could be foo, or somewhere in the scheduler

	re := regexp.MustCompile(`#.* runtime\.sigtramp `)
	if found := re.Find(got) != nil; !found {
		t.Fatalf("could not find sigtramp in backtrace")
	}

	re = regexp.MustCompile("#.* <signal handler called>")
	loc := re.FindIndex(got)
	if loc == nil {
		t.Fatalf("could not find signal handler marker in backtrace")
	}
	rest := got[loc[1]:]

	// Look for any frames after the signal handler. We want to see
	// symbolized frames, not garbage unknown frames.
	//
	// Since the signal might not be delivered to the main thread we can't
	// look for main.main. Every thread should have a runtime frame though.
	re = regexp.MustCompile(`#.* runtime\.`)
	if found := re.Find(rest) != nil; !found {
		t.Fatalf("could not find runtime symbol in backtrace after signal handler:\n%s", rest)
	}
}

const coreCrashThreadSource = `
package main

/*
#cgo CFLAGS: -g -O0
#include <stdio.h>
#include <stddef.h>
void trigger_crash()
{
	int* ptr = NULL;
	*ptr = 1024;
}
*/
import "C"
import (
	"flag"
	"fmt"
	"os"
	"runtime/debug"
	"syscall"
)

func enableCore() {
	debug.SetTraceback("crash")

	var lim syscall.Rlimit
	err := syscall.Getrlimit(syscall.RLIMIT_CORE, &lim)
	if err != nil {
		panic(fmt.Sprintf("error getting rlimit: %v", err))
	}
	lim.Cur = lim.Max
	fmt.Fprintf(os.Stderr, "Setting RLIMIT_CORE = %+#v\n", lim)
	err = syscall.Setrlimit(syscall.RLIMIT_CORE, &lim)
	if err != nil {
		panic(fmt.Sprintf("error setting rlimit: %v", err))
	}
}

func main() {
	flag.Parse()

	enableCore()

	C.trigger_crash()
}
`

// TestGdbCoreCrashThreadBacktrace tests that runtime could let the fault thread to crash process
// and make fault thread as number one thread while gdb in a core file
func TestGdbCoreCrashThreadBacktrace(t *testing.T) {
	if runtime.GOOS != "linux" {
		// N.B. This test isn't fundamentally Linux-only, but it needs
		// to know how to enable/find core files on each OS.
		t.Skip("Test only supported on Linux")
	}
	if runtime.GOARCH != "386" && runtime.GOARCH != "amd64" {
		// TODO(go.dev/issue/25218): Other architectures use sigreturn
		// via VDSO, which we somehow don't handle correctly.
		t.Skip("Backtrace through signal handler only works on 386 and amd64")
	}

	testenv.MustHaveCGO(t)
	checkGdbEnvironment(t)
	t.Parallel()
	checkGdbVersion(t)

	coreUsesPID := canGenerateCore(t)

	// Build the source code.
	dir := t.TempDir()
	src := filepath.Join(dir, "main.go")
	err := os.WriteFile(src, []byte(coreCrashThreadSource), 0644)
	if err != nil {
		t.Fatalf("failed to create file: %v", err)
	}
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", "a.exe", "main.go")
	cmd.Dir = dir
	out, err := testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("building source %v\n%s", err, out)
	}

	// Start the test binary.
	cmd = testenv.Command(t, "./a.exe")
	cmd.Dir = dir
	var output bytes.Buffer
	cmd.Stdout = &output // for test logging
	cmd.Stderr = &output

	if err := cmd.Start(); err != nil {
		t.Fatalf("error starting test binary: %v", err)
	}

	pid := cmd.Process.Pid

	err = cmd.Wait()
	t.Logf("child output:\n%s", output.String())
	if err == nil {
		t.Fatalf("Wait succeeded, want SIGABRT")
	}
	ee, ok := err.(*exec.ExitError)
	if !ok {
		t.Fatalf("Wait err got %T %v, want exec.ExitError", ee, ee)
	}
	ws, ok := ee.Sys().(syscall.WaitStatus)
	if !ok {
		t.Fatalf("Sys got %T %v, want syscall.WaitStatus", ee.Sys(), ee.Sys())
	}
	if ws.Signal() != syscall.SIGABRT {
		t.Fatalf("Signal got %d want SIGABRT", ws.Signal())
	}
	if !ws.CoreDump() {
		t.Fatalf("CoreDump got %v want true", ws.CoreDump())
	}

	coreFile := "core"
	if coreUsesPID {
		coreFile += fmt.Sprintf(".%d", pid)
	}

	// Execute gdb commands.
	args := []string{"-nx", "-batch",
		"-iex", "add-auto-load-safe-path " + filepath.Join(testenv.GOROOT(t), "src", "runtime"),
		"-ex", "backtrace",
		filepath.Join(dir, "a.exe"),
		filepath.Join(dir, coreFile),
	}
	cmd = testenv.Command(t, "gdb", args...)

	got, err := cmd.CombinedOutput()
	t.Logf("gdb output:\n%s", got)
	if err != nil {
		t.Fatalf("gdb exited with error: %v", err)
	}

	re := regexp.MustCompile(`#.* trigger_crash`)
	if found := re.Find(got) != nil; !found {
		t.Fatalf("could not find trigger_crash in backtrace")
	}
}
