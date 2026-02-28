// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"internal/testenv"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"testing"
	"time"
)

func TestProcessLiteral(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Process literals do not work on Windows. FindProcess/etc must initialize the process handle")
	}
	if runtime.GOARCH == "wasm" {
		t.Skip("Signals send + notify not fully supported om wasm port")
	}

	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)
	defer signal.Stop(c)

	p := &os.Process{Pid: os.Getpid()}
	if err := p.Signal(os.Interrupt); err != nil {
		t.Fatalf("Signal got err %v, want nil", err)
	}

	// Verify we actually received the signal.
	select {
	case <-time.After(1 * time.Second):
		t.Error("timeout waiting for signal")
	case <-c:
		// Good
	}
}

func TestProcessReleaseTwice(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("Pipe() got err %v, want nil", err)
	}
	defer r.Close()
	defer w.Close()

	p, err := os.StartProcess(testenv.GoToolPath(t), []string{"go"}, &os.ProcAttr{
		// N.B. On Windows, StartProcess requires exactly 3 Files. Pass
		// in a dummy pipe to avoid irrelevant output on the test stdout.
		Files: []*os.File{r, w, w},
	})
	if err != nil {
		t.Fatalf("starting test process: %v", err)
	}
	if err := p.Release(); err != nil {
		t.Fatalf("first Release: got err %v, want nil", err)
	}

	err = p.Release()

	// We want EINVAL from a second Release call only on Windows.
	var want error
	if runtime.GOOS == "windows" {
		want = syscall.EINVAL
	}

	if err != want {
		t.Fatalf("second Release: got err %v, want %v", err, want)
	}
}
