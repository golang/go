// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"errors"
	"internal/testenv"
	"os"
	"syscall"
	"testing"
)

func TestFindProcessViaPidfd(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	if err := os.CheckPidfdOnce(); err != nil {
		// Non-pidfd code paths tested in exec_unix_test.go.
		t.Skipf("skipping: pidfd not available: %v", err)
	}

	p, err := os.StartProcess(testenv.GoToolPath(t), []string{"go"}, &os.ProcAttr{})
	if err != nil {
		t.Fatalf("starting test process: %v", err)
	}
	p.Wait()

	// Use pid of a non-existing process.
	proc, err := os.FindProcess(p.Pid)
	// FindProcess should never return errors on Unix.
	if err != nil {
		t.Fatalf("FindProcess: got error %v, want <nil>", err)
	}
	// FindProcess should never return nil Process.
	if proc == nil {
		t.Fatal("FindProcess: got nil, want non-nil")
	}
	if proc.Status() != os.StatusDone {
		t.Fatalf("got process status: %v, want %d", proc.Status(), os.StatusDone)
	}

	// Check that all Process' public methods work as expected with
	// "done" Process.
	if err := proc.Kill(); err != os.ErrProcessDone {
		t.Errorf("Kill: got %v, want %v", err, os.ErrProcessDone)
	}
	if err := proc.Signal(os.Kill); err != os.ErrProcessDone {
		t.Errorf("Signal: got %v, want %v", err, os.ErrProcessDone)
	}
	if _, err := proc.Wait(); !errors.Is(err, syscall.ECHILD) {
		t.Errorf("Wait: got %v, want %v", err, os.ErrProcessDone)
	}
	// Release never returns errors on Unix.
	if err := proc.Release(); err != nil {
		t.Fatalf("Release: got %v, want <nil>", err)
	}
}
