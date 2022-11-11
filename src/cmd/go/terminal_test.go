// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"errors"
	"internal/testenv"
	"internal/testpty"
	"io"
	"os"
	"testing"

	"golang.org/x/term"
)

func TestTerminalPassthrough(t *testing.T) {
	// Check that if 'go test' is run with a terminal connected to stdin/stdout,
	// then the go command passes that terminal down to the test binary
	// invocation (rather than, e.g., putting a pipe in the way).
	//
	// See issue 18153.
	testenv.MustHaveGoBuild(t)

	// Start with a "self test" to make sure that if we *don't* pass in a
	// terminal, the test can correctly detect that. (cmd/go doesn't guarantee
	// that it won't add a terminal in the middle, but that would be pretty weird.)
	t.Run("pipe", func(t *testing.T) {
		r, w, err := os.Pipe()
		if err != nil {
			t.Fatalf("pipe failed: %s", err)
		}
		defer r.Close()
		defer w.Close()
		stdout, stderr := runTerminalPassthrough(t, r, w)
		if stdout {
			t.Errorf("stdout is unexpectedly a terminal")
		}
		if stderr {
			t.Errorf("stderr is unexpectedly a terminal")
		}
	})

	// Now test with a read PTY.
	t.Run("pty", func(t *testing.T) {
		r, processTTY, err := testpty.Open()
		if errors.Is(err, testpty.ErrNotSupported) {
			t.Skipf("%s", err)
		} else if err != nil {
			t.Fatalf("failed to open test PTY: %s", err)
		}
		defer r.Close()
		w, err := os.OpenFile(processTTY, os.O_RDWR, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer w.Close()
		stdout, stderr := runTerminalPassthrough(t, r, w)
		if !stdout {
			t.Errorf("stdout is not a terminal")
		}
		if !stderr {
			t.Errorf("stderr is not a terminal")
		}
	})
}

func runTerminalPassthrough(t *testing.T, r, w *os.File) (stdout, stderr bool) {
	cmd := testenv.Command(t, testGo, "test", "-run=^$")
	cmd.Env = append(cmd.Environ(), "GO_TEST_TERMINAL_PASSTHROUGH=1")
	cmd.Stdout = w
	cmd.Stderr = w
	t.Logf("running %s", cmd)
	err := cmd.Start()
	if err != nil {
		t.Fatalf("starting subprocess: %s", err)
	}
	w.Close()
	// Read the subprocess output. The behavior of reading from a PTY after the
	// child closes it is very strange (e.g., on Linux, read returns EIO), so we
	// ignore errors as long as we get everything we need. We still try to read
	// all of the output so we can report it in case of failure.
	buf, err := io.ReadAll(r)
	if len(buf) != 2 || !(buf[0] == '1' || buf[0] == 'X') || !(buf[1] == '2' || buf[1] == 'X') {
		t.Errorf("expected exactly 2 bytes matching [1X][2X]")
		if err != nil {
			// An EIO here might be expected depending on OS.
			t.Errorf("error reading from subprocess: %s", err)
		}
	}
	err = cmd.Wait()
	if err != nil {
		t.Errorf("suprocess failed with: %s", err)
	}
	if t.Failed() {
		t.Logf("subprocess output:\n%s", string(buf))
		t.FailNow()
	}
	return buf[0] == '1', buf[1] == '2'
}

func init() {
	if os.Getenv("GO_TEST_TERMINAL_PASSTHROUGH") == "" {
		return
	}

	if term.IsTerminal(1) {
		print("1")
	} else {
		print("X")
	}
	if term.IsTerminal(2) {
		print("2")
	} else {
		print("X")
	}
	os.Exit(0)
}
