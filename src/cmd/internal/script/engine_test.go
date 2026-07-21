// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package script

import (
	"context"
	"fmt"
	"internal/testenv"
	"io"
	"os"
	"os/exec"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestInterruptCmd(t *testing.T) {
	if runtime.GOOS == "js" || runtime.GOOS == "wasip1" {
		t.Skip(runtime.GOOS + " does not support executables")
	}

	const msg = "Hello world\n"

	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		fmt.Printf(msg)
		time.Sleep(24 * time.Hour) // sleep forever
		os.Exit(3)
	}

	exe := testenv.Executable(t)
	cmd := testenv.CommandContext(t, t.Context(), exe, "-test.run=^TestInterruptCmd$")
	cmd = testenv.CleanCmdEnv(cmd)
	cmd.Env = append(cmd.Env, "GO_WANT_HELPER_PROCESS=1")

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}

	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}

	buf := make([]byte, len(msg))
	n, err := io.ReadFull(stdout, buf)
	if n != len(buf) || err != nil || string(buf) != msg {
		t.Fatalf("ReadFull = %d, %v, %q", n, err, buf[:n])
	}

	interruptCmdErr := make(chan error)
	go func() {
		interruptCmdErr <- InterruptCmd(cmd)
	}()

	err = cmd.Wait()
	if err == nil {
		t.Fatal("expected Wait failure")
	} else if err, ok := err.(*exec.ExitError); ok {
		checkInterruptCmdError(t, err)
	} else {
		t.Fatalf("unexpected error while running executable: %s\n%s", err, string(buf))
	}

	err = <-interruptCmdErr
	if err != nil {
		t.Errorf("InterruptCmd failed: %v", err)
	}
}

// checkInterruptCmdError calls t.Fatal if err is interrupt cmd error.
func checkInterruptCmdError(t *testing.T, err error) {
	errstr := err.Error()
	if runtime.GOOS == "plan9" {
		expectedPrefixError := "exit status: "
		if !strings.HasPrefix(errstr, expectedPrefixError) {
			t.Fatalf("unexpected error prefix while exiting executable: got=%q, want=%q", errstr, expectedPrefixError)
		}
		expectedSuffixError := ": interrupt'"
		if !strings.HasSuffix(errstr, expectedSuffixError) {
			t.Fatalf("unexpected error suffix while exiting executable: got=%q, want=%q", errstr, expectedSuffixError)
		}
		return
	}

	expectedError := "signal: interrupt"
	if runtime.GOOS == "windows" {
		expectedError = "exit status 1"
	}
	if errstr != expectedError {
		t.Fatalf("unexpected error while exiting executable: got=%q, want=%q", errstr, expectedError)
	}
}

func FuzzQuoteArgs(f *testing.F) {
	state, err := NewState(context.Background(), f.TempDir(), nil /* env */)
	if err != nil {
		f.Fatalf("failed to create state: %v", err)
	}

	f.Add("foo")
	f.Add(`"foo"`)
	f.Add(`'foo'`)
	f.Fuzz(func(t *testing.T, s string) {
		give := []string{s}
		quoted := quoteArgs(give)
		cmd, err := parse("file.txt", 42, "cmd "+quoted)
		if err != nil {
			t.Fatalf("quoteArgs(%q) = %q cannot be parsed: %v", give, quoted, err)
		}
		args := expandArgs(state, cmd.rawArgs, nil /* regexpArgs */)

		if !slices.Equal(give, args) {
			t.Fatalf("quoteArgs failed to round-trip.\ninput:\n\t%#q\nquoted:\n\t%q\nparsed:\n\t%#q", give, quoted, args)
		}
	})
}
