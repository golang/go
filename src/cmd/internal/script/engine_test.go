// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package script

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"internal/testenv"
	"io"
	"os"
	"os/exec"
	"runtime"
	"slices"
	"strings"
	"testing"
	"testing/synctest"
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

func TestCommandErrorDuration(t *testing.T) {
	errFail := errors.New("FAIL")
	// fail1 fails after 1 second in its run func.
	fail1 := Command(CmdUsage{Async: true}, func(*State, ...string) (WaitFunc, error) {
		time.Sleep(1 * time.Second)
		return nil, errFail
	})
	// fail2 fails after 2 seconds, with 1.5s spent in its WaitFunc.
	fail2 := Command(CmdUsage{Async: true}, func(*State, ...string) (WaitFunc, error) {
		time.Sleep(500 * time.Millisecond)
		failTime := time.Now().Add(1500 * time.Millisecond)
		return func(*State) (string, string, error) {
			time.Sleep(time.Until(failTime))
			return "", "", errFail
		}, nil
	})
	e := &Engine{Cmds: map[string]Cmd{"fail1": fail1, "fail2": fail2, "sleep": Sleep(), "wait": Wait()}}

	tests := []struct {
		name   string
		script []string
		want   string
	}{
		// We report durations for failing foreground commands.
		{"fail 1s", []string{"fail1 x"}, "t.txt:1: fail1 x (took 1.000s): FAIL"},
		{"fail 2s", []string{"fail2 x"}, "t.txt:1: fail2 x (took 2.000s): FAIL"},
		// We don't know the duration of a background command.
		{"background 1s", []string{"fail1 x &"}, "t.txt:1: fail1 x: FAIL"},
		// A wait error reports how long it waited for a failing background command.
		// This might be zero wait time if the command finished before wait was called.
		{"wait 1.5s", []string{"fail2 x &", "wait"}, "t.txt:2: wait (took 1.500s): t.txt:1: fail2 x: FAIL"},
		{"wait 0s", []string{"fail2 x &", "sleep 10s", "wait"}, "t.txt:3: wait (took 0.000s): t.txt:1: fail2 x: FAIL"},
		// Background commands that have no WaitFunc don't actually wait.
		{"wait no WaitFunc", []string{"fail1 x &", "wait"}, "t.txt:1: fail1 x: FAIL"},
		// Sleeping longer than the 10m context timeout reports a duration.
		{"sleep 1h", []string{"sleep 1h"}, "t.txt:1: sleep 1h (took 600.000s): context deadline exceeded"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ctx, cancel := context.WithTimeout(t.Context(), 10*time.Minute)
				defer cancel()
				s, err := NewState(ctx, t.TempDir(), nil)
				if err != nil {
					t.Fatal(err)
				}
				script := strings.Join(tt.script, "\n")
				err = e.Execute(s, "t.txt", bufio.NewReader(strings.NewReader(script)), io.Discard)
				if err == nil {
					t.Fatalf("got nil error, want %q", tt.want)
				}
				if err.Error() != tt.want {
					t.Fatalf("\n got %q\nwant %q", err, tt.want)
				}
				if err, ok := errors.AsType[*CommandError](err); ok && err.Background && err.Duration != 0 {
					t.Fatalf("CommandError for background command has Duration %v, want 0", err.Duration)
				}
			})
		})
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
