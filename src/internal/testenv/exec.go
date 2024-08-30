// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testenv

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// MustHaveExec checks that the current system can start new processes
// using os.StartProcess or (more commonly) exec.Command.
// If not, MustHaveExec calls t.Skip with an explanation.
//
// On some platforms MustHaveExec checks for exec support by re-executing the
// current executable, which must be a binary built by 'go test'.
// We intentionally do not provide a HasExec function because of the risk of
// inappropriate recursion in TestMain functions.
//
// To check for exec support outside of a test, just try to exec the command.
// If exec is not supported, testenv.SyscallIsNotSupported will return true
// for the resulting error.
func MustHaveExec(t testing.TB) {
	if err := tryExec(); err != nil {
		t.Helper()
		t.Skipf("skipping test: cannot exec subprocess on %s/%s: %v", runtime.GOOS, runtime.GOARCH, err)
	}
}

var tryExec = sync.OnceValue(func() error {
	switch runtime.GOOS {
	case "wasip1", "js", "ios":
	default:
		// Assume that exec always works on non-mobile platforms and Android.
		return nil
	}

	// ios has an exec syscall but on real iOS devices it might return a
	// permission error. In an emulated environment (such as a Corellium host)
	// it might succeed, so if we need to exec we'll just have to try it and
	// find out.
	//
	// As of 2023-04-19 wasip1 and js don't have exec syscalls at all, but we
	// may as well use the same path so that this branch can be tested without
	// an ios environment.

	if !testing.Testing() {
		// This isn't a standard 'go test' binary, so we don't know how to
		// self-exec in a way that should succeed without side effects.
		// Just forget it.
		return errors.New("can't probe for exec support with a non-test executable")
	}

	// We know that this is a test executable. We should be able to run it with a
	// no-op flag to check for overall exec support.
	exe, err := exePath()
	if err != nil {
		return fmt.Errorf("can't probe for exec support: %w", err)
	}
	cmd := exec.Command(exe, "-test.list=^$")
	cmd.Env = origEnv
	return cmd.Run()
})

// Executable is a wrapper around [MustHaveExec] and [os.Executable].
// It returns the path name for the executable that started the current process,
// or skips the test if the current system can't start new processes,
// or fails the test if the path can not be obtained.
func Executable(t testing.TB) string {
	MustHaveExec(t)

	exe, err := exePath()
	if err != nil {
		t.Fatalf("os.Executable error: %v", err)
	}
	return exe
}

var exePath = sync.OnceValues(func() (string, error) {
	return os.Executable()
})

var execPaths sync.Map // path -> error

// MustHaveExecPath checks that the current system can start the named executable
// using os.StartProcess or (more commonly) exec.Command.
// If not, MustHaveExecPath calls t.Skip with an explanation.
func MustHaveExecPath(t testing.TB, path string) {
	MustHaveExec(t)

	err, found := execPaths.Load(path)
	if !found {
		_, err = exec.LookPath(path)
		err, _ = execPaths.LoadOrStore(path, err)
	}
	if err != nil {
		t.Helper()
		t.Skipf("skipping test: %s: %s", path, err)
	}
}

// CleanCmdEnv will fill cmd.Env with the environment, excluding certain
// variables that could modify the behavior of the Go tools such as
// GODEBUG and GOTRACEBACK.
//
// If the caller wants to set cmd.Dir, set it before calling this function,
// so PWD will be set correctly in the environment.
func CleanCmdEnv(cmd *exec.Cmd) *exec.Cmd {
	if cmd.Env != nil {
		panic("environment already set")
	}
	for _, env := range cmd.Environ() {
		// Exclude GODEBUG from the environment to prevent its output
		// from breaking tests that are trying to parse other command output.
		if strings.HasPrefix(env, "GODEBUG=") {
			continue
		}
		// Exclude GOTRACEBACK for the same reason.
		if strings.HasPrefix(env, "GOTRACEBACK=") {
			continue
		}
		cmd.Env = append(cmd.Env, env)
	}
	return cmd
}

// CommandContext is like exec.CommandContext, but:
//   - skips t if the platform does not support os/exec,
//   - sends SIGQUIT (if supported by the platform) instead of SIGKILL
//     in its Cancel function
//   - if the test has a deadline, adds a Context timeout and WaitDelay
//     for an arbitrary grace period before the test's deadline expires,
//   - fails the test if the command does not complete before the test's deadline, and
//   - sets a Cleanup function that verifies that the test did not leak a subprocess.
func CommandContext(t testing.TB, ctx context.Context, name string, args ...string) *exec.Cmd {
	t.Helper()
	MustHaveExec(t)

	var (
		cancelCtx   context.CancelFunc
		gracePeriod time.Duration // unlimited unless the test has a deadline (to allow for interactive debugging)
	)

	if t, ok := t.(interface {
		testing.TB
		Deadline() (time.Time, bool)
	}); ok {
		if td, ok := t.Deadline(); ok {
			// Start with a minimum grace period, just long enough to consume the
			// output of a reasonable program after it terminates.
			gracePeriod = 100 * time.Millisecond
			if s := os.Getenv("GO_TEST_TIMEOUT_SCALE"); s != "" {
				scale, err := strconv.Atoi(s)
				if err != nil {
					t.Fatalf("invalid GO_TEST_TIMEOUT_SCALE: %v", err)
				}
				gracePeriod *= time.Duration(scale)
			}

			// If time allows, increase the termination grace period to 5% of the
			// test's remaining time.
			testTimeout := time.Until(td)
			if gp := testTimeout / 20; gp > gracePeriod {
				gracePeriod = gp
			}

			// When we run commands that execute subprocesses, we want to reserve two
			// grace periods to clean up: one for the delay between the first
			// termination signal being sent (via the Cancel callback when the Context
			// expires) and the process being forcibly terminated (via the WaitDelay
			// field), and a second one for the delay between the process being
			// terminated and the test logging its output for debugging.
			//
			// (We want to ensure that the test process itself has enough time to
			// log the output before it is also terminated.)
			cmdTimeout := testTimeout - 2*gracePeriod

			if cd, ok := ctx.Deadline(); !ok || time.Until(cd) > cmdTimeout {
				// Either ctx doesn't have a deadline, or its deadline would expire
				// after (or too close before) the test has already timed out.
				// Add a shorter timeout so that the test will produce useful output.
				ctx, cancelCtx = context.WithTimeout(ctx, cmdTimeout)
			}
		}
	}

	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Cancel = func() error {
		if cancelCtx != nil && ctx.Err() == context.DeadlineExceeded {
			// The command timed out due to running too close to the test's deadline.
			// There is no way the test did that intentionally — it's too close to the
			// wire! — so mark it as a test failure. That way, if the test expects the
			// command to fail for some other reason, it doesn't have to distinguish
			// between that reason and a timeout.
			t.Errorf("test timed out while running command: %v", cmd)
		} else {
			// The command is being terminated due to ctx being canceled, but
			// apparently not due to an explicit test deadline that we added.
			// Log that information in case it is useful for diagnosing a failure,
			// but don't actually fail the test because of it.
			t.Logf("%v: terminating command: %v", ctx.Err(), cmd)
		}
		return cmd.Process.Signal(Sigquit)
	}
	cmd.WaitDelay = gracePeriod

	t.Cleanup(func() {
		if cancelCtx != nil {
			cancelCtx()
		}
		if cmd.Process != nil && cmd.ProcessState == nil {
			t.Errorf("command was started, but test did not wait for it to complete: %v", cmd)
		}
	})

	return cmd
}

// Command is like exec.Command, but applies the same changes as
// testenv.CommandContext (with a default Context).
func Command(t testing.TB, name string, args ...string) *exec.Cmd {
	t.Helper()
	return CommandContext(t, context.Background(), name, args...)
}
