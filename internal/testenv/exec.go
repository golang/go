// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testenv

import (
	"context"
	"flag"
	"os"
	"os/exec"
	"reflect"
	"runtime"
	"strconv"
	"sync"
	"testing"
	"time"
)

// HasExec reports whether the current system can start new processes
// using os.StartProcess or (more commonly) exec.Command.
func HasExec() bool {
	switch runtime.GOOS {
	case "aix",
		"android",
		"darwin",
		"dragonfly",
		"freebsd",
		"illumos",
		"linux",
		"netbsd",
		"openbsd",
		"plan9",
		"solaris",
		"windows":
		// Known OS that isn't ios or wasm; assume that exec works.
		return true

	case "ios", "js", "wasip1":
		// ios has an exec syscall but on real iOS devices it might return a
		// permission error. In an emulated environment (such as a Corellium host)
		// it might succeed, so try it and find out.
		//
		// As of 2023-04-19 wasip1 and js don't have exec syscalls at all, but we
		// may as well use the same path so that this branch can be tested without
		// an ios environment.
		fallthrough

	default:
		tryExecOnce.Do(func() {
			exe, err := os.Executable()
			if err != nil {
				return
			}
			if flag.Lookup("test.list") == nil {
				// We found the executable, but we don't know how to run it in a way
				// that should succeed without side-effects. Just forget it.
				return
			}
			// We know that a test executable exists and can run, because we're
			// running it now. Use it to check for overall exec support, but be sure
			// to remove any environment variables that might trigger non-default
			// behavior in a custom TestMain.
			cmd := exec.Command(exe, "-test.list=^$")
			cmd.Env = []string{}
			if err := cmd.Run(); err == nil {
				tryExecOk = true
			}
		})
		return tryExecOk
	}
}

var (
	tryExecOnce sync.Once
	tryExecOk   bool
)

// NeedsExec checks that the current system can start new processes
// using os.StartProcess or (more commonly) exec.Command.
// If not, NeedsExec calls t.Skip with an explanation.
func NeedsExec(t testing.TB) {
	if !HasExec() {
		t.Skipf("skipping test: cannot exec subprocess on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
}

// CommandContext is like exec.CommandContext, but:
//   - skips t if the platform does not support os/exec,
//   - if supported, sends SIGQUIT instead of SIGKILL in its Cancel function
//   - if the test has a deadline, adds a Context timeout and (if supported) WaitDelay
//     for an arbitrary grace period before the test's deadline expires,
//   - if Cmd has the Cancel field, fails the test if the command is canceled
//     due to the test's deadline, and
//   - if supported, sets a Cleanup function that verifies that the test did not
//     leak a subprocess.
func CommandContext(t testing.TB, ctx context.Context, name string, args ...string) *exec.Cmd {
	t.Helper()
	NeedsExec(t)

	var (
		cancelCtx   context.CancelFunc
		gracePeriod time.Duration // unlimited unless the test has a deadline (to allow for interactive debugging)
	)

	if td, ok := Deadline(t); ok {
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
		// field), and a second one for the delay becween the process being
		// terminated and and the test logging its output for debugging.
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

	cmd := exec.CommandContext(ctx, name, args...)

	// Use reflection to set the Cancel and WaitDelay fields, if present.
	// TODO(bcmills): When we no longer support Go versions below 1.20,
	// remove the use of reflect and assume that the fields are always present.
	rc := reflect.ValueOf(cmd).Elem()

	if rCancel := rc.FieldByName("Cancel"); rCancel.IsValid() {
		rCancel.Set(reflect.ValueOf(func() error {
			if cancelCtx != nil && ctx.Err() == context.DeadlineExceeded {
				// The command timed out due to running too close to the test's deadline
				// (because we specifically set a shorter Context deadline for that
				// above). There is no way the test did that intentionally — it's too
				// close to the wire! — so mark it as a test failure. That way, if the
				// test expects the command to fail for some other reason, it doesn't
				// have to distinguish between that reason and a timeout.
				t.Errorf("test timed out while running command: %v", cmd)
			} else {
				// The command is being terminated due to ctx being canceled, but
				// apparently not due to an explicit test deadline that we added.
				// Log that information in case it is useful for diagnosing a failure,
				// but don't actually fail the test because of it.
				t.Logf("%v: terminating command: %v", ctx.Err(), cmd)
			}
			return cmd.Process.Signal(Sigquit)
		}))
	}

	if rWaitDelay := rc.FieldByName("WaitDelay"); rWaitDelay.IsValid() {
		rWaitDelay.Set(reflect.ValueOf(gracePeriod))
	}

	// t.Cleanup was added in Go 1.14; for earlier Go versions,
	// we just let the Context leak.
	type Cleanupper interface {
		Cleanup(func())
	}
	if ct, ok := t.(Cleanupper); ok {
		ct.Cleanup(func() {
			if cancelCtx != nil {
				cancelCtx()
			}
			if cmd.Process != nil && cmd.ProcessState == nil {
				t.Errorf("command was started, but test did not wait for it to complete: %v", cmd)
			}
		})
	}

	return cmd
}

// Command is like exec.Command, but applies the same changes as
// testenv.CommandContext (with a default Context).
func Command(t testing.TB, name string, args ...string) *exec.Cmd {
	t.Helper()
	return CommandContext(t, context.Background(), name, args...)
}
