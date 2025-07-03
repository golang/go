// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package synctest_test

import (
	"fmt"
	"internal/testenv"
	"os"
	"regexp"
	"testing"
	"testing/synctest"
)

// Tests for interactions between synctest bubbles and the testing package.
// Other bubble behaviors are tested in internal/synctest.

func TestSuccess(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
	})
}

func TestFatal(t *testing.T) {
	runTest(t, nil, func() {
		synctest.Test(t, func(t *testing.T) {
			t.Fatal("fatal")
		})
	}, `^--- FAIL: TestFatal.*
    synctest_test.go:.* fatal
FAIL
$`)
}

func TestError(t *testing.T) {
	runTest(t, nil, func() {
		synctest.Test(t, func(t *testing.T) {
			t.Error("error")
		})
	}, `^--- FAIL: TestError.*
    synctest_test.go:.* error
FAIL
$`)
}

func TestVerboseError(t *testing.T) {
	runTest(t, []string{"-test.v"}, func() {
		synctest.Test(t, func(t *testing.T) {
			t.Error("error")
		})
	}, `^=== RUN   TestVerboseError
    synctest_test.go:.* error
--- FAIL: TestVerboseError.*
FAIL
$`)
}

func TestSkip(t *testing.T) {
	runTest(t, nil, func() {
		synctest.Test(t, func(t *testing.T) {
			t.Skip("skip")
		})
	}, `^PASS
$`)
}

func TestVerboseSkip(t *testing.T) {
	runTest(t, []string{"-test.v"}, func() {
		synctest.Test(t, func(t *testing.T) {
			t.Skip("skip")
		})
	}, `^=== RUN   TestVerboseSkip
    synctest_test.go:.* skip
--- PASS: TestVerboseSkip.*
PASS
$`)
}

func TestCleanup(t *testing.T) {
	done := false
	synctest.Test(t, func(t *testing.T) {
		ch := make(chan struct{})
		t.Cleanup(func() {
			// This cleanup function should execute inside the test's bubble.
			// (If it doesn't the runtime will panic.)
			close(ch)
		})
		// synctest.Test will wait for this goroutine to exit before returning.
		// The cleanup function signals the goroutine to exit before the wait starts.
		go func() {
			<-ch
			done = true
		}()
	})
	if !done {
		t.Fatalf("background goroutine did not return")
	}
}

func TestContext(t *testing.T) {
	state := "not started"
	synctest.Test(t, func(t *testing.T) {
		go func() {
			state = "waiting on context"
			<-t.Context().Done()
			state = "done"
		}()
		// Wait blocks until the goroutine above is blocked on t.Context().Done().
		synctest.Wait()
		if got, want := state, "waiting on context"; got != want {
			t.Fatalf("state = %q, want %q", got, want)
		}
	})
	// t.Context() is canceled before the test completes,
	// and synctest.Test does not return until the goroutine has set its state to "done".
	if got, want := state, "done"; got != want {
		t.Fatalf("state = %q, want %q", got, want)
	}
}

func TestDeadline(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		defer wantPanic(t, "testing: t.Deadline called inside synctest bubble")
		_, _ = t.Deadline()
	})
}

func TestParallel(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		defer wantPanic(t, "testing: t.Parallel called inside synctest bubble")
		t.Parallel()
	})
}

func TestRun(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		defer wantPanic(t, "testing: t.Run called inside synctest bubble")
		t.Run("subtest", func(t *testing.T) {
		})
	})
}

func TestHelper(t *testing.T) {
	runTest(t, []string{"-test.v"}, func() {
		synctest.Test(t, func(t *testing.T) {
			helperLog(t, "log in helper")
		})
	}, `^=== RUN   TestHelper
    synctest_test.go:.* log in helper
--- PASS: TestHelper.*
PASS
$`)
}

func wantPanic(t *testing.T, want string) {
	if e := recover(); e != nil {
		if got := fmt.Sprint(e); got != want {
			t.Errorf("got panic message %q, want %q", got, want)
		}
	} else {
		t.Errorf("got no panic, want one")
	}
}

func runTest(t *testing.T, args []string, f func(), pattern string) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		f()
		return
	}
	t.Helper()
	re := regexp.MustCompile(pattern)
	testenv.MustHaveExec(t)
	cmd := testenv.Command(t, testenv.Executable(t), "-test.run=^"+regexp.QuoteMeta(t.Name())+"$", "-test.count=1")
	cmd.Args = append(cmd.Args, args...)
	cmd = testenv.CleanCmdEnv(cmd)
	cmd.Env = append(cmd.Env, "GO_WANT_HELPER_PROCESS=1")
	out, _ := cmd.CombinedOutput()
	if !re.Match(out) {
		t.Errorf("got output:\n%s\nwant matching:\n%s", out, pattern)
	}
}
